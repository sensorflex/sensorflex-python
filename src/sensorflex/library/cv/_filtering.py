"""A library for sensor data filtering."""

import time

import numpy as np
from numpy.typing import NDArray

from sensorflex import Node, Port


def rotation_mat_to_quat_wxyz(R: NDArray[np.float32]) -> NDArray[np.float32]:
    """Convert 3x3 rotation matrix to quaternion [w,x,y,z]."""
    # Robust-ish branch method
    m00, m01, m02 = float(R[0, 0]), float(R[0, 1]), float(R[0, 2])
    m10, m11, m12 = float(R[1, 0]), float(R[1, 1]), float(R[1, 2])
    m20, m21, m22 = float(R[2, 0]), float(R[2, 1]), float(R[2, 2])

    tr = m00 + m11 + m22
    if tr > 0.0:
        S = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * S
        x = (m21 - m12) / S
        y = (m02 - m20) / S
        z = (m10 - m01) / S
    elif (m00 > m11) and (m00 > m22):
        S = np.sqrt(1.0 + m00 - m11 - m22) * 2.0
        w = (m21 - m12) / S
        x = 0.25 * S
        y = (m01 + m10) / S
        z = (m02 + m20) / S
    elif m11 > m22:
        S = np.sqrt(1.0 + m11 - m00 - m22) * 2.0
        w = (m02 - m20) / S
        x = (m01 + m10) / S
        y = 0.25 * S
        z = (m12 + m21) / S
    else:
        S = np.sqrt(1.0 + m22 - m00 - m11) * 2.0
        w = (m10 - m01) / S
        x = (m02 + m20) / S
        y = (m12 + m21) / S
        z = 0.25 * S

    q = np.array([w, x, y, z], dtype=np.float32)
    n = float(np.linalg.norm(q))
    if n < 1e-8:
        return np.array([1, 0, 0, 0], dtype=np.float32)
    return (q / n).astype(np.float32)


def quat_wxyz_to_rotation_mat(q: NDArray[np.float32]) -> NDArray[np.float32]:
    """Convert quaternion [w,x,y,z] to 3x3 rotation matrix."""
    q = q.astype(np.float32)
    n = float(np.linalg.norm(q))
    if n < 1e-8:
        return np.eye(3, dtype=np.float32)
    q = q / n
    w, x, y, z = map(float, q)

    ww, xx, yy, zz = w * w, x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    R = np.array(
        [
            [ww + xx - yy - zz, 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), ww - xx + yy - zz, 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), ww - xx - yy + zz],
        ],
        dtype=np.float32,
    )
    return R


def ensure_shortest_quat(
    q_meas: NDArray[np.float32], q_ref: NDArray[np.float32]
) -> NDArray[np.float32]:
    """Flip sign to avoid q and -q discontinuity (choose closest to reference)."""
    if float(np.dot(q_meas, q_ref)) < 0.0:
        return (-q_meas).astype(np.float32)
    return q_meas.astype(np.float32)


class PoseKalmanFilterNode(Node):
    """
    Kalman filter for SE(3) pose (4x4) using:
      - Position KF: state [p(3), v(3)]
      - Quaternion KF: state [q(4), qdot(4)] with renormalization

    Input:
      i_pose4x4: (4,4) float32, camera->face (or whatever your upstream produces)
                Can be None or invalid -> predict only.

    Output:
      o_pose4x4: (4,4) float32 filtered pose
    """

    i_pose4x4: Port[NDArray]
    o_pose4x4: Port[NDArray]

    def __init__(
        self,
        # Position tuning
        pos_meas_std: float = 0.02,  # meters (measurement noise)
        pos_accel_std: float = 1.0,  # m/s^2 (process noise)
        # Rotation tuning
        rot_meas_std: float = 0.05,  # "quaternion component" noise (pragmatic)
        rot_accel_std: float = 1.0,  # qddot noise (pragmatic)
        dt_min: float = 1e-3,
        dt_max: float = 0.1,
        name: str | None = None,
    ) -> None:
        super().__init__(name)
        self.i_pose4x4 = Port(None)
        self.o_pose4x4 = Port(np.eye(4, dtype=np.float32))

        self._pos_meas_var = float(pos_meas_std * pos_meas_std)
        self._pos_accel_var = float(pos_accel_std * pos_accel_std)

        self._rot_meas_var = float(rot_meas_std * rot_meas_std)
        self._rot_accel_var = float(rot_accel_std * rot_accel_std)

        self._dt_min = float(dt_min)
        self._dt_max = float(dt_max)

        # Position KF: x = [p(3), v(3)]
        self._xp = np.zeros((6, 1), dtype=np.float32)
        self._Pp = np.eye(6, dtype=np.float32) * 1.0

        # Quaternion KF: x = [q(4), qdot(4)]
        self._xq = np.zeros((8, 1), dtype=np.float32)
        self._xq[0, 0] = 1.0
        self._Pq = np.eye(8, dtype=np.float32) * 1.0

        self._last_t = None
        self._last_q = np.array([1, 0, 0, 0], dtype=np.float32)

    def _compute_dt(self) -> float:
        t = time.perf_counter()
        if self._last_t is None:
            self._last_t = t
            return 1.0 / 30.0
        dt = float(t - self._last_t)
        self._last_t = t
        dt = max(self._dt_min, min(self._dt_max, dt))
        return dt

    def _kf_predict(
        self,
        x: NDArray[np.float32],
        P: NDArray[np.float32],
        F: NDArray[np.float32],
        Q: NDArray[np.float32],
    ):
        x = F @ x
        P = (F @ P @ F.T + Q).astype(np.float32)
        return x, P

    def _kf_update(
        self,
        x: NDArray[np.float32],
        P: NDArray[np.float32],
        H: NDArray[np.float32],
        R: NDArray[np.float32],
        z: NDArray[np.float32],
    ):
        y = z - (H @ x)
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        x = (x + K @ y).astype(np.float32)
        I = np.eye(P.shape[0], dtype=np.float32)
        P = ((I - K @ H) @ P).astype(np.float32)
        return x, P

    def forward(self) -> None:
        dt = self._compute_dt()

        # -------------------------
        # Predict (position)
        # -------------------------
        Fp = np.eye(6, dtype=np.float32)
        Fp[0, 3] = dt
        Fp[1, 4] = dt
        Fp[2, 5] = dt

        # Continuous white-acceleration model (discrete Q)
        q = self._pos_accel_var
        dt2, dt3, dt4 = dt * dt, dt * dt * dt, dt * dt * dt * dt
        Qp = np.zeros((6, 6), dtype=np.float32)
        Qp[0:3, 0:3] = np.eye(3, dtype=np.float32) * (dt4 / 4.0) * q
        Qp[0:3, 3:6] = np.eye(3, dtype=np.float32) * (dt3 / 2.0) * q
        Qp[3:6, 0:3] = np.eye(3, dtype=np.float32) * (dt3 / 2.0) * q
        Qp[3:6, 3:6] = np.eye(3, dtype=np.float32) * (dt2) * q

        self._xp, self._Pp = self._kf_predict(self._xp, self._Pp, Fp, Qp)

        # -------------------------
        # Predict (quaternion components)
        # xq = [q(4), qdot(4)]
        # -------------------------
        Fq = np.eye(8, dtype=np.float32)
        Fq[0:4, 4:8] = np.eye(4, dtype=np.float32) * dt

        qq = self._rot_accel_var
        Qq = np.zeros((8, 8), dtype=np.float32)
        Qq[0:4, 0:4] = np.eye(4, dtype=np.float32) * (dt4 / 4.0) * qq
        Qq[0:4, 4:8] = np.eye(4, dtype=np.float32) * (dt3 / 2.0) * qq
        Qq[4:8, 0:4] = np.eye(4, dtype=np.float32) * (dt3 / 2.0) * qq
        Qq[4:8, 4:8] = np.eye(4, dtype=np.float32) * (dt2) * qq

        self._xq, self._Pq = self._kf_predict(self._xq, self._Pq, Fq, Qq)

        # -------------------------
        # Measurement update if we have a pose
        # -------------------------
        T = ~self.i_pose4x4
        assert T is not None
        has_meas = (
            isinstance(T, np.ndarray) and T.shape == (4, 4) and np.isfinite(T).all()
        )

        if has_meas:
            T = T.astype(np.float32)
            p_meas = T[:3, 3].reshape(3, 1)

            Rm = T[:3, :3]
            q_meas = rotation_mat_to_quat_wxyz(Rm)
            q_pred = self._xq[0:4, 0].astype(np.float32)
            q_meas = ensure_shortest_quat(q_meas, q_pred)

            # Position update: z = p
            Hp = np.zeros((3, 6), dtype=np.float32)
            Hp[0, 0] = 1.0
            Hp[1, 1] = 1.0
            Hp[2, 2] = 1.0
            Rp = np.eye(3, dtype=np.float32) * self._pos_meas_var
            self._xp, self._Pp = self._kf_update(
                self._xp, self._Pp, Hp, Rp, p_meas.astype(np.float32)
            )

            # Quaternion update: z = q
            Hq = np.zeros((4, 8), dtype=np.float32)
            Hq[0:4, 0:4] = np.eye(4, dtype=np.float32)
            Rq = np.eye(4, dtype=np.float32) * self._rot_meas_var
            zq = q_meas.reshape(4, 1).astype(np.float32)
            self._xq, self._Pq = self._kf_update(self._xq, self._Pq, Hq, Rq, zq)

        # -------------------------
        # Normalize quaternion + build output pose
        # -------------------------
        q_f = self._xq[0:4, 0].astype(np.float32)
        # Keep continuity by matching last quaternion sign
        q_f = ensure_shortest_quat(q_f, self._last_q)
        self._last_q = q_f.copy()

        n = float(np.linalg.norm(q_f))
        if n < 1e-8:
            q_f = np.array([1, 0, 0, 0], dtype=np.float32)
        else:
            q_f = (q_f / n).astype(np.float32)

        R_f = quat_wxyz_to_rotation_mat(q_f)
        p_f = self._xp[0:3, 0].astype(np.float32)

        Tout = np.eye(4, dtype=np.float32)
        Tout[:3, :3] = R_f
        Tout[:3, 3] = p_f

        self.o_pose4x4 <<= Tout
