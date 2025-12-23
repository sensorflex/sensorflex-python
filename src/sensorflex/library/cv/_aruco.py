"""A library for AruCo marker detection."""

import cv2
import numpy as np
from numpy.typing import NDArray

from sensorflex import Node, Port


class AruCoPostEstimationNode(Node):
    """
    ArUco-based camera pose estimation (OpenCV >= 4.8 safe).

    Input:
        i_frame: NDArray[H, W, 3]  (uint8)

    Output:
        o_pose: NDArray[4, 4]      (float32, marker -> camera)
    """

    i_frame: Port[NDArray]
    o_pose: Port[NDArray]

    def __init__(
        self,
        camera_matrix: NDArray,
        dist_coeffs: NDArray,
        marker_size: float = 0.05,  # meters
        aruco_dict: int = cv2.aruco.DICT_4X4_50,
        name: str | None = None,
    ) -> None:
        super().__init__(name)
        self.i_frame = Port(None)
        self.o_pose = Port(np.eye(4, dtype=np.float32))

        self._K = camera_matrix.astype(np.float32)
        self._D = dist_coeffs.astype(np.float32)

        half = marker_size / 2.0

        # 3D marker corners in marker coordinate frame
        self._obj_pts = np.array(
            [
                [-half, half, 0.0],
                [half, half, 0.0],
                [half, -half, 0.0],
                [-half, -half, 0.0],
            ],
            dtype=np.float32,
        )

        dictionary = cv2.aruco.getPredefinedDictionary(aruco_dict)
        parameters = cv2.aruco.DetectorParameters()

        self._detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    def forward(self) -> None:
        frame = self.i_frame.value
        if frame is None:
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame

        corners, ids, _ = self._detector.detectMarkers(gray)

        if ids is None or len(corners) == 0:
            self.o_pose <<= ~self.o_pose
            return

        # Use first detected marker
        img_pts = corners[0].reshape(4, 2).astype(np.float32)

        ok, rvec, tvec = cv2.solvePnP(
            self._obj_pts,
            img_pts,
            self._K,
            self._D,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,  # best for planar square
        )

        if not ok:
            self.o_pose <<= ~self.o_pose
            return

        R, _ = cv2.Rodrigues(rvec)

        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = R
        T[:3, 3] = tvec.reshape(3)

        self.o_pose <<= T
