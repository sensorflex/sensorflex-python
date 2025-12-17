"""A simple example for using a distributed graph."""

from typing import Union

import cv2
import asyncio
import numpy as np

from numpy.typing import NDArray
from sensorflex import Node, Graph, Port
from sensorflex.library.cv import WebcamNode
from sensorflex.library.net import WebSocketServerNode, WebSocketServerConfig
from sensorflex.utils.logging import configure_default_logging

configure_default_logging()


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

    o_draw: Port[NDArray]

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
        self.o_draw = Port(None)

        self._K = camera_matrix.astype(np.float32)
        self._D = dist_coeffs.astype(np.float32)

        self.marker_size = marker_size

        half = marker_size / 2.0

        # 3D marker corners in marker coordinate frame
        self._obj_pts = np.array(
            [
                [-half, -half, 0.0],  # bottom-left
                [half, -half, 0.0],  # bottom-right
                [half, half, 0.0],  # top-right
                [-half, half, 0.0],  # top-left
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

        # DEBUG: Check frame size matches calibration
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
            flags=cv2.SOLVEPNP_ITERATIVE,  # best for planar square
        )

        if not ok:
            self.o_pose <<= ~self.o_pose
            return

        R, _ = cv2.Rodrigues(rvec)

        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = R
        T[:3, 3] = tvec.reshape(3)

        # Draw visualization
        frame_draw = frame.copy()
        cv2.aruco.drawDetectedMarkers(frame_draw, corners, ids)
        cv2.drawFrameAxes(
            frame_draw,
            self._K,
            self._D,
            rvec,
            tvec,
            self.marker_size * 0.5,
        )
        frame_draw = cv2.cvtColor(frame_draw, cv2.COLOR_BGR2RGB)
        self.o_draw <<= frame_draw

        self.o_pose <<= marker_to_unity_camera_pose(T)


def invert_se3(T_co: np.ndarray) -> np.ndarray:
    """
    Invert an SE(3) transform.

    Args:
        T_co: (4,4) homogeneous transform (object -> camera)

    Returns:
        T_oc: (4,4) homogeneous transform (camera -> object)
    """
    assert T_co.shape == (4, 4)

    R = T_co[:3, :3]
    t = T_co[:3, 3]

    T_oc = np.eye(4, dtype=T_co.dtype)
    T_oc[:3, :3] = R.T
    T_oc[:3, 3] = -R.T @ t

    return T_oc


def marker_to_unity_camera_pose(T_marker_to_cam_cv: np.ndarray) -> np.ndarray:
    """
    Input:  T (4x4) marker->camera in OpenCV coords (x right, y down, z forward)
    Output: (4x4) camera->world in Unity coords, assuming world == marker frame
    """
    T = T_marker_to_cam_cv.astype(np.float32)

    # 1) invert: camera->marker (still in OpenCV axis convention)
    T_cam_to_marker_cv = invert_se3(T)

    # 2) axis conversion OpenCV -> Unity: flip Y and Z
    C = np.diag([1.0, -1.0, 1.0, 1.0]).astype(np.float32)

    # camera->world (world == marker) in Unity convention
    T_cam_to_world_unity = C @ T_cam_to_marker_cv @ C
    return T_cam_to_world_unity


def jpeg_encode_bgr(img_bgr: NDArray, quality: int = 80) -> NDArray:
    """
    img_bgr: uint8 NumPy array in BGR order (OpenCV default), shape (H,W,3) or (H,W)
    returns: JPEG bytes
    """
    if img_bgr.dtype != np.uint8:
        raise ValueError("img_bgr must be uint8")

    ok, buf = cv2.imencode(
        ".jpg",
        img_bgr,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)],
    )
    if not ok:
        raise RuntimeError("cv2.imencode failed")

    return buf


class FrameEncoderNode(Node):
    i_frame: Port[cv2.typing.MatLike]
    o_frame: Port[bytes]

    def __init__(self, name: Union[str, None] = None) -> None:
        super().__init__(name)

        self.i_frame = Port(None)
        self.o_frame = Port(None)

    def forward(self):
        payload = ~self.i_frame
        payload = jpeg_encode_bgr(payload, quality=50)
        payload = np.array(payload).tobytes()
        self.o_frame <<= payload


class PayloadFactoryNode(Node):
    i_frame: Port[bytes]
    i_pose4x4: Port[NDArray]

    o_payload: Port[bytes]

    def __init__(self, name: str | None = None) -> None:
        super().__init__(name)

        self.i_frame = Port(None)
        self.i_pose4x4 = Port(None)
        self.o_payload = Port(None)

    def forward(self):
        b_frame = ~self.i_frame

        T = ~self.i_pose4x4
        # T_unity = marker_to_unity_camera_pose(T_cv)
        b_pose4x4 = T.astype(np.float32).tobytes(order="F")

        b_payload = b_pose4x4 + b_frame
        self.o_payload <<= b_payload


def get_graph():
    camera_matrix = np.array(
        [
            [1143.0742696887514, 0.0, 940.2203465406869],
            [0.0, 1143.2048955398304, 558.9794033660755],
            [0.0, 0.0, 1.0],
        ]
    )
    distortion_coefficients = np.array(
        [
            [
                0.19395459813454857,
                -0.5035552148706698,
                -0.0001805203192460468,
                -0.0006628657834761212,
                0.38188726336468726,
            ]
        ]
    )
    # fx, fy = (1143.07, 1143.20)
    # cx, cy = (940.22, 558.98)

    g = Graph()

    # Run the following code on a computer with webcam.
    g += (cam_node := WebcamNode())
    g += (enc_node := FrameEncoderNode())
    g += (
        aruco_node := AruCoPostEstimationNode(
            camera_matrix=camera_matrix,
            dist_coeffs=distortion_coefficients,
            marker_size=0.066,
        )
    )
    g += (fac_node := PayloadFactoryNode())
    g += (
        wss_node := WebSocketServerNode(
            config=WebSocketServerConfig(max_size=0, compression=None)
        )
    )
    # g += (vis_node := RerunVideoVisNode())

    _ = +cam_node.o_frame + (
        aruco_node
        + (cam_node.o_frame >> aruco_node.i_frame)
        + enc_node
        + (cam_node.o_frame >> enc_node.i_frame)
        + fac_node
        + (enc_node.o_frame >> fac_node.i_frame)
        + (aruco_node.o_pose >> fac_node.i_pose4x4)
        # + vis_node
        # + (aruco_node.o_draw >> vis_node.i_frame)
        + wss_node
        + (fac_node.o_payload >> wss_node.i_broadcast_message)
    )

    return g


async def main():
    g = get_graph()
    await g.wait_forever()


if __name__ == "__main__":
    # init_rerun_context("Aruco")
    asyncio.run(main())
