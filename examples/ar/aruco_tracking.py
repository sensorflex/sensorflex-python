"""A simple example for using a distributed graph."""

from typing import Union, Tuple, Optional

import cv2
import asyncio
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from numpy.typing import NDArray
from sensorflex import Node, Graph, Port
from sensorflex.library.cv import WebcamNode
from sensorflex.library.vis import init_rerun_context
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


# If you have these in your framework already:
# from sensorflex.core.runtime import Node, Port
# from numpy.typing import NDArray
# import numpy as np


class RerunRGBVisNode(Node):
    """
    Logs:
      - camera/image        -> Pinhole (static) + Image (per-frame)
      - camera              -> Transform3D (per-frame extrinsics)
      - camera/pose_axes    -> Arrows3D (static glyph, moved by camera transform)
      - world/origin_axes   -> Arrows3D (static glyph)

    Also sends a blueprint to show:
      - a 3D view rooted at /world with eye controls looking at the world origin
      - a 2D view showing camera/image
    """

    i_frame: Port[NDArray]
    i_pose4x4: Port[NDArray]

    _i: int

    def __init__(
        self,
        # Camera intrinsics (recommended for 3D->2D overlays if you later add projections)
        resolution: Tuple[int, int],  # (width, height)
        focal_length: Union[Tuple[float, float], float],  # (fx, fy) or scalar
        principal_point: Tuple[float, float],  # (cx, cy)
        # Visualization
        axis_length: float = 0.1,  # camera local axes length
        world_axis_length: Optional[
            float
        ] = None,  # world origin axes length (defaults to 2x axis_length)
        # Pose interpretation:
        # True if T is child-from-parent (world->camera). False if parent-from-child (camera->world).
        pose_is_child_from_parent: bool = True,
        # Blueprint / view layout
        send_blueprint: bool = True,
        world_view_origin: str = "/world",
        image_view_origin: str = "camera/image",
        name: str | None = None,
    ) -> None:
        super().__init__(name)
        self._i = 0

        self.i_frame = Port(None)
        self.i_pose4x4 = Port(None)

        self._axis_length = float(axis_length)
        self._world_axis_length = (
            float(world_axis_length)
            if world_axis_length is not None
            else float(axis_length) * 2.0
        )
        self._pose_is_child_from_parent = bool(pose_is_child_from_parent)

        # Make navigation nicer
        rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

        # -------------------------
        # Static world origin axes
        # -------------------------
        WL = self._world_axis_length
        rr.log(
            "world/origin_axes",
            rr.Arrows3D(
                origins=[[0.0, 0.0, 0.0]] * 3,
                vectors=[[WL, 0.0, 0.0], [0.0, WL, 0.0], [0.0, 0.0, WL]],
                colors=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                labels=["+X", "+Y", "+Z"],
            ),
            static=True,
        )

        # -------------------------
        # Static camera local axes glyph (moved by `camera` transform)
        # -------------------------
        L = self._axis_length
        rr.log(
            "camera/pose_axes",
            rr.Arrows3D(
                origins=[[0.0, 0.0, 0.0]] * 3,
                vectors=[[L, 0.0, 0.0], [0.0, L, 0.0], [0.0, 0.0, L]],
                colors=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                labels=["+X", "+Y", "+Z"],
            ),
            static=True,
        )

        # -------------------------
        # Pinhole intrinsics for camera/image
        # -------------------------
        w, h = resolution
        rr.log(
            image_view_origin,
            rr.Pinhole(
                resolution=[w, h],
                focal_length=focal_length,
                principal_point=principal_point,
            ),
            static=True,
        )

        # -------------------------
        # Blueprint: 3D world view + 2D image view
        # -------------------------
        if send_blueprint:
            blueprint = rrb.Blueprint(
                rrb.Horizontal(
                    rrb.Spatial3DView(
                        name="World (fixed)",
                        origin=world_view_origin,
                        spatial_information=rrb.SpatialInformation(
                            show_axes=True,
                            show_bounding_box=False,
                        ),
                        eye_controls=rrb.EyeControls3D(
                            # tweak these if your scene scale differs
                            position=(0.5, -0.8, 0.5),
                            look_target=(0.0, 0.0, 0.0),
                            eye_up=(0.0, 0.0, 1.0),
                            speed=5.0,
                            tracking_entity=None,  # critical: do NOT follow the camera entity
                        ),
                    ),
                    rrb.Spatial2DView(
                        name="Camera Image",
                        origin=image_view_origin,
                    ),
                ),
                collapse_panels=False,
            )
            rr.send_blueprint(blueprint, make_active=True)

        self._image_path = image_view_origin

    def forward(self) -> None:
        rgb = ~self.i_frame
        if rgb is None:
            return

        rr.set_time("frame_idx", sequence=self._i)

        # Show camera image (2D view)
        rr.log(self._image_path, rr.Image(rgb, color_model="BGR"))

        # Log camera pose (3D view)
        T = ~self.i_pose4x4
        if T is not None:
            R = T[:3, :3]
            t = T[:3, 3]
            rr.log(
                "camera",
                rr.Transform3D(
                    translation=t,
                    mat3x3=R,
                    relation=(
                        rr.TransformRelation.ChildFromParent
                        if self._pose_is_child_from_parent
                        else rr.TransformRelation.ParentFromChild
                    ),
                ),
            )

        self._i += 1


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
    fx, fy = (1143.07, 1143.20)
    cx, cy = (940.22, 558.98)

    g = Graph()

    # Run the following code on a computer with webcam.
    g += (cam_node := WebcamNode())
    g += (
        aruco_node := AruCoPostEstimationNode(
            camera_matrix=camera_matrix,
            dist_coeffs=distortion_coefficients,
            marker_size=0.066,
        )
    )
    g += (
        vis_node := RerunRGBVisNode(
            resolution=(1920, 1080), focal_length=(fx, fy), principal_point=(cx, cy)
        )
    )

    _ = +cam_node.o_frame + (
        aruco_node
        + (cam_node.o_frame >> aruco_node.i_frame)
        + vis_node
        + (cam_node.o_frame >> vis_node.i_frame)
        + (aruco_node.o_pose >> vis_node.i_pose4x4)
    )

    return g


async def main():
    g = get_graph()
    await g.wait_forever()


if __name__ == "__main__":
    init_rerun_context("AruCo Marker Tracking.")
    asyncio.run(main())
