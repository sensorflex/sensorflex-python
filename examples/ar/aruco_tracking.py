"""A simple example for using a distributed graph."""

import asyncio
from typing import Any

import numpy as np

from sensorflex import Graph, Node, Port
from sensorflex.library.cv import (
    AruCoPostEstimationNode,
    PoseKalmanFilterNode,
    WebcamNode,
)
from sensorflex.utils.logging import configure_default_logging

configure_default_logging()
np.set_printoptions(suppress=True)


class PrinterNode(Node):
    i_info: Port[Any]

    def __init__(self, name: str | None = None) -> None:
        super().__init__(name)
        self.i_info = Port(np.eye(4, dtype=np.float32))

    def forward(self) -> None:
        print(~self.i_info)


def get_graph():
    # You should change these to your own camera calibration results.
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
    g += (
        aruco_node := AruCoPostEstimationNode(
            camera_matrix=camera_matrix,
            dist_coeffs=distortion_coefficients,
            marker_size=0.066,
        )
    )
    g += (kalman_node := PoseKalmanFilterNode())
    g += (print_node := PrinterNode())

    cam_node.o_frame += (
        (cam_node.o_frame >> aruco_node.i_frame)
        + aruco_node
        + (aruco_node.o_pose >> kalman_node.i_pose4x4)
        + kalman_node
        + (kalman_node.o_pose4x4 >> print_node.i_info)
        + print_node
    )

    return g


async def main():
    g = get_graph()
    await g.wait_forever()


if __name__ == "__main__":
    asyncio.run(main())
