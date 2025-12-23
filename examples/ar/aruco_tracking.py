"""A simple example for using a distributed graph."""

import asyncio

import numpy as np

from sensorflex import Graph
from sensorflex.library.cv import AruCoPostEstimationNode, WebcamNode
from sensorflex.library.vis import init_rerun_context
from sensorflex.utils.logging import configure_default_logging

configure_default_logging()


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

    cam_node.o_frame += aruco_node + (cam_node.o_frame >> aruco_node.i_frame)

    return g


async def main():
    g = get_graph()
    await g.wait_forever()


if __name__ == "__main__":
    init_rerun_context("AruCo Marker Tracking.")
    asyncio.run(main())
