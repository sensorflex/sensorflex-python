"""A simple example for using a distributed graph."""

import cv2
import asyncio
import numpy as np
import rerun as rr
from typing import Any

from numpy.typing import NDArray
from sensorflex import Node, Graph, Port
from sensorflex.library.cv import WebcamNode
from sensorflex.library.web import (
    WebSocketServerNode,
    WebSocketClientNode,
    WebSocketMessage,
)
from sensorflex.library.visualization import init_rerun_context
from sensorflex.utils.logging import configure_default_logging

configure_default_logging()


class RerunRGBVisNode(Node):
    """
    Video track handler that logs frames to Rerun.
    """

    i_frame: Port[NDArray]

    _i: int

    def __init__(self) -> None:
        super().__init__()
        self._i = 0
        self.i_frame = Port(None)

    def forward(self) -> None:
        rgb = ~self.i_frame
        rr.set_time("frame_idx", sequence=self._i)
        rr.log(self.name, rr.Image(rgb, color_model="BGR"))
        self._i += 1


class VisNode(Node):
    i_arr: Port[NDArray]

    def __init__(self, name: str | None = None) -> None:
        super().__init__(name)
        self.i_arr = Port(None)

    def forward(self):
        arr = ~self.i_arr
        cv2.imshow("Print", arr)
        # t0 = asyncio.get_running_loop().time()
        cv2.waitKey(1)  # Important for OpenCV GUI.
        # dt = (asyncio.get_running_loop().time() - t0) * 1000
        # print(f"opencv took {dt:.4f} ms.")


class FrameEncoderNode(Node):
    i_frame: Port[cv2.typing.MatLike]
    o_frame: Port[bytes]

    def __init__(self, name: str | None = None) -> None:
        super().__init__(name)

        self.i_frame = Port(None)
        self.o_frame = Port(None)

    def forward(self):
        payload = ~self.i_frame
        payload = np.array(payload).tobytes()
        self.o_frame <<= payload


class FrameDecoderNode(Node):
    i_frame: Port[WebSocketMessage]
    o_frame: Port[NDArray]

    def __init__(self, name: str | None = None) -> None:
        super().__init__(name)

        self.i_frame = Port(None)
        self.o_frame = Port(None)

    def forward(self):
        payload = (~self.i_frame).payload
        payload = payload["data"]
        assert type(payload) is bytes
        payload = np.frombuffer(payload, dtype=np.uint8).reshape((1080, 1920, 3))

        self.o_frame <<= payload


def get_graph():
    g = Graph()
    g += (
        (s_node := WebSocketServerNode())
        + (c_node := WebSocketClientNode())
        + (cam_node := WebcamNode())
        + (e_node := FrameEncoderNode())
        + (d_node := FrameDecoderNode())
        + (v_node := RerunRGBVisNode())
    )

    _ = +cam_node.o_frame + (
        e_node
        + (cam_node.o_frame >> e_node.i_frame)
        + c_node
        + (e_node.o_frame >> c_node.i_message)
    )

    _ = +s_node.o_message + (
        d_node
        + (s_node.o_message >> d_node.i_frame)
        + v_node
        + (d_node.o_frame >> v_node.i_frame)
    )

    return g


async def main():
    g = get_graph()
    await g.wait_forever()


if __name__ == "__main__":
    init_rerun_context("Distributed view.")
    asyncio.run(main())
