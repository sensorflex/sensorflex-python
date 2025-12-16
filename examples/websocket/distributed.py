"""A simple example for using a distributed graph."""

import cv2
import asyncio
import numpy as np
import rerun as rr
from typing import Union

from numpy.typing import NDArray
from sensorflex import Node, Graph, Port
from sensorflex.library.cv import WebcamNode
from sensorflex.library.net import (
    WebSocketServerNode,
    WebSocketClientNode,
    WebSocketMessageEnvelope,
    WebSocketClientConfig,
    WebSocketServerConfig,
)
from sensorflex.library.vis import init_rerun_context
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

    def __init__(self, name: Union[str, None] = None) -> None:
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

    def __init__(self, name: Union[str, None] = None) -> None:
        super().__init__(name)

        self.i_frame = Port(None)
        self.o_frame = Port(None)

    def forward(self):
        payload = ~self.i_frame
        payload = np.array(payload).tobytes()
        self.o_frame <<= payload


def jpeg_decode_bgr(jpeg_bytes: bytes) -> NDArray:
    """
    jpeg_bytes: JPEG-encoded bytes
    returns: uint8 NumPy array in BGR order (H, W, 3)
    """
    buf = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)

    if img is None:
        raise RuntimeError("cv2.imdecode failed")

    return img


def png_decode(png_bytes: bytes, flags=cv2.IMREAD_UNCHANGED) -> NDArray:
    """
    png_bytes: PNG-encoded bytes
    flags:
      - cv2.IMREAD_UNCHANGED (preserve channels, incl. alpha)
      - cv2.IMREAD_COLOR
      - cv2.IMREAD_GRAYSCALE
    returns: uint8 NumPy array
    """
    buf = np.frombuffer(png_bytes, dtype=np.uint8)
    img = cv2.imdecode(buf, flags)

    if img is None:
        raise RuntimeError("cv2.imdecode failed")

    return img


class FrameDecoderNode(Node):
    i_frame: Port[WebSocketMessageEnvelope]
    o_frame: Port[NDArray]

    def __init__(self, name: Union[str, None] = None) -> None:
        super().__init__(name)

        self.i_frame = Port(None)
        self.o_frame = Port(None)

    def forward(self):
        payload = (~self.i_frame).payload
        assert type(payload) is bytes
        payload = jpeg_decode_bgr(payload)
        payload = payload.reshape((1080, 1920, 3))

        self.o_frame <<= payload


def get_graph():
    g = Graph()

    # Run the following code on a computer with webcam.
    g += (
        (
            c_node := WebSocketClientNode(
                uri="ws://localhost:8765",
                config=WebSocketClientConfig(
                    max_size=None,
                    compression=None,
                ),
            )
        )
        + (cam_node := WebcamNode())
        + (e_node := FrameEncoderNode())
    )

    _ = +cam_node.o_frame + (
        e_node
        + (cam_node.o_frame >> e_node.i_frame)
        + c_node
        + (e_node.o_frame >> c_node.i_message)
    )

    # Run this part on another computer.
    g += (
        (
            s_node := WebSocketServerNode(
                config=WebSocketServerConfig(
                    max_size=None,
                    compression=None,
                )
            )
        )
        + (d_node := FrameDecoderNode())
        + (v_node := RerunRGBVisNode())
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
