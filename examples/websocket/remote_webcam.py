"""A simple example for using a distributed graph."""

import asyncio

import cv2
from numpy.typing import NDArray

from sensorflex import Graph, Node, Port
from sensorflex.library.cv import (
    ImageCodec,
    ImageDecodeNode,
    ImageEncodeNode,
    WebcamNode,
)
from sensorflex.library.net import (
    WebSocketClientConfig,
    WebSocketClientNode,
    WebSocketMessageEnvelope,
    WebSocketServerConfig,
    WebSocketServerNode,
)
from sensorflex.library.vis import RerunVideoVisNode, init_rerun_context
from sensorflex.utils.logging import configure_default_logging

configure_default_logging()


class ServerUnpackNode(Node):
    i_message: Port[WebSocketMessageEnvelope]
    o_bytes: Port[bytes]

    def __init__(self, name: str | None = None) -> None:
        super().__init__(name)
        self.i_message = Port(None)
        self.o_bytes = Port(None)

    def forward(self) -> None:
        msg = ~self.i_message
        data = msg.payload
        assert isinstance(data, bytes)
        self.o_bytes <<= data


class RGB2BGRNode(Node):
    i_img: Port[NDArray]
    o_img: Port[NDArray]

    def __init__(self, name: str | None = None) -> None:
        super().__init__(name)
        self.i_img = Port(None)
        self.o_img = Port(None)

    def forward(self) -> None:
        img = ~self.i_img
        self.o_img <<= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def get_graph():
    g = Graph()

    # Run the following code on a computer with webcam.
    g += (
        (
            c_node := WebSocketClientNode(
                uri="ws://localhost:8765",  # Change this.
                config=WebSocketClientConfig(
                    max_size=None,
                    compression=None,
                ),
            )
        )
        + (cam_node := WebcamNode())
        + (e_node := ImageEncodeNode(ImageCodec.JPEG))
    )

    cam_node.o_frame += (
        (cam_node.o_frame >> e_node.i_img)
        + e_node
        + (e_node.o_buf >> c_node.i_message)
        + c_node
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
        + (m_node := ServerUnpackNode())
        + (d_node := ImageDecodeNode(ImageCodec.JPEG))
        + (t_node := RGB2BGRNode())
        + (v_node := RerunVideoVisNode())
    )

    s_node.o_message += (
        (s_node.o_message >> m_node.i_message)
        # + (s_node.o_message >> d_node.i_buf)  # Type checker will report error.
        + m_node
        + (m_node.o_bytes >> d_node.i_buf)
        + d_node
        + (d_node.o_img >> t_node.i_img)
        + t_node
        + (t_node.o_img >> v_node.i_frame)
        + v_node
    )

    return g


async def main():
    g = get_graph()
    await g.wait_forever()


if __name__ == "__main__":
    init_rerun_context("Distributed view.")
    asyncio.run(main())
