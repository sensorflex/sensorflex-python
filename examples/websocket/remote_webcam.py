"""A simple example for using a distributed graph."""

import asyncio
from typing import cast

from sensorflex import Graph
from sensorflex.library.cv import ImageCodec, WebcamNode, get_image_coder
from sensorflex.library.net import (
    WebSocketClientConfig,
    WebSocketClientNode,
    WebSocketServerConfig,
    WebSocketServerNode,
)
from sensorflex.library.vis import RerunVideoVisNode, init_rerun_context
from sensorflex.utils.logging import configure_default_logging

configure_default_logging()


def get_graph():
    encode, decode = get_image_coder(ImageCodec.JPEG)

    g = Graph()

    # Run the following code on a computer with webcam.
    g += (
        c_node := WebSocketClientNode(
            uri="ws://localhost:8765",  # Change this.
            config=WebSocketClientConfig(
                max_size=None,
                compression=None,  # Important! Avoid using compression on encoded data.
            ),
        )
    ) + (cam_node := WebcamNode())

    cam_node.o_frame += (cam_node.o_frame.map(encode) > c_node.i_message) + c_node

    # Run this part on another computer.
    g += (
        s_node := WebSocketServerNode(
            config=WebSocketServerConfig(
                max_size=None,
                compression=None,  # Important! Avoid using compression on encoded data.
            )
        )
    ) + (v_node := RerunVideoVisNode())

    s_node.o_message += (
        (img := s_node.o_message.map(lambda v: decode(cast(bytes, v))))
        # avoid using cv2.cvtColor for better performance
        + (img.map(lambda x: x[:, :, ::-1]) > v_node.i_frame)
        + v_node
    )

    return g


async def main():
    g = get_graph()
    await g.wait_forever()


if __name__ == "__main__":
    init_rerun_context("Distributed view.")
    asyncio.run(main())
