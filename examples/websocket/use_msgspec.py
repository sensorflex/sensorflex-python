"""A simple example for using multiple pipelines in a graph."""

import asyncio
from typing import List, Union

import msgspec

from sensorflex import Graph, Node, Port
from sensorflex.library.net import (
    WebSocketClientNode,
    WebSocketServerNode,
    get_msgpack_encoder_decoder_nodes,
)
from sensorflex.utils.logging import configure_default_logging

configure_default_logging()


class User(msgspec.Struct):
    id: int
    name: str
    scores: List[float]


class Pose(msgspec.Struct):
    position: tuple[float, float, float]
    rotation: tuple[float, float, float, float]  # quaternion


class PrintUserNode(Node):
    i_user: Port[User]

    def __init__(self, name: Union[str, None] = None) -> None:
        super().__init__(name)
        self.i_user = Port(None)

    def forward(self):
        msg = ~self.i_user
        print(f"Received user: {msg}")


class PrintPoseNode(Node):
    i_pose: Port[User]

    def __init__(self, name: Union[str, None] = None) -> None:
        super().__init__(name)
        self.i_pose = Port(None)

    def forward(self):
        msg = ~self.i_pose
        print(f"Received pose: {msg}")


def get_graph():
    encoder_node, decoder_node = get_msgpack_encoder_decoder_nodes([User, Pose])

    g = Graph()
    g += (
        (s_node := WebSocketServerNode())
        + (c_node := WebSocketClientNode())
        + encoder_node
        + decoder_node
        + (pu_node := PrintUserNode())
        + (pp_node := PrintPoseNode())
    )

    g.main_pipeline += encoder_node + (encoder_node.o_bytes >> c_node.i_message)

    s_node.o_message += (
        (s_node.o_message.map(lambda x: x.payload) >> decoder_node.i_bytes)
        + decoder_node
        + decoder_node.o_data.match(
            lambda v: type(v),
            {
                User: ((decoder_node.o_data >> pu_node.i_user) + pu_node),
                Pose: (decoder_node.o_data >> pp_node.i_pose) + pp_node,
            },
        )
    )

    return g, encoder_node


async def main():
    g, e_node = get_graph()
    t = g.wait_forever_as_task()

    await asyncio.sleep(1)
    e_node.i_data <<= Pose((1.0, 2.0, 3.0), (1.0, 2.0, 3.0, 4.0))
    g.main_pipeline.run()

    await asyncio.sleep(1)
    e_node.i_data <<= User(0, "John", [1, 2, 3])
    g.main_pipeline.run()

    # await asyncio.Future()
    await asyncio.sleep(1)
    t.cancel()


if __name__ == "__main__":
    asyncio.run(main())
