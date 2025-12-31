"""A simple example for using multiple pipelines in a graph."""

import asyncio
from typing import List, Union
from uuid import UUID

import msgspec

from sensorflex import Graph, Node, Port
from sensorflex.library.net import (
    WebSocketClientNode,
    WebSocketServerNode,
    get_msgpack_coder_transforms,
)
from sensorflex.utils.logging import configure_default_logging

configure_default_logging()


# ############################################################################
# Define your data models here.
# ############################################################################
class User(msgspec.Struct):
    id: int
    name: str
    scores: List[float]


class Pose(msgspec.Struct):
    position: tuple[float, float, float]
    rotation: tuple[float, float, float, float]  # quaternion


# ############################################################################
# Custom nodes.
# ############################################################################
class SenderNode(Node):
    i_data: Port[User | Pose]
    o_data: Port[User | Pose]

    def __init__(self, name: str | None = None) -> None:
        super().__init__(name)
        self.i_data = Port(None)
        self.o_data = Port(None)

    def forward(self) -> None:
        self.o_data <<= ~self.i_data


class PrintUserNode(Node):
    i_user: Port[User, UUID]

    def __init__(self, name: Union[str, None] = None) -> None:
        super().__init__(name)
        self.i_user = Port(None)

    def forward(self):
        msg = ~self.i_user
        print(f"Received user: {msg} from client, port meta is:", self.i_user.meta)


class PrintPoseNode(Node):
    i_pose: Port[Pose]

    def __init__(self, name: Union[str, None] = None) -> None:
        super().__init__(name)
        self.i_pose = Port(None)

    def forward(self):
        msg = ~self.i_pose
        print(f"Received pose: {msg}")


def get_graph():
    encode, decode = get_msgpack_coder_transforms(User, Pose)

    g = Graph()
    g += (
        (s_node := WebSocketServerNode())
        + (c_node := WebSocketClientNode())
        + (sd_node := SenderNode())
        + (pu_node := PrintUserNode())
        + (pp_node := PrintPoseNode())
    )

    g.main_pipeline += sd_node + (sd_node.o_data.map(encode) > c_node.i_message)

    s_node.o_message += (
        (msg := s_node.o_message.map(decode))
        + msg.isinstance(User, lambda o_data: (o_data > pu_node.i_user) + pu_node)
        + msg.isinstance(Pose, lambda o_data: (o_data > pp_node.i_pose) + pp_node)
    )

    return g, sd_node


async def main():
    g, sd_node = get_graph()
    t = g.wait_forever_as_task()

    await asyncio.sleep(1)
    sd_node.i_data <<= Pose((1.0, 2.0, 3.0), (1.0, 2.0, 3.0, 4.0))
    g.main_pipeline.run()

    await asyncio.sleep(1)
    sd_node.i_data <<= User(0, "John", [1, 2, 3])
    g.main_pipeline.run()

    # await asyncio.Future()
    await asyncio.sleep(1)
    t.cancel()


if __name__ == "__main__":
    asyncio.run(main())
