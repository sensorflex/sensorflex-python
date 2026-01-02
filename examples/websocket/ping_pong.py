"""A simple example for using multiple pipelines in a graph."""

import asyncio
import json
import time
from typing import Any, Union

from websockets.typing import Data

from sensorflex import Graph, Node, Port
from sensorflex.library.net import (
    WebSocketClientNode,
    WebSocketServerNode,
)
from sensorflex.utils.logging import configure_default_logging

configure_default_logging()


class DelayNode(Node):
    i_value: Port[Data]
    o_value: Port[Any]

    def __init__(self, name: Union[str, None] = None) -> None:
        super().__init__(name)
        self.i_value = Port(None)
        self.o_value = Port(None)

    def forward(self):
        time.sleep(0.5)
        msg = ~self.i_value
        msg = json.loads(msg)
        msg["i"] = msg["i"] + 1
        self.o_value <<= json.dumps(msg)


def get_graph():
    g = Graph()
    g += (
        (s_node := WebSocketServerNode()),
        (c_node := WebSocketClientNode()),
        (d_node := DelayNode()),
    )

    s_node.o_message += (
        d_node[(s_node.o_message.print() > d_node.i_value)],
        c_node[(d_node.o_value > c_node.i_message)],
    )

    return g, c_node


async def main():
    g, c_node = get_graph()
    t = g.wait_forever_as_task()

    await asyncio.sleep(1)

    c_node.i_message <<= json.dumps({"Hello": "World", "i": 0})

    await asyncio.Future()
    t.cancel()


if __name__ == "__main__":
    asyncio.run(main())
