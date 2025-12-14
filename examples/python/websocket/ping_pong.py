"""A simple example for using multiple pipelines in a graph."""

import time
import asyncio
from typing import Any

from sensorflex import Node, Graph, Port
from sensorflex.library.web import (
    WebSocketServerNode,
    WebSocketClientNode,
    WebSocketMessage,
)
from sensorflex.utils.logging import configure_default_logging

configure_default_logging()


class PrintNode(Node):
    field: Port[Any]

    def __init__(self, name: str | None = None) -> None:
        super().__init__(name)
        self.field = Port(None)

    def forward(self):
        print(~self.field)


class DelayNode(Node):
    i_value: Port[WebSocketMessage]
    o_value: Port[Any]

    def __init__(self, name: str | None = None) -> None:
        super().__init__(name)
        self.i_value = Port(None)
        self.o_value = Port(None)

    def forward(self):
        time.sleep(0.5)
        msg = ~self.i_value
        msg.payload["i"] = msg.payload["i"] + 1
        self.o_value <<= msg.payload


def get_graph():
    g = Graph()

    g += (s_node := WebSocketServerNode())
    ps = +s_node.o_message
    ps += (c_node := WebSocketClientNode())
    ps += (p_node := PrintNode())
    ps += s_node.o_message >> p_node.field

    ps += (d_node := DelayNode())
    ps += s_node.o_message >> d_node.i_value
    ps += d_node.o_value >> c_node.i_message

    return g, c_node


async def main():
    g, c_node = get_graph()
    t = g.wait_forever_as_task()

    await asyncio.sleep(1)

    c_node.i_message <<= {"Hello": "World", "i": 0}
    g.run_main_pipeline()

    await asyncio.Future()
    t.cancel()


if __name__ == "__main__":
    asyncio.run(main())
