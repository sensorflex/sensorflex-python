"""A simple example for using multiple pipelines in a graph."""

import asyncio
import json
import time
from typing import Any, Union

from sensorflex import Graph, Node, Port
from sensorflex.library.net import (
    WebSocketClientNode,
    WebSocketMessageEnvelope,
    WebSocketServerNode,
)
from sensorflex.utils.logging import configure_default_logging

configure_default_logging()


class BytesPrintNode(Node):
    field: Port[WebSocketMessageEnvelope]

    def __init__(self, name: Union[str, None] = None) -> None:
        super().__init__(name)
        self.field = Port(None)

    def forward(self):
        msg = ~self.field
        msg = msg.payload
        assert isinstance(msg, bytes)
        print(f"Received bytes: {msg}")


class PrintNode(Node):
    field: Port[WebSocketMessageEnvelope]

    def __init__(self, name: Union[str, None] = None) -> None:
        super().__init__(name)
        self.field = Port(None)

    def forward(self):
        msg = ~self.field
        msg = msg.payload
        assert isinstance(msg, str)
        print(f"Received text: {msg}")


class DelayNode(Node):
    i_value: Port[WebSocketMessageEnvelope]
    o_value: Port[Any]

    def __init__(self, name: Union[str, None] = None) -> None:
        super().__init__(name)
        self.i_value = Port(None)
        self.o_value = Port(None)

    def forward(self):
        time.sleep(0.5)
        msg = ~self.i_value
        msg = json.loads(msg.payload)
        msg["i"] = msg["i"] + 1
        self.o_value <<= json.dumps(msg)


def get_graph():
    g = Graph()
    g += (
        (s_node := WebSocketServerNode())
        + (p_node := PrintNode())
        + (ep_node := BytesPrintNode())
        + (c_node := WebSocketClientNode())
    )

    s_node.o_message.match(
        lambda v: type(v.payload),
        {
            str: ((s_node.o_message >> p_node.field) + p_node),
            bytes: (s_node.o_message >> ep_node.field) + ep_node,
        },
    )

    return g, c_node


async def main():
    g, c_node = get_graph()
    t = g.wait_forever_as_task()

    await asyncio.sleep(1)

    c_node.i_message <<= json.dumps({"Hello": "World", "i": 0})

    await asyncio.sleep(1)
    c_node.i_message <<= bytes([0, 1, 2])

    await asyncio.Future()
    t.cancel()


if __name__ == "__main__":
    asyncio.run(main())
