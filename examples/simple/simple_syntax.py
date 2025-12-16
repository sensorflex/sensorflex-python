"""A simple WebRTC server."""

import asyncio
from typing import Any

from sensorflex import Graph, Node, Port
from sensorflex.library.net import WebSocketServerNode


class PrintNode(Node):
    field: Port[Any]

    def __init__(self, name: str | None = None) -> None:
        super().__init__(name)
        self.field = Port(None)

    def forward(self):
        print(~self.field)


async def main():
    g = Graph()
    n1 = g.add_node(WebSocketServerNode())
    wp = n1.o_message.get_branched_pipeline()

    n2 = PrintNode()
    wp.add(n2)
    wp.add(n1.o_message.connect(n2.field))
    wp.add(n2)

    await g.wait_forever()


if __name__ == "__main__":
    asyncio.run(main())
