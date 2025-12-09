"""A simple WebRTC server."""

import asyncio
from typing import Any

from sensorflex import Graph, Node, Port
from sensorflex.library.web import WebSocketServerNode


class PrintNode(Node):
    field: Port[Any]

    def __init__(self, name: str | None = None) -> None:
        super().__init__(name)
        self.field = Port(None)

    def forward(self):
        print(~self.field)


async def main():
    g = Graph()

    g += (n1 := WebSocketServerNode())
    p = n1.message.event_pipeline

    p += (n2 := PrintNode())
    p += n1.message >> n2.field

    await g.wait_forever()


if __name__ == "__main__":
    asyncio.run(main())
