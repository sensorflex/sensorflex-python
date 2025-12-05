"""A simple WebRTC server."""

import asyncio
from typing import Any

from sensorflex import Graph, Node, Port, WebSocketServer


class PrintNode(Node):
    field: Port[Any]

    def __init__(self, name: str | None = None) -> None:
        super().__init__(name)
        self.field = Port(None)

    def forward(self):
        print(~self.field)


async def main():
    g = Graph()
    n1 = g << WebSocketServer()
    n2 = g << PrintNode()
    g <<= n1.last_message >> n2.field

    g.watch(n1.last_message)

    # await g.run_and_wait_forever()
    # _ = g.run_and_wait_forever()
    t = g.run_and_wait_forever_as_task()

    for i in range(5):
        await asyncio.sleep(3)

    t.cancel()


if __name__ == "__main__":
    asyncio.run(main())
