"""A simple example for using multiple pipelines in a graph."""

import asyncio
from typing import Union

from sensorflex import Graph, Node, Port
from sensorflex.utils.logging import configure_default_logging

configure_default_logging()


class PrintGreaterThan10Node(Node):
    i_number: Port[int]

    def __init__(self, name: Union[str, None] = None) -> None:
        super().__init__(name)
        self.i_number = Port(None)

    def forward(self):
        print(f">10: {~self.i_number}")


class PrintLEQThan10Node(Node):
    i_number: Port[int]

    def __init__(self, name: Union[str, None] = None) -> None:
        super().__init__(name)
        self.i_number = Port(None)

    def forward(self):
        print(f"<=10: {~self.i_number}")


class NumberGeneratorNode(Node):
    o_number: Port[int]

    _i: int

    def __init__(self, name: Union[str, None] = None) -> None:
        super().__init__(name)
        self.o_number = Port(None)
        self._i = 0

    def forward(self):
        self.o_number <<= self._i
        self._i += 5


def get_graph():
    g = Graph()
    g += (
        (n_node := NumberGeneratorNode())
        + (pleq_node := PrintLEQThan10Node())
        + (pgt_node := PrintGreaterThan10Node())
    )

    g.main_pipeline += n_node + n_node.o_number.match(
        lambda v: v <= 10,
        {
            True: ((n_node.o_number >> pleq_node.i_number) + pleq_node),
            False: (n_node.o_number >> pgt_node.i_number) + pgt_node,
        },
    )

    return g


async def main():
    g = get_graph()
    t = g.wait_forever_as_task()

    for _ in range(20):
        g.main_pipeline.run()
        await asyncio.sleep(1)

    # await asyncio.Future()
    await asyncio.sleep(1)
    print("Exit.")
    t.cancel()


if __name__ == "__main__":
    asyncio.run(main())
