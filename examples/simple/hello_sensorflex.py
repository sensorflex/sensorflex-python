"""Hello SensorFlex, your first example!"""

import asyncio
from typing import Any

import numpy as np
from numpy.typing import NDArray

from sensorflex import Graph, Node, Port


class ArrayInitNode(Node):
    i_i: Port[int]
    o_arr: Port[NDArray]

    def __init__(self, name: str | None = None) -> None:
        super().__init__(name)
        self.i_i = Port(None)
        self.o_arr = Port(None)

    def forward(self):
        # do something with ~self.path
        x = np.array([0], dtype=np.uint8) + ~self.i_i
        self.o_arr <<= x


class PrintNode(Node):
    i_field: Port[Any]

    def __init__(self, name: str | None = None) -> None:
        super().__init__(name)
        self.i_field = Port(None)

    def forward(self):
        print(f"arr is current at: {~self.i_field}")


async def main():
    # Define a graph
    mp = (g := Graph()).main_pipeline

    mp += (n1 := ArrayInitNode())
    mp += (n2 := PrintNode())
    mp += n1.o_arr >> n2.i_field

    # Now execute
    for i in range(5):
        n1.i_i <<= i
        mp.run()


if __name__ == "__main__":
    asyncio.run(main())
