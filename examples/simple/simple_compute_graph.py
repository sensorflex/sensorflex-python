"""A simple compute graph example."""

import asyncio
import time
from typing import Any, Union

import numpy as np
from numpy.typing import NDArray

from sensorflex import Graph, Node, Port


# Define your nodes
class ImageLoadingNode(Node):
    # path: Port[str] = Port(None)
    i: Port[int]
    bgr: Port[NDArray]

    def __init__(self, name: Union[str, None] = None) -> None:
        super().__init__(name)
        self.i = Port(0)
        self.bgr = Port(None)

    def forward(self):
        # do something with ~self.path
        x = np.array([3, 2, 1], dtype=np.uint8) + ~self.i
        x = x.reshape((1, 1, 3))
        self.bgr <<= x


class ImageTransformationNode(Node):
    bgr: Port[NDArray]
    rgb: Port[NDArray]

    def __init__(self, name: Union[str, None] = None) -> None:
        super().__init__(name)
        self.bgr = Port(None)
        self.rgb = Port(None)

    def forward(self):
        bgr = ~self.bgr
        rgb = bgr[:, :, [2, 1, 0]]
        self.rgb <<= rgb


class WaitNode(Node):
    wait_seconds: int

    def __init__(self, wait_seconds: int, name: Union[str, None] = None) -> None:
        super().__init__(name)
        self.wait_seconds = wait_seconds

    def forward(self):
        time.sleep(self.wait_seconds)


class PrintNode(Node):
    field: Port[Any]

    def __init__(self, name: Union[str, None] = None) -> None:
        super().__init__(name)
        self.field = Port(None)

    def forward(self):
        print(~self.field)


# Define a graph
def get_graph(i):
    mp = (g := Graph()).main_pipeline

    mp += (n1 := ImageLoadingNode())
    mp += (n2 := ImageTransformationNode())
    mp += n1.bgr >> n2.bgr

    mp += WaitNode(i)
    mp += (n3 := PrintNode())
    mp += n3.field << n2.bgr

    return g, n1, mp


async def run_in_thread():
    threads = []

    # Now execute
    for i in range(5):
        g, n1, mp = get_graph(i)
        n1.i <<= i
        t = g.run_in_thread()
        threads.append(t)

    for t in threads:
        t.join()


if __name__ == "__main__":
    asyncio.run(run_in_thread())
