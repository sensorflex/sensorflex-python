"""A simple compute graph example."""

import time
import numpy as np
import multiprocessing as mp
from typing import Any
from numpy.typing import NDArray

from sensorflex import Node, Graph, Port


# Define your nodes
class ImageLoadingNode(Node):
    # path: Port[str] = Port(None)
    i: Port[int]
    bgr: Port[NDArray]

    def __init__(self, name: str | None = None) -> None:
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

    def __init__(self, name: str | None = None) -> None:
        super().__init__(name)
        self.bgr = Port(None)
        self.rgb = Port(None)

    def forward(self):
        bgr = ~self.bgr
        rgb = bgr[:, :, [2, 1, 0]]
        self.rgb <<= rgb


class WaitNode(Node):
    wait_seconds: int

    def __init__(self, wait_seconds: int, name: str | None = None) -> None:
        super().__init__(name)
        self.wait_seconds = wait_seconds

    def forward(self):
        time.sleep(self.wait_seconds)


class PrintNode(Node):
    field: Port[Any]

    def __init__(self, name: str | None = None) -> None:
        super().__init__(name)
        self.field = Port(None)

    def forward(self):
        print(~self.field)


def get_graph(i):
    g = Graph()

    n1 = g << ImageLoadingNode()

    n2 = g << ImageTransformationNode()
    g <<= n1.bgr >> n2.bgr

    __ = g << WaitNode(i)

    n3 = g << PrintNode()
    g <<= n3.field << n2.bgr

    return g, n1


def run(i: int):
    g, n1 = get_graph(i)
    n1.i <<= i
    g.run()


if __name__ == "__main__":
    p = mp.Pool(5)
    p.map(run, range(5))
