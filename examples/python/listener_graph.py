"""A simple compute graph example."""

from sensorflex import Node, Port, ListenerGraph

import time
import numpy as np
from numpy.typing import NDArray

from typing import Any


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


# Define a graph
def get_graph():
    g = ListenerGraph()
    n1 = g << ImageLoadingNode()
    g.watch(n1.i)

    n2 = g << ImageTransformationNode()
    g <<= n1.bgr >> n2.bgr

    # __ = g << WaitNode(2)

    n3 = g << PrintNode()
    g <<= n3.field << n2.bgr

    return g, n1


if __name__ == "__main__":
    g, n1 = get_graph()
    g.run_and_wait_in_thread()

    for i in range(5):
        with g.update():
            n1.i <<= i

        time.sleep(2)

    g.stop()
