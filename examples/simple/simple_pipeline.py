"""A simple example for using multiple pipelines in a graph."""

import time
import asyncio
import numpy as np
from typing import Any
from numpy.typing import NDArray

from sensorflex import Node, Graph, Port
from sensorflex.library.web import WebSocketServerNode
from sensorflex.utils.logging import configure_default_logging

configure_default_logging()


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


class PrintEventNode(Node):
    field: Port[Any]

    def __init__(self, name: str | None = None) -> None:
        super().__init__(name)
        self.field = Port(None)

    def on_field_change(self):
        print(~self.field)


# Define a graph
def get_graph():
    mp = (g := Graph()).main_pipeline

    mp += (n1 := ImageLoadingNode())
    mp += (n2 := ImageTransformationNode())
    mp += n1.bgr >> n2.bgr

    mp += (__ := WaitNode(1))
    mp += (n3 := PrintNode())
    mp += n3.field << n2.bgr

    g += (nw := WebSocketServerNode())
    wp = +nw.o_message
    wp += nw.o_message >> nw.i_message
    wp += n3 + (nw.o_message >> n3.field)

    return g


async def main():
    g = get_graph()
    t = g.wait_forever_as_task()

    await asyncio.sleep(15)
    g.run_main_pipeline()

    t.cancel()


if __name__ == "__main__":
    asyncio.run(main())
