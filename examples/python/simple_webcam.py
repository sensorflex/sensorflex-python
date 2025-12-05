"""A simple compute graph example."""

import cv2
import asyncio
from numpy.typing import NDArray

from sensorflex import Node, Graph, Port, WebcamNode


class VFXNode(Node):
    frame: Port[NDArray]
    output: Port[NDArray]

    def __init__(self, name: str | None = None) -> None:
        super().__init__(name)
        self.frame = Port(None)
        self.output = Port(None)

    def forward(self):
        # do something with ~self.path
        x = (~self.frame).copy()
        x = x[:, :, ::-1]
        self.output <<= x


class PrintShapeNode(Node):
    arr: Port[NDArray]

    def __init__(self, name: str | None = None) -> None:
        super().__init__(name)
        self.arr = Port(None)

    def forward(self):
        arr = ~self.arr
        cv2.imshow("Print", arr)
        cv2.waitKey(1)  # Important for OpenCV GUI.
        print(arr.shape)


# Define a graph
def get_graph():
    g = Graph()
    n1 = g << WebcamNode()
    n2 = g << VFXNode()
    g <<= n1.last_frame >> n2.frame
    n3 = g << PrintShapeNode()
    g <<= n2.output >> n3.arr

    g.watch(n1.last_frame)

    return g


async def main():
    g = get_graph()
    await g.run_and_wait_forever()


if __name__ == "__main__":
    asyncio.run(main())
