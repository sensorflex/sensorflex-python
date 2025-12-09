"""A simple compute graph example."""

import cv2
import asyncio
from numpy.typing import NDArray

from sensorflex import Node, Graph, Port
from sensorflex.library.cv import WebcamNode


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

    g += (n1 := WebcamNode())
    p = n1.frame.event_pipeline

    p += (n2 := VFXNode())
    p += n1.frame >> n2.frame

    p += (n3 := PrintShapeNode())
    p += n2.output >> n3.arr

    return g


async def main():
    g = get_graph()
    await g.wait_forever()


if __name__ == "__main__":
    asyncio.run(main())
