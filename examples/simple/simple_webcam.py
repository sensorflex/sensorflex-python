"""A simple compute graph example."""

import asyncio
from typing import Union

import cv2
from numpy.typing import NDArray

from sensorflex import Graph, Node, Port
from sensorflex.library.cv import WebcamNode
from sensorflex.utils.logging import configure_default_logging

configure_default_logging()


class VFXNode(Node):
    i_frame: Port[NDArray]
    o_frame: Port[NDArray]

    def __init__(self, name: Union[str, None] = None) -> None:
        super().__init__(name)
        self.i_frame = Port(None)
        self.o_frame = Port(None)

    def forward(self):
        # do something with ~self.path
        x = (~self.i_frame).copy()
        x = x[:, :, ::-1]
        self.o_frame <<= x


class PrintShapeNode(Node):
    i_arr: Port[NDArray]

    def __init__(self, name: Union[str, None] = None) -> None:
        super().__init__(name)
        self.i_arr = Port(None)

    def forward(self):
        arr = ~self.i_arr
        cv2.imshow("Print", arr)
        cv2.waitKey(1)  # Important for OpenCV GUI.
        print(arr.shape)


# Define a graph
def get_graph():
    g = Graph()

    g += (n1 := WebcamNode())

    p = +n1.o_frame
    p += n1
    p += (n2 := VFXNode())
    p += n1.o_frame >> n2.i_frame
    p += (n3 := PrintShapeNode())
    p += n2.o_frame >> n3.i_arr

    return g


async def main():
    g = get_graph()
    await g.wait_forever()


if __name__ == "__main__":
    asyncio.run(main())
