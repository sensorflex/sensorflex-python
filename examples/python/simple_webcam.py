"""A simple compute graph example."""

import cv2
import numpy as np
from numpy.typing import NDArray

from sensorflex import Node, Graph, Port


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
        print(arr.shape)


# Define a graph
def get_graph():
    g = Graph()
    n1 = g << VFXNode()
    n2 = g << PrintShapeNode()
    g <<= n1.output >> n2.arr

    return g, n1


def main():
    g, n1 = get_graph()

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        # Apply graph
        n1.frame <<= frame
        g.run()

        cv2.imshow("Webcam", ~n1.output)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


if __name__ == "__main__":
    main()
