"""Hello SensorFlex, your first example!"""

import asyncio
import numpy as np
from numpy.typing import NDArray
from typing import Any

from sensorflex import Node, Port, Graph


class ImageLoadingNode(Node):
    # path: Port[str] = Port(None)
    i: Port[int] = Port(0)
    bgr: Port[NDArray] = Port(None)

    def forward(self):
        # do something with ~self.path
        x = np.array([3, 2, 1], dtype=np.uint8) + ~self.i
        x = x.reshape((1, 1, 3))
        self.bgr <<= x


class ImageTransformationNode(Node):
    bgr: Port[NDArray] = Port(None)
    rgb: Port[NDArray] = Port(None)

    def forward(self):
        bgr = ~self.bgr
        rgb = bgr[:, :, [2, 1, 0]]
        self.rgb <<= rgb


class PrintNode(Node):
    field: Port[Any] = Port(None)

    def forward(self):
        print(~self.field)


async def main():
    # Define a graph
    mp = (g := Graph()).main_pipeline

    mp += (n1 := ImageLoadingNode())
    mp += (n2 := ImageTransformationNode())
    mp += n1.bgr >> n2.bgr

    mp += (n3 := PrintNode())
    mp += n3.field << n2.bgr

    # Now execute
    for i in range(5):
        n1.i <<= i
        g.run_main_pipeline()


if __name__ == "__main__":
    asyncio.run(main())
