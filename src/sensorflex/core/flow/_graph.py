"""A library for building SensorFlex computation graph."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Any, TypeVar, Tuple, List, Dict

from sensorflex.core.flow._node import Node, Port


G = TypeVar("G", bound=Node)


class Graph:
    def __init__(self) -> None:
        self.nodes: List[Node] = []
        self.edges = []

        self.__node_edge_map: Dict[Node, List[Tuple[Port, Port]]] = {}

    def add(self, node: G) -> G:
        self.nodes.append(node)
        return node

    def connect(self, left_node, right_node) -> None:
        pass

    def run(self):
        for node in self.nodes:
            if node in self.__node_edge_map:
                if self.__node_edge_map[node] is not None:
                    for port_a, port_b in self.__node_edge_map[node]:
                        # Flow data between ports
                        port_b.value = port_a.value

            node.forward()

    def __lshift__(self, node: G) -> G:
        self.add(node)
        return node

    def __ilshift__(self, edge: Tuple[Port, Port]) -> Graph:
        # Process edges
        self.edges.append(edge)

        receiver_node = edge[1].parent_node
        if receiver_node not in self.__node_edge_map:
            self.__node_edge_map[receiver_node] = [edge]
        else:
            self.__node_edge_map[receiver_node].append(edge)

        return self

    def __enter__(self) -> Graph:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


if __name__ == "__main__":

    class ImageLoadingNode(Node):
        # path: Port[str] = Port(None)
        bgr: Port[NDArray] = Port(None)

        def forward(self):
            # do something with ~self.path
            x = np.array([3, 2, 1], dtype=np.uint8)
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

    with Graph() as g:
        n1 = g << ImageLoadingNode()
        n2 = g << ImageTransformationNode()
        g <<= n1.bgr >> n2.bgr

        n3 = g << PrintNode()
        g <<= n3.field << n2.bgr

        g.run()
