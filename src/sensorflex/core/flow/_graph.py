"""A library for SensorFlex computation graph."""

from __future__ import annotations
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from typing import (
    Any,
    Generic,
    TypeVar,
    get_type_hints,
    get_origin,
    Tuple,
    List,
    Optional,
)


T = TypeVar("T")
N = TypeVar("N")
M = TypeVar("M")


class Port(Generic[T]):
    value: Optional[T]
    parent_node: Node

    def __init__(self, value: Optional[T]) -> None:
        self.value = value

    def __ilshift__(self, value: T) -> Port[T]:
        # print("binding successful.")
        # print(self.parent_node)

        self.value = value

        return self

    def __invert__(self) -> T:
        print("invert called")
        assert self.value is not None
        return self.value

    def __rshift__(self, other: Port[N]) -> Tuple[Port[T], Port[N]]:
        return (self, other)


class Operator:
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("Must implement forward.")


@dataclass
class BgrToRgbOp(Operator):
    bgr: NDArray
    use_resizing: bool = False

    def forward(self) -> NDArray:
        rgb = self.bgr[::, [2, 1, 0]]
        return rgb


class Node:
    name: str

    def __init__(self, name: Optional[str] = None) -> None:
        super().__init__()
        self.name = name if name is not None else self.__class__.__name__

        self.__post_init__()

    def __post_init__(self) -> None:
        self._ports: dict[str, Port[Any]] = {}

        # pull *resolved* type hints from the class
        hints = get_type_hints(type(self))

        for name, hint in hints.items():
            # detect Port[...] from the annotation
            if get_origin(hint) is Port and hasattr(self, name):
                port_obj = getattr(self, name)
                print(port_obj)
                # optionally also check isinstance(port_obj, Port)
                if isinstance(port_obj, Port):
                    self._ports[name] = port_obj

                    # Assign port obj's parent to self
                    port_obj.parent_node = self
            # elif not hasattr(self, name):
            # print(hint)
            # setattr(self, name, Port())

        # print(hints)

    def forward(self):
        pass


G = TypeVar("G", bound=Node)


class ImageLoadingNode(Node):
    path: Port[str] = Port(None)
    bgr: Port[NDArray] = Port(None)

    def forward(self):
        # do something with ~self.path
        self.bgr <<= np.zeros((1080, 1920, 3), dtype=np.uint8)


class ImageTransformationNode(Node):
    bgr: Port[NDArray] = Port(None)
    rgb: Port[NDArray] = Port(None)

    def forward(self):
        bgr = ~self.bgr
        rgb = bgr[::, [2, 1, 0]]
        self.rgb <<= rgb


class Graph:
    def __init__(self) -> None:
        self.nodes: List[Node] = []
        self.edges = []

    def add(self, node: G) -> G:
        self.nodes.append(node)
        return node

    def connect(self, left_node, right_node) -> None:
        pass

    def run(self):
        prev_node = None
        for node in self.nodes:
            if prev_node is not None:
                # TODO: Implement actual data binding.
                pass

            node.forward()
            prev_node = node

    def __lshift__(self, node: G) -> G:
        self.add(node)
        return node

    def __ilshift__(self, edge: Tuple[Port, Port]) -> Graph:
        self.edges.append(edge)
        return self

    def __enter__(self) -> Graph:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


if __name__ == "__main__":
    with Graph() as g:
        n1 = g << ImageLoadingNode()
        n2 = g << ImageTransformationNode()
        g <<= n1.bgr >> n2.bgr

        g.run()
