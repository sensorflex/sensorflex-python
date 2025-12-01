"""A library for session management."""

from __future__ import annotations

from typing import Any, Generic, TypeVar, get_type_hints, get_origin, Tuple, Optional


T = TypeVar("T")
N = TypeVar("N")


class Port(Generic[T]):
    value: Optional[T]
    parent_node: Node

    def __init__(self, value: Optional[T]) -> None:
        self.value = value

    def __ilshift__(self, value: T) -> Port[T]:
        self.value = value
        return self

    def __invert__(self) -> T:
        assert self.value is not None
        return self.value

    def __rshift__(self, other: Port[N]) -> Tuple[Port[T], Port[N]]:
        return (self, other)

    def __lshift__(self, other: Port[N]) -> Tuple[Port[N], Port[T]]:
        return (other, self)


class Node:
    name: str

    def __init__(self, name: Optional[str] = None) -> None:
        super().__init__()
        self.name = name if name is not None else self.__class__.__name__

        self.__post_init__()

    def __post_init__(self) -> None:
        self._ports: dict[str, Port[Any]] = {}

        hints = get_type_hints(type(self))

        for name, hint in hints.items():
            if get_origin(hint) is Port and hasattr(self, name):
                port_obj = getattr(self, name)

                if isinstance(port_obj, Port):
                    self._ports[name] = port_obj
                    port_obj.parent_node = self

        # print(hints)

    def forward(self):
        pass
