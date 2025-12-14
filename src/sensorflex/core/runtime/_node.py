"""A library for session management."""

from __future__ import annotations

from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ._flow import Port, Edge, GraphPartGroup
    from ._graph import Graph


class Node:
    name: str
    parent_graph: Graph | None

    _ports: dict[str, Port[Any]]

    def __init__(self, name: Optional[str] = None) -> None:
        super().__init__()
        self.name = name if name is not None else self.__class__.__name__
        self.parent_graph = None
        self._ports = {}

    def forward(self) -> None:
        pass

    def __add__(self, items: Node | Edge | GraphPartGroup) -> GraphPartGroup:
        from ._flow import GraphPartGroup

        if isinstance(items, GraphPartGroup):
            return self + items
        else:
            return GraphPartGroup([self, items])


class IntegratedNode:
    """Integrate graph into a node."""

    __graph: Graph

    pass
