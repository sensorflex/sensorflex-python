"""A library for session management."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

if TYPE_CHECKING:
    from ._flow import GraphPart, GraphPartGroup, Port
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

    def forward(self) -> None: ...

    def __add__(self, items: GraphPart | GraphPartGroup) -> GraphPartGroup:
        from ._flow import GraphPartGroup

        if isinstance(items, GraphPartGroup):
            parts = items._parts
            parts = [self] + parts
            return GraphPartGroup(parts)  # type: ignore
        else:
            return GraphPartGroup([self, items])


class RouterNode(Node):
    i_data: Port[Any]

    _eval_func: Callable[[Any, Any], bool]
    _cond_map: Dict[Any, Port[Any]]

    def __init__(
        self,
        eval_func: Callable[[Any, Any], bool],
        name: str | None = None,
    ) -> None:
        super().__init__(name)
        self._eval_func = eval_func
        self._cond_map = {}

        from ._flow import Port

        self.i_data = Port(None)

    def forward(self) -> None:
        data = ~self.i_data

        for cond in self._cond_map.keys():
            if self._eval_func(data, cond):
                self._cond_map[cond] <<= data

    def on(self, cond: Any, callback: Callable[[Port], GraphPartGroup]):
        from ._flow import Port

        cond_port = Port(None)
        cond_port.parent_node = self
        self._cond_map[cond] = cond_port
        cond_port += callback(cond_port)


class IntegratedNode:
    """Integrate graph into a node."""

    __graph: Graph

    pass
