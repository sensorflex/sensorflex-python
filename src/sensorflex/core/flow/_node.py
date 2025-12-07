"""A library for session management."""

from __future__ import annotations

from typing import Any, Optional, TYPE_CHECKING
from ._operator import Port, Action

if TYPE_CHECKING:
    from ._graph import Graph


class Node:
    name: str
    parent_graph: Graph

    _ports: dict[str, Port[Any]]

    def __init__(self, name: Optional[str] = None) -> None:
        super().__init__()
        self.name = name if name is not None else self.__class__.__name__

    def __register_ports__(self) -> None:
        """
        Discover all Port[...] attributes for this node instance and
        attach them to self._ports, setting parent_node on each Port.
        """
        self._ports = {}

        cls = type(self)

        # 1) Preferred path: your transform already tells us which attributes
        #    are "static" (like dataclass fields), including Ports.
        static_attrs = getattr(cls, "__static_attributes__", None)

        if static_attrs is not None:
            # static_attrs is something like ('bgr', 'i')
            for name in static_attrs:
                port_obj = getattr(self, name, None)
                if isinstance(port_obj, Port):
                    self._ports[name] = port_obj
                    port_obj.parent_node = self

                if isinstance(port_obj, Action):
                    port_obj.parent_node = self

            return  # we're done

        # 2) Fallback if there is no __static_attributes__: scan instance attrs
        for name, value in self.__dict__.items():
            if isinstance(value, Port):
                self._ports[name] = value
                value.parent_node = self

            if isinstance(value, Action):
                value.parent_node = self

    def forward(self) -> None:
        pass


class IntegratedNode:
    """Integrate graph into a node."""

    __graph: Graph

    pass
