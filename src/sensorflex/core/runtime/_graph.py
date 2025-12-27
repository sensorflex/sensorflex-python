"""A library for building SensorFlex computation graph."""

from __future__ import annotations

import asyncio
import inspect
from threading import Thread
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    TypeVar,
    cast,
    overload,
)

from sensorflex.utils.logging import get_logger

from ._flow import Block, Edge, Empty, GraphPartGroup, Port
from ._node import Node

if TYPE_CHECKING:
    from ._flow import GraphPart

logger = get_logger("Graph")


NP = TypeVar("NP", bound=Node)


def _call_func(bind: Callable[[], Awaitable[Any]] | Awaitable[Any] | None) -> None:
    if bind is None:
        return

    # Case A: bind is already an awaitable (e.g., coroutine object / Task / Future)
    if inspect.isawaitable(bind):
        asyncio.ensure_future(bind)
        return

    # Case B: bind is a callable; call it
    result = bind()

    # If callable returns an awaitable, schedule it
    if inspect.isawaitable(result):
        asyncio.ensure_future(result)


class Pipeline:
    """A pipeline is a part of graph that describes the execution order of nodes."""

    nodes: List[Node]
    edges: List[Edge]
    parent_graph: Graph

    _instructions: List[GraphPart]

    _node_edge_map: Dict[Node, List[Edge]]
    _exec_condition: Callable[..., bool] | None = None

    def __init__(
        self,
        parent_graph: Graph,
        exec_condition: Callable | None = None,
    ) -> None:
        self.nodes = []
        self.edges = []
        self._node_edge_map = {}
        self.parent_graph = parent_graph

        self._instructions = []
        self._exec_condition = exec_condition

    def check_edges(self):
        for edge in self.edges:
            if (
                not edge.src._is_branched_pipeline_head
                and edge.src.parent_node not in self.nodes
            ):
                logger.warning(
                    "Adding an edge from a node that is not part of the pipeline."
                    + f"From: {edge.src.parent_node.name}, "
                    + f"To: {edge.dst.parent_node.name}"
                )
            if edge.dst.parent_node not in self.nodes:
                logger.warning(
                    "Adding an edge to a node that is not part of the pipeline."
                    + f"From: {edge.src.parent_node.name}, "
                    + f"To: {edge.dst.parent_node.name}"
                )

    def add_edge(self, edge: Edge):
        self.edges.append(edge)

        parent_node = edge.src.parent_node
        if parent_node in self._node_edge_map:
            self._node_edge_map[parent_node].append(edge)
        else:
            self._node_edge_map[parent_node] = [edge]

    def run(self):
        if self._exec_condition is not None:
            if not self._exec_condition():
                return

        def _exec(instructions: List[GraphPart]):
            for i in instructions:
                if isinstance(i, Edge):
                    i.forward()
                    self.parent_graph._on_port_change(
                        cast(Port, i.dst), by_sensorflex=True
                    )
                elif isinstance(i, Node):
                    i.forward()
                elif isinstance(i, Block):
                    if i.condition is None or i.condition():
                        _exec(i.instructions._parts)
                else:
                    raise ValueError("Unrecognized graph part type.")

        _exec(self._instructions)

    def add(self, node_or_edge):
        self += node_or_edge

    @overload
    def __iadd__(self, item: GraphPart) -> Pipeline: ...

    @overload
    def __iadd__(self, item: GraphPartGroup) -> Pipeline: ...

    def __iadd__(self, item: GraphPart | GraphPartGroup) -> Pipeline:
        if isinstance(item, Edge):
            edge: Edge = item
            self.add_edge(edge)
            self._instructions.append(edge)
        elif isinstance(item, Node):
            node: Node = item

            if node.parent_graph is None:
                node = self.parent_graph.add_node(node)

            self.nodes.append(node)
            self._instructions.append(node)

        elif isinstance(item, Empty):
            _: Empty = item
            # TODO: maybe its not a good idea to have Empty.

        elif isinstance(item, Block):
            self._instructions.append(item)

        elif isinstance(item, GraphPartGroup):
            for v in item:
                self += v

        return self

    def __add__(self, part: Node | Edge | GraphPartGroup) -> Pipeline:
        self += part
        return self


G = TypeVar("G", bound=Node)


class GraphExecMixin:
    main_pipeline: Pipeline
    _event_queue: asyncio.Queue[Port]

    _node_edge_map: Dict[Node, List[Edge]]
    _port_pipeline_map: Dict[Port, List[Pipeline]]

    _loop: asyncio.AbstractEventLoop

    def _on_port_change(self, port: Port, by_sensorflex: bool = False):
        if port in self._port_pipeline_map:
            self.schedule_exec(port)

        if by_sensorflex:
            if port.on_change is not None:
                if isinstance(port.on_change, list):
                    for func in port.on_change:
                        _call_func(func)
                else:
                    _call_func(port.on_change)

    def _exec_pipelines(self, port: Port):
        for pipeline in self._port_pipeline_map[port]:
            pipeline.run()

    def schedule_exec(self, port: Port):
        self._loop.call_soon_threadsafe(self._exec_pipelines, port)

    async def wait_forever(self):
        while True:
            _ = await self._event_queue.get()

    def wait_forever_as_task(self) -> asyncio.Task:
        return asyncio.create_task(self.wait_forever())

    def run_in_thread(self) -> Thread:
        t = Thread(target=self.main_pipeline.run)
        t.start()
        return t


class Graph(GraphExecMixin):
    nodes: List[Node]
    edges: List[Edge]

    main_pipeline: Pipeline

    _batching: bool
    _node_edge_map: Dict[Node, List[Edge]]

    _loop: asyncio.AbstractEventLoop
    _event_queue: asyncio.Queue[Port]

    def __init__(self) -> None:
        self.nodes = []
        self.edges = []

        self.main_pipeline = Pipeline(self)

        # Ports we care about
        self._node_edge_map: Dict[Node, List[Edge]] = {}
        self._port_pipeline_map: Dict[Port, List[Pipeline]] = {}

        # For batched (sync) updates
        self._batching = False

        # For async/event-driven mode in a dedicated thread
        self._loop = asyncio.get_event_loop()
        self._event_queue = asyncio.Queue()

    def _register_ports(self, node: Node) -> bool:
        for name in dir(node):
            obj = getattr(node, name)

            if isinstance(obj, Port):
                node._ports[name] = obj
                obj.parent_node = node

        return True

    def add_node(self, node: G) -> G:
        node.parent_graph = cast(Graph, self)
        self._register_ports(node)
        self.nodes.append(node)
        return node

    def add_pipeline(self, port: Port, pipeline: Pipeline):
        if port in self._port_pipeline_map:
            self._port_pipeline_map[port].append(pipeline)
        else:
            self._port_pipeline_map[port] = [pipeline]

    def __iadd__(self, part: Node | GraphPartGroup) -> Graph:
        if isinstance(part, Node):
            self.add_node(part)
        else:
            self = self + part
        return self

    def __add__(self, part: Node | GraphPartGroup) -> Graph:
        if isinstance(part, GraphPartGroup):
            for n in part:
                assert isinstance(n, Node)
                self.add_node(n)
        else:
            self.add_node(part)

        return self

    def __lshift__(self, node: G) -> G:
        """To support: g << SomeNode()"""
        node = self.add_node(node)
        return node

    def connect(self, left_port: Port, right_port: Port) -> Edge:
        return left_port >> right_port
