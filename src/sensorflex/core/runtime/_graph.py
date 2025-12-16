"""A library for building SensorFlex computation graph."""

from __future__ import annotations

import asyncio
import inspect
from threading import Thread
from typing import TypeVar, List, Dict, Self, overload, cast, Awaitable

from ._node import Node
from ._flow import Edge, Port, GraphPartGroup
from sensorflex.utils.logging import get_logger, Perf

logger = get_logger("Graph")

NP = TypeVar("NP", bound=Node)


def _call_func(bind) -> None:
    if bind is None:
        return

    # Case 1: bind is already a coroutine object
    if inspect.iscoroutine(bind):
        loop = asyncio.get_running_loop()
        loop.create_task(bind)
        return

    # Case 2: bind is a function: call it
    result = bind()

    # Case 3: result is an awaitable (async func)
    if inspect.isawaitable(result):
        assert result is Awaitable
        loop = asyncio.get_running_loop()
        loop.create_task(result)


class Pipeline:
    """A pipeline is a part of graph that describes the execution order of nodes."""

    nodes: List[Node]
    edges: List[Edge]
    parent_graph: Graph

    _from_node: Node | None
    _node_edge_map: Dict[Node, List[Edge]]

    def __init__(self, parent_graph: Graph, from_node: Node | None = None) -> None:
        self.nodes = []
        self.edges = []
        self._node_edge_map = {}
        self.parent_graph = parent_graph
        self._from_node = from_node

    def add_edge(self, edge: Edge):
        if (
            not edge.src._is_branched_pipeline_head
            and edge.src.parent_node not in self.nodes
        ):
            logger.warning(
                f"Adding an edge from a node that is not part of the pipeline. From: {edge.src.parent_node.name}, To: {edge.dst.parent_node.name}"
            )
        if edge.dst.parent_node not in self.nodes:
            logger.warning(
                f"Adding an edge to a node that is not part of the pipeline. From: {edge.src.parent_node.name}, To: {edge.dst.parent_node.name}"
            )

        self.edges.append(edge)

        parent_node = edge.src.parent_node
        if parent_node in self._node_edge_map:
            self._node_edge_map[parent_node].append(edge)
        else:
            self._node_edge_map[parent_node] = [edge]

    def run(self):
        if self._from_node:
            self.push_data_for_node(self._from_node)

        for node in self.nodes:
            with Perf(f"[{node.name}] forward."):
                # t0 = asyncio.get_running_loop().time()
                # with Perf("forward"):
                node.forward()
            # dt = (asyncio.get_running_loop().time() - t0) * 1000
            # print(f"{node.name} forward took {dt:.4f} ms.")

            # t0 = asyncio.get_running_loop().time()
            self.push_data_for_node(node)
            # dt = (asyncio.get_running_loop().time() - t0) * 1000
            # print(f"{node.name} push_data took {dt:.4f} ms.")

    def push_data_for_node(self, node: Node):
        if edges := self._node_edge_map.get(node):
            for edge in edges:
                # Flow data between ports
                edge.dst.value = edge.src.value
                self.parent_graph._on_port_change(edge.dst, by_sensorflex=True)

    def add(self, node_or_edge):
        self += node_or_edge

    @overload
    def __iadd__(self, items: Node) -> Self: ...

    @overload
    def __iadd__(self, items: Edge) -> Self: ...

    @overload
    def __iadd__(self, items: GraphPartGroup) -> Self: ...

    def __iadd__(self, items: Node | Edge | GraphPartGroup) -> Self:
        if isinstance(items, Edge):
            edge: Edge = items
            self.add_edge(edge)
        elif isinstance(items, Node):
            node: Node = items

            if node.parent_graph is None:
                node = self.parent_graph.add_node(node)

            self.nodes.append(node)
        elif isinstance(items, GraphPartGroup):
            for v in items:
                self += v

        return self

    def __add__(self, part: Node | Edge | GraphPartGroup) -> Self:
        self += part
        return self


G = TypeVar("G", bound=Node)


class GraphSyntaxMixin:
    nodes: List[Node]
    edges: List[Edge]

    _port_pipeline_map: Dict[Port, List[Pipeline]]

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

    def __iadd__(self, part: Node | GraphPartGroup) -> Self:
        if isinstance(part, Node):
            self.add_node(part)
        else:
            self = self + part
        return self

    def __add__(self, part: Node | GraphPartGroup) -> Self:
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
                    _call_func(port.on_change())

    def _exec_pipelines(self, port: Port):
        for pipeline in self._port_pipeline_map[port]:
            pipeline.run()

    def schedule_exec(self, port: Port):
        self._loop.call_soon_threadsafe(self._exec_pipelines, port)

    def run_main_pipeline(self):
        def _exec():
            self.main_pipeline.run()

        self._loop.call_soon_threadsafe(_exec)

    async def wait_forever(self):
        while True:
            _ = await self._event_queue.get()

    def wait_forever_as_task(self) -> asyncio.Task:
        return asyncio.create_task(self.wait_forever())

    def run_in_thread(self) -> Thread:
        t = Thread(target=self.run_main_pipeline)
        t.start()
        return t


class Graph(GraphSyntaxMixin, GraphExecMixin):
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
