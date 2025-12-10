"""A library for building SensorFlex computation graph."""

from __future__ import annotations

import asyncio
import inspect
from threading import Thread
from typing import TypeVar, List, Dict, Self, overload, cast, Awaitable

from ._node import Node
from ._flow import Edge, Port

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

    _node_edge_map: Dict[Node, List[Edge]]

    def __init__(self, parent_graph: Graph) -> None:
        self.nodes = []
        self.edges = []
        self._node_edge_map = {}
        self.parent_graph = parent_graph

    def add_edge(self, edge_from: Port, edge_to: Port):
        t = Edge(edge_from, edge_to)
        self.edges.append(t)

        if edge_to.parent_node in self._node_edge_map:
            self._node_edge_map[edge_to.parent_node].append(t)
        else:
            self._node_edge_map[edge_to.parent_node] = [t]

    def run(self):
        for node in self.nodes:
            if edges := self._node_edge_map.get(node):
                for edge in edges:
                    # Flow data between ports
                    edge.dst.value = edge.src.value
                    self.parent_graph._on_port_change(edge.dst, by_sensorflex=True)

            node.forward()

    def add(self, node_or_edge):
        self += node_or_edge

    @overload
    def __iadd__(self, node_or_edge: Node) -> Self: ...

    @overload
    def __iadd__(self, node_or_edge: Edge) -> Self: ...

    def __iadd__(self, node_or_edge: Node | Edge) -> Self:
        if isinstance(node_or_edge, Edge):
            edge: Edge = node_or_edge
            self.edges.append(edge)

            receiver_node = edge.dst.parent_node

            if receiver_node not in self._node_edge_map:
                self._node_edge_map[receiver_node] = [edge]
            else:
                self._node_edge_map[receiver_node].append(edge)

        else:
            node: Node = node_or_edge

            if node.parent_graph is None:
                node = self.parent_graph.add_node(node)

            self.nodes.append(node)

        return self

    def __add__(self, node: Node) -> Self:
        if node.parent_graph is None:
            node = self.parent_graph.add_node(node)
        self.nodes.append(node)
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

    def __iadd__(self, node: Node) -> Self:
        self.add_node(node)
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

    def _exec_node(self, node: Node):
        # async with asyncio.TaskGroup() as tg:
        if node in self._node_edge_map:
            if self._node_edge_map[node] is not None:
                for edge in self._node_edge_map[node]:
                    # Flow data between ports
                    edge.dst.value = edge.src.value

        node.forward()

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
