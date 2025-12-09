"""A library for building SensorFlex computation graph."""

from __future__ import annotations

import asyncio
from threading import Thread
from typing import TypeVar, Tuple, List, Dict, Self, overload, cast

from ._node import Node
from ._operator import Port, Event

NP = TypeVar("NP", bound=Node)


class Pipeline:
    """A pipeline is a part of graph that describes the execution order of nodes."""

    nodes: List[Node]
    edges: List[Tuple[Port | Event, Port]]
    parent_graph: Graph

    _node_edge_map: Dict[Node, List[Tuple[Port | Event, Port]]]

    def __init__(self, parent_graph: Graph) -> None:
        self.nodes = []
        self.edges = []
        self._node_edge_map = {}
        self.parent_graph = parent_graph

    def add_edge(self, edge_from: Event | Port, edge_to: Port):
        t = (edge_from, edge_to)
        self.edges.append(t)

        if edge_to.parent_node in self._node_edge_map:
            self._node_edge_map[edge_to.parent_node].append(t)
        else:
            self._node_edge_map[edge_to.parent_node] = [t]

    def run(self):
        for node in self.nodes:
            # async with asyncio.TaskGroup() as tg:
            if node in self._node_edge_map:
                if self._node_edge_map[node] is not None:
                    for port_a, port_b in self._node_edge_map[node]:
                        # Flow data between ports
                        port_b.value = port_a.value

            node.forward()

    def add(self, node_or_edge):
        self += node_or_edge

    @overload
    def __iadd__(self, node_or_edge: Node) -> Self: ...

    @overload
    def __iadd__(self, node_or_edge: Tuple[Port | Event, Port]) -> Self: ...

    def __iadd__(self, node_or_edge: Node | Tuple[Port | Event, Port]) -> Self:
        if isinstance(node_or_edge, tuple):
            edge: Tuple[Port | Event, Port] = node_or_edge
            self.edges.append(edge)

            receiver_node = edge[1].parent_node
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
    edges: List[Tuple[Port, Port]]

    _node_edge_map: Dict[Node, List[Tuple[Port, Port]]]
    _action_pipeline_map: Dict[Event, List[Pipeline]]

    def _register_ports(self, node: Node) -> bool:
        for name in dir(node):
            obj = getattr(node, name)

            if isinstance(obj, Port):
                node._ports[name] = obj
                obj.parent_node = node

            if isinstance(obj, Event):
                obj.parent_node = node

        return True

    def add_node(self, node: G) -> G:
        node.parent_graph = cast(Graph, self)
        self._register_ports(node)
        self.nodes.append(node)
        return node

    def add_pipeline(self, action: Event, pipeline: Pipeline):
        if action in self._action_pipeline_map:
            self._action_pipeline_map[action].append(pipeline)
        else:
            self._action_pipeline_map[action] = [pipeline]

    def __iadd__(self, node: Node) -> Self:
        self.add_node(node)
        return self

    def __lshift__(self, node: G) -> G:
        """To support: g << SomeNode()"""
        node = self.add_node(node)
        return node

    def connect(self, left_port: Port, right_port: Port) -> Tuple[Port, Port]:
        return left_port >> right_port


class GraphExecMixin:
    main_pipeline: Pipeline
    _event_queue: asyncio.Queue[Port]

    _node_edge_map: Dict[Node, List[Tuple[Port, Port]]]
    _action_pipeline_map: Dict[Event, List[Pipeline]]

    _loop: asyncio.AbstractEventLoop

    def _exec_node(self, node: Node):
        # async with asyncio.TaskGroup() as tg:
        if node in self._node_edge_map:
            if self._node_edge_map[node] is not None:
                for port_a, port_b in self._node_edge_map[node]:
                    # Flow data between ports
                    port_b.value = port_a.value

        node.forward()

    def _exec_pipelines(self, action: Event):
        if action in self._action_pipeline_map:
            for pipeline in self._action_pipeline_map[action]:
                pipeline.run()

    def schedule_exec(self, action: Event):
        self._loop.call_soon_threadsafe(self._exec_pipelines, action)

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
    edges: List[Tuple[Port, Port]]

    main_pipeline: Pipeline

    _batching: bool
    _node_edge_map: Dict[Node, List[Tuple[Port, Port]]]

    _loop: asyncio.AbstractEventLoop
    _event_queue: asyncio.Queue[Port]

    def __init__(self) -> None:
        self.nodes = []
        self.edges = []

        self.main_pipeline = Pipeline(self)

        # Ports we care about
        self._node_edge_map: Dict[Node, List[Tuple[Port, Port]]] = {}
        self._action_pipeline_map: Dict[Event, List[Pipeline]] = {}

        # For batched (sync) updates
        self._batching = False

        # For async/event-driven mode in a dedicated thread
        self._loop = asyncio.get_event_loop()
        self._event_queue = asyncio.Queue()
