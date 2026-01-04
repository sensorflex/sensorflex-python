"""A library for building SensorFlex computation graph."""

from __future__ import annotations

import asyncio
import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass
from threading import Thread
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    cast,
    overload,
    runtime_checkable,
)

from typing_extensions import Generic, TypeVar

from sensorflex.utils.logging import get_logger

logger = get_logger("Graph")


class Instruction(ABC):
    @abstractmethod
    def forward(self) -> None:
        pass

    def __add__(self, items: Instruction | List[Instruction]) -> List[Instruction]:
        if isinstance(items, list):
            return [self] + items
        else:
            return [self, items]


T_co = TypeVar("T_co", covariant=True)
T_contra = TypeVar("T_contra", contravariant=True)


@runtime_checkable
class ReceivablePort(Protocol[T_contra]):
    value: Optional[Any]
    meta: Any
    parent_node: Node

    def __lshift__(self, other: TransmissiblePort[T_contra]) -> Edge: ...


@runtime_checkable
class TransmissiblePort(Protocol[T_co]):
    value: Optional[Any]
    meta: Any
    parent_node: Node

    def __rshift__(self, other: ReceivablePort[T_co]) -> Edge: ...


TP = TypeVar("TP")
TM = TypeVar("TM", default=None)  # type: ignore
TT = TypeVar("TT")


class Port(Generic[TP, TM]):
    value: Optional[TP]
    meta: TM

    parent_node: Node

    on_change: Callable[[], Awaitable[Any]] | Awaitable[Any] | None

    def __init__(
        self,
        value: Optional[TP],
        # meta: TM | None = None,
        on_change: Callable[[], Awaitable[Any]] | Awaitable[Any] | None = None,
    ) -> None:
        self.value = value
        # self.meta = meta
        self.on_change = on_change

    def __ilshift__(self, value: TP | Tuple[TP, TM]) -> Port[TP, TM]:
        """port <<= value: set values of a port."""
        if isinstance(value, tuple) and len(value) == 2:
            self.value, self.meta = value
        else:
            self.value = value

        g = self.parent_node.parent_graph
        assert g is not None
        g._on_port_change(self, True)

        return self

    def __invert__(self) -> TP:
        """~port: get values from a port."""
        assert self.value is not None
        return self.value

    def get_branched_pipeline(self) -> Pipeline:
        from ._graph import Pipeline

        g = self.parent_node.parent_graph
        assert g is not None

        p = Pipeline(g)
        g.add_pipeline(self, p)

        return p

    def add_pipeline(self, others: Instruction | Sequence[Instruction]) -> Port[TP, TM]:
        p = self.get_branched_pipeline()
        p += others
        p.check_edges()
        return self

    def __iadd__(self, others: Instruction | Sequence[Instruction]) -> Port[TP, TM]:
        return self.add_pipeline(others)

    def connect(self, other: ReceivablePort[TP], transfer_meta: bool = False) -> Edge:
        return Edge(self, other, transfer_meta)

    def __gt__(self, other: ReceivablePort[TP]) -> Edge:
        return self.connect(other)

    def __rshift__(self, other: ReceivablePort[TP]) -> Edge:
        return self.connect(other, transfer_meta=True)

    def __le__(self, other: TransmissiblePort[TP]) -> Edge:
        return Edge(other, self, False)

    def __lshift__(self, other: TransmissiblePort[TP]) -> Edge:
        # return other.connect(self, transfer_meta=True)
        return Edge(other, self, True)

    def print(self) -> PortView[TP, TM]:
        def t(v):
            print(self.value)
            return v

        return PortView(self, t)

    def map(self, func: Callable[[TP], TT]) -> PortView[TT, TM]:
        return PortView(self, func)

    def match(
        self,
        func: Callable[[TP], Any],
        branches: Dict[Any, List[Instruction]],
    ) -> List[Instruction]:
        results: List[Instruction] = []
        for k in branches.keys():

            def cond(k=k):
                v = self.value
                assert v is not None
                return func(v) == k

            results.append(Block(branches[k], condition=cond))

        return results

    def isinstance(
        self,
        data_type: type[TT],
        branch_func: Callable[[Port[TT]], Instruction | Sequence[Instruction]],
    ) -> Block:
        # Important: avoid binding self.value at definition time.
        def cond(t=data_type):
            return isinstance(self.value, t)

        group = branch_func(self)  # type: ignore

        if isinstance(group, Instruction):
            return Block([group], condition=cond)
        else:
            return Block(list(group), condition=cond)


class PortView(Instruction, TransmissiblePort, Generic[TP, TM]):
    host: Port[Any, TM] | PortView[Any, TM]
    view_transform: Callable[[Any], TP]

    parent_node: Node

    _value_cache: TP | None = None

    def __init__(
        self,
        host: Port[Any, TM] | PortView[Any, TM],
        view_transform: Callable[[Any], TP],
    ) -> None:
        super().__init__()
        self.host = host
        self.view_transform = view_transform

        self.parent_node = host.parent_node

    @property
    def value(self) -> TP:
        if self._value_cache is not None:
            return self._value_cache

        t = self.host.value
        assert t is not None
        t = self.view_transform(t)
        return t

    @property
    def meta(self) -> TM | None:
        return self.host.meta

    def forward(self) -> None:
        t = self.host.value
        assert t is not None
        t = self.view_transform(t)
        self._value_cache = t

    def connect(self, other: ReceivablePort[TP], transfer_meta: bool = False) -> Edge:
        return Edge(self, other, transfer_meta)

    def __gt__(self, other: ReceivablePort[TP]) -> Edge:
        return self.connect(other)

    def __rshift__(self, other: ReceivablePort[TP]) -> Edge:
        return self.connect(other, transfer_meta=True)

    def print(self) -> PortView[TP, TM]:
        def t(v):
            print(self.value, self.meta)
            return v

        return PortView(self, t)

    def map(self, func: Callable[[TP], TT]) -> PortView[TT, TM]:
        return PortView(self, func)

    def isinstance(
        self,
        data_type: type[TT],
        branch_func: Callable[[PortView[TT, TM]], Instruction | Sequence[Instruction]],
    ) -> Block:
        # Important: avoid binding self.value at definition time.
        def cond(t=data_type):
            return isinstance(self.value, t)

        group = branch_func(self)  # type: ignore

        if isinstance(group, Instruction):
            return Block([group], condition=cond)
        else:
            return Block(list(group), condition=cond)

    def match(
        self,
        func: Callable[[TP], Any],
        branches: Dict[Any, List[Instruction]],
    ) -> List[Instruction]:
        results: List[Instruction] = []
        for k in branches.keys():

            def cond(k=k):
                v = self.value
                assert v is not None
                return func(v) == k

            results.append(Block(branches[k], condition=cond))

        return results


@dataclass(frozen=True)
class Edge(Instruction):
    src: TransmissiblePort[Any]
    dst: ReceivablePort[Any]

    _transfer_meta: bool = False

    def forward(self):
        self.dst.value = self.src.value

        if self._transfer_meta and hasattr(self.src, "meta"):
            self.dst.meta = self.src.meta


@dataclass(frozen=True)
class Block(Instruction):
    parts: List[Instruction]
    condition: Callable | None = None

    def forward(self):
        # if self.condition is None or self.condition():
        for i in self.parts:
            i.forward()


# TODO: support better syntax.
@dataclass(frozen=True)
class Condition:
    cond_func: Callable[[], bool]

    def eval(self):
        return self.cond_func()


CT = TypeVar("CT")


@dataclass(frozen=True)
class TypeCondition(Generic[CT]):
    obj: Any
    target_type: type[CT]

    def eval(self):
        return isinstance(self.obj, self.target_type)


@overload
def when(
    condition: TypeCondition[CT], branch: Callable[[CT], List[Instruction]]
) -> Block: ...
@overload
def when(condition: Condition, branch: Callable[[], List[Instruction]]) -> Block: ...


def when(condition, branch) -> Block:
    if isinstance(condition, TypeCondition):
        return Block(branch(condition.obj), condition)
    else:
        return Block(branch(), condition)


class Node(Instruction):
    name: str
    parent_graph: Graph | None

    _ports: dict[str, Port[Any]]

    _pre_ins: List[Instruction]
    _pos_ins: List[Instruction]

    _exec_cond: Callable[[Any], bool] | None = None

    def __init__(self, name: Optional[str] = None) -> None:
        super().__init__()
        self.name = name if name is not None else self.__class__.__name__
        self.parent_graph = None
        self._ports = {}

        self._pre_ins = []
        self._pos_ins = []

    def forward(self) -> None: ...

    def _as_list(self, x: Instruction | Sequence[Instruction]) -> list[Instruction]:
        # Convert any Sequence[Instruction] (list/tuple/...) to list[Instruction]
        if isinstance(x, Instruction):
            return [x]
        # x is a Sequence[Instruction]
        return list(x)

    def __getitem__(
        self,
        ins: Instruction
        | Sequence[Instruction]
        | Tuple[
            None | Instruction | Sequence[Instruction],
            Instruction | Sequence[Instruction],
        ],
    ):
        # reset or keep? (I'm assuming "set" semantics each call)
        self._pre_ins = []
        self._pos_ins = []

        if isinstance(ins, tuple):
            pre, pos = ins

            if pre is not None:
                self._pre_ins = self._as_list(pre)

            self._pos_ins = self._as_list(pos)
            return self

        # non-tuple cases
        if isinstance(ins, Instruction):
            self._pre_ins = [ins]
        else:
            # Sequence[Instruction]
            self._pre_ins = list(ins)

        return self


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

    _instructions: List[Instruction]

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
        # TODO: rethink how to design this feature.
        #
        # for edge in self.edges:
        #     if (
        #         not edge.src._is_branched_pipeline_head
        #         and edge.src.parent_node not in self.nodes
        #     ):
        #         logger.warning(
        #             "Adding an edge from a node that is not part of the pipeline."
        #             + f"From: {edge.src.parent_node.name}, "
        #             + f"To: {edge.dst.parent_node.name}"
        #         )
        #     if edge.dst.parent_node not in self.nodes:
        #         logger.warning(
        #             "Adding an edge to a node that is not part of the pipeline."
        #             + f"From: {edge.src.parent_node.name}, "
        #             + f"To: {edge.dst.parent_node.name}"
        #         )
        pass

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

        def _exec(instructions: List[Instruction]):
            pvs: List[PortView] = []

            for i in instructions:
                if isinstance(i, PortView):
                    # Forward on PortView evaluates the transformation
                    # and cache the result
                    i.forward()
                    pvs.append(i)
                elif isinstance(i, Edge):
                    i.forward()
                    self.parent_graph._on_port_change(
                        cast(Port, i.dst), by_sensorflex=True
                    )
                elif isinstance(i, Node):
                    _exec(i._pre_ins)
                    i.forward()
                    _exec(i._pos_ins)

                elif isinstance(i, Block):
                    if i.condition is None or i.condition():
                        _exec(i.parts)
                else:
                    raise ValueError("Unrecognized graph part type.")

            # Clear cache after pipeline execution.
            for pv in pvs:
                pv._value_cache = None

        _exec(self._instructions)

    def add(self, node_or_edge):
        self += node_or_edge

    @overload
    def __iadd__(self, item: Instruction) -> Pipeline: ...

    @overload
    def __iadd__(self, item: Sequence[Instruction]) -> Pipeline: ...

    def __iadd__(self, item: Instruction | Sequence[Instruction]) -> Pipeline:
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

        elif isinstance(item, PortView):
            self._instructions.append(item)

        elif isinstance(item, Block):
            self._instructions.append(item)

        elif isinstance(item, tuple) or isinstance(item, list):
            for v in item:
                self += v

        return self


G = TypeVar("G", bound=Node)


class Graph:
    nodes: List[Node]
    edges: List[Edge]

    main_pipeline: Pipeline

    _batching: bool
    _node_edge_map: Dict[Node, List[Edge]]

    _loop: asyncio.AbstractEventLoop

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

    def _register_ports(self, node: Node) -> bool:
        for name in dir(node):
            obj = getattr(node, name)

            if isinstance(obj, Port):
                node._ports[name] = obj
                obj.parent_node = node

        return True

    def _on_port_change(self, port: Port[Any, Any], by_sensorflex: bool = False):
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
        await asyncio.Future()

    def wait_forever_as_task(self) -> asyncio.Task:
        return asyncio.create_task(self.wait_forever())

    def run_in_thread(self) -> Thread:
        t = Thread(target=self.main_pipeline.run)
        t.start()
        return t

    def add_node(self, node: G) -> G:
        node.parent_graph = self
        self._register_ports(node)
        self.nodes.append(node)
        return node

    def add_pipeline(self, port: Port[Any, Any], pipeline: Pipeline):
        if port in self._port_pipeline_map:
            self._port_pipeline_map[port].append(pipeline)
        else:
            self._port_pipeline_map[port] = [pipeline]

    def __iadd__(self, part: Node | Sequence[Node]) -> Graph:
        if isinstance(part, Node):
            self.add_node(part)
        else:
            for n in part:
                self.add_node(n)
        return self

    def __lshift__(self, node: G) -> G:
        """To support: g << SomeNode()"""
        node = self.add_node(node)
        return node

    def connect(self, left_port: Port, right_port: Port) -> Edge:
        return left_port >> right_port
