"""Library for computation flow management."""

from __future__ import annotations

from asyncio import Task, create_task
from dataclasses import dataclass
from enum import Enum, auto
from threading import Thread
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Callable,
    Coroutine,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    Protocol,
    TypeVar,
    runtime_checkable,
)

if TYPE_CHECKING:
    from ._graph import Pipeline
    from ._node import Node


class GraphPartGroup:
    _parts: List[GraphPart]

    def __init__(self, parts: List[GraphPart]) -> None:
        self._parts = parts

    def __iter__(self) -> Iterator[GraphPart]:
        return iter(self._parts)

    def __add__(self, items: GraphPart | GraphPartGroup) -> GraphPartGroup:
        if isinstance(items, GraphPartGroup):
            return GraphPartGroup(self._parts + items._parts)
        else:
            return GraphPartGroup(self._parts + [items])


T_co = TypeVar("T_co", covariant=True)
T_contra = TypeVar("T_contra", contravariant=True)


@runtime_checkable
class InPort(Protocol[T_contra]):
    value: Optional[Any]
    parent_node: Node

    def __lshift__(self, other: OutPort[T_contra]) -> Edge: ...


@runtime_checkable
class OutPort(Protocol[T_co]):
    value: Optional[Any]
    parent_node: Node

    _is_branched_pipeline_head: bool

    def __rshift__(self, other: InPort[T_co]) -> Edge: ...


class Empty:
    def forward(self):
        pass


@dataclass(frozen=True)
class Edge:
    src: OutPort[Any]
    dst: InPort[Any]

    def __add__(self, items: GraphPart | GraphPartGroup) -> GraphPartGroup:
        if isinstance(items, GraphPartGroup):
            parts = items._parts
            parts = [self] + parts
            return GraphPartGroup(parts)  # type: ignore
        else:
            return GraphPartGroup([self, items])

    def forward(self):
        v = self.src.value
        self.dst.value = v


@dataclass(frozen=True)
class Block:
    instructions: GraphPartGroup
    condition: Callable | None = None

    def __add__(self, items: GraphPart | GraphPartGroup) -> GraphPartGroup:
        if isinstance(items, GraphPartGroup):
            parts = items._parts
            parts = [self] + parts
            return GraphPartGroup(parts)  # type: ignore
        else:
            return GraphPartGroup([self, items])

    def forward(self):
        for i in self.instructions:
            i.forward()


if TYPE_CHECKING:
    GraphPart = Empty | Node | Edge | Block


TP = TypeVar("TP")
TM = TypeVar("TM")


class PortView(OutPort, Generic[TP]):
    host: Port[Any] | PortView[Any]
    view_transform: Callable[[Any], TP]

    parent_node: Node
    _is_branched_pipeline_head: bool

    def __init__(
        self, host: Port[Any] | PortView[Any], view_transform: Callable[[Any], TP]
    ) -> None:
        super().__init__()
        self.host = host
        self.view_transform = view_transform

        self.parent_node = host.parent_node
        self._is_branched_pipeline_head = host._is_branched_pipeline_head

    @property
    def value(self) -> TP:
        t = self.host.value
        assert t is not None
        t = self.view_transform(t)
        return t

    def connect(self, other: InPort[TP]) -> Edge:
        return Edge(self, other)

    def __rshift__(self, other: InPort[TP]) -> Edge:
        return self.connect(other)

    def print(self) -> PortView[TP]:
        def t(v):
            print(self.value)
            return v

        return PortView(self, t)

    def map(self, func: Callable[[TP], TM]) -> PortView[TM]:
        return PortView(self, func)

    def isinstance(
        self, data_type: type[TM], branch_func: Callable[[Port[TM]], GraphPartGroup]
    ) -> Block:
        # Important: avoid binding self.value at definition time.
        def cond(t=data_type):
            return isinstance(self.value, t)

        group = branch_func(self)  # type: ignore

        return Block(group, condition=cond)


class Port(Generic[TP]):
    value: Optional[TP]
    parent_node: Node

    on_change: Callable[[], Awaitable[Any]] | Awaitable[Any] | None

    _is_branched_pipeline_head: bool

    def __init__(
        self,
        value: Optional[TP],
        on_change: Callable[[], Awaitable[Any]] | Awaitable[Any] | None = None,
    ) -> None:
        self.value = value
        self.on_change = on_change
        self._is_branched_pipeline_head = False

    def __ilshift__(self, value: TP) -> Port[TP]:
        """port <<= value: set values of a port."""
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

        self._is_branched_pipeline_head = True

        return p

    def __pos__(self) -> Pipeline:
        return self.get_branched_pipeline()

    def add_pipeline(self, others: GraphPart | GraphPartGroup) -> Port:
        self += others
        return self

    def __iadd__(self, others: GraphPart | GraphPartGroup) -> Port:
        p = self.get_branched_pipeline()
        p += others
        p.check_edges()
        return self

    def connect(self, other: InPort[TP]) -> Edge:
        return Edge(self, other)

    def __rshift__(self, other: InPort[TP]) -> Edge:
        return self.connect(other)

    def __lshift__(self, other: OutPort[TP]) -> Edge:
        return Edge(other, self)

    def print(self) -> PortView[TP]:
        def t(v):
            print(self.value)
            return v

        return PortView(self, t)

    def map(self, func: Callable[[TP], TM]) -> PortView[TM]:
        return PortView(self, func)

    def match(
        self,
        func: Callable[[TP], Any],
        branches: Dict[Any, GraphPartGroup],
    ) -> GraphPartGroup:
        results = []
        for k in branches.keys():

            def cond(k=k):
                v = self.value
                assert v is not None
                return func(v) == k

            results.append(Block(branches[k], condition=cond))

        return GraphPartGroup(results)

    def isinstance(
        self, data_type: type[TM], branch_func: Callable[[Port[TM]], GraphPartGroup]
    ) -> Block:
        # Important: avoid binding self.value at definition time.
        def cond(t=data_type):
            return isinstance(self.value, t)

        group = branch_func(self)  # type: ignore

        return Block(group, condition=cond)


class FutureState(Enum):
    INITIALIZED = auto()

    # We just created/scheduled the task in this step()
    STARTED = auto()

    # Task exists but is not done yet (pending/running)
    RUNNING = auto()

    # Task finished successfully (has result, no exception, not cancelled)
    COMPLETED = auto()

    # Task finished with an exception
    FAILED = auto()

    # Task was cancelled
    CANCELLED = auto()


T = TypeVar("T")


class FutureOp(Generic[T]):
    coroutine: Callable[[], Coroutine[Any, Any, T]]
    task: Task[T] | None = None

    def __init__(self, coroutine: Callable[[], Coroutine[Any, Any, T]]) -> None:
        super().__init__()
        self.coroutine = coroutine

    def start(self) -> FutureState:
        self.task = create_task(self.coroutine())
        return FutureState.STARTED

    def step(self, restart: bool = True) -> FutureState:
        if self.task is None:
            if not restart:
                return FutureState.INITIALIZED

            self.task = create_task(self.coroutine())
            return FutureState.STARTED

        # 2) We have a task
        if not self.task.done():
            return FutureState.RUNNING

        # 3) Task is done -> classify outcome
        if self.task.cancelled():
            return FutureState.CANCELLED

        exc = self.task.exception()
        if exc is not None:
            return FutureState.FAILED

        return FutureState.COMPLETED

    def cancel(self) -> bool:
        if self.task is not None:
            return self.task.cancel()
        return False

    def reset(self) -> None:
        self.task = None

    def get_result(self) -> T | None:
        if self.task is not None and self.task.done():
            return self.task.result()
        return None


# New: dedicated thread state (no CANCELLED because Python can't really do that)
class ThreadState(Enum):
    INITIALIZED = auto()  # Thread not created or started yet
    STARTED = auto()  # Thread object created and start() just called
    RUNNING = auto()  # Thread is alive and running
    COMPLETED = auto()  # Thread finished with no exception
    FAILED = auto()  # Thread finished and raised an exception


class ThreadOp(Generic[T]):
    """
    Small wrapper around a Thread that:
      - Tracks state explicitly (ThreadState)
      - Can be restarted by creating a new Thread object
      - Captures result/exception from the target callable
    """

    _target: Callable[[], T]
    _thread: Thread | None
    _state: ThreadState
    _result: T | None
    _exc: BaseException | None

    def __init__(self, target: Callable[[], T], daemon: bool = True) -> None:
        super().__init__()
        self._target = target
        self._thread = None
        self._state = ThreadState.INITIALIZED
        self._result = None
        self._exc = None
        self._daemon = daemon

    # Internal wrapper to capture result/exception
    def _run_wrapper(self) -> None:
        try:
            self._result = self._target()
        except BaseException as e:  # noqa: BLE001 – we want to catch anything
            self._exc = e

    def step(self, start_new: bool = True) -> ThreadState:
        # No thread object yet, or we reset after completion
        if self._thread is None:
            if not start_new:
                # We haven't started anything
                self._state = ThreadState.INITIALIZED
                return self._state

            self._thread = Thread(target=self._run_wrapper, daemon=self._daemon)
            self._thread.start()
            self._state = ThreadState.STARTED
            return self._state

        # We have a thread object
        if self._thread.is_alive():
            self._state = ThreadState.RUNNING
            return self._state

        # Thread finished; classify outcome
        if self._exc is not None:
            self._state = ThreadState.FAILED
        else:
            self._state = ThreadState.COMPLETED

        # Optional: if you want `step(start_new=True)` to allow a restart,
        # you can recreate the thread *only* after we've reported completion once.
        return self._state

    def reset(self) -> None:
        """
        Reset the operator so it can run again:
        - Clears result/exception
        - Creates a fresh Thread on the next step(start_new=True)
        """
        self._thread = None
        self._result = None
        self._exc = None
        self._state = ThreadState.INITIALIZED

    def join(self, timeout: float | None = None) -> bool:
        """
        Join the underlying thread. Returns True if it finished
        before the timeout, False otherwise.
        """
        if self._thread is None:
            return True  # nothing to join

        self._thread.join(timeout)
        return not self._thread.is_alive()

    def get_result(self) -> T | None:
        """
        Return the result if the thread has finished successfully.
        Raises the stored exception if the thread failed.
        """
        # Ensure we only expose result when we're done
        if self._thread is None or self._thread.is_alive():
            return None

        if self._exc is not None:
            # You can also choose to just return None and not raise,
            # depending on how you want to handle errors at the graph level.
            raise self._exc
        return self._result

    def cancel(self) -> bool:
        """
        Threads in Python cannot be forcibly cancelled.
        This always returns False and exists only to match the FutureOp API.
        """
        # You could add a cooperative "stop flag" here if your target checks it.
        return False

    @property
    def state(self) -> ThreadState:
        return self._state
