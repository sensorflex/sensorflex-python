from __future__ import annotations

from dataclasses import dataclass
from threading import Thread
from typing import (
    Any,
    Callable,
    Generic,
    TypeVar,
    Optional,
    TYPE_CHECKING,
    List,
    Iterator,
)
from enum import Enum, auto

from asyncio import create_task, Task
from typing import Coroutine


if TYPE_CHECKING:
    from ._graph import Pipeline
    from ._node import Node

TP = TypeVar("TP")
NP = TypeVar("NP")


class GraphPartGroup:
    _parts: List[Node | Edge]

    def __init__(self, parts: List[Node | Edge]) -> None:
        self._parts = parts

    def __iter__(self) -> Iterator[Node | Edge]:
        return iter(self._parts)

    def __add__(self, items: Node | Edge | GraphPartGroup) -> GraphPartGroup:
        if isinstance(items, GraphPartGroup):
            return GraphPartGroup(self._parts + items._parts)
        else:
            return GraphPartGroup(self._parts + [items])


@dataclass
class Edge:
    src: Port
    dst: Port

    def __add__(self, items: Node | Edge | GraphPartGroup) -> GraphPartGroup:
        if isinstance(items, GraphPartGroup):
            return self + items
        else:
            return GraphPartGroup([self, items])


class Port(Generic[TP]):
    value: Optional[TP]
    parent_node: Node

    on_change: Callable | List[Callable] | None

    _is_branched_pipeline_head: bool

    def __init__(
        self, value: Optional[TP], on_change: Callable | List[Callable] | None = None
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
        p = Pipeline(g, self.parent_node)

        # The self node should not be automatically added to the pipeline.
        # p += self.parent_node
        g.add_pipeline(self, p)

        self._is_branched_pipeline_head = True

        return p

    def __pos__(self):
        return self.get_branched_pipeline()

    def connect(self, other: Port[NP]) -> Edge:
        return Edge(self, other)

    def __rshift__(self, other: Port[NP]) -> Edge:
        return self.connect(other)

    def __lshift__(self, other: Port[NP]) -> Edge:
        return other.connect(self)


T = TypeVar("T")


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
