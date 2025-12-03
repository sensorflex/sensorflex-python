"""A library for operators"""

from __future__ import annotations

from typing import Any, TypeVar, Callable, Generic, Coroutine

from enum import Enum, auto
from asyncio import create_task, Task


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

    def step(self, start_new: bool = True) -> FutureState:
        if self.task is None:
            if not start_new:
                return FutureState.INITIALIZED

            self.task = create_task(self.coroutine())
            return FutureState.STARTED

        # 2) We have a task
        if not self.task.done():
            return FutureState.RUNNING

        # 3) Task is done → classify outcome
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

    def reset(self):
        self.task = None

    def get_result(self) -> T | None:
        if self.task is not None and self.task.done():
            return self.task.result()
        return None
