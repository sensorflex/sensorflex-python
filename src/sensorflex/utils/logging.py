"""The default logging utility."""

import logging
import asyncio

ROOT_LOGGER_NAME = "SensorFlex"


class Perf:
    def __init__(self, name: str) -> None:
        self._name = name
        self._loop = asyncio.get_running_loop()

    def __enter__(self):
        self._t0 = self._loop.time()

    def __exit__(self, a, b, c):
        dt = (self._loop.time() - self._t0) * 1000
        print(f"Task {self._name} took {dt:.4f} ms")


def get_logger(subsystem: str | None = None) -> logging.Logger:
    """
    Get a logger under the 'SensorFlex' namespace.
    Examples:
        get_logger()                -> 'SensorFlex'
        get_logger("webrtc")        -> 'SensorFlex.webrtc'
        get_logger("handlers.png")  -> 'SensorFlex.handlers.png'
    """
    if subsystem:
        name = f"{ROOT_LOGGER_NAME}.{subsystem}"
    else:
        name = ROOT_LOGGER_NAME
    return logging.getLogger(name)


def configure_default_logging(level: int = logging.INFO) -> None:
    """
    Convenience helper for apps / demos.
    Framework code should NOT call this; only top-level scripts.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
