"""The default logging utility."""

import logging

ROOT_LOGGER_NAME = "SensorFlex"


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
