"""The default logging utility."""

import logging

ROOT_LOGGER_NAME = "sensorflex"


def get_logger(subsystem: str | None = None) -> logging.Logger:
    """
    Get a logger under the 'sensorflex' namespace.
    Examples:
        get_logger()                -> 'sensorflex'
        get_logger("webrtc")        -> 'sensorflex.webrtc'
        get_logger("handlers.png")  -> 'sensorflex.handlers.png'
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
