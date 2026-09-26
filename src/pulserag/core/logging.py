"""Logging configuration for the process."""

from __future__ import annotations

import logging
import sys

__all__ = ["configure_logging"]
_NOISY_LOGGERS = {
    "httpcore": logging.WARNING,
    "httpx": logging.WARNING,
    "urllib3": logging.WARNING,
    "qdrant_client": logging.WARNING,
    "openai": logging.WARNING,
}


def configure_logging(level: str = "INFO") -> None:
    """Install the root logging config for this process."""
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
    )

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())

    for logger_name, logger_level in _NOISY_LOGGERS.items():
        logging.getLogger(logger_name).setLevel(logger_level)
