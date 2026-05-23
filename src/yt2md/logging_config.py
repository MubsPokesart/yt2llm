"""Structlog configuration. Configures once at CLI entry; loggers obtained via get_logger()."""

from __future__ import annotations

import logging
import sys

import structlog

_VERBOSITY_TO_LEVEL = {
    0: logging.WARNING,
    1: logging.INFO,
    2: logging.DEBUG,
}


def configure_logging(verbosity: int = 0) -> None:
    """Configure structlog to emit JSON to stderr at the level mapped from verbosity.

    verbosity: 0 = WARNING (default), 1 = INFO (-v), 2 = DEBUG (-vv).
    """
    level = _VERBOSITY_TO_LEVEL.get(verbosity, logging.DEBUG)
    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)
    logging.basicConfig(level=level, stream=sys.stderr, format="%(message)s")

    structlog.reset_defaults()
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a structlog BoundLogger by name."""
    return structlog.get_logger(name)  # type: ignore[no-any-return]
