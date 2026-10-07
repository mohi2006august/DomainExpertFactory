"""Structured logging (CLAUDE.md: log with structured fields)."""

from __future__ import annotations

import logging
import os
import sys

import structlog

_configured = False


def configure_logging(level: str | None = None, json_logs: bool | None = None) -> None:
    global _configured
    if _configured:
        return
    level_name = (level or os.environ.get("FACTORY_LOG_LEVEL", "INFO")).upper()
    if json_logs is None:
        json_logs = os.environ.get("FACTORY_LOG_JSON", "0") in {"1", "true", "yes"}
    renderer: structlog.typing.Processor = (
        structlog.processors.JSONRenderer()
        if json_logs
        else structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level_name, logging.INFO)
        ),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=False,
    )
    _configured = True


def get_logger(name: str = "factory", **fields: object) -> structlog.typing.FilteringBoundLogger:
    configure_logging()
    logger: structlog.typing.FilteringBoundLogger = structlog.get_logger(name)
    return logger.bind(**fields) if fields else logger
