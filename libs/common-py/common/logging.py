"""
Structured logging configuration using structlog.

Supports two output modes selected via the LOG_MODE environment variable:
    - "json" (default) — JSON-formatted log lines for production / log aggregation.
    - "console" — coloured, human-readable output for local development.
"""

import logging
import os
import sys

import structlog

_LOG_MODE = os.getenv("LOG_MODE", "json")
_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()


def _configure_structlog() -> None:
    """Wire up structlog processors and renderer based on LOG_MODE."""

    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if _LOG_MODE == "console":
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer(colors=True),
        ]
    else:
        processors = shared_processors + [
            structlog.processors.JSONRenderer(),
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Also set the stdlib logging level so third-party libraries are quiet enough.
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, _LOG_LEVEL, logging.INFO),
    )


_configure_structlog()


def get_logger(name: str = __name__) -> structlog.stdlib.BoundLogger:
    """Return a structured logger instance bound to *name*."""
    return structlog.get_logger(name)


# ---------------------------------------------------------------------------
# Request-ID middleware helper
# ---------------------------------------------------------------------------

def bind_request_id(request_id: str) -> None:
    """Bind a request-id into structlog's thread-local context.

    Call this early in every request lifecycle (e.g. in a middleware) so that
    all subsequent log calls within the same request carry the id.
    """
    structlog.contextvars.bind_contextvars(request_id=request_id)


def clear_request_context() -> None:
    """Clear all context variables bound during a request."""
    structlog.contextvars.clear_contextvars()
