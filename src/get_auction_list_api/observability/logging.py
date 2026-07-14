"""Structured logging with defensive secret redaction."""

import logging
import re
import sys
from collections.abc import Mapping, Sequence
from typing import Any, cast

import structlog
from structlog.typing import EventDict, Processor

from get_auction_list_api.config import Settings

REDACTED = "[REDACTED]"
_SENSITIVE_KEY = re.compile(
    r"(?i)(authorization|cookie|password|secret|token|api[_-]?key|"
    r"service[_-]?role|database[_-]?url)"
)
_BEARER_VALUE = re.compile(r"(?i)\bbearer\s+[^\s,;]+")
_ASSIGNMENT_VALUE = re.compile(r"(?i)\b(password|secret|token|api[_-]?key)=([^&\s]+)")
_JWT_VALUE = re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b")
_EMAIL_VALUE = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
_PHONE_VALUE = re.compile(r"(?<!\d)(?:\+?1[-. ]?)?\(?\d{3}\)?[-. ]?\d{3}[-. ]?\d{4}(?!\d)")
_ADDRESS_VALUE = re.compile(
    r"\b\d{1,7}\s+[A-Za-z0-9.' -]{2,80}\s(?:st|street|rd|road|dr|drive|ln|lane|"
    r"ave|avenue|blvd|boulevard|ct|court|trl|trail)\b",
    re.IGNORECASE,
)


def redact(value: Any, *, key: str | None = None) -> Any:
    """Recursively redact values under sensitive keys and common token forms."""

    if key is not None and _SENSITIVE_KEY.search(key):
        return REDACTED
    if isinstance(value, Mapping):
        return {str(item_key): redact(item, key=str(item_key)) for item_key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [redact(item) for item in value]
    if isinstance(value, str):
        value = _BEARER_VALUE.sub(REDACTED, value)
        value = _ASSIGNMENT_VALUE.sub(lambda match: f"{match.group(1)}={REDACTED}", value)
        value = _JWT_VALUE.sub(REDACTED, value)
        value = _EMAIL_VALUE.sub(REDACTED, value)
        value = _PHONE_VALUE.sub(REDACTED, value)
        return _ADDRESS_VALUE.sub(REDACTED, value)
    return value


def redact_event(
    _logger: logging.Logger,
    _method_name: str,
    event_dict: EventDict,
) -> EventDict:
    """Structlog processor that sanitizes the complete event payload."""

    return cast(EventDict, redact(event_dict))


def configure_logging(settings: Settings) -> None:
    """Configure deterministic JSON or developer-console application logs."""

    renderer: Processor
    if settings.log_format == "json":
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=sys.stdout.isatty())

    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, settings.log_level),
        stream=sys.stdout,
        force=True,
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            redact_event,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, settings.log_level)),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )


def get_logger() -> structlog.stdlib.BoundLogger:
    """Return a structured logger."""

    return cast(structlog.stdlib.BoundLogger, structlog.get_logger())
