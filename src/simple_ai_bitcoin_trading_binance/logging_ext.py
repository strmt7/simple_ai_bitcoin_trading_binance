"""Structured logging with aggressive credential redaction.

The repo's existing output surfaces (stdout, JSON artifacts, TUI activity log)
already route through manual redaction helpers.  This module adds a stdlib
``logging``-based channel so long-running loops (autonomous trading, training
suite) can emit structured events to both console and a rotating file without
leaking secrets.

Design:

* The log handlers install a ``RedactionFilter`` that scrubs known sensitive
  tokens from every ``record.msg`` and ``record.args`` before emission.
* The file handler is ``RotatingFileHandler`` capped at 2 MiB × 5 files so the
  autonomous loop cannot fill a disk by accident.
* ``configure(path=...)`` is idempotent; calling it twice reuses the existing
  handlers.  Tests can call ``reset()`` between cases.
"""

from __future__ import annotations

import json
import logging
import re
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Callable, Iterable

LOGGER_NAME = "simple_ai_bitcoin_trading_binance"
_MAX_BYTES = 2 * 1024 * 1024
_BACKUP_COUNT = 5

# Patterns that must never appear in logs regardless of where they came from.
_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(ghp_|ghs_|github_pat_)[A-Za-z0-9_]{20,}"),
    re.compile(r"(sk-[A-Za-z0-9]{20,})"),
    re.compile(r"-----BEGIN [A-Z ]+PRIVATE KEY-----.*?-----END [A-Z ]+PRIVATE KEY-----", re.DOTALL),
    # Binance-style HMAC signature (64 hex chars)
    re.compile(r"\bsignature=[a-fA-F0-9]{40,}"),
    # signed-request fields passed through query strings
    re.compile(r"\b(timestamp|recvWindow|signature)=[^&\s\"']+"),
    # api_key/api_secret style key=value pairs (catch both CLI and JSON)
    re.compile(r"\"(api_key|api_secret|apiKey|apiSecret)\"\s*:\s*\"[^\"]*\""),
    re.compile(r"\b(api_key|api_secret|apiKey|apiSecret)\s*=\s*\S+"),
    re.compile(r"X-MBX-APIKEY:\s*\S+", re.IGNORECASE),
)

_Replacement = str | Callable[[re.Match[str]], str]

_REPLACEMENTS: tuple[tuple[re.Pattern[str], _Replacement], ...] = (
    (_SECRET_PATTERNS[0], "<redacted-pat>"),
    (_SECRET_PATTERNS[1], "<redacted-openai-key>"),
    (_SECRET_PATTERNS[2], "<redacted-private-key>"),
    (_SECRET_PATTERNS[3], "signature=<redacted>"),
    (_SECRET_PATTERNS[4], lambda match: f"{match.group(1)}=<redacted>"),
    (_SECRET_PATTERNS[5], lambda match: f'"{match.group(1)}":"<redacted>"'),
    (_SECRET_PATTERNS[6], lambda match: f"{match.group(1)}=<redacted>"),
    (_SECRET_PATTERNS[7], "X-MBX-APIKEY: <redacted>"),
)


def redact(text: str) -> str:
    """Return ``text`` with every sensitive token pattern replaced by placeholders.

    The replacements run in the declared order; each pattern is applied once so
    overlapping matches do not cascade.  Non-string inputs are returned verbatim
    so the filter can be composed with ``str.format`` / ``record.getMessage``.
    """

    if not isinstance(text, str):
        return text
    redacted = text
    for pattern, repl in _REPLACEMENTS:
        if callable(repl):
            redacted = pattern.sub(repl, redacted)
        else:
            redacted = pattern.sub(repl, redacted)
    return redacted


class RedactionFilter(logging.Filter):
    """A logging filter that rewrites message + args to strip secret material."""

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003 - stdlib name
        try:
            record.msg = redact(record.getMessage())
            record.args = ()
        except Exception:  # noqa: BLE001 - logging must never raise
            record.msg = "<redaction-error>"
            record.args = ()
        return True


class JsonLineFormatter(logging.Formatter):
    """Emit one JSON object per log record, stable schema, sorted keys."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        # include any extra fields the caller attached via LoggerAdapter / extra=
        for key, value in record.__dict__.items():
            if key in payload or key.startswith("_") or key in _LOG_RECORD_RESERVED:
                continue
            if not isinstance(value, (str, int, float, bool, type(None), list, dict)):
                continue
            payload[key] = value
        return json.dumps(payload, sort_keys=True)


_LOG_RECORD_RESERVED: frozenset[str] = frozenset({
    "args", "asctime", "created", "exc_info", "exc_text", "filename", "funcName",
    "levelname", "levelno", "lineno", "module", "msecs", "message", "msg",
    "name", "pathname", "process", "processName", "relativeCreated",
    "stack_info", "thread", "threadName",
})


_CONFIGURED_PATH: Path | None = None


def configure(*, path: Path | None = None, level: int = logging.INFO) -> logging.Logger:
    """Configure the package-wide logger.  Idempotent per ``path``.

    Call this from the entry point of any long-running operation (autonomous,
    training suite) so events land in ``data/logs/simple-ai-trading.log``.
    """

    global _CONFIGURED_PATH
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)

    # Always install the redaction filter exactly once.
    if not any(isinstance(flt, RedactionFilter) for flt in logger.filters):
        logger.addFilter(RedactionFilter())

    # Console handler.
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, RotatingFileHandler)
               for h in logger.handlers):
        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(console)

    if path is None:
        return logger

    resolved = Path(path)
    if _CONFIGURED_PATH == resolved:
        return logger

    # Remove any previous file handler — a re-configure with a new path replaces it.
    for handler in list(logger.handlers):
        if isinstance(handler, RotatingFileHandler):
            logger.removeHandler(handler)
            handler.close()

    resolved.parent.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        resolved,
        maxBytes=_MAX_BYTES,
        backupCount=_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setFormatter(JsonLineFormatter())
    logger.addHandler(file_handler)
    _CONFIGURED_PATH = resolved
    return logger


def reset() -> None:
    """Detach every handler/filter.  Used by tests to start from a clean slate."""

    global _CONFIGURED_PATH
    logger = logging.getLogger(LOGGER_NAME)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    for flt in list(logger.filters):
        logger.removeFilter(flt)
    _CONFIGURED_PATH = None


def get_logger(suffix: str | None = None) -> logging.Logger:
    """Return the package logger, optionally namespaced with ``suffix``."""

    name = LOGGER_NAME if not suffix else f"{LOGGER_NAME}.{suffix}"
    return logging.getLogger(name)


def describe_handlers(logger: logging.Logger | None = None) -> list[str]:
    """Return a human-readable list of active handlers for diagnostics."""

    logger = logger or logging.getLogger(LOGGER_NAME)
    out: list[str] = []
    for handler in logger.handlers:
        cls = type(handler).__name__
        tag = ""
        if isinstance(handler, RotatingFileHandler):
            tag = f" path={handler.baseFilename}"
        out.append(f"{cls}{tag}")
    return out


def iter_secret_placeholders() -> Iterable[str]:
    """Yield every placeholder string this module emits — handy for tests."""

    yield "<redacted-pat>"
    yield "<redacted-openai-key>"
    yield "<redacted-private-key>"
    yield "<redacted>"
