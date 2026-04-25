from __future__ import annotations

import io
import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

import pytest

from simple_ai_bitcoin_trading_binance import logging_ext


@pytest.fixture(autouse=True)
def _reset_logging():
    logging_ext.reset()
    yield
    logging_ext.reset()


# --------- redact() coverage ---------


def test_redact_non_string_returned_verbatim():
    assert logging_ext.redact(12345) == 12345  # type: ignore[arg-type]
    obj = object()
    assert logging_ext.redact(obj) is obj  # type: ignore[arg-type]


def test_redact_github_pat():
    raw = "token ghp_" + "a" * 30 + " trailing"
    out = logging_ext.redact(raw)
    assert "ghp_" not in out
    assert "<redacted-pat>" in out


def test_redact_openai_key():
    raw = "use sk-" + "B" * 30 + " please"
    out = logging_ext.redact(raw)
    assert "sk-" not in out
    assert "<redacted-openai-key>" in out


def test_redact_private_key_block():
    raw = (
        "-----BEGIN RSA PRIVATE KEY-----\n"
        "MIIBOwIBAAJBAK...\n"
        "-----END RSA PRIVATE KEY-----"
    )
    out = logging_ext.redact(raw)
    assert "BEGIN" not in out
    assert "<redacted-private-key>" in out


def test_redact_hmac_signature():
    raw = "GET /api?signature=" + "ab12" * 12  # 48 hex chars
    out = logging_ext.redact(raw)
    assert "signature=<redacted>" in out
    # ensure the raw hex does not linger
    assert "ab12ab12" not in out


def test_redact_timestamp_kv():
    raw = "timestamp=1700000000"
    out = logging_ext.redact(raw)
    assert "timestamp=<redacted>" in out


def test_redact_recv_window_kv():
    raw = "recvWindow=5000"
    out = logging_ext.redact(raw)
    assert "recvWindow=<redacted>" in out


def test_redact_json_api_key():
    raw = '{"api_key":"abc123"}'
    out = logging_ext.redact(raw)
    assert "abc123" not in out
    assert '"api_key":"<redacted>"' in out


def test_redact_cli_api_key():
    raw = "api_key=abc123"
    out = logging_ext.redact(raw)
    assert "abc123" not in out
    assert "api_key=<redacted>" in out


def test_redact_x_mbx_apikey_header():
    raw = "X-MBX-APIKEY: deadbeefcafebabe"
    out = logging_ext.redact(raw)
    assert "deadbeef" not in out
    assert "X-MBX-APIKEY: <redacted>" in out


# --------- RedactionFilter ---------


def test_redaction_filter_happy_path_rewrites_message():
    flt = logging_ext.RedactionFilter()
    record = logging.LogRecord(
        name="x",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="token ghp_%s here",
        args=("A" * 30,),
        exc_info=None,
    )
    assert flt.filter(record) is True
    assert "ghp_" not in record.msg
    assert "<redacted-pat>" in record.msg
    assert record.args == ()


def test_redaction_filter_swallows_getmessage_errors():
    flt = logging_ext.RedactionFilter()
    # Build a normal record first, then poison it so ``getMessage`` raises
    # during formatting -- the filter must swallow the exception.
    record = logging.LogRecord(
        name="x",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="hello %d",
        args=("not-an-int",),  # %d with a string -> TypeError at getMessage
        exc_info=None,
    )
    assert flt.filter(record) is True
    assert record.msg == "<redaction-error>"
    assert record.args == ()


# --------- JsonLineFormatter ---------


def _make_record(**extra):
    rec = logging.LogRecord(
        name="mylogger",
        level=logging.WARNING,
        pathname=__file__,
        lineno=10,
        msg="hello %s",
        args=("world",),
        exc_info=None,
    )
    for k, v in extra.items():
        setattr(rec, k, v)
    return rec


def test_json_line_formatter_normal_record():
    f = logging_ext.JsonLineFormatter()
    rec = _make_record()
    out = f.format(rec)
    data = json.loads(out)
    assert data["level"] == "WARNING"
    assert data["logger"] == "mylogger"
    assert data["message"] == "hello world"
    assert "ts" in data
    # keys sorted
    assert list(data.keys()) == sorted(data.keys())


def test_json_line_formatter_with_exc_info():
    f = logging_ext.JsonLineFormatter()
    try:
        raise RuntimeError("boom")
    except RuntimeError:
        import sys
        exc_info = sys.exc_info()
    rec = logging.LogRecord(
        name="mylogger",
        level=logging.ERROR,
        pathname=__file__,
        lineno=1,
        msg="failed",
        args=(),
        exc_info=exc_info,
    )
    out = f.format(rec)
    data = json.loads(out)
    assert "exc" in data
    assert "RuntimeError" in data["exc"]


def test_json_line_formatter_includes_extra_and_omits_reserved():
    f = logging_ext.JsonLineFormatter()
    rec = _make_record()
    # attach an extra user field
    rec.user_field = "keepme"
    # attach a non-serializable value -> should be dropped
    rec.skip_me = object()
    # attach a reserved name -> must not appear
    rec.pathname = "/never/shown"
    # attach an underscore-prefixed name -> must not appear
    rec._private = "nope"
    out = f.format(rec)
    data = json.loads(out)
    assert data.get("user_field") == "keepme"
    assert "skip_me" not in data
    assert "pathname" not in data
    assert "_private" not in data


# --------- configure / reset / get_logger / describe_handlers ---------


def test_configure_without_path_installs_console_only(tmp_path):
    logger = logging_ext.configure()
    assert any(isinstance(h, logging.StreamHandler) for h in logger.handlers)
    assert not any(isinstance(h, RotatingFileHandler) for h in logger.handlers)
    # redaction filter installed
    assert any(isinstance(flt, logging_ext.RedactionFilter) for flt in logger.filters)


def test_configure_idempotent_does_not_duplicate_handlers(tmp_path):
    logger = logging_ext.configure()
    h1 = list(logger.handlers)
    f1 = list(logger.filters)
    logger2 = logging_ext.configure()
    assert logger is logger2
    assert list(logger.handlers) == h1
    assert list(logger.filters) == f1


def test_configure_with_path_installs_rotating_file_handler(tmp_path):
    log_path = tmp_path / "logs" / "trading.log"
    logger = logging_ext.configure(path=log_path, level=logging.DEBUG)
    assert logger.level == logging.DEBUG
    assert log_path.parent.exists()
    file_handlers = [h for h in logger.handlers if isinstance(h, RotatingFileHandler)]
    assert len(file_handlers) == 1
    assert Path(file_handlers[0].baseFilename) == log_path.resolve() or \
        Path(file_handlers[0].baseFilename) == log_path
    # calling again with the same path is a no-op swap
    logger2 = logging_ext.configure(path=log_path)
    file_handlers2 = [h for h in logger2.handlers if isinstance(h, RotatingFileHandler)]
    assert len(file_handlers2) == 1


def test_configure_reconfigure_with_new_path_swaps_file_handler(tmp_path):
    path_a = tmp_path / "a.log"
    path_b = tmp_path / "b.log"
    logger = logging_ext.configure(path=path_a)
    handlers_a = [h for h in logger.handlers if isinstance(h, RotatingFileHandler)]
    assert len(handlers_a) == 1
    first_file = handlers_a[0].baseFilename

    logger = logging_ext.configure(path=path_b)
    handlers_b = [h for h in logger.handlers if isinstance(h, RotatingFileHandler)]
    assert len(handlers_b) == 1
    assert handlers_b[0].baseFilename != first_file
    assert str(path_b) in handlers_b[0].baseFilename


def test_configure_level_respected(tmp_path):
    logger = logging_ext.configure(level=logging.WARNING)
    assert logger.level == logging.WARNING


def test_reset_clears_handlers_and_filters_then_allows_reconfigure(tmp_path):
    log_path = tmp_path / "r.log"
    logger = logging_ext.configure(path=log_path)
    assert logger.handlers
    assert logger.filters
    logging_ext.reset()
    assert logger.handlers == []
    assert logger.filters == []
    # re-configuring works after reset
    logger2 = logging_ext.configure(path=log_path)
    assert any(isinstance(h, RotatingFileHandler) for h in logger2.handlers)


def test_get_logger_default_and_suffix():
    parent = logging_ext.get_logger()
    assert parent.name == logging_ext.LOGGER_NAME
    child = logging_ext.get_logger("subsystem")
    assert child.name == f"{logging_ext.LOGGER_NAME}.subsystem"
    # empty suffix collapses to parent
    assert logging_ext.get_logger("").name == logging_ext.LOGGER_NAME


def test_describe_handlers_lists_classes_and_file_paths(tmp_path):
    log_path = tmp_path / "d.log"
    logger = logging_ext.configure(path=log_path)
    described = logging_ext.describe_handlers(logger)
    assert any("StreamHandler" in entry for entry in described)
    assert any("RotatingFileHandler" in entry and "path=" in entry for entry in described)


def test_describe_handlers_default_logger(tmp_path):
    log_path = tmp_path / "d2.log"
    logging_ext.configure(path=log_path)
    described = logging_ext.describe_handlers()
    assert any("RotatingFileHandler" in entry for entry in described)


def test_iter_secret_placeholders_yields_four():
    out = list(logging_ext.iter_secret_placeholders())
    assert out == [
        "<redacted-pat>",
        "<redacted-openai-key>",
        "<redacted-private-key>",
        "<redacted>",
    ]
