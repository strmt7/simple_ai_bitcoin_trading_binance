from __future__ import annotations

import json

import pytest

from simple_ai_bitcoin_trading_binance.api import Candle
import simple_ai_bitcoin_trading_binance.storage as storage
from simple_ai_bitcoin_trading_binance.market_data import clean_candles
from simple_ai_bitcoin_trading_binance.storage import write_json_atomic


def _candle(open_time: int, close: float = 100.0, close_time: int | None = None) -> Candle:
    return Candle(
        open_time=open_time,
        open=close,
        high=close + 1.0,
        low=close - 1.0,
        close=close,
        volume=1.0,
        close_time=open_time + 60_000 if close_time is None else close_time,
    )


def test_clean_candles_sorts_dedupes_invalid_and_unclosed() -> None:
    first = _candle(60_000, close=100.0)
    replacement = _candle(60_000, close=101.0)
    valid_early = _candle(0, close=99.0)
    invalid = Candle(120_000, open=5.0, high=4.0, low=6.0, close=5.0, volume=1.0, close_time=180_000)
    unclosed = _candle(180_000, close=102.0, close_time=10_000_000)

    cleaned = clean_candles([first, invalid, unclosed, valid_early, replacement], now_ms=240_000)

    assert [c.open_time for c in cleaned] == [0, 60_000]
    assert cleaned[-1].close == 101.0


def test_clean_candles_ignores_non_candles_and_can_keep_unclosed_rows() -> None:
    future = _candle(120_000, close=102.0, close_time=10_000_000)

    cleaned = clean_candles([object(), future], now_ms=0, drop_unclosed=False)  # type: ignore[list-item]

    assert cleaned == [future]


def test_write_json_atomic_replaces_payload_and_applies_mode(tmp_path) -> None:
    target = tmp_path / "nested" / "payload.json"
    write_json_atomic(target, {"b": 2}, sort_keys=True, mode=0o600)
    write_json_atomic(target, {"a": 1}, sort_keys=True, mode=0o600)

    assert json.loads(target.read_text(encoding="utf-8")) == {"a": 1}
    assert target.stat().st_mode & 0o777 == 0o600


def test_write_json_atomic_removes_temp_file_on_failed_write(tmp_path, monkeypatch) -> None:
    def fail_dump(*_args, **_kwargs) -> None:
        raise ValueError("boom")

    target = tmp_path / "payload.json"
    monkeypatch.setattr(storage.json, "dump", fail_dump)

    with pytest.raises(ValueError, match="boom"):
        write_json_atomic(target, {"x": 1})

    assert not target.exists()
    assert list(tmp_path.glob(".payload.json.*.tmp")) == []


def test_write_json_atomic_preserves_original_error_when_cleanup_fails(tmp_path, monkeypatch) -> None:
    def fail_dump(*_args, **_kwargs) -> None:
        raise ValueError("boom")

    def fail_unlink(_path) -> None:
        raise OSError("locked")

    target = tmp_path / "payload.json"
    with monkeypatch.context() as context:
        context.setattr(storage.json, "dump", fail_dump)
        context.setattr(storage.Path, "unlink", fail_unlink)
        with pytest.raises(ValueError, match="boom"):
            write_json_atomic(target, {"x": 1})

    for leftover in tmp_path.glob(".payload.json.*.tmp"):
        leftover.unlink()
