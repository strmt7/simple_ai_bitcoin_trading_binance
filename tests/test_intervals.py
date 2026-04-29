from __future__ import annotations

import pytest

from simple_ai_bitcoin_trading_binance import intervals


def test_supported_intervals_spot_includes_one_second():
    out = intervals.supported_intervals("spot")
    assert "1s" in out
    assert out is intervals.SPOT_INTERVALS


def test_supported_intervals_futures_excludes_one_second():
    out = intervals.supported_intervals("futures")
    assert "1s" not in out
    assert out is intervals.FUTURES_INTERVALS


def test_is_supported_true_and_false():
    assert intervals.is_supported("1m", "spot") is True
    assert intervals.is_supported("1s", "futures") is False
    assert intervals.is_supported("99h", "spot") is False


def test_validate_interval_valid_returns_interval():
    assert intervals.validate_interval("5m", "spot") == "5m"
    assert intervals.validate_interval("5m", "futures") == "5m"


def test_validate_interval_invalid_error_mentions_allowed_list():
    with pytest.raises(ValueError) as excinfo:
        intervals.validate_interval("7m", "spot")
    msg = str(excinfo.value)
    assert "7m" in msg
    assert "spot" in msg
    # make sure allowed list appears
    assert "1m" in msg
    assert "1M" in msg


def test_validate_interval_invalid_for_futures_mentions_market():
    with pytest.raises(ValueError) as excinfo:
        intervals.validate_interval("1s", "futures")
    assert "futures" in str(excinfo.value)


def test_max_limit_spot_and_futures():
    assert intervals.max_limit("spot") == intervals.MAX_LIMIT_SPOT == 1000
    assert intervals.max_limit("futures") == intervals.MAX_LIMIT_FUTURES == 1500


def test_interval_minutes_unknown_raises():
    with pytest.raises(ValueError) as excinfo:
        intervals.interval_minutes("banana")
    assert "banana" in str(excinfo.value)


def test_interval_minutes_known_values():
    assert intervals.interval_minutes("1m") == 1
    assert intervals.interval_minutes("1h") == 60
    assert intervals.interval_minutes("1d") == 1440
    assert intervals.interval_minutes("1s") == 1  # the special "budgeting" value


def test_interval_milliseconds_uses_exact_exchange_cadence():
    assert intervals.interval_milliseconds("1s") == 1_000
    assert intervals.interval_milliseconds("1m") == 60_000
    assert intervals.interval_milliseconds("15m") == 900_000
    assert intervals.interval_milliseconds("1M") == 2_592_000_000
    with pytest.raises(ValueError, match="banana"):
        intervals.interval_milliseconds("banana")


def test_minutes_between_end_leq_start_returns_zero():
    assert intervals.minutes_between(1000, 1000) == 0
    assert intervals.minutes_between(2000, 1000) == 0


def test_minutes_between_end_greater_than_start():
    # 5 minutes = 300_000 ms
    assert intervals.minutes_between(0, 300_000) == 5
    # partial minute floored
    assert intervals.minutes_between(0, 330_000) == 5


def test_estimate_candle_count_zero_span():
    assert intervals.estimate_candle_count("1m", 1000, 1000) == 0


def test_estimate_candle_count_reasonable_span():
    # 1 hour span with 5m interval -> 12 candles
    span_ms = 60 * 60 * 1000
    assert intervals.estimate_candle_count("5m", 0, span_ms) == 12


def test_estimate_candle_count_zero_step_guard(monkeypatch):
    # Exercise the defensive ``step <= 0`` branch by injecting a 0 minute
    # entry into the lookup table.  This is structurally unreachable through
    # real intervals (all minute values are >= 1) but the branch exists as
    # a belt-and-braces guard.
    monkeypatch.setitem(intervals._MINUTES_BY_INTERVAL, "0x", 0)
    assert intervals.estimate_candle_count("0x", 0, 10 * 60_000) == 0


def test_describe_returns_comma_separated_string():
    rendered = intervals.describe(["1m", "5m", "1h"])
    assert rendered == "1m, 5m, 1h"
    # also works for an empty iterable
    assert intervals.describe([]) == ""
