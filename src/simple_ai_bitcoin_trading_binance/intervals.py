"""Canonical Binance-supported kline intervals and duration helpers.

The tuples in this module are the source of truth used by every user-facing
interval prompt — CLI arguments, the backtest panel, the autonomous loop, and
the TUI dropdowns.  Updating an interval here is the only change required to
expose or retire an interval across the app.

Interval strings mirror Binance's REST documentation verbatim (case-sensitive):

    spot klines      (/api/v3/klines)   — includes ``1s`` and max limit 1000
    usd-m futures    (/fapi/v1/klines)  — max limit 1500

Both sets share the minute/hour/day/week/month cadence; only ``1s`` is spot-only.
Do not introduce an interval that is not on the exchange; requests would fail
with ``-1120`` (invalid interval) and that is not a contract we want to tolerate.
"""

from __future__ import annotations

from typing import Iterable

SPOT_INTERVALS: tuple[str, ...] = (
    "1s",
    "1m",
    "3m",
    "5m",
    "15m",
    "30m",
    "1h",
    "2h",
    "4h",
    "6h",
    "8h",
    "12h",
    "1d",
    "3d",
    "1w",
    "1M",
)

FUTURES_INTERVALS: tuple[str, ...] = (
    "1m",
    "3m",
    "5m",
    "15m",
    "30m",
    "1h",
    "2h",
    "4h",
    "6h",
    "8h",
    "12h",
    "1d",
    "3d",
    "1w",
    "1M",
)

MAX_LIMIT_SPOT = 1000
MAX_LIMIT_FUTURES = 1500

# canonical minute durations — used by fetch planning and P&L stat rollups
_MINUTES_BY_INTERVAL: dict[str, int] = {
    "1s": 1,  # stored as "1 minute equivalent" for budgeting; not a true minute
    "1m": 1,
    "3m": 3,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "2h": 120,
    "4h": 240,
    "6h": 360,
    "8h": 480,
    "12h": 720,
    "1d": 1440,
    "3d": 4320,
    "1w": 10080,
    "1M": 43200,
}

_MILLISECONDS_BY_INTERVAL: dict[str, int] = {
    "1s": 1_000,
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
    "6h": 21_600_000,
    "8h": 28_800_000,
    "12h": 43_200_000,
    "1d": 86_400_000,
    "3d": 259_200_000,
    "1w": 604_800_000,
    "1M": 2_592_000_000,
}


def supported_intervals(market_type: str) -> tuple[str, ...]:
    """Return the intervals Binance actually accepts for ``market_type``."""

    return SPOT_INTERVALS if market_type == "spot" else FUTURES_INTERVALS


def is_supported(interval: str, market_type: str) -> bool:
    """Return True when ``interval`` is valid on the given ``market_type``."""

    return interval in supported_intervals(market_type)


def validate_interval(interval: str, market_type: str) -> str:
    """Return ``interval`` if valid, otherwise raise ``ValueError``.

    The error message names the full allowed set so operators never have to
    guess a spelling.
    """

    if is_supported(interval, market_type):
        return interval
    allowed = ", ".join(supported_intervals(market_type))
    raise ValueError(
        f"Interval {interval!r} is not supported on {market_type}. "
        f"Allowed intervals: {allowed}"
    )


def max_limit(market_type: str) -> int:
    """Exchange cap on `limit` for a single klines request."""

    return MAX_LIMIT_SPOT if market_type == "spot" else MAX_LIMIT_FUTURES


def interval_minutes(interval: str) -> int:
    """Approximate minute duration for ``interval`` — used for range planning.

    ``1s`` is reported as 1 minute for budgeting purposes because the caller
    uses this value to estimate how many candles a time window will produce;
    over-budgeting by a factor of 60 for 1-second feeds is the less-bad
    approximation compared to rounding to zero.
    """

    if interval not in _MINUTES_BY_INTERVAL:
        raise ValueError(f"Unknown interval: {interval!r}")
    return _MINUTES_BY_INTERVAL[interval]


def interval_milliseconds(interval: str) -> int:
    """Exact millisecond duration for a Binance kline interval."""

    if interval not in _MILLISECONDS_BY_INTERVAL:
        raise ValueError(f"Unknown interval: {interval!r}")
    return _MILLISECONDS_BY_INTERVAL[interval]


def minutes_between(start_ms: int, end_ms: int) -> int:
    """Return the whole-minute span between two epoch-millisecond markers."""

    if end_ms <= start_ms:
        return 0
    return (int(end_ms) - int(start_ms)) // 60_000


def estimate_candle_count(interval: str, start_ms: int, end_ms: int) -> int:
    """Estimate how many candles exist in a time window — for fetch planning."""

    step = interval_minutes(interval)
    if step <= 0:
        return 0
    spans = minutes_between(start_ms, end_ms)
    return max(0, spans // step)


def describe(intervals: Iterable[str]) -> str:
    """Render an interval list as a human-readable string for help output."""

    return ", ".join(intervals)
