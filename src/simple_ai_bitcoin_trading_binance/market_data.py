"""Market-data normalization shared by training, backtests, and live signals."""

from __future__ import annotations

import math
import time
from collections.abc import Iterable

from .api import Candle


def _is_valid_ohlcv(candle: Candle) -> bool:
    values = (candle.open, candle.high, candle.low, candle.close, candle.volume)
    if not all(math.isfinite(float(value)) for value in values):
        return False
    if candle.open <= 0.0 or candle.high <= 0.0 or candle.low <= 0.0 or candle.close <= 0.0:
        return False
    if candle.volume < 0.0 or candle.open_time < 0 or candle.close_time < 0:
        return False
    if candle.close_time < candle.open_time:
        return False
    if candle.low > candle.high:
        return False
    if not (candle.low <= candle.open <= candle.high):
        return False
    if not (candle.low <= candle.close <= candle.high):
        return False
    return True


def clean_candles(
    candles: Iterable[Candle],
    *,
    now_ms: int | None = None,
    drop_unclosed: bool = True,
) -> list[Candle]:
    """Return sorted, deduplicated, fully closed, valid candles.

    Duplicate open times can appear when pages overlap or a live fetch includes
    a mutable in-progress kline. Keeping the last valid row for each open time
    mirrors exchange pagination while still making downstream feature rows
    deterministic.
    """

    if now_ms is None:
        now_ms = int(time.time() * 1000)

    by_open_time: dict[int, Candle] = {}
    for candle in candles:
        if not isinstance(candle, Candle):
            continue
        if not _is_valid_ohlcv(candle):
            continue
        if drop_unclosed and candle.close_time > now_ms:
            continue
        by_open_time[int(candle.open_time)] = candle
    return [by_open_time[key] for key in sorted(by_open_time)]
