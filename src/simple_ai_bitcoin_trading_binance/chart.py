"""ASCII chart and sparkline helpers for the operator shell and TUI.

Everything is stdlib-only — no numpy, no pandas — and deterministic for a
given input so tests can pin exact output.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

_SPARK_RAMP = "▁▂▃▄▅▆▇█"


def _finite(values: Iterable[float]) -> list[float]:
    return [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(float(v))]


def sparkline(values: Sequence[float]) -> str:
    """Return a single-row Unicode sparkline for ``values``.

    Empty or all-NaN input renders an empty string.  Flat series collapse to
    the mid-ramp glyph so the reader still sees that data was present.
    """

    finite = _finite(values)
    if not finite:
        return ""
    low = min(finite)
    high = max(finite)
    span = high - low
    mid = _SPARK_RAMP[len(_SPARK_RAMP) // 2]
    if span <= 0:
        return mid * len(finite)
    levels = len(_SPARK_RAMP) - 1
    chars: list[str] = []
    for value in finite:
        ratio = (value - low) / span
        index = int(round(ratio * levels))
        index = 0 if index < 0 else (levels if index > levels else index)
        chars.append(_SPARK_RAMP[index])
    return "".join(chars)


@dataclass(frozen=True)
class ChartLayout:
    width: int
    height: int


def _resample(values: Sequence[float], width: int) -> list[float]:
    finite = _finite(values)
    if not finite:
        return []
    if width <= 0:
        return []
    if len(finite) <= width:
        return finite[-width:] if width < len(finite) else list(finite)
    # decimate keeping the most recent samples aligned to the right edge
    bucket = len(finite) / width
    sampled: list[float] = []
    for i in range(width):
        start = int(i * bucket)
        # float rounding can collapse a bucket to zero width — force at least
        # one sample so the output length always equals ``width``.
        end = max(int((i + 1) * bucket), start + 1)
        slice_ = finite[start:end]
        sampled.append(sum(slice_) / len(slice_))
    return sampled


def equity_curve(
    values: Sequence[float],
    *,
    width: int = 60,
    height: int = 10,
    baseline: float | None = None,
) -> list[str]:
    """Render an ASCII equity curve.

    The curve uses ``*`` for points and ``·`` for the scale's horizontal axis
    baseline when supplied.  Width and height are clamped to a minimum of 4
    so the output remains visible in narrow TUIs.
    """

    width = max(4, int(width))
    height = max(4, int(height))
    samples = _resample(values, width)
    if not samples:
        return ["(no data)"]
    low = min(samples)
    high = max(samples)
    if baseline is not None:
        low = min(low, float(baseline))
        high = max(high, float(baseline))
    span = high - low
    if span <= 0:
        span = 1.0
    rows: list[list[str]] = [[" "] * width for _ in range(height)]
    for x, value in enumerate(samples):
        ratio = (value - low) / span
        y = height - 1 - int(round(ratio * (height - 1)))
        y = 0 if y < 0 else (height - 1 if y >= height else y)
        rows[y][x] = "*"
    if baseline is not None:
        ratio = (float(baseline) - low) / span
        yb = height - 1 - int(round(ratio * (height - 1)))
        yb = 0 if yb < 0 else (height - 1 if yb >= height else yb)
        for x in range(width):
            if rows[yb][x] == " ":
                rows[yb][x] = "·"
    return ["".join(row) for row in rows]


@dataclass(frozen=True)
class MiniCandle:
    open: float
    high: float
    low: float
    close: float


def mini_candles(
    candles: Sequence[MiniCandle],
    *,
    width: int = 40,
    height: int = 8,
) -> list[str]:
    """Render a very compact ASCII candle chart.

    Each candle occupies one column; the body is ``█`` for bullish and ``▓``
    for bearish bars; wicks are drawn as ``│``.
    """

    width = max(4, int(width))
    height = max(4, int(height))
    if not candles:
        return ["(no candles)"]
    sampled = candles[-width:]
    highs = [c.high for c in sampled]
    lows = [c.low for c in sampled]
    low = min(lows)
    high = max(highs)
    span = high - low
    if span <= 0:
        span = 1.0
    rows: list[list[str]] = [[" "] * len(sampled) for _ in range(height)]
    for x, candle in enumerate(sampled):
        body_top = max(candle.open, candle.close)
        body_bot = min(candle.open, candle.close)
        y_high = height - 1 - int(round((candle.high - low) / span * (height - 1)))
        y_low = height - 1 - int(round((candle.low - low) / span * (height - 1)))
        y_body_top = height - 1 - int(round((body_top - low) / span * (height - 1)))
        y_body_bot = height - 1 - int(round((body_bot - low) / span * (height - 1)))
        for y in range(min(y_high, y_body_top), max(y_low, y_body_bot) + 1):
            if y < 0 or y >= height:
                continue
            rows[y][x] = "│"
        glyph = "█" if candle.close >= candle.open else "▓"
        top = min(y_body_top, y_body_bot)
        bot = max(y_body_top, y_body_bot)
        for y in range(top, bot + 1):
            if y < 0 or y >= height:
                continue
            rows[y][x] = glyph
    return ["".join(row) for row in rows]


def format_equity_footer(
    starting_cash: float,
    ending_cash: float,
    realized_pnl: float,
    win_rate: float,
) -> str:
    """Compact scoreboard line shared between backtest + live report surfaces."""

    def _fmt(amount: float) -> str:
        sign = "+" if amount >= 0 else "-"
        return f"{sign}{abs(amount):,.2f}"

    return (
        f"start {starting_cash:,.2f}  end {ending_cash:,.2f}  "
        f"pnl {_fmt(realized_pnl)}  win {win_rate:.0%}"
    )
