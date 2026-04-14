"""Feature construction for training and inference."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

from .api import Candle


@dataclass(frozen=True)
class ModelRow:
    timestamp: int
    close: float
    features: Tuple[float, ...]
    label: int


def _sma(values: Sequence[float], window: int) -> float:
    if len(values) < window:
        return float("nan")
    return sum(values[-window:]) / float(window)


def _ema(values: Sequence[float], window: int) -> float:
    if len(values) < window:
        return float("nan")
    k = 2.0 / (window + 1)
    ema = values[0]
    for value in values[1:]:
        ema = value * k + ema * (1 - k)
    return ema


def _rsi(values: Sequence[float], window: int) -> float:
    if len(values) < window + 1:
        return float("nan")
    gains: List[float] = []
    losses: List[float] = []
    for i in range(len(values) - window, len(values)):
        delta = values[i] - values[i - 1]
        gains.append(max(0.0, delta))
        losses.append(max(0.0, -delta))
    avg_gain = sum(gains) / len(gains)
    avg_loss = sum(losses) / len(losses)
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - 100 / (1 + rs)


def make_rows(candles: Sequence[Candle], short_window: int, long_window: int,
              lookahead: int = 1) -> List[ModelRow]:
    closes = [c.close for c in candles]
    rows: List[ModelRow] = []
    if len(candles) < max(long_window, short_window, lookahead + 2):
        return rows

    for i in range(long_window + lookahead, len(candles) - lookahead):
        window = closes[: i + 1]
        short = _sma(window, short_window)
        long = _sma(window, long_window)
        ema = _ema(window, long_window)
        rsi = _rsi(window, 14)
        momentum = (window[-1] / window[-2] - 1.0) if window[-2] else 0.0
        if not all(math.isfinite(v) for v in (short, long, ema, rsi)):
            continue
        spread = (short - long) / long if long else 0.0
        vol = 0.0
        if i > 0:
            vol = _sma([abs(closes[j] - closes[j - 1]) / closes[j - 1] for j in range(1, i + 1)], min(20, i))

        features = (
            momentum,
            spread,
            rsi / 100.0,
            ema / window[-1] - 1.0,
            vol,
        )

        # label: 1 if future close increases by at least 0.1%, else 0
        future = closes[i + lookahead]
        present = closes[i]
        label = int(future >= present * 1.001)
        rows.append(ModelRow(timestamp=candles[i].close_time, close=present, features=features, label=label))
    return rows

