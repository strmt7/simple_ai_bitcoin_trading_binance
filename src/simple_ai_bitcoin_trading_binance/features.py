"""Feature construction for training and inference."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

from .api import Candle


FEATURE_VERSION = "v1"

# Ordered feature names and count used by persistence checks.
FEATURE_NAMES = (
    "momentum_1",
    "momentum_3",
    "momentum_10",
    "momentum_20",
    "ema_spread",
    "rsi",
    "ema_gap",
    "relative_atr",
    "volatility_20",
    "volume_ratio",
    "trend_acceleration",
    "gap_to_vwap",
    "volume_trend",
)


def normalize_enabled_features(enabled_features: Sequence[str] | None = None) -> tuple[str, ...]:
    if enabled_features is None:
        return tuple(FEATURE_NAMES)
    normalized: list[str] = []
    for name in enabled_features:
        feature_name = str(name)
        if feature_name not in FEATURE_NAMES:
            raise ValueError(f"Unknown feature: {feature_name}")
        if feature_name not in normalized:
            normalized.append(feature_name)
    if not normalized:
        raise ValueError("At least one feature must remain enabled")
    return tuple(normalized)


def _feature_indices(enabled_features: Sequence[str] | None = None) -> tuple[int, ...]:
    normalized = normalize_enabled_features(enabled_features)
    return tuple(FEATURE_NAMES.index(name) for name in normalized)


def feature_signature(
    short_window: int,
    long_window: int,
    label_threshold: float,
    *,
    feature_version: str = FEATURE_VERSION,
    enabled_features: Sequence[str] | None = None,
) -> str:
    """Return a deterministic signature for a feature configuration."""
    short_window = int(short_window)
    long_window = int(long_window)
    threshold = float(label_threshold)
    selected = normalize_enabled_features(enabled_features)
    return "|".join(
        [
            f"feature_version={feature_version}",
            f"feature_count={len(selected)}",
            f"feature_names={','.join(selected)}",
            f"short_window={short_window}",
            f"long_window={long_window}",
            f"label_threshold={threshold:.10g}",
        ]
    )


def _valid_ohlcv(candle: Candle) -> bool:
    if not math.isfinite(candle.open) or not math.isfinite(candle.high) or not math.isfinite(candle.low) or not math.isfinite(candle.close):
        return False
    if candle.open <= 0.0 or candle.high <= 0.0 or candle.low <= 0.0 or candle.close <= 0.0:
        return False
    if candle.volume < 0.0 or candle.open_time < 0 or candle.close_time < 0:
        return False
    if candle.low > candle.high:
        return False
    if not (candle.low <= candle.open <= candle.high):
        return False
    if not (candle.low <= candle.close <= candle.high):
        return False
    if candle.close_time < candle.open_time:
        return False
    return True


@dataclass(frozen=True)
class ModelRow:
    timestamp: int
    close: float
    features: Tuple[float, ...]
    label: int


def feature_dimension(enabled_features: Sequence[str] | None = None) -> int:
    return len(normalize_enabled_features(enabled_features))


def _safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    if denominator == 0:
        return default
    return numerator / denominator


def _pct(numerator: float, denominator: float) -> float:
    return _safe_div(numerator - denominator, denominator)


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


def _true_range(candles: Sequence[Candle], i: int) -> float:
    prev_close = candles[i - 1].close
    if prev_close <= 0:
        return 0.0
    high = candles[i].high
    low = candles[i].low
    return max(high - low, abs(high - prev_close), abs(low - prev_close))


def _safe_features(values: Sequence[float]) -> List[float]:
    return [0.0 if not math.isfinite(v) else float(v) for v in values]


def make_rows(
    candles: Sequence[Candle],
    short_window: int,
    long_window: int,
    *,
    lookahead: int = 1,
    label_threshold: float = 0.001,
    enabled_features: Sequence[str] | None = None,
) -> List[ModelRow]:
    if short_window <= 0 or long_window <= 0 or lookahead <= 0:
        raise ValueError("short_window, long_window, and lookahead must be positive")
    if long_window < short_window:
        raise ValueError("long_window must be greater than or equal to short_window")

    selected_indices = _feature_indices(enabled_features)
    candles = [c for c in candles if _valid_ohlcv(c)]
    candles = sorted(candles, key=lambda c: c.open_time)
    closes = [c.close for c in candles]
    rows: List[ModelRow] = []
    min_window = max(long_window, short_window, lookahead + 2, 2 * long_window)
    if len(candles) < min_window:
        return rows

    for i in range(long_window + lookahead, len(candles) - lookahead):
        window = closes[: i + 1]
        short = _sma(window, short_window)
        long = _sma(window, long_window)
        ema = _ema(window, long_window)
        rsi = _rsi(window, 14)

        if not all(math.isfinite(v) for v in (short, long, ema, rsi)):
            continue

        momentum = _pct(window[-1], window[-2]) if i >= 1 else 0.0
        momentum_3 = _pct(window[-1], window[-4]) if i >= 3 else 0.0
        momentum_10 = _pct(window[-1], window[-11]) if i >= 10 else 0.0
        momentum_20 = _pct(window[-1], window[-21]) if i >= 20 else 0.0

        spread = _safe_div(short - long, long)

        close_changes = [_pct(window[j], window[j - 1]) for j in range(1, i + 1)]
        vol_moment = _sma([abs(v) for v in close_changes[max(0, len(close_changes) - 20):]], 20)
        atr_window = [_true_range(candles[: i + 1], j) for j in range(1, i + 1)]
        atr = _sma(atr_window, min(14, len(atr_window)))
        rel_atr = _safe_div(atr, window[-1])

        ema_spread = _safe_div(ema - window[-1], window[-1])

        volume_window = [c.volume for c in candles[: i + 1]]
        prev_vol = _sma(volume_window[:-1], min(20, max(1, i)))
        vol_ratio = _safe_div(volume_window[-1] - prev_vol, prev_vol)

        prev_short = _sma(window[:-2], short_window)
        trend_accel = _safe_div(short - prev_short, prev_short) if prev_short else 0.0

        gap_to_vwap = _safe_div(
            window[-1] - _sma(window[max(0, len(window) - 5):], min(5, len(window))),
            window[-1],
        )

        vol_short = _sma(volume_window[-short_window:], min(short_window, len(volume_window)))
        vol_long = _sma(volume_window[-long_window:], min(long_window, len(volume_window)))
        volume_trend = _safe_div(vol_short - vol_long, vol_long)

        full_features = tuple(_safe_features([
            momentum,
            momentum_3,
            momentum_10,
            momentum_20,
            spread,
            rsi / 100.0,
            ema_spread,
            rel_atr,
            vol_moment,
            vol_ratio,
            trend_accel,
            gap_to_vwap,
            volume_trend,
        ]))
        features = tuple(full_features[index] for index in selected_indices)

        if not all(math.isfinite(v) for v in features):
            continue

        future = closes[i + lookahead]
        present = closes[i]
        label = int(_pct(future, present) >= label_threshold)
        rows.append(ModelRow(timestamp=candles[i].close_time, close=present, features=features, label=label))

    return rows


def make_rows_legacy(candles: Sequence[Candle], short_window: int, long_window: int,
                     lookahead: int = 1) -> List[ModelRow]:
    """Compatibility helper for existing integrations expecting 5-feature rows."""
    return make_rows(candles, short_window, long_window, lookahead=lookahead, label_threshold=0.001)
