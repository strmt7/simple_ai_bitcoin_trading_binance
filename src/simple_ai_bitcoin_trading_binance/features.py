"""Feature construction for training and inference."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence, Tuple

from .api import Candle
from .market_data import clean_candles


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
_FEATURE_INDEX = {name: index for index, name in enumerate(FEATURE_NAMES)}


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
    return tuple(_FEATURE_INDEX[name] for name in normalized)


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
    if not all(math.isfinite(value) for value in (candle.open, candle.high, candle.low, candle.close)):
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


def _prefix_sum(values: Sequence[float]) -> list[float]:
    total = 0.0
    prefix = [0.0]
    for value in values:
        total += value
        prefix.append(total)
    return prefix


def _window_mean(prefix: Sequence[float], start: int, end: int) -> float:
    if end < start:
        return float("nan")
    return (prefix[end + 1] - prefix[start]) / float(end - start + 1)


def _rolling_mean(prefix: Sequence[float], end: int, window: int) -> float:
    if window <= 0 or end < window - 1:
        return float("nan")
    return _window_mean(prefix, end - window + 1, end)


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
    gains: list[float] = []
    losses: list[float] = []
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


def _safe_features(values: Sequence[float]) -> list[float]:
    return [0.0 if not math.isfinite(v) else float(v) for v in values]


@dataclass(frozen=True)
class _FeatureCache:
    candles: list[Candle]
    closes: list[float]
    volumes: list[float]
    close_prefix: list[float]
    volume_prefix: list[float]
    abs_change_prefix: list[float]
    true_range_prefix: list[float]
    gain_prefix: list[float]
    loss_prefix: list[float]


def _build_feature_cache(candles: Sequence[Candle]) -> _FeatureCache:
    cleaned = [candle for candle in clean_candles(candles) if _valid_ohlcv(candle)]
    closes = [candle.close for candle in cleaned]
    volumes = [candle.volume for candle in cleaned]
    abs_changes = [0.0]
    true_ranges = [0.0]
    gains = [0.0]
    losses = [0.0]
    for index in range(1, len(cleaned)):
        previous = closes[index - 1]
        current = closes[index]
        delta = current - previous
        gains.append(max(0.0, delta))
        losses.append(max(0.0, -delta))
        abs_changes.append(abs(_pct(current, previous)))
        true_ranges.append(_true_range(cleaned, index))
    return _FeatureCache(
        candles=cleaned,
        closes=closes,
        volumes=volumes,
        close_prefix=_prefix_sum(closes),
        volume_prefix=_prefix_sum(volumes),
        abs_change_prefix=_prefix_sum(abs_changes),
        true_range_prefix=_prefix_sum(true_ranges),
        gain_prefix=_prefix_sum(gains),
        loss_prefix=_prefix_sum(losses),
    )


def _rsi_at(cache: _FeatureCache, end: int, window: int) -> float:
    if window <= 0 or end < window:
        return float("nan")
    start = end + 1 - window
    avg_gain = _window_mean(cache.gain_prefix, start, end)
    avg_loss = _window_mean(cache.loss_prefix, start, end)
    if avg_loss == 0:
        return 100.0
    return 100 - 100 / (1 + avg_gain / avg_loss)


def _build_full_features(
    cache: _FeatureCache,
    index: int,
    short_window: int,
    long_window: int,
) -> tuple[float, ...] | None:
    closes = cache.closes
    volumes = cache.volumes
    close = closes[index]
    short = _rolling_mean(cache.close_prefix, index, short_window)
    long = _rolling_mean(cache.close_prefix, index, long_window)
    ema = _ema(closes[max(0, index + 1 - (2 * long_window)): index + 1], long_window)
    rsi = _rsi_at(cache, index, 14)
    if not all(math.isfinite(value) for value in (short, long, ema, rsi)):
        return None

    momentum = _pct(close, closes[index - 1]) if index >= 1 else 0.0
    momentum_3 = _pct(close, closes[index - 3]) if index >= 3 else 0.0
    momentum_10 = _pct(close, closes[index - 10]) if index >= 10 else 0.0
    momentum_20 = _pct(close, closes[index - 20]) if index >= 20 else 0.0
    spread = _safe_div(short - long, long)

    vol_moment = (
        _window_mean(cache.abs_change_prefix, index - 19, index)
        if index >= 20
        else float("nan")
    )
    atr_count = min(14, index)
    atr = _window_mean(cache.true_range_prefix, index - atr_count + 1, index)
    rel_atr = _safe_div(atr, close)
    ema_spread = _safe_div(ema - close, close)

    prev_vol = _rolling_mean(cache.volume_prefix, index - 1, min(20, max(1, index)))
    vol_ratio = _safe_div(volumes[index] - prev_vol, prev_vol)
    prev_short = _rolling_mean(cache.close_prefix, index - 2, short_window)
    trend_accel = _safe_div(short - prev_short, prev_short) if prev_short else 0.0
    gap_average = _rolling_mean(cache.close_prefix, index, min(5, index + 1))
    gap_to_vwap = _safe_div(close - gap_average, close)
    vol_short = _rolling_mean(cache.volume_prefix, index, min(short_window, index + 1))
    vol_long = _rolling_mean(cache.volume_prefix, index, min(long_window, index + 1))
    volume_trend = _safe_div(vol_short - vol_long, vol_long)

    return tuple(_safe_features([
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


def make_rows(
    candles: Sequence[Candle],
    short_window: int,
    long_window: int,
    *,
    lookahead: int = 1,
    label_threshold: float = 0.001,
    enabled_features: Sequence[str] | None = None,
) -> list[ModelRow]:
    if short_window <= 0 or long_window <= 0 or lookahead <= 0:
        raise ValueError("short_window, long_window, and lookahead must be positive")
    if long_window < short_window:
        raise ValueError("long_window must be greater than or equal to short_window")

    selected_indices = _feature_indices(enabled_features)
    cache = _build_feature_cache(candles)
    rows: list[ModelRow] = []
    min_window = max(long_window, short_window, lookahead + 2, 2 * long_window)
    if len(cache.candles) < min_window:
        return rows

    for i in range(long_window + lookahead, len(cache.candles) - lookahead):
        full_features = _build_full_features(cache, i, short_window, long_window)
        if full_features is None:
            continue
        features = tuple(full_features[index] for index in selected_indices)

        future = cache.closes[i + lookahead]
        present = cache.closes[i]
        label = int(_pct(future, present) >= label_threshold)
        rows.append(ModelRow(timestamp=cache.candles[i].close_time, close=present, features=features, label=label))

    return rows


def make_rows_legacy(candles: Sequence[Candle], short_window: int, long_window: int,
                     lookahead: int = 1) -> list[ModelRow]:
    """Compatibility helper for existing integrations expecting 5-feature rows."""
    return make_rows(candles, short_window, long_window, lookahead=lookahead, label_threshold=0.001)
