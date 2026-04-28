"""Richer feature engineering and training pipeline on top of the base model.

The core model (`.model.TrainedModel`) is a 13-feature logistic regression.
This module keeps the same math but expands the feature vector with:

* Non-linear transforms (`tanh`, `log1p`) of the base features.
* Polynomial pairwise interactions among the top-K base features.
* Multi-window SMA / RSI / volatility snapshots anchored at configurable extra
  lookbacks so the model can see short, medium, and long regimes in one row.

Everything remains pure stdlib — no numpy, no sklearn — so the ``TrainedModel``
serializer already in the repo can persist the expanded model without changes.
The expansion parameters are deterministic from the strategy config alone, so
inference at test / live time recomputes the same feature vector every call.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

from .api import Candle
from .features import (
    FEATURE_NAMES,
    FEATURE_VERSION,
    ModelRow,
    feature_signature as base_feature_signature,
    make_rows as make_base_rows,
    normalize_enabled_features,
)
from .market_data import clean_candles
from .model import TrainedModel, train as train_logistic

ADVANCED_FEATURE_VERSION = "v2-advanced"


@dataclass(frozen=True)
class AdvancedFeatureConfig:
    """Parameters that drive feature expansion and must match at inference.

    All fields are part of the model's feature signature — changing any of
    them forces a retrain, which is correct because the feature space itself
    has changed.
    """

    base_features: tuple[str, ...]
    polynomial_degree: int = 2
    polynomial_top_features: int = 6
    extra_lookback_windows: tuple[int, ...] = (5, 20, 60)
    nonlinear_transforms: tuple[str, ...] = ("tanh", "log1p")
    short_window: int = 10
    long_window: int = 40
    label_threshold: float = 0.001


def _tanh(x: float) -> float:
    try:
        return math.tanh(x)
    except (OverflowError, ValueError):
        return 1.0 if x > 0 else -1.0


def _log1p_signed(x: float) -> float:
    return math.copysign(math.log1p(abs(x)), x) if x != 0.0 else 0.0


def _sma(values: Sequence[float], window: int) -> float:
    if window <= 0 or len(values) < window:
        return float("nan")
    return sum(values[-window:]) / float(window)


def _rsi(values: Sequence[float], window: int) -> float:
    if window <= 0 or len(values) < window + 1:
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


def _volatility(values: Sequence[float], window: int) -> float:
    if window <= 1 or len(values) < window:
        return float("nan")
    recent = values[-window:]
    mean = sum(recent) / len(recent)
    variance = sum((v - mean) ** 2 for v in recent) / max(1, len(recent) - 1)
    return math.sqrt(max(0.0, variance))


def _safe(x: float) -> float:
    return 0.0 if not math.isfinite(x) else float(x)


def _extra_window_features(closes: Sequence[float], windows: Sequence[int]) -> list[float]:
    features: list[float] = []
    for window in windows:
        sma = _sma(closes, window)
        rsi = _rsi(closes, window)
        vol = _volatility(closes, window)
        anchor = closes[-1] if closes else 0.0
        features.extend([
            _safe((anchor - sma) / sma) if math.isfinite(sma) and sma != 0 else 0.0,
            _safe((rsi / 100.0) if math.isfinite(rsi) else 0.0),
            _safe(vol / anchor) if anchor != 0 else 0.0,
        ])
    return features


def _nonlinear_expand(values: Sequence[float], transforms: Sequence[str]) -> list[float]:
    out: list[float] = []
    for name in transforms:
        if name == "tanh":
            out.extend(_tanh(v) for v in values)
        elif name == "log1p":
            out.extend(_log1p_signed(v) for v in values)
        else:
            raise ValueError(f"Unsupported transform: {name!r}")
    return out


def _polynomial_pairs(values: Sequence[float], top_k: int, degree: int) -> list[float]:
    """Return pairwise products (and optionally triples) of the first ``top_k`` features.

    For ``degree == 2`` we emit the upper-triangle pairwise products.  For
    ``degree == 3`` we additionally emit cubes and a small triple-interaction.
    """

    if top_k <= 1 or degree < 2:
        return []
    base = list(values)[:top_k]
    pairs: list[float] = []
    for i in range(len(base)):
        for j in range(i, len(base)):
            pairs.append(base[i] * base[j])
    if degree >= 3 and len(base) >= 3:
        pairs.append(base[0] * base[1] * base[2])
        pairs.extend(v ** 3 for v in base[:3])
    return [_safe(v) for v in pairs]


def advanced_feature_dimension(cfg: AdvancedFeatureConfig) -> int:
    """Dimension of the expanded feature vector — used for model load checks."""

    base = len(cfg.base_features)
    # each extra window contributes 3 derived features
    extras = 3 * len(cfg.extra_lookback_windows)
    transforms = base * len(cfg.nonlinear_transforms)
    pairs = 0
    if cfg.polynomial_degree >= 2 and cfg.polynomial_top_features > 1:
        k = min(cfg.polynomial_top_features, base)
        pairs = k * (k + 1) // 2
        if cfg.polynomial_degree >= 3 and k >= 3:
            pairs += 1 + 3
    return base + extras + transforms + pairs


def expand_row(row: ModelRow, candles: Sequence[Candle], cfg: AdvancedFeatureConfig,
               at_index: int) -> ModelRow:
    """Return ``row`` with its feature tuple expanded per ``cfg``.

    ``candles`` is the full candle sequence whose ``at_index`` corresponds to
    ``row`` — this is how multi-window lookups find history behind the row.
    """

    base = list(row.features)
    closes = [c.close for c in candles[: at_index + 1]]
    extras = _extra_window_features(closes, cfg.extra_lookback_windows)
    transforms = _nonlinear_expand(base, cfg.nonlinear_transforms)
    pairs = _polynomial_pairs(base, cfg.polynomial_top_features, cfg.polynomial_degree)
    expanded = tuple(_safe(v) for v in base + extras + transforms + pairs)
    return ModelRow(
        timestamp=row.timestamp,
        close=row.close,
        features=expanded,
        label=row.label,
    )


def make_advanced_rows(
    candles: Sequence[Candle],
    cfg: AdvancedFeatureConfig,
    *,
    lookahead: int = 1,
) -> List[ModelRow]:
    """Build expanded ``ModelRow`` objects for ``candles`` using ``cfg``."""

    enabled = normalize_enabled_features(cfg.base_features)
    base_rows = make_base_rows(
        candles,
        cfg.short_window,
        cfg.long_window,
        lookahead=lookahead,
        label_threshold=cfg.label_threshold,
        enabled_features=enabled,
    )
    if not base_rows:
        return []
    # reconstruct the index alignment used by make_rows
    valid_candles = _filter_valid(candles)
    index_by_time = {candle.close_time: idx for idx, candle in enumerate(valid_candles)}
    expanded: list[ModelRow] = []
    for row in base_rows:
        idx = index_by_time.get(row.timestamp)
        if idx is None:
            continue
        expanded.append(expand_row(row, valid_candles, cfg, idx))
    return expanded


def _filter_valid(candles: Sequence[Candle]) -> list[Candle]:
    return clean_candles(candles)


def advanced_feature_signature(cfg: AdvancedFeatureConfig) -> str:
    """Deterministic signature for the advanced feature space."""

    base = base_feature_signature(
        cfg.short_window,
        cfg.long_window,
        cfg.label_threshold,
        feature_version=FEATURE_VERSION,
        enabled_features=cfg.base_features,
    )
    return "|".join([
        f"advanced_version={ADVANCED_FEATURE_VERSION}",
        f"polynomial_degree={cfg.polynomial_degree}",
        f"polynomial_top_features={cfg.polynomial_top_features}",
        f"extra_lookback_windows={','.join(str(w) for w in cfg.extra_lookback_windows)}",
        f"nonlinear_transforms={','.join(cfg.nonlinear_transforms)}",
        base,
    ])


@dataclass(frozen=True)
class AdvancedTrainingReport:
    """What the training suite persists alongside an advanced model artifact."""

    feature_dim: int
    feature_signature: str
    epochs: int
    learning_rate: float
    l2_penalty: float
    seed: int
    row_count: int
    positive_rate: float


def train_advanced(
    rows: Sequence[ModelRow],
    cfg: AdvancedFeatureConfig,
    *,
    epochs: int,
    learning_rate: float,
    l2_penalty: float,
    seed: int = 7,
) -> Tuple[TrainedModel, AdvancedTrainingReport]:
    """Train a logistic regression on an expanded feature set.

    Returns the ``TrainedModel`` along with a small report describing the run
    so downstream code can persist reproducibility metadata.
    """

    if not rows:
        raise ValueError("No training rows available")
    signature = advanced_feature_signature(cfg)
    model = train_logistic(
        list(rows),
        epochs=epochs,
        learning_rate=learning_rate,
        seed=seed,
        l2_penalty=l2_penalty,
        feature_signature=signature,
    )
    positives = sum(1 for row in rows if row.label == 1)
    report = AdvancedTrainingReport(
        feature_dim=model.feature_dim,
        feature_signature=signature,
        epochs=epochs,
        learning_rate=learning_rate,
        l2_penalty=l2_penalty,
        seed=seed,
        row_count=len(rows),
        positive_rate=(positives / len(rows)) if rows else 0.0,
    )
    return model, report


def default_config_for(objective_name: str, strategy_feature_names: Sequence[str]) -> AdvancedFeatureConfig:
    """Build a starter ``AdvancedFeatureConfig`` tied to an objective name.

    Callers in the training suite layer their own per-objective overrides on
    top of the returned config; this helper keeps the defaults in one place.
    """

    names = normalize_enabled_features(strategy_feature_names or FEATURE_NAMES)
    if objective_name == "conservative":
        return AdvancedFeatureConfig(
            base_features=names,
            polynomial_degree=2,
            polynomial_top_features=5,
            extra_lookback_windows=(10, 30, 90),
        )
    if objective_name == "risky":
        return AdvancedFeatureConfig(
            base_features=names,
            polynomial_degree=3,
            polynomial_top_features=9,
            extra_lookback_windows=(3, 15, 45, 120),
        )
    return AdvancedFeatureConfig(
        base_features=names,
        polynomial_degree=2,
        polynomial_top_features=7,
        extra_lookback_windows=(5, 20, 60),
    )
