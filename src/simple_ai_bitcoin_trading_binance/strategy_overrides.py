"""Persist and apply model-selected execution strategy overrides."""

from __future__ import annotations

import math
from typing import Any

from .types import StrategyConfig

StrategyOverrideValue = int | float

EXECUTION_STRATEGY_OVERRIDE_KEYS: tuple[str, ...] = (
    "leverage",
    "risk_per_trade",
    "max_position_pct",
    "max_open_positions",
    "stop_loss_pct",
    "take_profit_pct",
    "signal_threshold",
    "cooldown_minutes",
    "max_trades_per_day",
    "max_drawdown_limit",
    "confidence_beta",
    "taker_fee_bps",
    "slippage_bps",
)


def _clean_override_value(value: Any) -> StrategyOverrideValue | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, float):
        return float(value) if math.isfinite(value) else None
    return None


def clean_strategy_overrides(raw: Any) -> dict[str, StrategyOverrideValue]:
    """Return JSON-safe execution overrides from an untrusted model payload."""

    if not isinstance(raw, dict):
        return {}
    allowed = set(EXECUTION_STRATEGY_OVERRIDE_KEYS)
    cleaned: dict[str, StrategyOverrideValue] = {}
    for key, value in raw.items():
        key_text = str(key)
        if key_text not in allowed:
            continue
        clean_value = _clean_override_value(value)
        if clean_value is not None:
            cleaned[key_text] = clean_value
    return cleaned


def strategy_overrides_from_config(strategy: StrategyConfig) -> dict[str, StrategyOverrideValue]:
    """Capture the execution fields needed to reproduce a model's backtest."""

    payload = strategy.asdict()
    return {
        key: value
        for key in EXECUTION_STRATEGY_OVERRIDE_KEYS
        if (value := _clean_override_value(payload.get(key))) is not None
    }


def apply_strategy_overrides(
    strategy: StrategyConfig,
    overrides: Any,
    *,
    protected_keys: set[str] | frozenset[str] | tuple[str, ...] = (),
) -> StrategyConfig:
    """Return ``strategy`` with safe persisted execution overrides applied."""

    cleaned = clean_strategy_overrides(overrides)
    protected = set(protected_keys)
    for key in protected:
        cleaned.pop(key, None)
    if not cleaned:
        return strategy
    payload = strategy.asdict()
    payload.update(cleaned)
    return StrategyConfig(**payload)


def apply_model_strategy_overrides(
    strategy: StrategyConfig,
    model: Any,
    *,
    protected_keys: set[str] | frozenset[str] | tuple[str, ...] = (),
) -> StrategyConfig:
    """Apply execution overrides embedded in a trained model artifact."""

    return apply_strategy_overrides(
        strategy,
        getattr(model, "strategy_overrides", {}),
        protected_keys=protected_keys,
    )
