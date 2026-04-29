"""Load/save and prompt user configuration."""

from __future__ import annotations

import json
from dataclasses import fields
from getpass import getpass
from pathlib import Path
from typing import Any, Callable

from .features import normalize_enabled_features
from .storage import write_json_atomic
from .types import RuntimeConfig, StrategyConfig, config_paths

SUPPORTED_SYMBOL = "BTCUSDC"
_RUNTIME_FIELD_NAMES = frozenset(field.name for field in fields(RuntimeConfig))
_STRATEGY_FIELD_NAMES = frozenset(field.name for field in fields(StrategyConfig))


def _read_config_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, TypeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    return dict(payload)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    write_json_atomic(path, payload, indent=2, sort_keys=True, mode=0o600)


def _known_payload(payload: dict[str, Any], allowed_fields: frozenset[str]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key in allowed_fields}


def _normalize_runtime_payload(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)
    symbol = str(normalized.get("symbol") or SUPPORTED_SYMBOL).upper()
    normalized["symbol"] = SUPPORTED_SYMBOL if symbol != SUPPORTED_SYMBOL else symbol
    return normalized


def load_runtime(overrides: dict[str, Any] | None = None) -> RuntimeConfig:
    paths = config_paths()
    payload: dict[str, Any] = {}
    if paths["runtime"].exists():
        payload.update(_read_config_json(paths["runtime"]))
    if overrides:
        payload.update(overrides)
    payload = _normalize_runtime_payload(payload)
    return RuntimeConfig(**_known_payload(payload, _RUNTIME_FIELD_NAMES))


def save_runtime(cfg: RuntimeConfig) -> RuntimeConfig:
    _write_json(config_paths()["runtime"], cfg.asdict())
    return cfg


def load_strategy() -> StrategyConfig:
    paths = config_paths()
    payload: dict[str, Any] = {}
    if paths["strategy"].exists():
        raw = _read_config_json(paths["strategy"])
        payload.update(raw)
    windows = payload.get("feature_windows")
    if isinstance(windows, (list, tuple)) and len(windows) == 2:
        try:
            payload["feature_windows"] = tuple(int(v) for v in windows)
        except (TypeError, ValueError):
            payload["feature_windows"] = (10, 40)
    else:
        payload["feature_windows"] = (10, 40)
    enabled_features = payload.get("enabled_features")
    if isinstance(enabled_features, (list, tuple)):
        try:
            payload["enabled_features"] = normalize_enabled_features(enabled_features)
        except ValueError:
            payload["enabled_features"] = normalize_enabled_features()
    else:
        payload["enabled_features"] = normalize_enabled_features()
    return StrategyConfig(**_known_payload(payload, _STRATEGY_FIELD_NAMES))


def save_strategy(cfg: StrategyConfig) -> StrategyConfig:
    _write_json(config_paths()["strategy"], cfg.asdict())
    return cfg


def _coalesce_prompt(value: str, current: str) -> str:
    candidate = value.strip()
    return candidate or current


def _prompt_bool(value: str, current: bool) -> bool:
    token = value.strip().lower()
    if token in {"y", "yes", "true", "1", "on"}:
        return True
    if token in {"n", "no", "false", "0", "off"}:
        return False
    return current


def prompt_runtime(current: RuntimeConfig, key_getter: Callable[[str], str] = input,
                  secret_getter: Callable[[str], str] = getpass) -> RuntimeConfig:
    """Collect Binance non-mainnet credentials and safety defaults from stdin."""

    market = _coalesce_prompt(
        key_getter(f"Market type [spot/futures] [{current.market_type}]: "),
        current.market_type,
    ).lower()
    if market not in {"spot", "futures"}:
        market = current.market_type

    symbol = _coalesce_prompt(
        key_getter(f"Trading symbol [{current.symbol}]: "),
        current.symbol,
    ).upper()
    if symbol != SUPPORTED_SYMBOL:
        symbol = SUPPORTED_SYMBOL

    return RuntimeConfig(
        symbol=symbol,
        interval=_coalesce_prompt(
            key_getter(f"Kline interval [{current.interval}]: "),
            current.interval,
        ),
        market_type=market,
        testnet=_prompt_bool(
            key_getter(f"Use Binance testnet? (y/n) [{'y' if current.testnet else 'n'}]: "),
            current.testnet,
        ),
        demo=_prompt_bool(
            key_getter(f"Use Binance Demo Trading API? (y/n) [{'y' if current.demo else 'n'}]: "),
            current.demo,
        ),
        api_key=_coalesce_prompt(
            secret_getter("Binance API key (blank to keep): "),
            current.api_key,
        ),
        api_secret=_coalesce_prompt(
            secret_getter("Binance API secret (blank to keep): "),
            current.api_secret,
        ),
        dry_run=_prompt_bool(
            key_getter(f"Paper-trading mode? (y/n) [{'y' if current.dry_run else 'n'}]: "),
            current.dry_run,
        ),
        validate_account=_prompt_bool(
            key_getter(f"Validate API credentials at startup? (y/n) [{'y' if current.validate_account else 'n'}]: "),
            current.validate_account,
        ),
        max_rate_calls_per_minute=current.max_rate_calls_per_minute,
        recv_window_ms=current.recv_window_ms,
        compute_backend=current.compute_backend,
        managed_usdc=current.managed_usdc,
        managed_btc=current.managed_btc,
    )
