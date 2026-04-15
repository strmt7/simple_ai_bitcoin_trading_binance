"""Load/save and prompt user configuration."""

from __future__ import annotations

import json
from getpass import getpass
from pathlib import Path
from typing import Callable, Dict

from .types import RuntimeConfig, StrategyConfig, config_paths

SUPPORTED_SYMBOL = "BTCUSDC"


def _read_config_json(path: Path) -> Dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, TypeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    return payload


def _write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    # keep key material local and less exposed
    path.chmod(0o600)


def _normalize_runtime_payload(payload: Dict[str, object]) -> Dict[str, object]:
    normalized = dict(payload)
    symbol = str(normalized.get("symbol") or SUPPORTED_SYMBOL).upper()
    normalized["symbol"] = SUPPORTED_SYMBOL if symbol != SUPPORTED_SYMBOL else symbol
    return normalized


def load_runtime(overrides: Dict[str, object] | None = None) -> RuntimeConfig:
    paths = config_paths()
    payload = {}
    if paths["runtime"].exists():
        payload.update(_read_config_json(paths["runtime"]))
    if overrides:
        payload.update(overrides)
    payload = _normalize_runtime_payload(payload)
    return RuntimeConfig(**{k: v for k, v in payload.items() if hasattr(RuntimeConfig, k)})


def save_runtime(cfg: RuntimeConfig) -> RuntimeConfig:
    _write_json(config_paths()["runtime"], cfg.asdict())
    return cfg


def load_strategy() -> StrategyConfig:
    paths = config_paths()
    payload = {}
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
    return StrategyConfig(**{k: v for k, v in payload.items() if hasattr(StrategyConfig, k)})


def save_strategy(cfg: StrategyConfig) -> StrategyConfig:
    _write_json(config_paths()["strategy"], cfg.asdict())
    return cfg


def _coalesce_prompt(value: str, current: str) -> str:
    candidate = value.strip()
    return candidate or current


def prompt_runtime(current: RuntimeConfig, key_getter: Callable[[str], str] = input,
                  secret_getter: Callable[[str], str] = getpass) -> RuntimeConfig:
    """Collect Binance testnet credentials and safety defaults from stdin."""

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
        testnet=key_getter(f"Use Binance testnet? (y/n) [{'y' if current.testnet else 'n'}]: ").strip().lower() != "n",
        api_key=_coalesce_prompt(
            secret_getter("Binance API key (blank to keep): "),
            current.api_key,
        ),
        api_secret=_coalesce_prompt(
            secret_getter("Binance API secret (blank to keep): "),
            current.api_secret,
        ),
        dry_run=key_getter(f"Paper-trading mode? (y/n) [{'y' if current.dry_run else 'n'}]: ").strip().lower() != "n",
        validate_account=key_getter(f"Validate API credentials at startup? (y/n) [{'y' if current.validate_account else 'n'}]: ").strip().lower() != "n",
    )
