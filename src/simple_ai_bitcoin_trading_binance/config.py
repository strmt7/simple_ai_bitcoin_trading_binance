"""Load/save and prompt user configuration."""

from __future__ import annotations

import json
from getpass import getpass
from pathlib import Path
from typing import Callable, Dict

from .types import RuntimeConfig, StrategyConfig, config_paths


def _write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    # keep key material local and less exposed
    path.chmod(0o600)


def load_runtime(overrides: Dict[str, object] | None = None) -> RuntimeConfig:
    paths = config_paths()
    payload = {}
    if paths["runtime"].exists():
        payload.update(json.loads(paths["runtime"].read_text(encoding="utf-8")))
    if overrides:
        payload.update(overrides)
    return RuntimeConfig(**{k: v for k, v in payload.items() if hasattr(RuntimeConfig, k)})


def save_runtime(cfg: RuntimeConfig) -> RuntimeConfig:
    _write_json(config_paths()["runtime"], cfg.asdict())
    return cfg


def load_strategy() -> StrategyConfig:
    paths = config_paths()
    payload = {}
    if paths["strategy"].exists():
        raw = json.loads(paths["strategy"].read_text(encoding="utf-8"))
        payload.update(raw)
    if payload.get("feature_windows") and isinstance(payload["feature_windows"], list):
        payload["feature_windows"] = tuple(payload["feature_windows"])
    return StrategyConfig(**{k: v for k, v in payload.items() if hasattr(StrategyConfig, k)})


def save_strategy(cfg: StrategyConfig) -> StrategyConfig:
    _write_json(config_paths()["strategy"], cfg.asdict())
    return cfg


def prompt_runtime(current: RuntimeConfig, key_getter: Callable[[str], str] = input,
                  secret_getter: Callable[[str], str] = getpass) -> RuntimeConfig:
    """Collect Binance testnet credentials and safety defaults from stdin."""

    market = key_getter(f"Market type [spot/futures] [{current.market_type}]: ") or current.market_type
    market = market.strip().lower()
    if market not in {"spot", "futures"}:
        market = current.market_type

    return RuntimeConfig(
        symbol=(key_getter(f"Trading symbol [{current.symbol}]: ") or current.symbol).upper(),
        interval=(key_getter(f"Kline interval [{current.interval}]: ") or current.interval),
        market_type=market,
        testnet=key_getter(f"Use Binance testnet? (y/n) [{'y' if current.testnet else 'n'}]: ").strip().lower() != "n",
        api_key=secret_getter("Binance API key (blank to keep): ") or current.api_key,
        api_secret=secret_getter("Binance API secret (blank to keep): ") or current.api_secret,
        dry_run=key_getter(f"Paper-trading mode? (y/n) [{'y' if current.dry_run else 'n'}]: ").strip().lower() != "n",
        validate_account=key_getter(f"Validate API credentials at startup? (y/n) [{'y' if current.validate_account else 'n'}]: ").strip().lower() != "n",
    )
