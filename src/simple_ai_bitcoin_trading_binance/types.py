"""Configuration and data structure definitions used across the CLI."""

from __future__ import annotations

from dataclasses import dataclass, field
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from .features import FEATURE_NAMES, FEATURE_VERSION, normalize_enabled_features


@dataclass
class RuntimeConfig:
    """Runtime configuration stored in the user profile."""

    symbol: str = "BTCUSDC"
    interval: str = "15m"
    market_type: str = "spot"
    testnet: bool = True
    api_key: str = ""
    api_secret: str = ""
    dry_run: bool = True
    validate_account: bool = True
    max_rate_calls_per_minute: int = 1100
    recv_window_ms: int = 5000
    compute_backend: str = "cpu"
    managed_usdc: float = 1000.0
    managed_btc: float = 0.0

    def asdict(self) -> Dict[str, Any]:
        return asdict(self)

    def public_dict(self) -> Dict[str, Any]:
        payload = self.asdict()
        for field_name in ("api_key", "api_secret"):
            if payload.get(field_name):
                payload[field_name] = "<redacted>"
        return payload


@dataclass
class StrategyConfig:
    """Tunable strategy inputs and risk controls."""

    leverage: float = 1.0
    risk_per_trade: float = 0.01
    max_position_pct: float = 0.20
    max_open_positions: int = 1
    stop_loss_pct: float = 0.02
    take_profit_pct: float = 0.03
    feature_windows: tuple[int, int] = (10, 40)
    signal_threshold: float = 0.58
    model_lookback: int = 250
    cooldown_minutes: int = 5
    max_trades_per_day: int = 24
    max_drawdown_limit: float = 0.25
    training_epochs: int = 250
    confidence_beta: float = 0.85
    taker_fee_bps: float = 1.0
    slippage_bps: float = 5.0
    label_threshold: float = 0.001
    feature_version: str = FEATURE_VERSION
    enabled_features: tuple[str, ...] = FEATURE_NAMES
    order_type: str = "MARKET"
    time_in_force: str = "GTC"
    post_only: bool = False
    reduce_only_on_close: bool = True

    def __post_init__(self) -> None:
        self.feature_windows = tuple(int(value) for value in self.feature_windows)
        self.enabled_features = normalize_enabled_features(self.enabled_features)

    def asdict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["feature_windows"] = list(self.feature_windows)
        payload["enabled_features"] = list(self.enabled_features)
        return payload


@dataclass
class RiskProfile:
    """Runtime stats and constraints that are not part of strategy tuning."""

    starting_cash: float = 1000.0
    max_daily_trades: int = 50
    max_open_positions: int = 1
    last_run: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")


def config_paths() -> Dict[str, Path]:
    """Return the default config directories used by the CLI."""

    base = Path.home() / ".config" / "simple_ai_bitcoin_trading_binance"
    return {
        "base": base,
        "runtime": base / "runtime.json",
        "strategy": base / "strategy.json",
    }
