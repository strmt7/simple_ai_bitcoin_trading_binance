"""Configuration and data structure definitions used across the CLI."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from .features import FEATURE_NAMES, FEATURE_VERSION, normalize_enabled_features


def _coerce_bool(value: object, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"y", "yes", "true", "1", "on"}:
            return True
        if token in {"n", "no", "false", "0", "off"}:
            return False
    return default


def _coerce_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return parsed


def _finite_float(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return parsed if math.isfinite(parsed) else default


def _normal_interval(value: object, default: str = "15m") -> str:
    candidate = str(value or "").strip()
    return candidate or default


@dataclass
class RuntimeConfig:
    """Runtime configuration stored in the user profile."""

    symbol: str = "BTCUSDC"
    interval: str = "15m"
    market_type: str = "spot"
    testnet: bool = True
    demo: bool = False
    api_key: str = ""
    api_secret: str = ""
    dry_run: bool = True
    validate_account: bool = True
    max_rate_calls_per_minute: int = 1100
    recv_window_ms: int = 5000
    compute_backend: str = "cpu"
    managed_usdc: float = 1000.0
    managed_btc: float = 0.0

    def __post_init__(self) -> None:
        self.symbol = "BTCUSDC" if str(self.symbol or "BTCUSDC").upper() != "BTCUSDC" else "BTCUSDC"
        self.interval = _normal_interval(self.interval)
        self.market_type = str(self.market_type or "spot").lower()
        self.testnet = _coerce_bool(self.testnet, True)
        self.demo = _coerce_bool(self.demo, False)
        self.dry_run = _coerce_bool(self.dry_run, True)
        self.validate_account = _coerce_bool(self.validate_account, True)
        self.api_key = str(self.api_key or "")
        self.api_secret = str(self.api_secret or "")
        self.max_rate_calls_per_minute = max(1, min(2000, _coerce_int(self.max_rate_calls_per_minute, 1100)))
        self.recv_window_ms = max(1, min(60000, _coerce_int(self.recv_window_ms, 5000)))
        self.compute_backend = str(self.compute_backend or "cpu")
        self.managed_usdc = max(0.0, _finite_float(self.managed_usdc, 1000.0))
        self.managed_btc = max(0.0, _finite_float(self.managed_btc, 0.0))

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
    external_signals_enabled: bool = False
    external_signal_max_adjustment: float = 0.04
    external_signal_min_providers: int = 2
    external_signal_ttl_seconds: int = 300
    external_signal_timeout_seconds: float = 3.0

    def __post_init__(self) -> None:
        windows = tuple(self.feature_windows) if isinstance(self.feature_windows, (list, tuple)) else (10, 40)
        if len(windows) != 2:
            windows = (10, 40)
        short_window = max(1, _coerce_int(windows[0], 10))
        long_window = max(short_window, _coerce_int(windows[1], 40))
        self.feature_windows = (short_window, long_window)
        self.leverage = max(1.0, _finite_float(self.leverage, 1.0))
        self.risk_per_trade = _finite_float(self.risk_per_trade, 0.01)
        self.max_position_pct = _finite_float(self.max_position_pct, 0.20)
        self.max_open_positions = max(0, _coerce_int(self.max_open_positions, 1))
        self.stop_loss_pct = _finite_float(self.stop_loss_pct, 0.02)
        self.take_profit_pct = _finite_float(self.take_profit_pct, 0.03)
        self.signal_threshold = min(0.99, max(0.01, _finite_float(self.signal_threshold, 0.58)))
        self.model_lookback = max(1, _coerce_int(self.model_lookback, 250))
        self.cooldown_minutes = max(0, _coerce_int(self.cooldown_minutes, 5))
        self.max_trades_per_day = max(0, _coerce_int(self.max_trades_per_day, 24))
        self.max_drawdown_limit = _finite_float(self.max_drawdown_limit, 0.25)
        self.training_epochs = max(1, _coerce_int(self.training_epochs, 250))
        self.confidence_beta = min(1.0, max(0.0, _finite_float(self.confidence_beta, 0.85)))
        self.taker_fee_bps = _finite_float(self.taker_fee_bps, 1.0)
        self.slippage_bps = _finite_float(self.slippage_bps, 5.0)
        self.label_threshold = _finite_float(self.label_threshold, 0.001)
        self.feature_version = str(self.feature_version or FEATURE_VERSION)
        self.order_type = str(self.order_type or "MARKET").upper()
        self.time_in_force = str(self.time_in_force or "GTC").upper()
        self.post_only = _coerce_bool(self.post_only, False)
        self.reduce_only_on_close = _coerce_bool(self.reduce_only_on_close, True)
        self.external_signals_enabled = _coerce_bool(self.external_signals_enabled, False)
        self.external_signal_max_adjustment = min(
            0.20,
            max(0.0, _finite_float(self.external_signal_max_adjustment, 0.04)),
        )
        self.external_signal_min_providers = max(0, _coerce_int(self.external_signal_min_providers, 2))
        self.external_signal_ttl_seconds = max(0, _coerce_int(self.external_signal_ttl_seconds, 300))
        self.external_signal_timeout_seconds = min(
            30.0,
            max(0.1, _finite_float(self.external_signal_timeout_seconds, 3.0)),
        )
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
    last_run: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))


def config_paths() -> Dict[str, Path]:
    """Return the default config directories used by the CLI."""

    base = Path.home() / ".config" / "simple_ai_bitcoin_trading_binance"
    return {
        "base": base,
        "runtime": base / "runtime.json",
        "strategy": base / "strategy.json",
    }
