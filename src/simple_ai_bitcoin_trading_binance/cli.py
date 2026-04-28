"""Entry point for the BTCUSDC test-trading CLI."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import sys
import time
from pathlib import Path
from typing import Iterable, Sequence

from .api import BinanceAPIError, BinanceClient
from .backtest import run_backtest
from .config import config_paths, load_runtime, load_strategy, prompt_runtime, save_runtime, save_strategy
from .dashboard import DashboardSnapshot, load_artifact_preview, render_dashboard
from .features import FEATURE_NAMES, feature_signature, make_rows, normalize_enabled_features
from .api import SymbolConstraints
from .market_data import clean_candles
from .model import (
    calibrate_threshold,
    confidence_adjusted_probability,
    evaluate_classification,
    evaluate,
    ModelFeatureMismatchError,
    ModelLoadError,
    load_model,
    model_decision_threshold,
    serialize_model,
    temporal_validation_split,
    train,
    walk_forward_report,
)
from .storage import write_json_atomic
from .types import RuntimeConfig, StrategyConfig


_TRAINING_PRESETS: dict[str, dict[str, object]] = {
    "custom": {},
    "quick": {
        "epochs": 80,
        "walk_forward": False,
        "calibrate_threshold": False,
    },
    "balanced": {
        "epochs": 180,
        "walk_forward": True,
        "walk_forward_train": 300,
        "walk_forward_test": 60,
        "walk_forward_step": 30,
        "calibrate_threshold": True,
    },
    "thorough": {
        "epochs": 350,
        "walk_forward": True,
        "walk_forward_train": 360,
        "walk_forward_test": 90,
        "walk_forward_step": 30,
        "calibrate_threshold": True,
    },
}


_STRATEGY_PROFILES: dict[str, dict[str, object]] = {
    "custom": {},
    "conservative": {
        "leverage": 1.0,
        "risk_per_trade": 0.005,
        "max_position_pct": 0.10,
        "stop_loss_pct": 0.015,
        "take_profit_pct": 0.025,
        "cooldown_minutes": 10,
        "max_open_positions": 1,
        "max_trades_per_day": 6,
        "signal_threshold": 0.64,
        "max_drawdown_limit": 0.12,
        "training_epochs": 180,
        "confidence_beta": 0.90,
    },
    "balanced": {
        "leverage": 2.0,
        "risk_per_trade": 0.01,
        "max_position_pct": 0.20,
        "stop_loss_pct": 0.02,
        "take_profit_pct": 0.03,
        "cooldown_minutes": 5,
        "max_open_positions": 1,
        "max_trades_per_day": 12,
        "signal_threshold": 0.58,
        "max_drawdown_limit": 0.20,
        "training_epochs": 250,
        "confidence_beta": 0.85,
    },
    "active": {
        "leverage": 3.0,
        "risk_per_trade": 0.015,
        "max_position_pct": 0.25,
        "stop_loss_pct": 0.025,
        "take_profit_pct": 0.04,
        "cooldown_minutes": 3,
        "max_open_positions": 1,
        "max_trades_per_day": 24,
        "signal_threshold": 0.55,
        "max_drawdown_limit": 0.25,
        "training_epochs": 300,
        "confidence_beta": 0.80,
    },
}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="simple-ai-trading",
        description="BTCUSDC test-trading CLI for Binance (spot + futures).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    parser_config = subparsers.add_parser("configure", help="configure Binance credentials and defaults")
    parser_config.set_defaults(func=command_configure)

    parser_connect = subparsers.add_parser("connect", help="validate credentials and connectivity")
    parser_connect.set_defaults(func=command_connect)

    parser_roundtrip = subparsers.add_parser(
        "spot-roundtrip",
        help="place a tiny signed spot-testnet roundtrip order with balance and filter prechecks",
    )
    parser_roundtrip.add_argument("--quantity", type=float, default=0.00008, help="BTC quantity to test")
    parser_roundtrip.add_argument(
        "--mode",
        choices=["auto", "buy-sell", "sell-buy"],
        default="auto",
        help="order sequence; auto uses buy-sell when USDC is available, otherwise sell-buy when BTC is available",
    )
    parser_roundtrip.add_argument("--yes", action="store_true", help="confirm signed testnet order placement")
    parser_roundtrip.set_defaults(func=command_spot_roundtrip)

    parser_doctor = subparsers.add_parser("doctor", help="run local readiness checks before paper or testnet trading")
    parser_doctor.add_argument("--input", default="data/historical_btcusdc.json")
    parser_doctor.add_argument("--model", default="data/model.json")
    parser_doctor.add_argument("--online", action="store_true", help="also check exchange connectivity")
    parser_doctor.set_defaults(func=command_doctor)

    parser_audit = subparsers.add_parser("audit", help="run local data/model/risk diagnostics without network calls")
    parser_audit.add_argument("--input", default="data/historical_btcusdc.json")
    parser_audit.add_argument("--model", default="data/model.json")
    parser_audit.set_defaults(func=command_audit)

    parser_report = subparsers.add_parser("report", help="show dashboard, artifacts, and optional readiness checks")
    parser_report.add_argument("--account", action="store_true", help="include authenticated account state")
    parser_report.add_argument("--doctor", action="store_true", help="include readiness checks")
    parser_report.add_argument("--no-doctor", action="store_false", dest="doctor", help="omit readiness checks")
    parser_report.add_argument("--online", action="store_true", help="include exchange connectivity in readiness checks")
    parser_report.add_argument("--input", default="data/historical_btcusdc.json")
    parser_report.add_argument("--model", default="data/model.json")
    parser_report.set_defaults(doctor=True)
    parser_report.set_defaults(func=command_report)

    parser_menu = subparsers.add_parser("menu", help="launch the interactive operator console")
    parser_menu.set_defaults(func=command_menu)

    parser_fetch = subparsers.add_parser("fetch", help="download BTCUSDC klines")
    parser_fetch.add_argument("--symbol", default=None)
    parser_fetch.add_argument("--interval", default=None)
    parser_fetch.add_argument("--limit", type=int, default=500)
    parser_fetch.add_argument("--batch-size", type=int, default=1000, help="klines per request (spot max 1000, futures max 1500)")
    parser_fetch.add_argument("--output", default="data/historical_btcusdc.json")
    parser_fetch.set_defaults(func=command_fetch)

    parser_train = subparsers.add_parser("train", help="train model from cached candles")
    parser_train.add_argument("--input", default="data/historical_btcusdc.json")
    parser_train.add_argument("--output", default="data/model.json")
    parser_train.add_argument("--preset", choices=sorted(_TRAINING_PRESETS), default="custom")
    parser_train.add_argument("--epochs", type=int, default=250)
    parser_train.add_argument("--learning-rate", type=float, default=0.05)
    parser_train.add_argument("--l2-penalty", type=float, default=1e-4)
    parser_train.add_argument("--seed", type=int, default=7)
    parser_train.add_argument("--walk-forward", action="store_true", help="run walk-forward validation before final training")
    parser_train.add_argument("--walk-forward-train", type=int, default=300)
    parser_train.add_argument("--walk-forward-test", type=int, default=60)
    parser_train.add_argument("--walk-forward-step", type=int, default=30)
    parser_train.add_argument("--calibrate-threshold", action="store_true", help="optimize a probability threshold on validation split")
    parser_train.set_defaults(func=command_train)

    parser_prepare = subparsers.add_parser("prepare", help="fetch, train, evaluate, backtest, then run readiness checks")
    parser_prepare.add_argument("--historical", default="data/historical_btcusdc.json")
    parser_prepare.add_argument("--model", default="data/model.json")
    parser_prepare.add_argument("--limit", type=int, default=500)
    parser_prepare.add_argument("--batch-size", type=int, default=1000, help="klines per fetch request (spot max 1000, futures max 1500)")
    parser_prepare.add_argument("--preset", choices=sorted(_TRAINING_PRESETS), default="balanced")
    parser_prepare.add_argument("--epochs", type=int, default=None, help="override preset training epochs")
    parser_prepare.add_argument("--learning-rate", type=float, default=0.05)
    parser_prepare.add_argument("--l2-penalty", type=float, default=1e-4)
    parser_prepare.add_argument("--seed", type=int, default=7)
    parser_prepare.add_argument("--start-cash", type=float, default=1000.0)
    parser_prepare.add_argument("--walk-forward", action="store_true", dest="walk_forward", help="force walk-forward validation")
    parser_prepare.add_argument("--no-walk-forward", action="store_false", dest="walk_forward", help="skip walk-forward validation")
    parser_prepare.add_argument("--walk-forward-train", type=int, default=None, help="override walk-forward training window")
    parser_prepare.add_argument("--walk-forward-test", type=int, default=None, help="override walk-forward test window")
    parser_prepare.add_argument("--walk-forward-step", type=int, default=None, help="override walk-forward step")
    parser_prepare.add_argument("--calibrate-threshold", action="store_true", dest="calibrate_threshold", help="force threshold calibration")
    parser_prepare.add_argument("--no-calibrate-threshold", action="store_false", dest="calibrate_threshold", help="skip threshold calibration")
    parser_prepare.set_defaults(walk_forward=None, calibrate_threshold=None)
    parser_prepare.add_argument("--online-doctor", action="store_true", help="include exchange connectivity in final readiness checks")
    parser_prepare.set_defaults(func=command_prepare)

    parser_tune = subparsers.add_parser("tune", help="perform a focused walk-forward tune over few risk parameters")
    parser_tune.add_argument("--input", default="data/historical_btcusdc.json")
    parser_tune.add_argument("--save-best", action="store_true")
    parser_tune.add_argument("--min-risk", type=float, default=0.002)
    parser_tune.add_argument("--max-risk", type=float, default=0.02)
    parser_tune.add_argument("--steps", type=int, default=5)
    parser_tune.add_argument("--min-leverage", type=float, default=1.0)
    parser_tune.add_argument("--max-leverage", type=float, default=20.0)
    parser_tune.add_argument("--min-threshold", type=float, default=0.52)
    parser_tune.add_argument("--max-threshold", type=float, default=0.88)
    parser_tune.add_argument("--min-take", type=float, default=0.01)
    parser_tune.add_argument("--max-take", type=float, default=0.06)
    parser_tune.add_argument("--min-stop", type=float, default=0.008)
    parser_tune.add_argument("--max-stop", type=float, default=0.04)
    parser_tune.add_argument("--lookback-days", type=int, default=None, help="use only the most recent N days of candles for tuning")
    parser_tune.add_argument("--from-date", default=None, help="inclusive start date for tuning window (YYYY-MM-DD)")
    parser_tune.add_argument("--to-date", default=None, help="inclusive end date for tuning window (YYYY-MM-DD)")
    parser_tune.set_defaults(func=command_tune)

    parser_backtest = subparsers.add_parser("backtest", help="run backtest against cached data")
    parser_backtest.add_argument("--input", default="data/historical_btcusdc.json")
    parser_backtest.add_argument("--model", default="data/model.json")
    parser_backtest.add_argument("--start-cash", type=float, default=1000.0)
    parser_backtest.set_defaults(func=command_backtest)

    parser_evaluate = subparsers.add_parser("evaluate", help="evaluate saved model against cached candles")
    parser_evaluate.add_argument("--input", default="data/historical_btcusdc.json")
    parser_evaluate.add_argument("--model", default="data/model.json")
    parser_evaluate.add_argument("--threshold", type=float, default=None)
    parser_evaluate.add_argument("--calibrate-threshold", action="store_true")
    parser_evaluate.set_defaults(func=command_evaluate)

    parser_live = subparsers.add_parser("live", help="run a conservative live loop on testnet or paper mode")
    parser_live.add_argument("--model", default="data/model.json")
    parser_live.add_argument("--steps", type=int, default=20)
    parser_live.add_argument("--sleep", type=int, default=5)
    parser_live.add_argument("--leverage", type=float, default=None, help="override leverage for this run (futures only)")
    parser_live.add_argument(
        "--retrain-interval",
        type=int,
        default=0,
        help="retrain model every N steps (0 disables, for adaptive paper/live behavior)",
    )
    parser_live.add_argument(
        "--retrain-window",
        type=int,
        default=300,
        help="number of recent rows used for each live retrain",
    )
    parser_live.add_argument(
        "--retrain-min-rows",
        type=int,
        default=240,
        help="minimum rows required before a retrain is attempted",
    )
    parser_live.add_argument(
        "--paper",
        action="store_true",
        help="force paper mode for this run even when runtime.dry_run is false",
    )
    parser_live.add_argument(
        "--live",
        action="store_true",
        help="force authenticated testnet execution even when runtime.dry_run is true",
    )
    parser_live.set_defaults(func=command_live)

    parser_status = subparsers.add_parser("status", help="show persisted runtime and strategy config")
    parser_status.set_defaults(func=command_status)

    parser_strategy = subparsers.add_parser("strategy", help="adjust strategy and risk parameters")
    parser_strategy.add_argument("--profile", choices=sorted(_STRATEGY_PROFILES), default="custom")
    parser_strategy.add_argument("--leverage", type=float, default=None)
    parser_strategy.add_argument("--risk", type=float, default=None)
    parser_strategy.add_argument("--max-position", type=float, default=None)
    parser_strategy.add_argument("--stop", type=float, default=None)
    parser_strategy.add_argument("--take", type=float, default=None)
    parser_strategy.add_argument("--cooldown", type=int, default=None)
    parser_strategy.add_argument("--max-open", type=int, default=None)
    parser_strategy.add_argument("--max-trades-per-day", type=int, default=None)
    parser_strategy.add_argument("--signal-threshold", type=float, default=None)
    parser_strategy.add_argument("--max-drawdown", type=float, default=None)
    parser_strategy.add_argument("--taker-fee-bps", type=float, default=None)
    parser_strategy.add_argument("--slippage-bps", type=float, default=None)
    parser_strategy.add_argument("--label-threshold", type=float, default=None)
    parser_strategy.add_argument("--model-lookback", type=int, default=None)
    parser_strategy.add_argument("--training-epochs", type=int, default=None)
    parser_strategy.add_argument("--confidence-beta", type=float, default=None)
    parser_strategy.add_argument("--feature-window-short", type=int, default=None)
    parser_strategy.add_argument("--feature-window-long", type=int, default=None)
    parser_strategy.add_argument("--set-features", default=None, help="comma-separated ordered feature list for retraining")
    parser_strategy.add_argument("--enable-feature", action="append", default=None, help="enable a feature by name")
    parser_strategy.add_argument("--disable-feature", action="append", default=None, help="disable a feature by name")
    parser_strategy.set_defaults(func=command_strategy)

    parser_shell = subparsers.add_parser("shell", help="launch the Claude-Code-inspired interactive shell")
    parser_shell.set_defaults(func=command_shell)

    parser_objectives = subparsers.add_parser("objectives", help="list registered training objectives")
    parser_objectives.set_defaults(func=command_objectives)

    parser_train_suite = subparsers.add_parser(
        "train-suite", help="train one advanced model per objective (Conservative/Default/Risky)",
    )
    parser_train_suite.add_argument("--input", default="data/historical_btcusdc.json")
    parser_train_suite.add_argument("--output-dir", default="data")
    parser_train_suite.add_argument("--starting-cash", type=float, default=1000.0)
    parser_train_suite.add_argument(
        "--objective", action="append", default=None,
        help="restrict suite to named objective(s); repeat to list multiple.",
    )
    parser_train_suite.set_defaults(func=command_train_suite)

    parser_backtest_panel = subparsers.add_parser(
        "backtest-panel", help="run a user-parameterized backtest and save a tagged report",
    )
    parser_backtest_panel.add_argument("--interval", required=True)
    parser_backtest_panel.add_argument("--market", default=None, help="override runtime market type")
    parser_backtest_panel.add_argument("--from-date", default=None)
    parser_backtest_panel.add_argument("--to-date", default=None)
    parser_backtest_panel.add_argument("--input", default="data/historical_btcusdc.json")
    parser_backtest_panel.add_argument("--model", default=None)
    parser_backtest_panel.add_argument("--objective", default=None)
    parser_backtest_panel.add_argument("--tag", default="")
    parser_backtest_panel.add_argument("--notes", default="")
    parser_backtest_panel.add_argument("--starting-cash", type=float, default=1000.0)
    parser_backtest_panel.set_defaults(func=command_backtest_panel)

    parser_autonomous = subparsers.add_parser(
        "autonomous", help="control the autonomous testnet loop (start/pause/resume/stop/status)",
    )
    parser_autonomous.add_argument(
        "action", choices=["start", "pause", "resume", "stop", "status"],
        help="autonomous action to perform",
    )
    parser_autonomous.add_argument("--objective", default="default")
    parser_autonomous.set_defaults(func=command_autonomous)

    parser_positions = subparsers.add_parser(
        "positions", help="list autonomous open positions and P&L stats",
    )
    parser_positions.add_argument("--stats", action="store_true", help="also print realized + unrealized stats")
    parser_positions.set_defaults(func=command_positions)

    parser_close = subparsers.add_parser(
        "close", help="close an autonomous position locally (ledger only, no exchange order)",
    )
    parser_close.add_argument("position_id", help="position id or 'all'")
    parser_close.set_defaults(func=command_close)

    return parser.parse_args(argv)


def _build_client(runtime):
    return BinanceClient(
        api_key=runtime.api_key,
        api_secret=runtime.api_secret,
        testnet=runtime.testnet,
        market_type=runtime.market_type,
        max_calls_per_minute=runtime.max_rate_calls_per_minute,
        recv_window_ms=getattr(runtime, "recv_window_ms", 5000),
    )


def _validate_runtime_connection(runtime, client) -> None:
    client.ping()
    client.ensure_btcusdc()
    if runtime.api_key and runtime.api_secret:
        client.get_account()


def _parse_form_bool(raw: str, default: bool) -> bool:
    token = raw.strip().lower()
    if token in {"y", "yes", "true", "1", "on"}:
        return True
    if token in {"n", "no", "false", "0", "off"}:
        return False
    return default


def _parse_optional_form_bool(raw: str) -> bool | None:
    token = raw.strip().lower()
    if not token:
        return None
    if token in {"y", "yes", "true", "1", "on"}:
        return True
    if token in {"n", "no", "false", "0", "off"}:
        return False
    raise ValueError(f"Expected yes/no/blank, got {raw!r}.")


def _parse_training_preset(raw: str) -> str:
    preset = (raw.strip().lower() or "custom")
    if preset not in _TRAINING_PRESETS:
        choices = "/".join(sorted(_TRAINING_PRESETS))
        raise ValueError(f"Preset must be one of: {choices}.")
    return preset


def _parse_strategy_profile(raw: str) -> str:
    profile = (raw.strip().lower() or "custom")
    if profile not in _STRATEGY_PROFILES:
        choices = "/".join(sorted(_STRATEGY_PROFILES))
        raise ValueError(f"Profile must be one of: {choices}.")
    return profile


def _apply_training_preset(args: argparse.Namespace) -> argparse.Namespace:
    preset = _parse_training_preset(str(getattr(args, "preset", "custom") or "custom"))
    for key, value in _TRAINING_PRESETS[preset].items():
        setattr(args, key, value)
    setattr(args, "preset", preset)
    return args


def _parse_form_int(raw: str, *, label: str, default: int, minimum: int | None = None, maximum: int | None = None) -> int:
    value = default if not raw.strip() else int(raw.strip())
    if minimum is not None and value < minimum:
        raise ValueError(f"{label} must be >= {minimum}.")
    if maximum is not None and value > maximum:
        raise ValueError(f"{label} must be <= {maximum}.")
    return value


def _parse_optional_form_int(raw: str, *, label: str, minimum: int | None = None, maximum: int | None = None) -> int | None:
    if not raw.strip():
        return None
    return _parse_form_int(raw, label=label, default=0, minimum=minimum, maximum=maximum)


def _parse_form_float(
    raw: str,
    *,
    label: str,
    default: float,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    value = default if not raw.strip() else float(raw.strip())
    if minimum is not None and value < minimum:
        raise ValueError(f"{label} must be >= {minimum}.")
    if maximum is not None and value > maximum:
        raise ValueError(f"{label} must be <= {maximum}.")
    return value


def _unchanged_form_value(payload: dict[str, str], key: str, current: object) -> bool:
    return payload.get(key, "").strip() == str(current)


def _profile_field_value(profile: str, payload: dict[str, str], key: str, current: object, parser) -> object | None:
    if profile != "custom" and _unchanged_form_value(payload, key, current):
        return None
    return parser(payload[key])


async def _ui_edit_runtime(ui, current) -> object:
    from .tui import FormField

    payload = await ui.form(
        "Runtime settings",
        [
            FormField("market_type", "Market type [spot/futures]", current.market_type),
            FormField("interval", "Kline interval", current.interval),
            FormField("testnet", "Use Binance testnet [yes/no]", "yes" if current.testnet else "no"),
            FormField("api_key", "Binance API key [blank keeps current]", "", password=True),
            FormField("api_secret", "Binance API secret [blank keeps current]", "", password=True),
            FormField("dry_run", "Paper trading mode [yes/no]", "yes" if current.dry_run else "no"),
            FormField("validate_account", "Validate credentials at startup [yes/no]", "yes" if current.validate_account else "no"),
            FormField("max_rate_calls_per_minute", "Max REST calls per minute", str(current.max_rate_calls_per_minute)),
            FormField("recv_window_ms", "Request recvWindow (ms, 1-60000)", str(getattr(current, "recv_window_ms", 5000))),
        ],
    )
    if payload is None:
        return current
    market_type = payload["market_type"].strip().lower()
    if market_type not in {"spot", "futures"}:
        market_type = current.market_type
    interval = payload["interval"].strip() or current.interval
    testnet = _parse_form_bool(payload["testnet"], current.testnet)
    api_key = payload["api_key"].strip() or current.api_key
    api_secret = payload["api_secret"].strip() or current.api_secret
    dry_run = _parse_form_bool(payload["dry_run"], current.dry_run)
    validate_account = _parse_form_bool(payload["validate_account"], current.validate_account)
    max_rate = _parse_form_int(
        payload["max_rate_calls_per_minute"],
        label="Max REST calls per minute",
        default=current.max_rate_calls_per_minute,
        minimum=1,
    )
    recv_window_ms = _parse_form_int(
        payload.get("recv_window_ms", "5000"),
        label="Request recvWindow",
        default=getattr(current, "recv_window_ms", 5000),
        minimum=1,
        maximum=60000,
    )
    return RuntimeConfig(
        symbol="BTCUSDC",
        interval=interval,
        market_type=market_type,
        testnet=testnet,
        api_key=api_key,
        api_secret=api_secret,
        dry_run=dry_run,
        validate_account=validate_account,
        max_rate_calls_per_minute=max_rate,
        recv_window_ms=recv_window_ms,
        compute_backend=getattr(current, "compute_backend", "cpu"),
        managed_usdc=getattr(current, "managed_usdc", 1000.0),
        managed_btc=getattr(current, "managed_btc", 0.0),
    )


async def _ui_edit_strategy_args(ui, cfg: StrategyConfig) -> argparse.Namespace:
    from .tui import FormField

    selected_features = await ui.multi_select(
        "Model feature selection",
        list(FEATURE_NAMES),
        list(cfg.enabled_features),
        help_text="Space toggles a feature. Save commits the selection used during retraining.",
    )
    if selected_features is None:
        return argparse.Namespace(
            profile="custom",
            leverage=None,
            risk=None,
            max_position=None,
            stop=None,
            take=None,
            cooldown=None,
            max_open=None,
            max_trades_per_day=None,
            signal_threshold=None,
            max_drawdown=None,
            taker_fee_bps=None,
            slippage_bps=None,
            label_threshold=None,
            model_lookback=None,
            training_epochs=None,
            confidence_beta=None,
            feature_window_short=None,
            feature_window_long=None,
            set_features=None,
            enable_feature=None,
            disable_feature=None,
        )
    enabled_features = normalize_enabled_features(selected_features)
    payload = await ui.form(
        "Strategy settings",
        [
            FormField("profile", "Risk profile [custom/conservative/balanced/active]", "custom"),
            FormField("leverage", "Leverage", str(cfg.leverage)),
            FormField("risk", "Risk per trade", str(cfg.risk_per_trade)),
            FormField("max_position", "Max position percent", str(cfg.max_position_pct)),
            FormField("stop", "Stop loss percent", str(cfg.stop_loss_pct)),
            FormField("take", "Take profit percent", str(cfg.take_profit_pct)),
            FormField("cooldown", "Cooldown minutes", str(cfg.cooldown_minutes)),
            FormField("max_open", "Max open positions", str(cfg.max_open_positions)),
            FormField("max_trades_per_day", "Max trades per day", str(cfg.max_trades_per_day)),
            FormField("signal_threshold", "Signal threshold", str(cfg.signal_threshold)),
            FormField("max_drawdown", "Max drawdown limit", str(cfg.max_drawdown_limit)),
            FormField("taker_fee_bps", "Taker fee bps", str(cfg.taker_fee_bps)),
            FormField("slippage_bps", "Slippage bps", str(cfg.slippage_bps)),
            FormField("label_threshold", "Label threshold", str(cfg.label_threshold)),
            FormField("model_lookback", "Model lookback rows", str(cfg.model_lookback)),
            FormField("training_epochs", "Training epochs", str(cfg.training_epochs)),
            FormField("confidence_beta", "Confidence beta", str(cfg.confidence_beta)),
            FormField("feature_window_short", "Feature window short", str(cfg.feature_windows[0])),
            FormField("feature_window_long", "Feature window long", str(cfg.feature_windows[1])),
        ],
    )
    if payload is None:
        return argparse.Namespace(
            profile="custom",
            leverage=None,
            risk=None,
            max_position=None,
            stop=None,
            take=None,
            cooldown=None,
            max_open=None,
            max_trades_per_day=None,
            signal_threshold=None,
            max_drawdown=None,
            taker_fee_bps=None,
            slippage_bps=None,
            label_threshold=None,
            model_lookback=None,
            training_epochs=None,
            confidence_beta=None,
            feature_window_short=None,
            feature_window_long=None,
            set_features=None,
            enable_feature=None,
            disable_feature=None,
        )
    profile = _parse_strategy_profile(payload["profile"])

    def field_float(key: str, current: float, label: str, *, minimum=None, maximum=None):
        return _profile_field_value(
            profile,
            payload,
            key,
            current,
            lambda raw: _parse_form_float(raw, label=label, default=current, minimum=minimum, maximum=maximum),
        )

    def field_int(key: str, current: int, label: str, *, minimum=None, maximum=None):
        return _profile_field_value(
            profile,
            payload,
            key,
            current,
            lambda raw: _parse_form_int(raw, label=label, default=current, minimum=minimum, maximum=maximum),
        )

    feature_window_short = field_int("feature_window_short", cfg.feature_windows[0], "Feature window short", minimum=1)
    feature_window_floor = int(feature_window_short or cfg.feature_windows[0])
    feature_window_long = _profile_field_value(
        profile,
        payload,
        "feature_window_long",
        cfg.feature_windows[1],
        lambda raw: _parse_form_int(
            raw,
            label="Feature window long",
            default=max(cfg.feature_windows[1], feature_window_floor + 1),
            minimum=feature_window_floor + 1,
        ),
    )
    return argparse.Namespace(
        profile=profile,
        leverage=field_float("leverage", cfg.leverage, "Leverage", minimum=1.0),
        risk=field_float("risk", cfg.risk_per_trade, "Risk per trade", minimum=0.0001),
        max_position=field_float("max_position", cfg.max_position_pct, "Max position percent", minimum=0.0, maximum=1.0),
        stop=field_float("stop", cfg.stop_loss_pct, "Stop loss percent", minimum=0.0, maximum=0.99),
        take=field_float("take", cfg.take_profit_pct, "Take profit percent", minimum=0.0, maximum=0.99),
        cooldown=field_int("cooldown", cfg.cooldown_minutes, "Cooldown minutes", minimum=0),
        max_open=field_int("max_open", cfg.max_open_positions, "Max open positions", minimum=0),
        max_trades_per_day=field_int("max_trades_per_day", cfg.max_trades_per_day, "Max trades per day", minimum=0),
        signal_threshold=field_float("signal_threshold", cfg.signal_threshold, "Signal threshold", minimum=0.01, maximum=0.99),
        max_drawdown=field_float("max_drawdown", cfg.max_drawdown_limit, "Max drawdown limit", minimum=0.0, maximum=1.0),
        taker_fee_bps=field_float("taker_fee_bps", cfg.taker_fee_bps, "Taker fee bps", minimum=0.0),
        slippage_bps=field_float("slippage_bps", cfg.slippage_bps, "Slippage bps", minimum=0.0),
        label_threshold=field_float("label_threshold", cfg.label_threshold, "Label threshold", minimum=0.0001),
        model_lookback=field_int("model_lookback", cfg.model_lookback, "Model lookback rows", minimum=10),
        training_epochs=field_int("training_epochs", cfg.training_epochs, "Training epochs", minimum=1),
        confidence_beta=field_float("confidence_beta", cfg.confidence_beta, "Confidence beta", minimum=0.0, maximum=1.0),
        feature_window_short=feature_window_short,
        feature_window_long=feature_window_long,
        set_features=",".join(enabled_features),
        enable_feature=None,
        disable_feature=None,
    )


def _recent_artifacts(*, base_dir: Path = Path("data"), limit: int = 8) -> list[Path]:
    if not base_dir.exists():
        return []
    paths = [path for path in base_dir.glob("*.json") if path.is_file()]
    paths.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return paths[:limit]


def _artifact_summary(path: Path) -> str:
    return load_artifact_preview(path)


def _show_recent_artifacts() -> int:
    artifacts = _recent_artifacts()
    if not artifacts:
        print("No recent artifacts under data/.")
        return 0
    print("Recent artifacts:")
    for path in artifacts:
        print(f"- {_artifact_summary(path)}")
    return 0


def _render_operator_report(
    *,
    with_account: bool,
    doctor: bool,
    online: bool,
    input_path: str,
    model_path: str,
    width: int = 90,
) -> str:
    sections = [render_dashboard(_dashboard_snapshot(with_account=with_account), width=width)]
    if doctor:
        ok, lines = _readiness_report(input_path=input_path, model_path=model_path, online=online)
        status = "ok" if ok else "fix"
        sections.append("\n".join([f"Readiness report ({status})", *lines]))
    return "\n\n".join(sections)


def _account_overview_lines(runtime) -> list[str]:
    if not runtime.api_key or not runtime.api_secret:
        return ["No API credentials configured."]
    client = _build_client(runtime)
    try:
        account = client.get_account()
    except BinanceAPIError as exc:
        return [f"Account overview failed: {exc}"]
    balances = account.get("balances", []) if isinstance(account, dict) else []
    assets = account.get("assets", []) if isinstance(account, dict) else []
    positions = account.get("positions", []) if isinstance(account, dict) else []
    interesting = []
    for item in balances:
        if not isinstance(item, dict):
            continue
        asset = str(item.get("asset", ""))
        free = str(item.get("free", "0"))
        locked = str(item.get("locked", "0"))
        if asset in {"BTC", "USDC"} or free not in {"0", "0.0", "0.00000000"} or locked not in {"0", "0.0", "0.00000000"}:
            interesting.append(f"{asset}: free={free} locked={locked}")
    for item in assets:
        if not isinstance(item, dict):
            continue
        asset = str(item.get("asset", ""))
        wallet = str(item.get("walletBalance", item.get("availableBalance", "0")))
        available = str(item.get("availableBalance", "0"))
        unrealized = str(item.get("unrealizedProfit", "0"))
        if asset in {"BTC", "USDC", "USDT"} or wallet not in {"0", "0.0", "0.00000000"} or unrealized not in {"0", "0.0", "0.00000000"}:
            interesting.append(f"{asset}: wallet={wallet} available={available} unrealized={unrealized}")
    for item in positions:
        if not isinstance(item, dict):
            continue
        symbol = str(item.get("symbol", ""))
        amount = str(item.get("positionAmt", item.get("positionAmount", "0")))
        entry = str(item.get("entryPrice", "0"))
        unrealized = str(item.get("unrealizedProfit", "0"))
        if amount not in {"0", "0.0", "0.00000000"} or unrealized not in {"0", "0.0", "0.00000000"}:
            interesting.append(f"{symbol}: position={amount} entry={entry} unrealized={unrealized}")
    if not interesting:
        return [f"market={runtime.market_type} testnet={runtime.testnet}", "No non-zero balances found."]
    return [f"market={runtime.market_type} testnet={runtime.testnet}", *interesting[:20]]


def _show_account_overview() -> int:
    runtime = load_runtime()
    lines = _account_overview_lines(runtime)
    print("Account overview")
    for line in lines:
        print(f"- {line}" if ":" in line and not line.startswith("market=") else line)
    if lines == ["No API credentials configured."]:
        return 2
    if lines and lines[0].startswith("Account overview failed:"):
        return 2
    return 0


def _dashboard_snapshot(*, with_account: bool) -> DashboardSnapshot:
    runtime = load_runtime()
    strategy = load_strategy()
    notes = [
        "Operate the system from the interactive console actions and modal forms.",
        "Use authenticated execution only on testnet and only after checking runtime state.",
    ]
    return DashboardSnapshot(
        runtime=runtime.public_dict(),
        strategy=strategy.asdict(),
        artifacts=[_artifact_summary(path) for path in _recent_artifacts()],
        account_lines=_account_overview_lines(runtime) if with_account else ["Load Account or Connect to fetch balances."],
        notes=notes,
    )


def _connection_status_line() -> str:
    runtime = load_runtime()
    market = f"{runtime.market_type}/{'testnet' if runtime.testnet else 'mainnet'}"
    mode = "paper-default" if runtime.dry_run else "testnet-live-default"
    checked_at = datetime.now(timezone.utc).strftime("%H:%M:%SZ")
    client = _build_client(runtime)
    try:
        client.ping()
        client.get_exchange_time()
        auth_label = "auth missing"
        if runtime.api_key and runtime.api_secret:
            account = client.get_account()
            auth_label = "auth ok" if isinstance(account, dict) else "auth response ok"
        return f"Connection {checked_at}: online {market} {mode} server-time ok {auth_label}"
    except BinanceAPIError as exc:
        return f"Connection {checked_at}: offline {market} ({exc})"


def _readiness_report(*, input_path: str, model_path: str, online: bool = False) -> tuple[bool, list[str]]:
    runtime = load_runtime()
    strategy = load_strategy()
    checks: list[tuple[bool, str, str]] = []

    def add(ok: bool, label: str, detail: str) -> None:
        checks.append((ok, label, detail))

    mode = "paper" if runtime.dry_run else "authenticated testnet"
    add(runtime.testnet, "safety target", "testnet enabled" if runtime.testnet else "testnet is disabled")
    add(runtime.market_type in {"spot", "futures"}, "market type", runtime.market_type)
    add(
        runtime.dry_run or bool(runtime.api_key and runtime.api_secret),
        "execution credentials",
        f"{mode}; credentials {'present' if runtime.api_key and runtime.api_secret else 'not configured'}",
    )
    add(0.01 <= strategy.signal_threshold <= 0.99, "signal threshold", f"{strategy.signal_threshold:.3f}")
    add(strategy.risk_per_trade > 0 and strategy.max_position_pct > 0, "risk sizing", f"risk={strategy.risk_per_trade} max_position={strategy.max_position_pct}")
    add(bool(strategy.enabled_features), "feature set", ",".join(strategy.enabled_features))

    data_file = Path(input_path)
    if data_file.exists():
        candles = _load_rows_for_command(str(data_file), label="Readiness data load failed")
        candle_count = len(candles) if candles is not None else 0
        add(candle_count > max(strategy.feature_windows), "training data", f"{candle_count} candles at {data_file}")
    else:
        add(False, "training data", f"missing {data_file}")

    model_file = Path(model_path)
    if model_file.exists():
        try:
            model = _load_runtime_model(model_file, strategy)
        except (OSError, ModelFeatureMismatchError, ModelLoadError, ValueError) as exc:
            add(False, "model artifact", f"{model_file} is not usable with current strategy ({exc})")
        else:
            add(True, "model artifact", f"{model_file} dim={model.feature_dim}")
    else:
        add(False, "model artifact", f"missing {model_file}")

    if online:
        line = _connection_status_line()
        add("online" in line and "offline" not in line and "failed" not in line, "exchange connectivity", line)

    lines = [f"[{'ok' if ok else 'fix'}] {label}: {detail}" for ok, label, detail in checks]
    return all(ok for ok, _label, _detail in checks), lines


_COMPUTE_BACKEND_CHOICES = ("cpu", "cuda", "rocm", "auto")


def _funds_summary(runtime) -> str:
    return (
        f"USDC available: {runtime.managed_usdc:.4f}  "
        f"BTC available: {runtime.managed_btc:.8f}"
    )


def _apply_funds_change(action: str, amount: float) -> tuple[object, str]:
    """Apply a Funds-menu mutation. Returns (new runtime, log message)."""
    runtime = load_runtime()
    if action == "deposit_usdc":
        runtime.managed_usdc = max(0.0, float(runtime.managed_usdc) + abs(amount))
        msg = f"Deposited {amount:.4f} USDC."
    elif action == "withdraw_usdc":
        avail = float(runtime.managed_usdc)
        take = min(avail, abs(amount))
        runtime.managed_usdc = max(0.0, avail - take)
        msg = f"Withdrew {take:.4f} USDC (capped to available {avail:.4f})."
    elif action == "deposit_btc":
        runtime.managed_btc = max(0.0, float(runtime.managed_btc) + abs(amount))
        msg = f"Deposited {amount:.8f} BTC."
    elif action == "withdraw_btc":
        avail = float(runtime.managed_btc)
        take = min(avail, abs(amount))
        runtime.managed_btc = max(0.0, avail - take)
        msg = f"Withdrew {take:.8f} BTC (capped to available {avail:.8f})."
    elif action == "reset":
        runtime.managed_usdc = 1000.0
        runtime.managed_btc = 0.0
        msg = "Reset funds to 1000.000 USDC and 0.000 BTC."
    else:  # pragma: no cover - guarded by menu options
        return runtime, f"Unknown funds action {action!r}."
    save_runtime(runtime)
    return runtime, msg


async def _ui_funds_menu(ui) -> int:
    from .tui import FormField

    while True:
        runtime = load_runtime()
        choice = await ui.menu(
            "Funds — virtual BTCUSDC allocation",
            [
                ("deposit_usdc", "Deposit USDC"),
                ("withdraw_usdc", "Withdraw USDC"),
                ("deposit_btc", "Deposit BTC"),
                ("withdraw_btc", "Withdraw BTC"),
                ("reset", "Reset to 1000 USDC / 0 BTC"),
                ("show", "Show current allocation"),
                ("close", "Close"),
            ],
            help_text=(
                f"Spot BTCUSDC only. {_funds_summary(runtime)}. "
                "Trading consumes from this allocation; the testnet wallet is a hard ceiling."
            ),
        )
        if choice in (None, "close"):
            return 0
        if choice == "show":
            ui.append_log(_funds_summary(runtime))
            continue
        if choice == "reset":
            confirmed = await ui.confirm("Reset allocation to 1000.0 USDC and 0.0 BTC?")
            if not confirmed:
                continue
            _, msg = _apply_funds_change("reset", 0.0)
            ui.append_log(msg)
            continue
        is_btc = choice.endswith("btc")
        unit = "BTC" if is_btc else "USDC"
        default_amount = "0.0001" if is_btc else "100"
        payload = await ui.form(
            f"{choice.replace('_', ' ').title()}",
            [
                FormField("amount", f"Amount in {unit}", default_amount),
            ],
        )
        if payload is None:
            continue
        try:
            amount = _parse_form_float(
                payload["amount"],
                label=f"Amount ({unit})",
                default=0.0,
                minimum=0.0,
            )
        except ValueError as exc:
            ui.append_log(f"Funds change rejected: {exc}")
            continue
        if amount <= 0:
            ui.append_log("Amount must be > 0.")
            continue
        _, msg = _apply_funds_change(choice, amount)
        ui.append_log(msg)


async def _ui_edit_execution(ui) -> int:
    from .tui import FormField

    cfg = load_strategy()
    payload = await ui.form(
        "Execution settings",
        [
            FormField(
                "order_type",
                "Order type [MARKET / LIMIT / LIMIT_MAKER]",
                str(getattr(cfg, "order_type", "MARKET")),
            ),
            FormField(
                "time_in_force",
                "Time in force [GTC / IOC / FOK]",
                str(getattr(cfg, "time_in_force", "GTC")),
            ),
            FormField(
                "post_only",
                "Post-only (LIMIT_MAKER) [yes/no]",
                "yes" if getattr(cfg, "post_only", False) else "no",
            ),
            FormField(
                "reduce_only_on_close",
                "Reduce-only when closing [yes/no]",
                "yes" if getattr(cfg, "reduce_only_on_close", True) else "no",
            ),
        ],
    )
    if payload is None:
        ui.append_log("Execution settings cancelled.")
        return 0
    order_type = payload["order_type"].strip().upper() or "MARKET"
    if order_type not in {"MARKET", "LIMIT", "LIMIT_MAKER"}:
        ui.append_log(f"Unsupported order type {order_type!r}; keeping {cfg.order_type!r}.")
        order_type = cfg.order_type
    tif = payload["time_in_force"].strip().upper() or "GTC"
    if tif not in {"GTC", "IOC", "FOK"}:
        ui.append_log(f"Unsupported timeInForce {tif!r}; keeping {cfg.time_in_force!r}.")
        tif = cfg.time_in_force
    post_only = _parse_form_bool(payload["post_only"], getattr(cfg, "post_only", False))
    reduce_only = _parse_form_bool(
        payload["reduce_only_on_close"],
        getattr(cfg, "reduce_only_on_close", True),
    )
    cfg.order_type = order_type
    cfg.time_in_force = tif
    cfg.post_only = post_only
    cfg.reduce_only_on_close = reduce_only
    save_strategy(cfg)
    ui.append_log(
        f"Saved execution: order_type={order_type} tif={tif} "
        f"post_only={post_only} reduce_only_on_close={reduce_only}"
    )
    return 0


async def _ui_edit_compute(ui) -> int:
    from .compute import describe_backend, resolve_backend
    from .tui import FormField

    runtime = load_runtime()
    current = getattr(runtime, "compute_backend", "cpu")
    payload = await ui.form(
        "Compute backend",
        [
            FormField(
                "backend",
                "Backend [cpu / cuda / rocm / auto]",
                current,
            ),
        ],
    )
    if payload is None:
        ui.append_log("Compute backend selection cancelled.")
        return 0
    requested = payload["backend"].strip().lower() or "cpu"
    if requested not in _COMPUTE_BACKEND_CHOICES:
        ui.append_log(f"Unknown backend {requested!r}; keeping {current!r}.")
        return 2
    info = await ui.run_blocking(resolve_backend, requested)
    runtime.compute_backend = requested
    save_runtime(runtime)
    ui.append_log(
        f"Saved compute_backend={requested}. Runtime status: {describe_backend(info)}"
    )
    return 0


async def _ui_settings_menu(ui) -> int:
    while True:
        choice = await ui.menu(
            "Settings",
            [
                ("runtime", "Runtime — credentials and connection"),
                ("strategy", "Strategy — risk, thresholds, features"),
                ("execution", "Execution — order type and routing"),
                ("compute", "Compute backend — CPU / GPU / auto"),
                ("close", "Close"),
            ],
            help_text="Centralized configuration. Up/Down to choose, Enter to open, Escape to close.",
        )
        if choice in (None, "close"):
            return 0
        if choice == "runtime":
            current = load_runtime()
            try:
                next_runtime = await _ui_edit_runtime(ui, current)
            except ValueError as exc:
                ui.append_log(f"Runtime settings invalid: {exc}")
                continue
            save_runtime(next_runtime)
            ui.append_log("Runtime settings saved.")
            continue
        if choice == "strategy":
            try:
                args = await _ui_edit_strategy_args(ui, load_strategy())
            except ValueError as exc:
                ui.append_log(f"Strategy settings invalid: {exc}")
                continue
            if args.set_features is None and args.leverage is None and getattr(args, "profile", "custom") == "custom":
                ui.append_log("Strategy update cancelled.")
                continue
            await ui.run_blocking(command_strategy, args)
            continue
        if choice == "execution":
            await _ui_edit_execution(ui)
            continue
        if choice == "compute":
            await _ui_edit_compute(ui)
            continue


def _tui_actions():
    from .tui import FormField, TUIAction
    async def _overview(ui):
        ui.append_log(await ui.run_blocking(lambda: render_dashboard(_dashboard_snapshot(with_account=True))))
        return 0

    async def _help(ui):
        ui.append_log(
            "\n".join(
                [
                    "Operator help — simple-ai-trading",
                    "==================================",
                    "",
                    "Scope: BTCUSDC spot trading on the Binance testnet only.",
                    "",
                    "First-time setup",
                    "----------------",
                    "  1. Settings -> Runtime — paste your Binance testnet API key and secret.",
                    "  2. Connect — pings the exchange and validates credentials.",
                    "  3. Funds — set how much virtual USDC and BTC the strategy is allowed to use.",
                    "  4. Readiness check — confirms safety flags, data, model, and connectivity.",
                    "",
                    "End-to-end paper run",
                    "--------------------",
                    "  1. Prepare system — fetches candles, trains, evaluates, backtests.",
                    "  2. Paper loop — runs the strategy without placing real orders.",
                    "  3. Operator report — prints dashboard, artifacts, and readiness summary.",
                    "",
                    "Manual pipeline (full control)",
                    "------------------------------",
                    "  Fetch candles -> Settings -> Strategy -> Train model -> Evaluate -> Backtest -> Tune strategy",
                    "",
                    "Authenticated testnet execution",
                    "-------------------------------",
                    "  * Always run Readiness check first.",
                    "  * Spot roundtrip is the smallest signed test (BUY then SELL).",
                    "  * Testnet loop runs the strategy with signed orders against the testnet.",
                    "  * The strategy is capped by the Funds allocation, never the raw testnet wallet.",
                    "",
                    "Keyboard",
                    "--------",
                    "  Tab / Shift-Tab     change active panel (the active panel has a green border)",
                    "  Up / Down           move within the active panel",
                    "  Enter               run the selected command",
                    "  r                   refresh the snapshot panel",
                    "  <  >                shrink / grow the command list",
                    "  -  +                shrink / grow the activity log",
                    "  Ctrl-L              clear the activity log",
                    "  q                   quit",
                    "  Inside any modal: Tab cycles fields, Enter saves, Escape cancels.",
                    "",
                    "Centralized configuration",
                    "-------------------------",
                    "  Settings opens a hub with: Runtime, Strategy, Execution, Compute backend.",
                    "  Funds is independent of trading — deposit / withdraw virtual USDC / BTC there.",
                    "  Compute backend selects CPU (default), CUDA, ROCm, or auto-detect.",
                    "",
                    "Safety",
                    "------",
                    "  testnet=true is the default and required for signed live execution in this build.",
                    "  Paper mode never places real orders, even when credentials are present.",
                    "  Credentials are stored at ~/.config/simple_ai_bitcoin_trading_binance/runtime.json (mode 600).",
                ]
            )
        )
        return 0

    async def _runtime(ui):
        current = load_runtime()
        try:
            next_runtime = await _ui_edit_runtime(ui, current)
        except ValueError as exc:
            print(f"Runtime settings invalid: {exc}", file=sys.stderr)
            return 2
        save_runtime(next_runtime)
        if next_runtime.validate_account and next_runtime.api_key and next_runtime.api_secret:
            client = _build_client(next_runtime)
            try:
                await ui.run_blocking(_validate_runtime_connection, next_runtime, client)
            except BinanceAPIError as exc:
                print(f"Configuration saved, but validation failed: {exc}", file=sys.stderr)
                return 2
        print("Runtime config saved to", config_paths()["runtime"])
        print(f"market={next_runtime.market_type} testnet={next_runtime.testnet} paper={next_runtime.dry_run}")
        return 0

    async def _strategy(ui):
        try:
            args = await _ui_edit_strategy_args(ui, load_strategy())
        except ValueError as exc:
            print(f"Strategy settings invalid: {exc}", file=sys.stderr)
            return 2
        if args.set_features is None and args.leverage is None and getattr(args, "profile", "custom") == "custom":
            print("Strategy update cancelled.")
            return 0
        return await ui.run_blocking(command_strategy, args)

    async def _connect(_ui):
        return await _ui.run_blocking(command_connect, argparse.Namespace())

    async def _doctor(ui):
        payload = await ui.form(
            "Readiness check",
            [
                FormField("input", "Training input path", "data/historical_btcusdc.json"),
                FormField("model", "Model path", "data/model.json"),
                FormField("online", "Include exchange connectivity [yes/no]", "yes"),
            ],
        )
        if payload is None:
            print("Readiness check cancelled.")
            return 0
        return await ui.run_blocking(
            command_doctor,
            argparse.Namespace(
                input=payload["input"].strip() or "data/historical_btcusdc.json",
                model=payload["model"].strip() or "data/model.json",
                online=_parse_form_bool(payload["online"], True),
            ),
        )

    async def _account(ui):
        return await ui.run_blocking(_show_account_overview)

    async def _audit(ui):
        payload = await ui.form(
            "Local audit",
            [
                FormField("input", "Training input path", "data/historical_btcusdc.json"),
                FormField("model", "Model path", "data/model.json"),
            ],
        )
        if payload is None:
            print("Audit cancelled.")
            return 0
        return await ui.run_blocking(
            command_audit,
            argparse.Namespace(
                input=payload["input"].strip() or "data/historical_btcusdc.json",
                model=payload["model"].strip() or "data/model.json",
            ),
        )

    async def _fetch(ui):
        runtime = load_runtime()
        max_batch_size = 1500 if runtime.market_type == "futures" else 1000
        payload = await ui.form(
            "Fetch candles",
            [
                FormField("limit", "Fetch limit", "500"),
                FormField("batch_size", f"Klines per request [max {max_batch_size}]", "1000"),
                FormField("output", "Candle output path", "data/historical_btcusdc.json"),
            ],
        )
        if payload is None:
            print("Fetch cancelled.")
            return 0
        try:
            limit = _parse_form_int(payload["limit"], label="Fetch limit", default=500, minimum=1)
            batch_size = _parse_form_int(payload.get("batch_size", "1000"), label="Klines per request", default=1000, minimum=1, maximum=max_batch_size)
        except ValueError as exc:
            print(f"Fetch settings invalid: {exc}", file=sys.stderr)
            return 2
        return await ui.run_blocking(
            command_fetch,
            argparse.Namespace(
                symbol=runtime.symbol,
                interval=runtime.interval,
                limit=limit,
                batch_size=batch_size,
                output=payload["output"].strip() or "data/historical_btcusdc.json",
            ),
        )

    async def _train(ui):
        payload = await ui.form(
            "Train model",
            [
                FormField("input", "Training input path", "data/historical_btcusdc.json"),
                FormField("output", "Model output path", "data/model.json"),
                FormField("preset", "Preset [custom/quick/balanced/thorough]", "custom"),
                FormField("epochs", "Training epochs", "250"),
                FormField("learning_rate", "Learning rate", "0.05"),
                FormField("l2_penalty", "L2 penalty", "0.0001"),
                FormField("seed", "Training seed", "7"),
                FormField("walk_forward", "Run walk-forward validation [yes/no]", "no"),
                FormField("walk_forward_train", "Walk-forward train window", "300"),
                FormField("walk_forward_test", "Walk-forward test window", "60"),
                FormField("walk_forward_step", "Walk-forward step", "30"),
                FormField("calibrate_threshold", "Calibrate threshold [yes/no]", "yes"),
            ],
        )
        if payload is None:
            print("Training cancelled.")
            return 0
        try:
            preset = _parse_training_preset(payload["preset"])
            epochs = _parse_form_int(payload["epochs"], label="Training epochs", default=250, minimum=1)
            learning_rate = _parse_form_float(payload.get("learning_rate", "0.05"), label="Learning rate", default=0.05, minimum=0.000001)
            l2_penalty = _parse_form_float(payload.get("l2_penalty", "0.0001"), label="L2 penalty", default=0.0001, minimum=0.0)
            seed = _parse_form_int(payload["seed"], label="Training seed", default=7, minimum=0)
            walk_forward_train = _parse_form_int(payload["walk_forward_train"], label="Walk-forward train window", default=300, minimum=2)
            walk_forward_test = _parse_form_int(payload["walk_forward_test"], label="Walk-forward test window", default=60, minimum=1)
            walk_forward_step = _parse_form_int(payload["walk_forward_step"], label="Walk-forward step", default=30, minimum=1)
        except ValueError as exc:
            print(f"Training settings invalid: {exc}", file=sys.stderr)
            return 2
        return await ui.run_blocking(
            command_train,
            argparse.Namespace(
                input=payload["input"].strip() or "data/historical_btcusdc.json",
                output=payload["output"].strip() or "data/model.json",
                preset=preset,
                epochs=epochs,
                learning_rate=learning_rate,
                l2_penalty=l2_penalty,
                seed=seed,
                walk_forward=_parse_form_bool(payload["walk_forward"], False),
                walk_forward_train=walk_forward_train,
                walk_forward_test=walk_forward_test,
                walk_forward_step=walk_forward_step,
                calibrate_threshold=_parse_form_bool(payload["calibrate_threshold"], True),
            ),
        )

    async def _tune(ui):
        payload = await ui.form(
            "Tune strategy",
            [
                FormField("input", "Tune input path", "data/historical_btcusdc.json"),
                FormField("window_mode", "Window mode [all/lookback/range]", "all"),
                FormField("lookback_days", "Lookback days", "30"),
                FormField("from_date", "From date YYYY-MM-DD", ""),
                FormField("to_date", "To date YYYY-MM-DD", ""),
                FormField("save_best", "Persist the best strategy [yes/no]", "no"),
                FormField("min_risk", "Minimum risk", "0.002"),
                FormField("max_risk", "Maximum risk", "0.02"),
                FormField("steps", "Grid steps", "5"),
                FormField("min_leverage", "Minimum leverage", "1.0"),
                FormField("max_leverage", "Maximum leverage", "20.0"),
                FormField("min_threshold", "Minimum threshold", "0.52"),
                FormField("max_threshold", "Maximum threshold", "0.88"),
                FormField("min_take", "Minimum take profit", "0.01"),
                FormField("max_take", "Maximum take profit", "0.06"),
                FormField("min_stop", "Minimum stop loss", "0.008"),
                FormField("max_stop", "Maximum stop loss", "0.04"),
            ],
        )
        if payload is None:
            print("Tune cancelled.")
            return 0
        mode = payload["window_mode"].strip().lower()
        lookback_days = None
        from_date = None
        to_date = None
        try:
            if mode == "lookback":
                lookback_days = _parse_form_int(payload["lookback_days"], label="Lookback days", default=30, minimum=1)
            elif mode == "range":
                from_date = payload["from_date"].strip() or None
                to_date = payload["to_date"].strip() or None
            elif mode not in {"", "all"}:
                raise ValueError("Window mode must be all, lookback, or range.")
            steps = _parse_form_int(payload["steps"], label="Grid steps", default=5, minimum=1)
            min_risk = _parse_form_float(payload["min_risk"], label="Minimum risk", default=0.002, minimum=0.0001)
            max_risk = _parse_form_float(payload["max_risk"], label="Maximum risk", default=0.02, minimum=0.0001)
            min_leverage = _parse_form_float(payload["min_leverage"], label="Minimum leverage", default=1.0, minimum=1.0)
            max_leverage = _parse_form_float(payload["max_leverage"], label="Maximum leverage", default=20.0, minimum=1.0)
            min_threshold = _parse_form_float(payload["min_threshold"], label="Minimum threshold", default=0.52, minimum=0.01, maximum=0.99)
            max_threshold = _parse_form_float(payload["max_threshold"], label="Maximum threshold", default=0.88, minimum=0.01, maximum=0.99)
            min_take = _parse_form_float(payload["min_take"], label="Minimum take profit", default=0.01, minimum=0.0, maximum=0.99)
            max_take = _parse_form_float(payload["max_take"], label="Maximum take profit", default=0.06, minimum=0.0, maximum=0.99)
            min_stop = _parse_form_float(payload["min_stop"], label="Minimum stop loss", default=0.008, minimum=0.0, maximum=0.99)
            max_stop = _parse_form_float(payload["max_stop"], label="Maximum stop loss", default=0.04, minimum=0.0, maximum=0.99)
        except ValueError as exc:
            print(f"Tune settings invalid: {exc}", file=sys.stderr)
            return 2
        return await ui.run_blocking(
            command_tune,
            argparse.Namespace(
                input=payload["input"].strip() or "data/historical_btcusdc.json",
                save_best=_parse_form_bool(payload["save_best"], False),
                min_risk=min_risk,
                max_risk=max_risk,
                steps=steps,
                min_leverage=min_leverage,
                max_leverage=max_leverage,
                min_threshold=min_threshold,
                max_threshold=max_threshold,
                min_take=min_take,
                max_take=max_take,
                min_stop=min_stop,
                max_stop=max_stop,
                lookback_days=lookback_days,
                from_date=from_date,
                to_date=to_date,
            ),
        )

    async def _backtest(ui):
        payload = await ui.form(
            "Backtest",
            [
                FormField("input", "Backtest input path", "data/historical_btcusdc.json"),
                FormField("model", "Model path", "data/model.json"),
                FormField("start_cash", "Starting cash", "1000"),
            ],
        )
        if payload is None:
            print("Backtest cancelled.")
            return 0
        try:
            start_cash = _parse_form_float(payload["start_cash"], label="Starting cash", default=1000.0, minimum=1.0)
        except ValueError as exc:
            print(f"Backtest settings invalid: {exc}", file=sys.stderr)
            return 2
        return await ui.run_blocking(
            command_backtest,
            argparse.Namespace(
                input=payload["input"].strip() or "data/historical_btcusdc.json",
                model=payload["model"].strip() or "data/model.json",
                start_cash=start_cash,
            ),
        )

    async def _evaluate(ui):
        payload = await ui.form(
            "Evaluate model",
            [
                FormField("input", "Evaluation input path", "data/historical_btcusdc.json"),
                FormField("model", "Model path", "data/model.json"),
                FormField("threshold", "Evaluation threshold [blank=strategy default]", ""),
                FormField("calibrate_threshold", "Calibrate threshold [yes/no]", "no"),
            ],
        )
        if payload is None:
            print("Evaluation cancelled.")
            return 0
        threshold_raw = payload["threshold"].strip()
        try:
            threshold = float(threshold_raw) if threshold_raw else None
        except ValueError as exc:
            print(f"Evaluation settings invalid: {exc}", file=sys.stderr)
            return 2
        return await ui.run_blocking(
            command_evaluate,
            argparse.Namespace(
                input=payload["input"].strip() or "data/historical_btcusdc.json",
                model=payload["model"].strip() or "data/model.json",
                threshold=threshold,
                calibrate_threshold=_parse_form_bool(payload["calibrate_threshold"], False),
            ),
        )

    async def _prepare(ui):
        runtime = load_runtime()
        max_batch_size = 1500 if runtime.market_type == "futures" else 1000
        payload = await ui.form(
            "Prepare system",
            [
                FormField("historical", "Historical candle path", "data/historical_btcusdc.json"),
                FormField("model", "Model artifact path", "data/model.json"),
                FormField("limit", "Fetch limit", "500"),
                FormField("batch_size", f"Klines per request [max {max_batch_size}]", "1000"),
                FormField("preset", "Training preset [custom/quick/balanced/thorough]", "balanced"),
                FormField("epochs", "Training epochs [blank=preset]", ""),
                FormField("learning_rate", "Learning rate", "0.05"),
                FormField("l2_penalty", "L2 penalty", "0.0001"),
                FormField("seed", "Training seed", "7"),
                FormField("walk_forward", "Walk-forward validation [yes/no/blank=preset]", ""),
                FormField("walk_forward_train", "Walk-forward train window [blank=preset]", ""),
                FormField("walk_forward_test", "Walk-forward test window [blank=preset]", ""),
                FormField("walk_forward_step", "Walk-forward step [blank=preset]", ""),
                FormField("calibrate_threshold", "Calibrate threshold [yes/no/blank=preset]", ""),
                FormField("start_cash", "Backtest starting cash", "1000"),
                FormField("online_doctor", "Include exchange connectivity in final check [yes/no]", "no"),
            ],
        )
        if payload is None:
            print("Prepare cancelled.")
            return 0
        historical = payload["historical"].strip() or "data/historical_btcusdc.json"
        model = payload["model"].strip() or "data/model.json"
        try:
            limit = _parse_form_int(payload["limit"], label="Fetch limit", default=500, minimum=1)
            batch_size = _parse_form_int(payload.get("batch_size", "1000"), label="Klines per request", default=1000, minimum=1, maximum=max_batch_size)
            preset = _parse_training_preset(payload["preset"])
            epochs = _parse_optional_form_int(payload.get("epochs", ""), label="Training epochs", minimum=1)
            learning_rate = _parse_form_float(payload.get("learning_rate", "0.05"), label="Learning rate", default=0.05, minimum=0.000001)
            l2_penalty = _parse_form_float(payload.get("l2_penalty", "0.0001"), label="L2 penalty", default=0.0001, minimum=0.0)
            seed = _parse_form_int(payload["seed"], label="Training seed", default=7, minimum=0)
            walk_forward_train = _parse_optional_form_int(payload.get("walk_forward_train", ""), label="Walk-forward train window", minimum=2)
            walk_forward_test = _parse_optional_form_int(payload.get("walk_forward_test", ""), label="Walk-forward test window", minimum=1)
            walk_forward_step = _parse_optional_form_int(payload.get("walk_forward_step", ""), label="Walk-forward step", minimum=1)
            walk_forward = _parse_optional_form_bool(payload.get("walk_forward", ""))
            calibrate_threshold = _parse_optional_form_bool(payload.get("calibrate_threshold", ""))
            start_cash = _parse_form_float(payload["start_cash"], label="Backtest starting cash", default=1000.0, minimum=1.0)
        except ValueError as exc:
            print(f"Prepare settings invalid: {exc}", file=sys.stderr)
            return 2
        return await ui.run_blocking(
            command_prepare,
            argparse.Namespace(
                historical=historical,
                model=model,
                limit=limit,
                batch_size=batch_size,
                preset=preset,
                epochs=epochs,
                learning_rate=learning_rate,
                l2_penalty=l2_penalty,
                seed=seed,
                walk_forward=walk_forward,
                walk_forward_train=walk_forward_train,
                walk_forward_test=walk_forward_test,
                walk_forward_step=walk_forward_step,
                calibrate_threshold=calibrate_threshold,
                start_cash=start_cash,
                online_doctor=_parse_form_bool(payload["online_doctor"], False),
            ),
        )

    async def _paper(ui):
        payload = await ui.form(
            "Paper loop",
            [
                FormField("model", "Model path", "data/model.json"),
                FormField("steps", "Paper loop steps", "20"),
                FormField("sleep", "Sleep seconds", "5"),
                FormField("retrain_interval", "Retrain interval", "0"),
                FormField("retrain_window", "Retrain window", "300"),
                FormField("retrain_min_rows", "Retrain minimum rows", "240"),
            ],
        )
        if payload is None:
            print("Paper loop cancelled.")
            return 0
        try:
            steps = _parse_form_int(payload["steps"], label="Paper loop steps", default=20, minimum=1)
            sleep = _parse_form_int(payload["sleep"], label="Sleep seconds", default=5, minimum=0)
            retrain_interval = _parse_form_int(payload["retrain_interval"], label="Retrain interval", default=0, minimum=0)
            retrain_window = _parse_form_int(payload["retrain_window"], label="Retrain window", default=300, minimum=1)
            retrain_min_rows = _parse_form_int(payload["retrain_min_rows"], label="Retrain minimum rows", default=240, minimum=1)
        except ValueError as exc:
            print(f"Paper loop settings invalid: {exc}", file=sys.stderr)
            return 2
        return await ui.run_blocking(
            command_live,
            argparse.Namespace(
                steps=steps,
                model=payload["model"].strip() or "data/model.json",
                sleep=sleep,
                leverage=None,
                retrain_interval=retrain_interval,
                retrain_window=retrain_window,
                retrain_min_rows=retrain_min_rows,
                paper=True,
                live=False,
            ),
        )

    async def _live(ui):
        payload = await ui.form(
            "Testnet loop",
            [
                FormField("model", "Model path", "data/model.json"),
                FormField("steps", "Live steps", "1"),
                FormField("sleep", "Sleep seconds", "5"),
                FormField("retrain_interval", "Retrain interval", "0"),
                FormField("retrain_window", "Retrain window", "300"),
                FormField("retrain_min_rows", "Retrain minimum rows", "240"),
            ],
        )
        if payload is None:
            print("Testnet loop cancelled.")
            return 0
        try:
            steps = _parse_form_int(payload["steps"], label="Live steps", default=1, minimum=1)
            sleep = _parse_form_int(payload["sleep"], label="Sleep seconds", default=5, minimum=0)
            retrain_interval = _parse_form_int(payload["retrain_interval"], label="Retrain interval", default=0, minimum=0)
            retrain_window = _parse_form_int(payload["retrain_window"], label="Retrain window", default=300, minimum=1)
            retrain_min_rows = _parse_form_int(payload["retrain_min_rows"], label="Retrain minimum rows", default=240, minimum=1)
        except ValueError as exc:
            print(f"Testnet loop settings invalid: {exc}", file=sys.stderr)
            return 2
        model = payload["model"].strip() or "data/model.json"
        if not await ui.confirm(f"Run authenticated testnet execution with model={model}, steps={steps}, sleep={sleep}s?"):
            print("Testnet execution cancelled.")
            return 0
        return await ui.run_blocking(
            command_live,
            argparse.Namespace(
                steps=steps,
                model=model,
                sleep=sleep,
                leverage=None,
                retrain_interval=retrain_interval,
                retrain_window=retrain_window,
                retrain_min_rows=retrain_min_rows,
                paper=False,
                live=True,
            ),
        )

    async def _roundtrip(ui):
        payload = await ui.form(
            "Spot roundtrip",
            [
                FormField("quantity", "Order quantity", "0.00008"),
                FormField("mode", "Mode [auto/buy-sell/sell-buy]", "auto"),
            ],
        )
        if payload is None:
            print("Spot test order cancelled.")
            return 0
        try:
            quantity = _parse_form_float(payload["quantity"], label="Order quantity", default=0.00008, minimum=0.00001)
        except ValueError as exc:
            print(f"Spot roundtrip settings invalid: {exc}", file=sys.stderr)
            return 2
        mode = payload["mode"].strip().lower() or "auto"
        if mode not in {"auto", "buy-sell", "sell-buy"}:
            print("Spot roundtrip mode must be auto, buy-sell, or sell-buy.", file=sys.stderr)
            return 2
        if not await ui.confirm(f"Place {mode} spot testnet roundtrip for quantity={quantity:.8f}?"):
            print("Spot test order cancelled.")
            return 0
        return await ui.run_blocking(
            command_spot_roundtrip,
            argparse.Namespace(quantity=quantity, mode=mode, yes=True),
        )

    async def _report(ui):
        payload = await ui.form(
            "Operator report",
            [
                FormField("input", "Training input path", "data/historical_btcusdc.json"),
                FormField("model", "Model path", "data/model.json"),
                FormField("readiness", "Include readiness report [yes/no]", "yes"),
                FormField("online", "Include exchange connectivity [yes/no]", "no"),
                FormField("account", "Include account state [yes/no]", "no"),
            ],
        )
        if payload is None:
            print("Operator report cancelled.")
            return 0
        return await ui.run_blocking(
            command_report,
            argparse.Namespace(
                input=payload["input"].strip() or "data/historical_btcusdc.json",
                model=payload["model"].strip() or "data/model.json",
                doctor=_parse_form_bool(payload["readiness"], True),
                online=_parse_form_bool(payload["online"], False),
                account=_parse_form_bool(payload["account"], False),
            ),
        )

    async def _funds(ui):
        return await _ui_funds_menu(ui)

    async def _settings(ui):
        return await _ui_settings_menu(ui)

    return [
        TUIAction("1", "Overview", "Print the latest runtime, strategy, funds, model, and recent artifacts into the activity log.", _overview),
        TUIAction("2", "Connect", "Ping the testnet exchange and validate that the configured credentials work.", _connect),
        TUIAction("3", "Account", "Read authenticated balances and open positions from the exchange.", _account),
        TUIAction("4", "Readiness check", "Verify safety flags, training data, model compatibility, and optionally exchange connectivity.", _doctor),
        TUIAction("5", "Local audit", "Check candle quality, feature stability, model metadata, and risk posture without network calls.", _audit),
        TUIAction("6", "Funds", "Manage the virtual USDC / BTC allocation that caps live trading. Independent of trading itself.", _funds),
        TUIAction("7", "Fetch candles", "Download fresh BTCUSDC klines from the testnet into a local dataset.", _fetch),
        TUIAction("8", "Train model", "Train or retrain the model on cached candles using the current strategy features.", _train),
        TUIAction("9", "Evaluate", "Score the saved model against cached candles (classification metrics + thresholds).", _evaluate),
        TUIAction("10", "Backtest", "Simulate trading on cached candles using the saved model; estimates PnL, fees, and drawdown.", _backtest),
        TUIAction("11", "Tune strategy", "Grid search execution parameters across all data, a recent lookback, or a date range.", _tune),
        TUIAction("12", "Prepare system", "One-shot pipeline: fetch candles, train, evaluate, backtest, audit, then readiness checks.", _prepare),
        TUIAction("13", "Paper loop", "Run the live loop in paper mode — no real orders; supports retraining controls.", _paper),
        TUIAction("14", "Testnet loop", "Run authenticated testnet execution with real signed orders against the testnet exchange.", _live),
        TUIAction("15", "Spot roundtrip", "Place a minimal BUY then SELL on spot testnet; smallest signed execution check.", _roundtrip),
        TUIAction("16", "Operator report", "Print the dashboard, recent artifacts, readiness report, and optional account state.", _report),
        # Backwards-compatible aliases for direct configuration shortcuts.
        TUIAction("17", "Runtime settings", "Shortcut to the Runtime panel inside Settings (API keys, market, testnet, recvWindow).", _runtime),
        TUIAction("18", "Strategy settings", "Shortcut to the Strategy panel inside Settings (risk, thresholds, model windows, features).", _strategy),
        TUIAction("19", "Settings", "Centralized configuration: Runtime, Strategy, Execution, Compute backend.", _settings),
        TUIAction("20", "Help", "Detailed help: workflow, keyboard shortcuts, safety notes, configuration tour.", _help),
    ]


def command_menu(_: argparse.Namespace) -> int:
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        print("Interactive console requires a real terminal (TTY).", file=sys.stderr)
        return 2

    from .tui import launch_tui

    return launch_tui(
        title="simple-ai-trading interactive console",
        actions=_tui_actions(),
        snapshot_provider=lambda width=72: render_dashboard(_dashboard_snapshot(with_account=False), width=width),
        connection_provider=_connection_status_line,
    )


def _load_json_candles(path: str) -> list[dict[str, object]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected candle list in JSON file: {path}")
    return payload


def _clamp(value: float, low: float, high: float) -> float:
    if value < low:
        return low
    if value > high:
        return high
    return value


def _rows_from_json(path: str):
    candles_raw = _load_json_candles(path)
    from .api import Candle

    rows: list[Candle] = []
    for item in candles_raw:
        if not isinstance(item, dict):
            continue
        try:
            rows.append(
                Candle(
                    open_time=int(item["open_time"]),
                    open=float(item["open"]),
                    high=float(item["high"]),
                    low=float(item["low"]),
                    close=float(item["close"]),
                    volume=float(item["volume"]),
                    close_time=int(item["close_time"]),
                )
            )
        except (TypeError, ValueError, KeyError):
            continue
    return rows


def _load_rows_for_command(path: str, *, label: str) -> list | None:
    try:
        return _rows_from_json(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"{label}: {exc}", file=sys.stderr)
        return None


def _parse_date_boundary(raw: str, *, end_of_day: bool) -> int:
    dt = datetime.strptime(raw, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    if end_of_day:
        dt = dt + timedelta(days=1) - timedelta(milliseconds=1)
    return int(dt.timestamp() * 1000)


def _filter_candles_for_time_window(
    candles: Sequence[object],
    *,
    lookback_days: int | None = None,
    from_date: str | None = None,
    to_date: str | None = None,
) -> list[object]:
    if lookback_days is not None and lookback_days <= 0:
        raise ValueError("lookback_days must be > 0")
    if lookback_days is not None and (from_date or to_date):
        raise ValueError("lookback_days cannot be combined with from_date/to_date")

    start_ms: int | None = None
    end_ms: int | None = None
    if from_date:
        start_ms = _parse_date_boundary(from_date, end_of_day=False)
    if to_date:
        end_ms = _parse_date_boundary(to_date, end_of_day=True)
    if start_ms is not None and end_ms is not None and start_ms > end_ms:
        raise ValueError("from_date must be <= to_date")

    filtered = list(candles)
    if lookback_days is not None and filtered:
        latest_close = max(int(getattr(candle, "close_time")) for candle in filtered)
        start_ms = latest_close - (lookback_days * 24 * 60 * 60 * 1000)

    if start_ms is not None:
        filtered = [candle for candle in filtered if int(getattr(candle, "open_time")) >= start_ms]
    if end_ms is not None:
        filtered = [candle for candle in filtered if int(getattr(candle, "open_time")) <= end_ms]
    return filtered


def _artifact_path(kind: str, *, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{kind}_run_{int(time.time() * 1_000_000)}.json"


def _persist_run_artifact(kind: str, output_dir: Path, payload: dict[str, object]) -> Path:
    path = _artifact_path(kind, output_dir=output_dir)
    write_json_atomic(path, payload, indent=2, sort_keys=True)
    return path


def _public_runtime_payload(runtime) -> dict[str, object]:
    return runtime.public_dict()


def _build_model_rows(candles: Sequence[object], strategy: StrategyConfig):
    candles = clean_candles(c for c in candles if hasattr(c, "open_time"))
    return make_rows(
        candles,
        strategy.feature_windows[0],
        strategy.feature_windows[1],
        lookahead=1,
        label_threshold=strategy.label_threshold,
        enabled_features=strategy.enabled_features,
    )


def _effective_leverage(cfg: StrategyConfig, market_type: str) -> float:
    if market_type != "futures":
        return 1.0
    return float(max(1.0, min(125.0, cfg.leverage)))


def _resolve_futures_leverage(runtime, cfg: StrategyConfig) -> float:
    """Resolve leverage from runtime+strategy with an exchange-side clamp when possible."""
    requested = _effective_leverage(cfg, runtime.market_type)
    if runtime.market_type != "futures":
        return requested
    if not runtime.api_key or not runtime.api_secret:
        return requested
    client = _build_client(runtime)
    try:
        max_leverage = client.get_max_leverage(runtime.symbol)
        if max_leverage > 0:
            return float(min(requested, max_leverage))
    except BinanceAPIError:
        return requested
    return requested


def _resolve_symbol_constraints(runtime, client) -> SymbolConstraints | None:
    try:
        return client.get_symbol_constraints(runtime.symbol)
    except BinanceAPIError:
        return None


def _strategy_feature_signature(strategy: StrategyConfig) -> str:
    return feature_signature(
        strategy.feature_windows[0],
        strategy.feature_windows[1],
        strategy.label_threshold,
        feature_version=strategy.feature_version,
        enabled_features=strategy.enabled_features,
    )


def _load_runtime_model(model_path: Path, strategy: StrategyConfig):
    strategy_signature = _strategy_feature_signature(strategy)
    return load_model(
        model_path,
        expected_feature_version=strategy.feature_version,
        expected_feature_signature=strategy_signature,
        expected_feature_dim=None,
    )


def _target_notional(
    cash: float,
    strategy: StrategyConfig,
    market_type: str,
    *,
    leverage: float | None = None,
) -> float:
    if cash <= 0:
        return 0.0
    if leverage is None:
        leverage = _effective_leverage(strategy, market_type)
    risk_exposure = strategy.risk_per_trade * leverage
    risk_exposure = min(risk_exposure, strategy.max_position_pct * leverage)
    risk_exposure = min(risk_exposure, 1.0)
    return max(0.0, cash * risk_exposure)


def _build_order_notional(
    cash: float,
    price: float,
    cfg: StrategyConfig,
    market_type: str,
    leverage: float,
    client,
    *,
    constraints: SymbolConstraints | None = None,
) -> tuple[float, float]:
    """Build and return adjusted (notional, qty) for a desired trade.

    Returns (notional, qty) after constraints are enforced.
    """
    if price <= 0:
        return 0.0, 0.0

    requested_notional = _target_notional(cash, cfg, market_type, leverage=leverage)
    if requested_notional <= 0:
        return 0.0, 0.0

    qty = requested_notional / price
    if constraints is None:
        return requested_notional, abs(qty)

    normalized_qty, parsed_constraints = client.normalize_quantity(constraints.symbol, abs(qty))
    if normalized_qty <= 0:
        return 0.0, 0.0

    requested_notional = normalized_qty * price

    if normalized_qty < parsed_constraints.min_qty:
        return 0.0, 0.0

    if parsed_constraints.min_notional > 0 and requested_notional < parsed_constraints.min_notional:
        return 0.0, 0.0

    if parsed_constraints.max_notional > 0 and requested_notional > parsed_constraints.max_notional:
        capped_notional = parsed_constraints.max_notional
        capped_qty, _ = client.normalize_quantity(parsed_constraints.symbol, capped_notional / price)
        if capped_qty <= 0:
            return 0.0, 0.0
        requested_notional = capped_qty * price
        normalized_qty = capped_qty

    return requested_notional, abs(normalized_qty)


def _safe_day_ms(timestamp_ms: int) -> int:
    return int(timestamp_ms // (24 * 60 * 60 * 1000))


def _resolve_live_retrain_rows(
    rows: list,
    *,
    retrain_window: int,
    retrain_min_rows: int,
) -> list:
    if len(rows) < retrain_min_rows:
        return []
    if len(rows) <= retrain_window:
        return rows
    return rows[-retrain_window:]


def _build_live_model(
    rows: list,
    *,
    model: object | None = None,
    retrain_every: int,
    step: int,
    cfg: StrategyConfig,
    retrain_window: int,
    retrain_min_rows: int,
) -> object | None:
    if model is not None:
        if retrain_every <= 0:
            return model
        if retrain_every > 0 and step % retrain_every != 0:
            return model

    train_rows = _resolve_live_retrain_rows(rows, retrain_window=retrain_window, retrain_min_rows=retrain_min_rows)
    if not train_rows:
        return model

    epochs = max(20, int(cfg.training_epochs * 0.4))
    return train(train_rows, epochs=epochs, feature_signature=_strategy_feature_signature(cfg))


def _score_to_direction(score: float, cfg: StrategyConfig, market_type: str, threshold: float | None = None) -> int:
    threshold = cfg.signal_threshold if threshold is None else _clamp(float(threshold), 0.0, 1.0)
    if market_type == "futures":
        if score >= threshold:
            return 1
        if score <= (1.0 - threshold):
            return -1
        return 0
    return 1 if score >= threshold else 0


def _safe_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _detect_existing_position(runtime, client, *, leverage: float, reference_price: float | None = None) -> dict[str, float | int | str] | None:
    if not hasattr(client, "get_account"):
        return None
    account = client.get_account()
    if not isinstance(account, dict):
        return None
    if runtime.market_type == "futures":
        for item in account.get("positions", []) or []:
            if not isinstance(item, dict) or item.get("symbol") != runtime.symbol:
                continue
            amount = _safe_float(item.get("positionAmt"))
            if amount == 0.0:
                continue
            entry_price = _safe_float(item.get("entryPrice")) or _safe_float(item.get("markPrice")) or reference_price or 0.0
            qty = abs(amount)
            margin = (
                _safe_float(item.get("positionInitialMargin"))
                or _safe_float(item.get("initialMargin"))
                or _safe_float(item.get("isolatedWallet"))
                or (qty * entry_price / max(1.0, leverage) if entry_price > 0.0 else 0.0)
            )
            return {
                "market": "futures",
                "side": 1 if amount > 0.0 else -1,
                "qty": qty,
                "entry_price": entry_price,
                "notional": qty * entry_price,
                "margin": margin,
            }
        return None

    base_asset = runtime.symbol.removesuffix("USDC")
    managed_btc = max(0.0, _safe_float(getattr(runtime, "managed_btc", 0.0)))
    if managed_btc <= 0.0:
        return None
    for item in account.get("balances", []) or []:
        if not isinstance(item, dict) or item.get("asset") != base_asset:
            continue
        qty = _safe_float(item.get("free")) + _safe_float(item.get("locked"))
        if qty <= 0.0:
            continue
        qty = min(qty, managed_btc)
        price = reference_price
        if price is None:
            price, _timestamp = client.get_symbol_price(runtime.symbol)
        return {
            "market": "spot",
            "side": 1,
            "qty": qty,
            "entry_price": float(price),
            "notional": qty * float(price),
            "margin": qty * float(price),
        }
    return None


def _paper_or_live_order(
    client: BinanceClient,
    runtime,
    strategy: StrategyConfig,
    *,
    side: str,
    size: float,
    dry_run: bool,
    leverage: float | None = None,
    reduce_only: bool = False,
) -> None:
    if leverage is None:
        leverage = _effective_leverage(strategy, runtime.market_type)
    kwargs = {"dry_run": dry_run, "leverage": leverage}
    if reduce_only and not dry_run:
        kwargs["reduce_only"] = True
    response = client.place_order(runtime.symbol, side, size, **kwargs)
    if dry_run:
        print("paper order:", json.dumps(response, indent=2))
        return
    print(f"live order: {side} {size:.8f} {runtime.symbol}")
    print(json.dumps(response, indent=2))


def _asset_free_balance(account: object, asset: str) -> float:
    if not isinstance(account, dict):
        return 0.0
    total = 0.0
    for item in account.get("balances", []) or []:
        if not isinstance(item, dict) or item.get("asset") != asset:
            continue
        total += _safe_float(item.get("free"))
    return total


def _order_executed_qty(order: object) -> float:
    if not isinstance(order, dict):
        return 0.0
    return _safe_float(order.get("executedQty") or order.get("origQty"))


def _roundtrip_quantity(client, symbol: str, requested: float, price: float) -> tuple[float, SymbolConstraints, float]:
    if requested <= 0.0:
        raise ValueError("Roundtrip quantity must be > 0.")
    if price <= 0.0:
        raise BinanceAPIError("Cannot resolve a positive BTCUSDC mark price")
    quantity, constraints = client.normalize_quantity(symbol, requested)
    min_notional = float(constraints.min_notional)
    if min_notional > 0.0 and quantity * price < min_notional:
        step = max(0.0, _safe_float(getattr(constraints, "step_size", 0.0)))
        target = (min_notional / price) + step
        quantity, constraints = client.normalize_quantity(symbol, max(target, float(constraints.min_qty)))
    notional = quantity * price
    if quantity <= 0.0:
        raise BinanceAPIError("Requested quantity is below BTCUSDC exchange filters")
    if min_notional > 0.0 and notional < min_notional:
        raise BinanceAPIError(
            f"Requested quantity notional {notional:.2f} is below exchange minimum {min_notional:.2f}"
        )
    if constraints.max_notional > 0.0 and notional > constraints.max_notional:
        raise BinanceAPIError(
            f"Requested quantity notional {notional:.2f} exceeds exchange maximum {constraints.max_notional:.2f}"
        )
    return quantity, constraints, notional


def _roundtrip_second_quantity(client, symbol: str, side: str, executed_qty: float, account: object, price: float) -> float:
    base_asset = symbol.removesuffix("USDC")
    if side == "SELL":
        available = _asset_free_balance(account, base_asset)
        target = min(max(0.0, executed_qty), available)
    else:
        available_quote = _asset_free_balance(account, "USDC")
        target = min(max(0.0, executed_qty), (available_quote / price) * 0.995 if price > 0.0 else 0.0)
    quantity, _constraints = client.normalize_quantity(symbol, target)
    return quantity


def command_spot_roundtrip(args: argparse.Namespace) -> int:
    runtime = load_runtime()
    if not getattr(args, "yes", False):
        print("Pass --yes to confirm signed spot-testnet order placement.", file=sys.stderr)
        return 2
    if runtime.market_type != "spot":
        print("Spot roundtrip requires runtime.market_type=spot.", file=sys.stderr)
        return 2
    if not runtime.testnet:
        print("Spot roundtrip requires testnet=true.", file=sys.stderr)
        return 2
    if not runtime.api_key or not runtime.api_secret:
        print("Spot roundtrip requires configured API credentials.", file=sys.stderr)
        return 2

    mode = str(getattr(args, "mode", "auto"))
    quantity_requested = float(getattr(args, "quantity", 0.0))
    client = _build_client(runtime)
    try:
        client.ensure_btcusdc()
        price, _timestamp = client.get_symbol_price(runtime.symbol)
        quantity, _constraints, notional = _roundtrip_quantity(
            client,
            runtime.symbol,
            quantity_requested,
            float(price),
        )
        before = client.get_account()
        btc_free = _asset_free_balance(before, "BTC")
        usdc_free = _asset_free_balance(before, "USDC")
        if mode == "auto":
            mode = "buy-sell" if usdc_free >= notional * 1.01 else "sell-buy"
        if mode == "buy-sell":
            if usdc_free < notional * 1.01:
                raise BinanceAPIError(f"Insufficient USDC for BUY leg: need about {notional:.2f}, have {usdc_free:.2f}")
            first_side, second_side = "BUY", "SELL"
        elif mode == "sell-buy":
            if btc_free < quantity:
                raise BinanceAPIError(f"Insufficient BTC for SELL leg: need {quantity:.8f}, have {btc_free:.8f}")
            first_side, second_side = "SELL", "BUY"
        else:
            raise ValueError(f"Unsupported roundtrip mode: {mode}")

        first = client.place_order(runtime.symbol, first_side, quantity, dry_run=False)
        executed = _order_executed_qty(first) or quantity
        mid = client.get_account()
        second_quantity = _roundtrip_second_quantity(client, runtime.symbol, second_side, executed, mid, float(price))
        if second_quantity <= 0.0:
            raise BinanceAPIError(f"Could not size {second_side} leg from post-{first_side} balances")
        second = client.place_order(runtime.symbol, second_side, second_quantity, dry_run=False)
        after = client.get_account()
    except (BinanceAPIError, ValueError) as exc:
        print(f"Spot roundtrip failed: {exc}", file=sys.stderr)
        return 2

    payload = {
        "command": "spot-roundtrip",
        "timestamp": int(time.time()),
        "runtime": _public_runtime_payload(runtime),
        "symbol": runtime.symbol,
        "mode": mode,
        "price_reference": float(price),
        "quantity_requested": float(quantity_requested),
        "quantity_first": float(quantity),
        "quantity_second": float(second_quantity),
        "notional_reference": float(notional),
        "balances_before": {
            "BTC": _asset_free_balance(before, "BTC"),
            "USDC": _asset_free_balance(before, "USDC"),
        },
        "balances_after": {
            "BTC": _asset_free_balance(after, "BTC"),
            "USDC": _asset_free_balance(after, "USDC"),
        },
        "first_order": {
            "side": first_side,
            "status": first.get("status") if isinstance(first, dict) else None,
            "orderId": first.get("orderId") if isinstance(first, dict) else None,
            "executedQty": first.get("executedQty") if isinstance(first, dict) else None,
        },
        "second_order": {
            "side": second_side,
            "status": second.get("status") if isinstance(second, dict) else None,
            "orderId": second.get("orderId") if isinstance(second, dict) else None,
            "executedQty": second.get("executedQty") if isinstance(second, dict) else None,
        },
    }
    artifact_path = _persist_run_artifact("spot_roundtrip", Path("data"), payload)
    print("Spot test roundtrip complete.")
    print(json.dumps({**payload, "artifact": str(artifact_path)}, indent=2))
    return 0


def command_configure(_: argparse.Namespace) -> int:
    current = load_runtime()
    next_runtime = prompt_runtime(current)
    save_runtime(next_runtime)

    if next_runtime.validate_account and next_runtime.api_key and next_runtime.api_secret:
        client = _build_client(next_runtime)
        try:
            _validate_runtime_connection(next_runtime, client)
        except BinanceAPIError as exc:
            print(f"Configuration saved, but validation failed: {exc}", file=sys.stderr)
            return 2

    print("Runtime config saved to", config_paths()["runtime"])
    print(f"market={next_runtime.market_type} testnet={next_runtime.testnet} paper={next_runtime.dry_run}")
    if next_runtime.market_type == "futures":
        print("futures-mode enabled; leverage can be set via strategy.leverage")
    return 0


def command_connect(_: argparse.Namespace) -> int:
    runtime = load_runtime()
    client = _build_client(runtime)
    try:
        server_time = client.get_exchange_time()
        client.ensure_btcusdc()
        account = None
        if runtime.api_key and runtime.api_secret:
            account = client.get_account()
            if isinstance(account, dict):
                account = {
                    "updateTime": account.get("updateTime"),
                    "canTrade": account.get("canTrade"),
                    "accountType": account.get("accountType"),
                    "positions": account.get("positions"),
                    "assets": account.get("assets"),
                }
        if runtime.market_type == "futures" and runtime.api_key and runtime.api_secret:
            try:
                max_leverage = client.get_max_leverage(runtime.symbol)
            except BinanceAPIError as exc:
                print(f"unable to fetch leverage bracket: {exc}", file=sys.stderr)
            else:
                print(f"max leverage on {runtime.symbol}: {max_leverage}x")

        print("exchange: connected")
        print("market:", runtime.market_type)
        print("testnet:", runtime.testnet)
        print("endpoint:", client.base_url)
        print("server_time:", server_time.get("serverTime") if isinstance(server_time, dict) else server_time)
        if account is not None:
            print("account:", json.dumps(account, indent=2))
        return 0
    except BinanceAPIError as exc:
        print(f"Connection failed: {exc}", file=sys.stderr)
        return 2


def command_doctor(args: argparse.Namespace) -> int:
    ok, lines = _readiness_report(input_path=args.input, model_path=args.model, online=bool(args.online))
    print("Readiness report")
    for line in lines:
        print(line)
    return 0 if ok else 2


def command_audit(args: argparse.Namespace) -> int:
    from .audit import build_audit_report, render_audit_report

    runtime = load_runtime()
    strategy = load_strategy()
    candles = _load_rows_for_command(args.input, label="Audit data load failed")
    if candles is None:
        return 2
    report = build_audit_report(candles, runtime, strategy, model_path=Path(args.model))
    print(render_audit_report(report))
    return 0 if report.ok else 2


def command_report(args: argparse.Namespace) -> int:
    print(
        _render_operator_report(
            with_account=bool(args.account),
            doctor=bool(args.doctor),
            online=bool(args.online),
            input_path=args.input,
            model_path=args.model,
        )
    )
    return 0


def command_prepare(args: argparse.Namespace) -> int:
    try:
        runtime = load_runtime()
        max_batch_size = 1500 if runtime.market_type == "futures" else 1000
        preset = _parse_training_preset(str(args.preset))
        limit = int(args.limit)
        batch_size = int(getattr(args, "batch_size", 1000))
        seed = int(args.seed)
        learning_rate = float(getattr(args, "learning_rate", 0.05))
        l2_penalty = float(getattr(args, "l2_penalty", 1e-4))
        start_cash = float(args.start_cash)
        preset_args = _apply_training_preset(
            argparse.Namespace(
                preset=preset,
                epochs=250,
                walk_forward=False,
                walk_forward_train=300,
                walk_forward_test=60,
                walk_forward_step=30,
                calibrate_threshold=False,
            )
        )
        epochs = int(args.epochs if getattr(args, "epochs", None) is not None else preset_args.epochs)
        walk_forward = (
            bool(args.walk_forward)
            if getattr(args, "walk_forward", None) is not None
            else bool(preset_args.walk_forward)
        )
        walk_forward_train = int(
            args.walk_forward_train
            if getattr(args, "walk_forward_train", None) is not None
            else preset_args.walk_forward_train
        )
        walk_forward_test = int(
            args.walk_forward_test
            if getattr(args, "walk_forward_test", None) is not None
            else preset_args.walk_forward_test
        )
        walk_forward_step = int(
            args.walk_forward_step
            if getattr(args, "walk_forward_step", None) is not None
            else preset_args.walk_forward_step
        )
        calibrate_threshold = (
            bool(args.calibrate_threshold)
            if getattr(args, "calibrate_threshold", None) is not None
            else bool(preset_args.calibrate_threshold)
        )
        if limit < 1:
            raise ValueError("Fetch limit must be >= 1.")
        if batch_size < 1:
            raise ValueError("Fetch batch size must be >= 1.")
        batch_size = min(batch_size, max_batch_size)
        if epochs < 1:
            raise ValueError("Training epochs must be >= 1.")
        if seed < 0:
            raise ValueError("Training seed must be >= 0.")
        if learning_rate <= 0.0:
            raise ValueError("Learning rate must be > 0.")
        if l2_penalty < 0.0:
            raise ValueError("L2 penalty must be >= 0.")
        if start_cash < 1:
            raise ValueError("Backtest starting cash must be >= 1.")
        if walk_forward_train < 2:
            raise ValueError("Walk-forward train window must be >= 2.")
        if walk_forward_test < 1:
            raise ValueError("Walk-forward test window must be >= 1.")
        if walk_forward_step < 1:
            raise ValueError("Walk-forward step must be >= 1.")
    except (TypeError, ValueError) as exc:
        print(f"Prepare settings invalid: {exc}", file=sys.stderr)
        return 2

    historical = str(args.historical)
    model = str(args.model)
    sequence = [
        (
            "Fetch candles",
            command_fetch,
            argparse.Namespace(symbol=runtime.symbol, interval=runtime.interval, limit=limit, batch_size=batch_size, output=historical),
        ),
        (
            "Train model",
            command_train,
            argparse.Namespace(
                input=historical,
                output=model,
                preset="custom",
                requested_preset=preset,
                epochs=epochs,
                learning_rate=learning_rate,
                l2_penalty=l2_penalty,
                seed=seed,
                walk_forward=walk_forward,
                walk_forward_train=walk_forward_train,
                walk_forward_test=walk_forward_test,
                walk_forward_step=walk_forward_step,
                calibrate_threshold=calibrate_threshold,
            ),
        ),
        (
            "Evaluate",
            command_evaluate,
            argparse.Namespace(input=historical, model=model, threshold=None, calibrate_threshold=calibrate_threshold),
        ),
        (
            "Backtest",
            command_backtest,
            argparse.Namespace(input=historical, model=model, start_cash=start_cash),
        ),
        (
            "Local audit",
            command_audit,
            argparse.Namespace(input=historical, model=model),
        ),
        (
            "Readiness check",
            command_doctor,
            argparse.Namespace(input=historical, model=model, online=bool(args.online_doctor)),
        ),
    ]
    for label, fn, step_args in sequence:
        print(f"== {label} ==")
        result = fn(step_args)
        if result != 0:
            print(f"Prepare stopped at {label}.", file=sys.stderr)
            return result
    return 0


def command_status(_: argparse.Namespace) -> int:
    runtime = load_runtime()
    strategy = load_strategy()
    print(json.dumps({"runtime": _public_runtime_payload(runtime), "strategy": strategy.asdict()}, indent=2))
    return 0


def command_strategy(args: argparse.Namespace) -> int:
    cfg = load_strategy()
    runtime = load_runtime()
    try:
        profile = _parse_strategy_profile(str(getattr(args, "profile", "custom") or "custom"))
    except ValueError as exc:
        print(f"Invalid strategy profile: {exc}", file=sys.stderr)
        return 2
    updates = dict(_STRATEGY_PROFILES[profile])
    if args.leverage is not None:
        requested = max(1.0, args.leverage)
        if runtime.market_type == "futures":
            if runtime.api_key and runtime.api_secret:
                try:
                    client = _build_client(runtime)
                    max_leverage = client.get_max_leverage(runtime.symbol)
                except BinanceAPIError:
                    max_leverage = 125
            else:
                max_leverage = 125
            requested = min(requested, float(max_leverage))
        updates["leverage"] = requested
    if args.risk is not None:
        updates["risk_per_trade"] = max(0.0001, args.risk)
    if args.max_position is not None:
        updates["max_position_pct"] = max(0.0, min(1.0, args.max_position))
    if args.stop is not None:
        updates["stop_loss_pct"] = max(0.0, min(0.99, args.stop))
    if args.take is not None:
        updates["take_profit_pct"] = max(0.0, min(0.99, args.take))
    if args.max_open is not None:
        updates["max_open_positions"] = max(0, args.max_open)
    if args.max_trades_per_day is not None:
        updates["max_trades_per_day"] = max(0, args.max_trades_per_day)
    if args.cooldown is not None:
        updates["cooldown_minutes"] = max(0, args.cooldown)
    if args.signal_threshold is not None:
        updates["signal_threshold"] = _clamp(args.signal_threshold, 0.01, 0.99)
    if args.max_drawdown is not None:
        updates["max_drawdown_limit"] = max(0.0, args.max_drawdown)
    if args.taker_fee_bps is not None:
        updates["taker_fee_bps"] = max(0.0, args.taker_fee_bps)
    if args.slippage_bps is not None:
        updates["slippage_bps"] = max(0.0, args.slippage_bps)
    if args.label_threshold is not None:
        updates["label_threshold"] = max(0.0001, args.label_threshold)
    if getattr(args, "model_lookback", None) is not None:
        updates["model_lookback"] = max(10, int(args.model_lookback))
    if getattr(args, "training_epochs", None) is not None:
        updates["training_epochs"] = max(1, int(args.training_epochs))
    if getattr(args, "confidence_beta", None) is not None:
        updates["confidence_beta"] = _clamp(float(args.confidence_beta), 0.0, 1.0)
    feature_window_short = getattr(args, "feature_window_short", None)
    feature_window_long = getattr(args, "feature_window_long", None)
    if feature_window_short is not None or feature_window_long is not None:
        short_window = max(1, int(feature_window_short if feature_window_short is not None else cfg.feature_windows[0]))
        long_window = max(short_window + 1, int(feature_window_long if feature_window_long is not None else cfg.feature_windows[1]))
        updates["feature_windows"] = (short_window, long_window)
    try:
        if getattr(args, "set_features", None):
            updates["enabled_features"] = normalize_enabled_features(
                [part.strip() for part in str(args.set_features).split(",") if part.strip()]
            )
        else:
            selected_features = list(cfg.enabled_features)
            for name in getattr(args, "enable_feature", []) or []:
                if name not in selected_features:
                    selected_features.append(name)
            for name in getattr(args, "disable_feature", []) or []:
                selected_features = [feature for feature in selected_features if feature != name]
            if getattr(args, "enable_feature", None) or getattr(args, "disable_feature", None):
                updates["enabled_features"] = normalize_enabled_features(selected_features)
    except ValueError as exc:
        print(f"Invalid feature selection: {exc}", file=sys.stderr)
        return 2

    cfg = StrategyConfig(**{**cfg.asdict(), **updates})
    save_strategy(cfg)
    print("Saved strategy settings.")
    print(json.dumps(cfg.asdict(), indent=2))
    return 0


def command_fetch(args: argparse.Namespace) -> int:
    runtime = load_runtime()
    symbol = (args.symbol or runtime.symbol).upper()
    interval = args.interval or runtime.interval
    output = Path(args.output)
    if symbol != "BTCUSDC":
        print("Error: this CLI supports BTCUSDC only", file=sys.stderr)
        return 2
    limit = max(1, int(args.limit))
    max_batch_size = 1500 if runtime.market_type == "futures" else 1000
    batch_size = max(1, min(max_batch_size, int(getattr(args, "batch_size", 1000))))

    client = _build_client(runtime)
    try:
        client.ensure_btcusdc()
        candles_by_open_time = {}
        end_time = None
        while len(candles_by_open_time) < limit:
            request_limit = min(batch_size, limit - len(candles_by_open_time))
            chunk = client.get_klines(symbol, interval, limit=request_limit, end_time=end_time)
            if not chunk:
                break
            before = len(candles_by_open_time)
            for candle in chunk:
                candles_by_open_time[candle.open_time] = candle
            earliest_open_time = min(c.open_time for c in chunk)
            end_time = earliest_open_time - 1
            if len(candles_by_open_time) == before or end_time < 0:
                break
            if len(chunk) < request_limit:
                break
    except BinanceAPIError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    candles = clean_candles(candles_by_open_time.values())[-limit:]

    payload = [
        {
            "open_time": c.open_time,
            "open": c.open,
            "high": c.high,
            "low": c.low,
            "close": c.close,
            "volume": c.volume,
            "close_time": c.close_time,
        }
        for c in candles
    ]
    write_json_atomic(output, payload, indent=2)
    print(f"saved {len(payload)} candles to {output}")
    return 0


def command_train(args: argparse.Namespace) -> int:
    try:
        args = _apply_training_preset(args)
    except ValueError as exc:
        print(f"Training settings invalid: {exc}", file=sys.stderr)
        return 2
    cfg = load_strategy()
    runtime = load_runtime()
    candles = _load_rows_for_command(args.input, label="Training data load failed")
    if candles is None:
        return 2
    rows = _build_model_rows(candles, cfg)
    if not rows:
        print("No rows produced. Fetch more data or increase lookback.")
        return 2

    wf = None
    seed = int(getattr(args, "seed", 7))
    learning_rate = float(getattr(args, "learning_rate", 0.05))
    l2_penalty = float(getattr(args, "l2_penalty", 1e-4))
    if learning_rate <= 0.0:
        print("Training settings invalid: learning_rate must be > 0.", file=sys.stderr)
        return 2
    if l2_penalty < 0.0:
        print("Training settings invalid: l2_penalty must be >= 0.", file=sys.stderr)
        return 2

    if args.walk_forward:
        try:
            wf = walk_forward_report(
                rows,
                train_window=args.walk_forward_train,
                test_window=args.walk_forward_test,
                step=args.walk_forward_step,
                epochs=max(50, args.epochs // 2),
                calibrate=args.calibrate_threshold,
                learning_rate=learning_rate,
                l2_penalty=l2_penalty,
            )
            print(
                f"walk-forward: folds={wf['folds']} avg_score={wf['average_score']:.4f} "
                f"(train={wf['train_window']} test={wf['test_window']} step={wf['step']})"
            )
            folds = wf["scores"]
            if folds:
                print(
                    "walk-forward fold scores: "
                    + ", ".join(f"{float(v):.3f}" for v in folds)
            )
        except ValueError as exc:
            print(f"walk-forward unavailable: {exc}")
            wf = None

    model_signature = _strategy_feature_signature(cfg)

    split = temporal_validation_split(rows)
    train_rows = split.train_rows
    calibration_rows = split.calibration_rows
    validation_rows = split.validation_rows

    model = train(
        train_rows,
        epochs=args.epochs,
        learning_rate=learning_rate,
        l2_penalty=l2_penalty,
        seed=seed,
        feature_signature=model_signature,
    )
    threshold = cfg.signal_threshold
    if args.calibrate_threshold and calibration_rows:
        threshold = calibrate_threshold(calibration_rows, model, start=0.05, end=0.95, steps=31)
    model.decision_threshold = float(threshold)
    model.calibration_size = len(calibration_rows) if args.calibrate_threshold else 0
    model.validation_size = len(validation_rows)
    model.training_cutoff_timestamp = train_rows[-1].timestamp if train_rows else None
    train_score = evaluate(train_rows, model, threshold=threshold)
    calibration_score = evaluate(calibration_rows, model, threshold=threshold) if calibration_rows else 0.0
    validation_score = evaluate(validation_rows, model, threshold=threshold) if validation_rows else 0.0
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    serialize_model(model, output)
    print(f"trained model saved to {output}")
    print(
        f"rows: {len(rows)} split train={len(train_rows)} "
        f"calibration={len(calibration_rows)} validation={len(validation_rows)}"
    )
    print(f"in-sample directional accuracy: {train_score:.3f}")
    print(f"out-of-sample directional accuracy: {validation_score:.3f}")
    if args.calibrate_threshold and calibration_rows:
        print(f"validated threshold: {threshold:.3f}")
        print(f"calibration directional accuracy: {calibration_score:.3f}")
    artifact = {
        "command": "train",
        "timestamp": int(time.time()),
        "seed": int(seed),
        "runtime": _public_runtime_payload(runtime),
        "strategy": cfg.asdict(),
        "train": {
            "input": str(args.input),
            "output": str(args.output),
            "rows_total": len(rows),
            "rows_train": len(train_rows),
            "rows_calibration": len(calibration_rows),
            "rows_validation": len(validation_rows),
            "epochs": int(args.epochs),
            "learning_rate": float(learning_rate),
            "l2_penalty": float(l2_penalty),
            "lookback_windows": list(cfg.feature_windows),
            "label_threshold": float(cfg.label_threshold),
            "preset": str(getattr(args, "requested_preset", args.preset)),
            "walk_forward": bool(args.walk_forward),
        },
        "walk_forward": wf if wf is not None else None,
        "metrics": {
            "in_sample_accuracy": float(train_score),
            "out_of_sample_accuracy": float(validation_score),
            "calibration_accuracy": float(calibration_score),
            "threshold": float(threshold),
            "tuned_threshold": float(threshold) if args.calibrate_threshold and calibration_rows else None,
            "calibrated_out_of_sample_accuracy": float(validation_score)
            if args.calibrate_threshold and validation_rows
            else None,
            "model_feature_version": model.feature_version,
            "model_feature_signature": model.feature_signature,
        },
        "model": {
            "path": str(args.output),
            "feature_dim": int(model.feature_dim),
            "feature_version": str(model.feature_version),
            "feature_signature": model.feature_signature,
            "decision_threshold": float(model.decision_threshold)
            if model.decision_threshold is not None
            else None,
            "calibration_size": int(model.calibration_size),
            "validation_size": int(model.validation_size),
            "training_cutoff_timestamp": int(model.training_cutoff_timestamp)
            if model.training_cutoff_timestamp is not None
            else None,
        },
        "market": runtime.market_type,
        "symbol": runtime.symbol,
    }
    resolved_leverage = _resolve_futures_leverage(runtime, cfg)
    print(f"market={runtime.market_type} leverage={resolved_leverage:.2f}")
    artifact_path = _persist_run_artifact("train", Path(args.output).parent, artifact)
    print(f"saved train artifact to {artifact_path}")
    return 0


def command_tune(args: argparse.Namespace) -> int:
    cfg = load_strategy()
    runtime = load_runtime()
    max_leverage = 125.0
    if runtime.market_type == "futures" and runtime.api_key and runtime.api_secret:
        try:
            client = _build_client(runtime)
            max_leverage = float(client.get_max_leverage(runtime.symbol))
        except BinanceAPIError:
            max_leverage = 125.0
    candles = _load_rows_for_command(args.input, label="Tune data load failed")
    if candles is None:
        return 2
    try:
        candles = _filter_candles_for_time_window(
            candles,
            lookback_days=getattr(args, "lookback_days", None),
            from_date=getattr(args, "from_date", None),
            to_date=getattr(args, "to_date", None),
        )
    except ValueError as exc:
        print(f"Tune window invalid: {exc}", file=sys.stderr)
        return 2
    rows = _build_model_rows(candles, cfg)
    if len(rows) < 40:
        print("Need more data rows to run tuning")
        return 2

    split = max(10, int(len(rows) * 0.7))
    train_rows = rows[:split]
    test_rows = rows[split:]

    risks: Iterable[float] = [args.min_risk + (args.max_risk - args.min_risk) * i / max(args.steps - 1, 1)
                              for i in range(args.steps)]
    levs: Iterable[float] = [args.min_leverage + (args.max_leverage - args.min_leverage) * i / max(args.steps - 1, 1)
                             for i in range(args.steps)]
    thrs: Iterable[float] = [args.min_threshold + (args.max_threshold - args.min_threshold) * i / max(args.steps - 1, 1)
                             for i in range(args.steps)]
    takes: Iterable[float] = [args.min_take + (args.max_take - args.min_take) * i / max(args.steps - 1, 1)
                              for i in range(args.steps)]
    stops: Iterable[float] = [args.min_stop + (args.max_stop - args.min_stop) * i / max(args.steps - 1, 1)
                              for i in range(args.steps)]
    best: StrategyConfig = cfg
    fallback: StrategyConfig | None = None
    fallback_score = float("-inf")
    tuned = False
    best_score = float("-inf")

    for risk in risks:
        for leverage in levs:
            for threshold in thrs:
                for take in takes:
                    for stop in stops:
                        candidate = StrategyConfig(
                            **{
                                **cfg.asdict(),
                                "risk_per_trade": max(0.0001, risk),
                                "leverage": max(1.0, min(float(max_leverage), leverage)),
                                "signal_threshold": max(0.05, min(0.95, threshold)),
                                "take_profit_pct": max(0.0, min(0.99, take)),
                                "stop_loss_pct": max(0.0, min(0.99, stop)),
                            },
                        )
                        model = train(train_rows, epochs=max(50, candidate.training_epochs // 2))
                        candidate_result = run_backtest(
                            test_rows,
                            model,
                            candidate,
                            market_type=runtime.market_type,
                            starting_cash=1000.0,
                        )
                        score = _tune_score(candidate_result, starting_cash=1000.0)
                        candidate_stopped = bool(getattr(candidate_result, "stopped_by_drawdown", False))
                        if candidate_stopped:
                            if score > fallback_score:
                                fallback_score = score
                                fallback = candidate
                            continue
                        if score > best_score:
                            best_score = score
                            best = candidate
                            tuned = True
        # no valid candidate should silently keep -inf
    if not tuned:
        if fallback is not None:
            best = fallback
            best_score = fallback_score
            print("Warning: all tune candidates hit drawdown limit; using best fallback score by risk-adjusted metric.")
        else:
            print("No valid candidates evaluated.")
            return 2

    print(f"tune best score: {best_score:.4f}")
    print(
        f"tune best config risk={best.risk_per_trade:.4f} take={best.take_profit_pct:.4f} "
        f"stop={best.stop_loss_pct:.4f} leverage={best.leverage:.1f} threshold={best.signal_threshold:.3f}"
    )
    if args.save_best:
        save_strategy(best)
        print("Saved tuned strategy.")
    return 0


def command_backtest(args: argparse.Namespace) -> int:
    runtime = load_runtime()
    cfg = load_strategy()
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Model file not found: {model_path}")
        return 2

    candles = _load_rows_for_command(args.input, label="Backtest data load failed")
    if candles is None:
        return 2
    rows = _build_model_rows(candles, cfg)
    try:
        model = _load_runtime_model(model_path, cfg)
    except (OSError, json.JSONDecodeError, ModelLoadError, ModelFeatureMismatchError) as exc:
        print(f"Model load failed: {exc}", file=sys.stderr)
        return 2
    decision_threshold = model_decision_threshold(model, cfg.signal_threshold)
    result = run_backtest(rows, model, cfg, starting_cash=args.start_cash, market_type=runtime.market_type)
    artifact = {
        "command": "backtest",
        "timestamp": int(time.time()),
        "runtime": _public_runtime_payload(runtime),
        "strategy": cfg.asdict(),
        "input": str(args.input),
        "model": str(model_path),
        "decision_threshold": float(decision_threshold),
        "starting_cash": float(args.start_cash),
        "rows": len(rows),
        "market": runtime.market_type,
        "symbol": runtime.symbol,
        "result": {
            "trades": int(result.trades),
            "win_rate": float(result.win_rate),
            "realized_pnl": float(result.realized_pnl),
            "fees": float(result.total_fees),
            "max_exposure": float(result.max_exposure),
            "ending_cash": float(result.ending_cash),
            "max_drawdown": float(result.max_drawdown),
            "stopped_by_drawdown": bool(result.stopped_by_drawdown),
            "trades_per_day_cap_hit": int(result.trades_per_day_cap_hit),
            "closed_trades": int(result.closed_trades),
            "gross_exposure": float(result.gross_exposure),
            "buy_hold_pnl": float(result.buy_hold_pnl),
            "edge_vs_buy_hold": float(result.edge_vs_buy_hold),
        },
    }
    _persist_run_artifact("backtest", model_path.parent, artifact)

    print(f"backtest BTCUSDC ({runtime.symbol})")
    print(f"market: {runtime.market_type}")
    print(f"trades: {result.trades}")
    print(f"win_rate: {result.win_rate:.2%}")
    print(f"realized_pnl: {result.realized_pnl:.2f}")
    print(f"fees: {result.total_fees:.2f}")
    print(f"max_exposure: {result.max_exposure:.2f}")
    print(f"starting_cash: {result.starting_cash:.2f}")
    print(f"ending_cash: {result.ending_cash:.2f}")
    print(f"buy_hold_pnl: {result.buy_hold_pnl:.2f}")
    print(f"edge_vs_buy_hold: {result.edge_vs_buy_hold:.2f}")
    print(f"max_drawdown: {result.max_drawdown:.2%}")
    print(f"stopped_by_drawdown: {result.stopped_by_drawdown}")
    return 0


def _tune_score(result: object, starting_cash: float = 1000.0) -> float:
    realized = float(getattr(result, "realized_pnl", 0.0))
    total_fees = float(getattr(result, "total_fees", 0.0))
    max_drawdown = float(getattr(result, "max_drawdown", 0.0))
    closed_trades = int(getattr(result, "closed_trades", 0))
    stopped_by_drawdown = bool(getattr(result, "stopped_by_drawdown", False))
    score = realized - total_fees
    score -= max_drawdown * starting_cash
    if stopped_by_drawdown:
        score -= starting_cash * 0.5
    if closed_trades <= 0:
        score -= 50.0
    return score


def command_evaluate(args: argparse.Namespace) -> int:
    cfg = load_strategy()
    runtime = load_runtime()
    candles = _load_rows_for_command(args.input, label="Evaluation data load failed")
    if candles is None:
        return 2
    rows = _build_model_rows(candles, cfg)
    if not rows:
        print("No rows available for evaluation. Fetch more data first.")
        return 2

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Model file not found: {model_path}")
        return 2

    split = temporal_validation_split(rows)
    train_rows = split.train_rows
    calibration_rows = split.calibration_rows
    test_rows = split.validation_rows

    try:
        model = _load_runtime_model(model_path, cfg)
    except (OSError, json.JSONDecodeError, ModelLoadError, ModelFeatureMismatchError) as exc:
        print(f"Model load failed: {exc}", file=sys.stderr)
        return 2
    threshold = model_decision_threshold(model, cfg.signal_threshold)
    if args.threshold is not None:
        threshold = float(args.threshold)
    elif test_rows:
        # make default threshold robust against short samples and class imbalance
        threshold = model_decision_threshold(model, cfg.signal_threshold)

    if args.calibrate_threshold and calibration_rows:
        threshold = calibrate_threshold(calibration_rows, model, start=0.05, end=0.95, steps=31)

    report = evaluate_classification(test_rows if test_rows else rows, model, threshold=threshold)
    train_report = evaluate_classification(train_rows, model, threshold=threshold) if train_rows else None
    artifact = {
        "command": "evaluate",
        "timestamp": int(time.time()),
        "runtime": _public_runtime_payload(runtime),
        "strategy": cfg.asdict(),
        "input": str(args.input),
        "model": str(args.model),
        "market": runtime.market_type,
        "symbol": runtime.symbol,
        "split": {
            "train_rows": len(train_rows),
            "calibration_rows": len(calibration_rows),
            "test_rows": len(test_rows),
        },
        "threshold": float(report.threshold),
        "calibrated": bool(args.calibrate_threshold),
        "rows": {
            "train": {
                "accuracy": float(train_report.accuracy) if train_report is not None else 0.0,
                "precision": float(train_report.precision) if train_report is not None else 0.0,
                "recall": float(train_report.recall) if train_report is not None else 0.0,
                "f1": float(train_report.f1) if train_report is not None else 0.0,
                "true_positive": int(train_report.true_positive) if train_report is not None else 0,
                "false_positive": int(train_report.false_positive) if train_report is not None else 0,
                "true_negative": int(train_report.true_negative) if train_report is not None else 0,
                "false_negative": int(train_report.false_negative) if train_report is not None else 0,
            },
            "test": {
                "accuracy": float(report.accuracy),
                "precision": float(report.precision),
                "recall": float(report.recall),
                "f1": float(report.f1),
                "true_positive": int(report.true_positive),
                "false_positive": int(report.false_positive),
                "true_negative": int(report.true_negative),
                "false_negative": int(report.false_negative),
            },
        },
    }
    _persist_run_artifact("evaluate", Path(args.model).parent, artifact)

    print(f"evaluate model={model_path}")
    print(f"threshold: {report.threshold:.3f}")
    print(
        "train_accuracy: "
        f"{train_report.accuracy:.3f} precision={train_report.precision:.3f} "
        f"recall={train_report.recall:.3f} f1={train_report.f1:.3f}"
    )
    print(
        "test_accuracy: "
        f"{report.accuracy:.3f} precision={report.precision:.3f} "
        f"recall={report.recall:.3f} f1={report.f1:.3f}"
    )
    print(
        "confusion tp={tp} fp={fp} tn={tn} fn={fn}".format(
            tp=report.true_positive, fp=report.false_positive, tn=report.true_negative, fn=report.false_negative
        )
    )
    return 0


def command_live(args: argparse.Namespace) -> int:
    runtime = load_runtime()
    cfg = load_strategy()
    if getattr(args, "paper", False) and getattr(args, "live", False):
        print("Choose either --paper or --live, not both.")
        return 2
    leverage_override = getattr(args, "leverage", None)
    if leverage_override is not None:
        requested = max(1.0, leverage_override)
        if runtime.market_type == "futures":
            cfg = StrategyConfig(**{**cfg.asdict(), "leverage": requested})
        else:
            print("Leverage override is spot-inactive; spot runs at 1x.")
    client = _build_client(runtime)
    model_path = Path(getattr(args, "model", "data/model.json"))

    if getattr(args, "live", False):
        effective_dry_run = False
    else:
        effective_dry_run = runtime.dry_run or args.paper
    if not runtime.testnet:
        print("Real-money execution is disabled in this phase. Set testnet=true to run.")
        return 2
    if not effective_dry_run and (not runtime.api_key or not runtime.api_secret):
        print("Live mode needs API key and secret. Run configure first or run with --paper.")
        return 2
    sleep_seconds = max(0, int(getattr(args, "sleep", 0)))
    if not effective_dry_run and sleep_seconds < 1:
        print("Authenticated live mode uses minimum sleep=1s.")
        sleep_seconds = 1

    leverage = _resolve_futures_leverage(runtime, cfg)
    if runtime.market_type == "futures" and not effective_dry_run:
        try:
            set_response = client.set_leverage(runtime.symbol, int(leverage))
            if isinstance(set_response, dict) and set_response.get("leverage"):
                leverage = float(set_response.get("leverage"))
        except BinanceAPIError as exc:
            print(f"Failed to set leverage: {exc}", file=sys.stderr)
            return 2

    cash = max(0.0, _safe_float(getattr(runtime, "managed_usdc", 1000.0)))
    position_notional = 0.0
    position_side = 0
    entry_price = 0.0
    margin_used = 0.0
    qty = 0.0
    wait_ticks = cfg.cooldown_minutes
    cooldown_left = 0
    if leverage > 125.0:
        leverage = 125.0
    elif leverage < 1.0:
        leverage = 1.0
    equity_peak = cash
    max_drawdown_seen = 0.0

    if model_path.exists():
        try:
            model = _load_runtime_model(model_path, cfg)
        except (ModelLoadError, ModelFeatureMismatchError) as exc:
            if not effective_dry_run:
                print(f"Live mode requires a compatible model: {exc}", file=sys.stderr)
                return 2
            print(f"Model load failed; regenerating: {exc}", file=sys.stderr)
            model = None
        except Exception:
            if not effective_dry_run:
                print(f"Live mode requires a readable model: {model_path}", file=sys.stderr)
                return 2
            model = None
    else:
        if not effective_dry_run:
            print(f"Live mode needs model file: {model_path}", file=sys.stderr)
            return 2
        model = None

    mode_label = "paper" if effective_dry_run else "live"
    print(f"Starting {mode_label} loop for {args.steps} steps on {runtime.symbol} {runtime.interval} [{runtime.market_type}]")
    if runtime.market_type == "futures":
        print(f"effective leverage: {leverage:.1f}x")

    fee_rate = max(0.0, cfg.taker_fee_bps) / 10_000.0
    slippage = max(0.0, cfg.slippage_bps) / 10_000.0
    constraints = _resolve_symbol_constraints(runtime, client)
    max_daily_trades = int(cfg.max_trades_per_day)
    if max_daily_trades <= 0:
        max_daily_trades = 0
    daily_trade_count: dict[int, int] = {}
    max_open_positions = int(cfg.max_open_positions)
    live_run = {
        "command": "live",
        "timestamp": int(time.time()),
        "runtime": _public_runtime_payload(runtime),
        "strategy": cfg.asdict(),
        "steps_total": int(args.steps),
        "market": runtime.market_type,
        "symbol": runtime.symbol,
        "model_path": str(model_path),
        "events": [],
        "model_signature": str(getattr(model, "feature_signature", "")) or None,
        "starting_cash": float(cash),
    }
    if not effective_dry_run:
        try:
            detected_position = _detect_existing_position(runtime, client, leverage=leverage)
        except BinanceAPIError as exc:
            print(f"Existing position check failed: {exc}", file=sys.stderr)
            return 2
        if detected_position is not None:
            position_side = int(detected_position["side"])
            qty = float(detected_position["qty"])
            entry_price = float(detected_position["entry_price"])
            position_notional = position_side * float(detected_position["notional"])
            margin_used = max(0.0, float(detected_position["margin"]))
            cash = max(0.0, cash - margin_used)
            print(
                "Detected existing exchange position: "
                f"{'long' if position_side > 0 else 'short'} qty={qty:.8f} entry={entry_price:.2f}"
            )
            live_run["events"].append(
                {
                    "step": 0,
                    "status": "resume_exchange_position",
                    "market": str(detected_position["market"]),
                    "direction": int(position_side),
                    "qty": float(qty),
                    "entry_price": float(entry_price),
                    "notional": float(position_notional),
                    "margin": float(margin_used),
                    "cash_after_resume": float(cash),
                }
            )
    drawdown_limit = float(cfg.max_drawdown_limit)
    halt_reason = "completed"
    steps_executed = 0
    entries = 0
    closes = 0
    skipped_entries = 0
    model_loads = 0 if model is None else 1
    exit_code = 0

    def record_order_error(step: int, side: str, size: float, exc: BinanceAPIError) -> None:
        nonlocal halt_reason, exit_code
        print(f"order error: {exc}", file=sys.stderr)
        halt_reason = "order_error"
        exit_code = 2
        live_run["events"].append(
            {
                "step": step,
                "status": "order_error",
                "side": side,
                "size": float(size),
                "error": str(exc),
            }
        )

    for i in range(args.steps):
        try:
            candles = client.get_klines(runtime.symbol, runtime.interval, limit=max(cfg.model_lookback, 300))
        except BinanceAPIError as exc:
            print(f"market error: {exc}", file=sys.stderr)
            halt_reason = "market_error"
            exit_code = 2
            live_run["events"].append({"step": i + 1, "status": "market_error", "error": str(exc)})
            break

        steps_executed += 1

        rows = _build_model_rows(candles, cfg)
        if not rows:
            print("not enough historical data for live signal")
            live_run["events"].append({"step": i + 1, "status": "no_rows"})
            time.sleep(sleep_seconds)
            continue

        live_run["events"].append({"step": i + 1, "status": "rows", "count": len(rows)})

        retrain_interval = getattr(args, "retrain_interval", 0)
        retrain_window = getattr(args, "retrain_window", 300)
        retrain_min_rows = getattr(args, "retrain_min_rows", 240)
        if retrain_min_rows <= 0:
            retrain_min_rows = max(1, 240)
        if retrain_window <= 0:
            retrain_window = max(1, 300)
        if retrain_interval < 0:
            retrain_interval = 0

        previous_model = model
        model = _build_live_model(
            rows,
            model=model,
            retrain_every=retrain_interval,
            step=i + 1,
            cfg=cfg,
            retrain_window=retrain_window,
            retrain_min_rows=retrain_min_rows,
        )
        if previous_model is None and model is not None:
            model_loads += 1
            if model is not None and model.__class__.__name__ != "TrainedModel":
                model_signature = None
            else:
                model_signature = getattr(model, "feature_signature", None)
                live_run["model_signature"] = str(model_signature) if model_signature else None
        elif previous_model is not None and model is not None and previous_model is not model:
            model_loads += 1
            model_signature = getattr(model, "feature_signature", None)
            live_run["model_signature"] = str(model_signature) if model_signature else None

        if model is None:
            model = train(rows, epochs=40, feature_signature=_strategy_feature_signature(cfg))
            model_loads += 1
        latest = rows[-1]
        try:
            raw_score = model.predict_proba(latest.features)
            threshold = model_decision_threshold(model, cfg.signal_threshold)
            score = confidence_adjusted_probability(raw_score, cfg.confidence_beta)
        except ValueError as exc:
            if not effective_dry_run:
                print(f"Live model incompatible with current rows: {exc}", file=sys.stderr)
                halt_reason = "model_incompatible"
                exit_code = 2
                live_run["events"].append({"step": i + 1, "status": "model_incompatible", "error": str(exc)})
                break
            print(f"paper model incompatible; retraining: {exc}", file=sys.stderr)
            model = train(rows, epochs=40, feature_signature=_strategy_feature_signature(cfg))
            model_loads += 1
            raw_score = model.predict_proba(latest.features)
            threshold = model_decision_threshold(model, cfg.signal_threshold)
            score = confidence_adjusted_probability(raw_score, cfg.confidence_beta)
        direction = _score_to_direction(score, cfg, runtime.market_type, threshold)

        # cooldown reduces immediate flip-flopping in choppy conditions
        if cooldown_left > 0:
            if runtime.market_type == "spot":
                direction = 0
            cooldown_left -= 1

        price = latest.close
        day = _safe_day_ms(latest.timestamp)
        trade_cap_reached = max_daily_trades > 0 and daily_trade_count.get(day, 0) >= max_daily_trades
        if position_side == 0 and direction != 0:
            if trade_cap_reached:
                print(f"step {i + 1:>2}: trade cap reached ({max_daily_trades}/day), skipping entry")
                skipped_entries += 1
                live_run["events"].append(
                    {
                        "step": i + 1,
                        "status": "skip_trade_cap",
                        "day": day,
                        "score": float(score),
                    }
                )
                time.sleep(sleep_seconds)
                continue
            if max_open_positions <= 0:
                print(f"step {i + 1:>2}: max open positions reached ({max_open_positions}), skipping entry")
                skipped_entries += 1
                live_run["events"].append(
                    {
                        "step": i + 1,
                        "status": "skip_max_open_positions",
                        "score": float(score),
                    }
                )
                time.sleep(sleep_seconds)
                continue

            notional, qty = _build_order_notional(
                cash,
                price,
                cfg,
                runtime.market_type,
                leverage,
                client,
                constraints=constraints,
            )
            if notional <= 0:
                print(f"step {i + 1:>2}: skipped entry due to order constraints")
                skipped_entries += 1
                live_run["events"].append(
                    {
                        "step": i + 1,
                        "status": "skip_constraints",
                        "score": float(score),
                        "notional": float(notional),
                    }
                )
                time.sleep(sleep_seconds)
                continue

            if runtime.market_type == "spot":
                margin = min(cash, abs(notional))
            else:
                margin = min(cash, abs(notional) / leverage)

            fee = abs(notional) * fee_rate
            total = margin + fee
            if total > cash:
                print(f"step {i + 1:>2}: insufficient cash for leverage-adjusted entry")
                skipped_entries += 1
                live_run["events"].append(
                    {
                        "step": i + 1,
                        "status": "skip_insufficient_cash_pre_fill",
                        "score": float(score),
                    }
                )
                time.sleep(sleep_seconds)
                continue

            side_sign = 1 if direction > 0 else -1
            fill = price * (1.0 + side_sign * slippage)
            if fill <= 0:
                time.sleep(sleep_seconds)
                continue

            notional = qty * fill
            fee = abs(notional) * fee_rate
            total = margin + fee
            if total > cash:
                print(f"step {i + 1:>2}: insufficient cash after fill adjustment")
                skipped_entries += 1
                live_run["events"].append(
                    {
                        "step": i + 1,
                        "status": "skip_insufficient_cash_after_fill",
                        "score": float(score),
                        "fill": float(fill),
                    }
                )
                time.sleep(sleep_seconds)
                continue

            side = "BUY" if side_sign > 0 else "SELL"
            try:
                _paper_or_live_order(
                    client,
                    runtime,
                    cfg,
                    side=side,
                    size=qty,
                    dry_run=effective_dry_run,
                    leverage=leverage,
                )
            except BinanceAPIError as exc:
                record_order_error(i + 1, side, qty, exc)
                break

            cash -= total
            position_side = direction
            position_notional = direction * notional
            qty = abs(qty)
            entry_price = fill
            margin_used = margin
            daily_trade_count[day] = daily_trade_count.get(day, 0) + 1

            print(f"step {i + 1:>2}: enter {'long' if position_side > 0 else 'short'} at {fill:.2f} qty={qty:.6f}")
            entries += 1
            live_run["events"].append(
                {
                    "step": i + 1,
                    "status": "enter",
                    "direction": int(position_side),
                    "score": float(score),
                    "price": float(fill),
                    "qty": float(qty),
                    "notional": float(notional),
                    "cash_after_entry": float(cash),
                }
            )
            cooldown_left = 0

        elif position_side != 0:
            pnl = position_side * (price - entry_price) * qty
            pnl_pct = ((price - entry_price) / entry_price) if position_side > 0 else ((entry_price - price) / entry_price)

            opposite_signal = direction != 0 and direction != position_side if runtime.market_type == "futures" else direction == 0
            should_close = opposite_signal or pnl_pct >= cfg.take_profit_pct or pnl_pct <= -cfg.stop_loss_pct

            if should_close:
                fill = price * (1.0 - position_side * slippage)
                realized = position_side * (fill - entry_price) * qty
                exit_fee = abs(fill * qty) * fee_rate

                side_to_close = "SELL" if position_side > 0 else "BUY"
                try:
                    _paper_or_live_order(
                        client,
                        runtime,
                        cfg,
                        side=side_to_close,
                        size=abs(qty),
                        dry_run=effective_dry_run,
                        leverage=leverage,
                        reduce_only=runtime.market_type == "futures",
                    )
                except BinanceAPIError as exc:
                    record_order_error(i + 1, side_to_close, abs(qty), exc)
                    break
                cash += margin_used + realized - exit_fee
                print(
                    f"step {i + 1:>2}: close {'long' if position_side > 0 else 'short'} "
                    f"pnl={pnl:.2f} cash={cash:.2f}"
                )
                closes += 1
                live_run["events"].append(
                    {
                        "step": i + 1,
                        "status": "close",
                        "direction": int(position_side),
                        "score": float(score),
                        "price": float(price),
                        "pnl": float(realized),
                        "cash_after": float(cash),
                    }
                )
                if realized > 0:
                    print("result: win")
                position_notional = 0.0
                position_side = 0
                qty = 0.0
                margin_used = 0.0
                entry_price = 0.0
                cooldown_left = max(0, wait_ticks)
            else:
                unrealized = margin_used + pnl
                print(f"step {i + 1:>2}: hold {'long' if position_side > 0 else 'short'} pnl={pnl_pct:.2%} cash={cash:.2f}")
                print(f"         unrealized equity={cash + unrealized:.2f}")
                live_run["events"].append(
                    {
                        "step": i + 1,
                        "status": "hold",
                        "direction": int(position_side),
                        "score": float(score),
                        "price": float(price),
                        "pnl": float(pnl),
                    }
                )

                if direction == 0:
                    cooldown_left = max(0, wait_ticks)

            # safety stop: drop out if drawdown exceeds configured cap
            if position_side != 0:
                equity = cash + margin_used + pnl
            else:
                equity = cash
            if equity > equity_peak:
                equity_peak = equity
            drawdown = (equity_peak - equity) / equity_peak if equity_peak else 0.0
            if drawdown > max_drawdown_seen:
                max_drawdown_seen = drawdown
            if cfg.max_drawdown_limit > 0.0 and drawdown >= cfg.max_drawdown_limit:
                if position_side != 0:
                    fill = price * (1.0 - position_side * slippage)
                    realized = position_side * (fill - entry_price) * qty
                    exit_fee = abs(fill * qty) * fee_rate

                    side_to_close = "SELL" if position_side > 0 else "BUY"
                    try:
                        _paper_or_live_order(
                            client,
                            runtime,
                            cfg,
                            side=side_to_close,
                            size=abs(qty),
                            dry_run=effective_dry_run,
                            leverage=leverage,
                            reduce_only=runtime.market_type == "futures",
                        )
                    except BinanceAPIError as exc:
                        record_order_error(i + 1, side_to_close, abs(qty), exc)
                        break
                    cash += margin_used + realized - exit_fee
                    print(
                        f"step {i + 1:>2}: emergency close from drawdown "
                        f"{drawdown:.2%}; cash={cash:.2f}"
                    )
                    position_notional = 0.0
                    position_side = 0
                    qty = 0.0
                    margin_used = 0.0
                    entry_price = 0.0
                live_run["events"].append(
                    {
                        "step": i + 1,
                        "status": "emergency_close",
                        "score": float(score),
                        "drawdown": float(drawdown),
                        "cash_after": float(cash),
                    }
                )
                print(f"step {i + 1:>2}: drawdown limit reached ({cfg.max_drawdown_limit:.1%}), stopping loop")
                halt_reason = "drawdown_limit"
                break

        if position_side == 0 and direction != 0:
            live_run["events"].append(
                {
                    "step": i + 1,
                    "status": "signal_no_entry",
                    "score": float(score),
                    "direction": int(direction),
                }
            )

        time.sleep(sleep_seconds)
    live_run["result"] = {
        "status": halt_reason,
        "steps_executed": steps_executed,
        "entries": entries,
        "closes": closes,
        "skipped_entries": skipped_entries,
        "model_loads": model_loads,
        "drawdown_seen": float(max_drawdown_seen),
        "ending_cash": float(cash),
        "equity_peak": float(equity_peak),
        "drawdown_limit": drawdown_limit,
    }
    _persist_run_artifact("live", model_path.parent, live_run)
    if max_drawdown_seen > 0.0:
        print(f"max_drawdown observed: {max_drawdown_seen:.2%}")
    print(f"finished loop market={runtime.market_type} cash={cash:.2f}")
    return exit_code


def command_shell(_: argparse.Namespace) -> int:
    from .shell import run_shell

    return int(run_shell([]))


def command_objectives(_: argparse.Namespace) -> int:
    from .objective import describe_objectives

    entries = describe_objectives()
    print(f"{'name':<14} {'label':<14} summary")
    for entry in entries:
        print(f"{entry['name']:<14} {entry['label']:<14} {entry['summary']}")
    return 0


def command_train_suite(args: argparse.Namespace) -> int:
    from .api import Candle
    from .objective import available_objectives
    from .training_suite import run_training_suite

    runtime = load_runtime()
    strategy = load_strategy()
    try:
        raw = _load_json_candles(args.input)
    except (OSError, ValueError) as err:
        print(f"failed to load candles from {args.input}: {err}", file=sys.stderr)
        return 2
    candles = []
    for entry in raw:
        try:
            candles.append(Candle(
                open_time=int(entry["open_time"]),
                open=float(entry["open"]),
                high=float(entry["high"]),
                low=float(entry["low"]),
                close=float(entry["close"]),
                volume=float(entry.get("volume", 0.0)),
                close_time=int(entry["close_time"]),
            ))
        except (KeyError, TypeError, ValueError):
            continue
    objectives = tuple(args.objective) if args.objective else available_objectives()
    report = run_training_suite(
        candles,
        strategy,
        objectives=objectives,
        market_type=runtime.market_type,
        starting_cash=args.starting_cash,
        output_dir=Path(args.output_dir),
    )
    print(f"training suite complete: {len(report.outcomes)} objective(s)")
    for outcome in report.outcomes:
        print(
            f"  {outcome.objective:<14} score={outcome.best_score:+.4f} "
            f"model={outcome.model_path}"
        )
    print(f"summary -> {report.summary_path}")
    return 0


def command_backtest_panel(args: argparse.Namespace) -> int:
    from .backtest_panel import BacktestRequest, parse_date_ms, run_panel

    runtime = load_runtime()
    strategy = load_strategy()
    market = args.market or runtime.market_type
    try:
        request = BacktestRequest(
            interval=args.interval,
            market_type=market,
            start_ms=parse_date_ms(args.from_date),
            end_ms=parse_date_ms(args.to_date, end_of_day=True),
            model_path=args.model,
            data_path=args.input,
            starting_cash=args.starting_cash,
            objective=args.objective,
            tag=args.tag,
            notes=args.notes,
        )
        report = run_panel(request, strategy)
    except ValueError as err:
        print(f"backtest-panel: {err}", file=sys.stderr)
        return 2
    print(f"backtest report -> data/backtests/{report.filename}")
    print(
        f"  trades={report.result.closed_trades} "
        f"realized_pnl={report.result.realized_pnl:+.2f} "
        f"max_dd={report.result.max_drawdown:.2%}"
    )
    if report.objective_score is not None:
        print(f"  objective={args.objective} score={report.objective_score:+.4f}")
    return 0


def command_autonomous(args: argparse.Namespace) -> int:
    from .autonomous import (
        STATE_PAUSED,
        STATE_RUNNING,
        STATE_STOPPING,
        AutonomousControl,
    )
    from .objective import available_objectives

    control = AutonomousControl()
    action = args.action
    if action == "start":
        if args.objective not in available_objectives():
            print(f"unknown objective {args.objective!r}", file=sys.stderr)
            return 2
        control.write(STATE_RUNNING, note=f"CLI start objective={args.objective}")
        print(f"autonomous: RUNNING (objective={args.objective})")
        return 0
    if action == "pause":
        control.write(STATE_PAUSED, note="CLI pause")
        print("autonomous: PAUSED")
        return 0
    if action == "resume":
        control.write(STATE_RUNNING, note="CLI resume")
        print("autonomous: RUNNING")
        return 0
    if action == "stop":
        control.write(STATE_STOPPING, note="CLI stop")
        print("autonomous: STOPPING")
        return 0
    # status
    payload = control.read()
    print(
        f"state={payload.get('state')} note={payload.get('note') or ''} "
        f"ts_ms={payload.get('ts_ms')}"
    )
    return 0


def command_positions(args: argparse.Namespace) -> int:
    from .positions import (
        PositionsStore,
        compute_stats,
        render_positions_table,
        render_stats_lines,
    )

    store = PositionsStore()
    opens = store.load_open()
    if not opens:
        print("(no open positions)")
    else:
        for row in render_positions_table(opens, mark_price=None):
            print(row)
    if args.stats:
        stats = compute_stats(store, mark_price=None)
        print("")
        for line in render_stats_lines(stats):
            print(line)
    return 0


def command_close(args: argparse.Namespace) -> int:
    from .positions import PositionsStore

    store = PositionsStore()
    target = args.position_id
    if target.lower() == "all":
        opens = store.load_open()
        for position in opens:
            store.remove_open(position.id)
        print(f"closed {len(opens)} positions (local ledger only)")
        return 0
    if store.remove_open(target):
        print(f"closed {target} (local ledger only)")
        return 0
    print(f"no open position with id {target!r}", file=sys.stderr)
    return 1


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if not argv:
        return command_menu(argparse.Namespace())
    args = _parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
