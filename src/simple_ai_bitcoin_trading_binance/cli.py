"""Entry point for the BTCUSDC test-trading CLI."""

from __future__ import annotations

import argparse
import builtins
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import random
import subprocess  # nosec B404
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence, TypeVar, cast

from .api import BinanceAPIError, BinanceClient, Candle
from .advanced_model import (
    AdvancedFeatureConfig,
    advanced_feature_signature,
    default_config_for,
    make_advanced_inference_rows,
    make_advanced_rows,
)
from .backtest import calibrate_threshold_for_backtest, risk_adjusted_backtest_score, run_backtest
from .config import config_paths, load_runtime, load_strategy, prompt_runtime, save_runtime, save_strategy
from .dashboard import DashboardSnapshot, load_artifact_preview, render_dashboard
from . import data_workflows
from .data_downloader import MarketDataSyncConfig, render_sync_result, sync_market_data
from .features import FEATURE_NAMES, ModelRow, feature_signature, make_inference_rows, make_rows, normalize_enabled_features
from .api import SymbolConstraints
from .external_signals import ExternalSignalReport, collect_external_signals, render_external_signal_report
from .live_artifacts import build_live_run_payload
from .market_data import clean_candles
from .model import (
    assess_probability_calibration,
    build_model_quality_report,
    calibrate_threshold,
    calibrate_probability_temperature,
    confidence_adjusted_probability,
    evaluate_classification,
    evaluate,
    feature_drift_report,
    ModelFeatureMismatchError,
    ModelLoadError,
    load_model,
    model_decision_threshold,
    serialize_model,
    temporal_validation_split,
    train,
    TrainedModel,
    walk_forward_report,
)
from .objective import available_objectives
from .risk_controls import EntryRiskDecision, assess_entry_risk, build_risk_policy_report, render_risk_policy_report
from . import risk_workflows
from .strategy_overrides import apply_model_strategy_overrides
from .storage import write_json_atomic
from .source_grading import grade_sources, render_source_grade_run
from .style import supports_ansi_terminal
from .types import RuntimeConfig, StrategyConfig

_JITTER_RANDOM = random.SystemRandom()


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
_T = TypeVar("_T")


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
        "external_signals_enabled": True,
        "external_signal_max_adjustment": 0.03,
        "external_signal_min_providers": 2,
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
        "external_signals_enabled": True,
        "external_signal_max_adjustment": 0.04,
        "external_signal_min_providers": 2,
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
        "external_signals_enabled": True,
        "external_signal_max_adjustment": 0.05,
        "external_signal_min_providers": 2,
    },
}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="simple-ai-trading",
        description="BTCUSDC non-mainnet trading CLI for Binance (spot + futures).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    parser_config = subparsers.add_parser("configure", help="configure Binance credentials and defaults")
    parser_config.set_defaults(func=command_configure)

    parser_connect = subparsers.add_parser("connect", help="validate credentials and connectivity")
    parser_connect.set_defaults(func=command_connect)

    parser_roundtrip = subparsers.add_parser(
        "spot-roundtrip",
        help="place a tiny signed spot testnet/demo roundtrip order with balance and filter prechecks",
    )
    parser_roundtrip.add_argument("--quantity", type=float, default=0.00008, help="BTC quantity to test")
    parser_roundtrip.add_argument(
        "--mode",
        choices=["auto", "buy-sell", "sell-buy"],
        default="auto",
        help="order sequence; auto uses buy-sell when USDC is available, otherwise sell-buy when BTC is available",
    )
    parser_roundtrip.add_argument("--yes", action="store_true", help="confirm signed testnet/demo order placement")
    parser_roundtrip.set_defaults(func=command_spot_roundtrip)

    parser_doctor = subparsers.add_parser(
        "doctor",
        help="run local readiness checks before paper or non-mainnet trading",
    )
    parser_doctor.add_argument("--input", default="data/historical_btcusdc.json")
    parser_doctor.add_argument("--model", default="data/model.json")
    parser_doctor.add_argument("--online", action="store_true", help="also check exchange connectivity")
    parser_doctor.set_defaults(func=command_doctor)

    parser_audit = subparsers.add_parser("audit", help="run local data/model/risk diagnostics without network calls")
    parser_audit.add_argument("--input", default="data/historical_btcusdc.json")
    parser_audit.add_argument("--model", default="data/model.json")
    parser_audit.set_defaults(func=command_audit)

    parser_risk = subparsers.add_parser("risk", help="show local risk policy before paper or live trading")
    parser_risk.add_argument("--model", default="data/model.json")
    parser_risk.add_argument("--paper", action="store_true", help="assess paper/dry-run execution")
    parser_risk.add_argument("--live", action="store_true", help="assess authenticated testnet/demo execution")
    parser_risk.add_argument("--leverage", type=float, default=None, help="optional futures leverage override")
    parser_risk.add_argument("--json", action="store_true")
    parser_risk.set_defaults(func=command_risk)

    parser_report = subparsers.add_parser("report", help="show dashboard, artifacts, and optional readiness checks")
    parser_report.add_argument("--account", action="store_true", help="include authenticated account state")
    parser_report.add_argument("--doctor", action="store_true", help="include readiness checks")
    parser_report.add_argument("--no-doctor", action="store_false", dest="doctor", help="omit readiness checks")
    parser_report.add_argument("--online", action="store_true", help="include exchange connectivity in readiness checks")
    parser_report.add_argument("--input", default="data/historical_btcusdc.json")
    parser_report.add_argument("--model", default="data/model.json")
    parser_report.set_defaults(doctor=True)
    parser_report.set_defaults(func=command_report)

    parser_menu = subparsers.add_parser("menu", help="launch the full-screen operator console")
    parser_menu.set_defaults(func=command_menu)

    parser_fetch = subparsers.add_parser("fetch", help="download BTCUSDC klines")
    parser_fetch.add_argument("--symbol", default=None)
    parser_fetch.add_argument("--interval", default=None)
    parser_fetch.add_argument("--limit", type=int, default=500)
    parser_fetch.add_argument("--batch-size", type=int, default=1000, help="klines per request (spot max 1000, futures max 1500)")
    parser_fetch.add_argument("--output", default="data/historical_btcusdc.json")
    parser_fetch.set_defaults(func=command_fetch)

    parser_data_sync = subparsers.add_parser(
        "data-sync",
        help="rate-limited Binance downloader for candles and auxiliary metrics into SQLite",
    )
    parser_data_sync.add_argument("--db", default="data/market_data.sqlite")
    parser_data_sync.add_argument("--symbol", default=None)
    parser_data_sync.add_argument("--interval", default=None)
    parser_data_sync.add_argument("--market", choices=["spot", "futures"], default=None)
    parser_data_sync.add_argument("--rows", type=int, default=500)
    parser_data_sync.add_argument("--batch-size", type=int, default=1000)
    parser_data_sync.add_argument("--include-futures-metrics", action="store_true", default=True)
    parser_data_sync.add_argument("--no-include-futures-metrics", action="store_false", dest="include_futures_metrics")
    parser_data_sync.add_argument("--loop", action="store_true", help="keep syncing in the foreground")
    parser_data_sync.add_argument("--iterations", type=int, default=1, help="foreground loop iterations; 0 means unlimited")
    parser_data_sync.add_argument("--sleep", type=int, default=300, help="seconds between loop iterations")
    parser_data_sync.add_argument("--background", action="store_true", help="start a detached downloader process")
    parser_data_sync.add_argument("--pid-file", default="data/market_data_sync.pid")
    parser_data_sync.add_argument("--log-file", default="data/market_data_sync.log")
    parser_data_sync.add_argument("--json", action="store_true")
    parser_data_sync.set_defaults(func=command_data_sync)

    parser_train = subparsers.add_parser("train", help="train model from cached candles")
    parser_train.add_argument("--input", default="data/historical_btcusdc.json")
    parser_train.add_argument("--output", default="data/model.json")
    parser_train.add_argument("--source", choices=["auto", "file", "db"], default="auto")
    parser_train.add_argument("--db", default="data/market_data.sqlite")
    parser_train.add_argument("--interval", default=None)
    parser_train.add_argument("--market", choices=["spot", "futures"], default=None)
    parser_train.add_argument("--min-rows", type=int, default=120)
    parser_train.add_argument("--download-missing", action="store_true")
    parser_train.add_argument("--preset", choices=sorted(_TRAINING_PRESETS), default="custom")
    parser_train.add_argument("--epochs", type=int, default=250)
    parser_train.add_argument("--learning-rate", type=float, default=0.05)
    parser_train.add_argument("--l2-penalty", type=float, default=1e-4)
    parser_train.add_argument("--seed", type=int, default=7)
    parser_train.add_argument(
        "--compute-backend",
        choices=_COMPUTE_BACKEND_CHOICES,
        default=None,
        help="training backend override; default uses saved runtime compute_backend",
    )
    parser_train.add_argument("--batch-size", type=int, default=8192, help="mini-batch size for GPU training")
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
    parser_backtest.add_argument(
        "--compute-backend",
        choices=_COMPUTE_BACKEND_CHOICES,
        default=None,
        help="model-scoring backend override; default uses saved runtime compute_backend",
    )
    parser_backtest.add_argument(
        "--score-batch-size",
        type=int,
        default=8192,
        help="batch size for GPU-assisted probability scoring",
    )
    parser_backtest.set_defaults(func=command_backtest)

    parser_evaluate = subparsers.add_parser("evaluate", help="evaluate saved model against cached candles")
    parser_evaluate.add_argument("--input", default="data/historical_btcusdc.json")
    parser_evaluate.add_argument("--model", default="data/model.json")
    parser_evaluate.add_argument("--threshold", type=float, default=None)
    parser_evaluate.add_argument("--calibrate-threshold", action="store_true")
    parser_evaluate.set_defaults(func=command_evaluate)

    parser_signals = subparsers.add_parser(
        "signals",
        help="fetch and cache free external BTC signal checks used by live mode",
    )
    parser_signals.add_argument("--model", default="data/model.json", help="model path used to derive default cache location")
    parser_signals.add_argument("--cache", default=None, help="signal cache path (default: model-adjacent data/signals)")
    parser_signals.add_argument("--ttl", type=int, default=300, help="cache TTL seconds")
    parser_signals.add_argument("--timeout", type=float, default=3.0, help="per-provider timeout seconds")
    parser_signals.add_argument("--max-adjustment", type=float, default=0.04, help="maximum model score adjustment")
    parser_signals.add_argument("--min-providers", type=int, default=2, help="minimum usable providers for positive boosts")
    parser_signals.add_argument(
        "--compute-backend",
        choices=_COMPUTE_BACKEND_CHOICES,
        default=None,
        help="optional backend for news keyword scoring",
    )
    parser_signals.add_argument(
        "--short-reaction-refresh",
        type=int,
        default=30,
        help="seconds after which cached short-horizon reaction news must refresh",
    )
    parser_signals.add_argument("--news-provider-limit", type=int, default=None, help="maximum RSS/news providers to poll")
    parser_signals.add_argument("--news-items-per-provider", type=int, default=None, help="feed items scored per news provider")
    parser_signals.add_argument("--provider-parallelism", type=int, default=None, help="maximum simultaneous news provider requests")
    parser_signals.add_argument("--provider-jitter", type=float, default=None, help="random per-provider delay ceiling in seconds")
    parser_signals.add_argument("--ollama-news", action="store_true", default=None, help="enable Ollama AI headline evaluation")
    parser_signals.add_argument("--no-ollama-news", action="store_false", dest="ollama_news", help="disable Ollama AI headline evaluation")
    parser_signals.add_argument("--ollama-model", default=None)
    parser_signals.add_argument("--ollama-url", default=None)
    parser_signals.add_argument("--ollama-timeout", type=float, default=None)
    parser_signals.add_argument("--telemetry-db", default=None, help="SQLite raw telemetry DB path")
    parser_signals.add_argument("--source-grade-max-age-hours", type=float, default=None, help="ignore source grades older than this; 0 disables the age cap")
    parser_signals.add_argument("--no-telemetry", action="store_true", help="do not journal raw provider/model payloads")
    parser_signals.add_argument("--loop", action="store_true", help="poll repeatedly with jitter instead of one collection")
    parser_signals.add_argument("--iterations", type=int, default=0, help="loop iterations; 0 means until interrupted")
    parser_signals.add_argument("--sleep", type=float, default=None, help="base loop interval seconds")
    parser_signals.add_argument("--jitter", type=float, default=None, help="random loop delay ceiling in seconds")
    parser_signals.add_argument("--refresh", action="store_true", help="ignore cache and fetch every provider")
    parser_signals.add_argument("--json", action="store_true", help="print machine-readable report")
    parser_signals.set_defaults(func=command_signals)

    parser_signals_benchmark = subparsers.add_parser(
        "signals-benchmark",
        help="benchmark provider polling limits, parallelism, and optional Ollama latency",
    )
    parser_signals_benchmark.add_argument("--provider-limit", action="append", type=int, default=None)
    parser_signals_benchmark.add_argument("--parallelism", action="append", type=int, default=None)
    parser_signals_benchmark.add_argument("--iterations", type=int, default=1)
    parser_signals_benchmark.add_argument("--timeout", type=float, default=3.0)
    parser_signals_benchmark.add_argument("--provider-jitter", type=float, default=0.0)
    parser_signals_benchmark.add_argument("--ollama-news", action="store_true", default=None)
    parser_signals_benchmark.add_argument("--no-ollama-news", action="store_false", dest="ollama_news")
    parser_signals_benchmark.add_argument("--ollama-model", default=None)
    parser_signals_benchmark.add_argument("--ollama-url", default=None)
    parser_signals_benchmark.add_argument("--ollama-timeout", type=float, default=None)
    parser_signals_benchmark.add_argument("--cache", default="data/signals/benchmark_external_signals.json")
    parser_signals_benchmark.add_argument("--no-telemetry", action="store_true")
    parser_signals_benchmark.add_argument("--json", action="store_true")
    parser_signals_benchmark.set_defaults(func=command_signals_benchmark)

    parser_source_grades = subparsers.add_parser(
        "source-grades",
        help="grade raw signal/news/model sources from telemetry with optional Ollama review",
    )
    parser_source_grades.add_argument("--db", default=None, help="SQLite raw telemetry DB path")
    parser_source_grades.add_argument("--window-hours", type=float, default=None)
    parser_source_grades.add_argument("--ollama", action="store_true", default=None, help="enable Ollama grading")
    parser_source_grades.add_argument("--no-ollama", action="store_false", dest="ollama", help="disable Ollama grading")
    parser_source_grades.add_argument("--ollama-model", default=None)
    parser_source_grades.add_argument("--ollama-url", default=None)
    parser_source_grades.add_argument("--ollama-timeout", type=float, default=None)
    parser_source_grades.add_argument("--json", action="store_true")
    parser_source_grades.set_defaults(func=command_source_grades)

    parser_live = subparsers.add_parser("live", help="run a conservative live loop on testnet/demo or paper mode")
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
    parser_live.add_argument(
        "--external-signals",
        action="store_true",
        default=None,
        help="enable cached free external signal adjustment for this run",
    )
    parser_live.add_argument(
        "--no-external-signals",
        action="store_false",
        dest="external_signals",
        help="disable cached free external signal adjustment for this run",
    )
    parser_live.set_defaults(func=command_live)

    parser_status = subparsers.add_parser("status", help="show persisted runtime and strategy config")
    parser_status.set_defaults(func=command_status)

    parser_compute = subparsers.add_parser("compute", help="show or set the model-training compute backend")
    parser_compute.add_argument("--backend", choices=_COMPUTE_BACKEND_CHOICES, default=None)
    parser_compute.set_defaults(func=command_compute)

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
    parser_strategy.add_argument("--external-signals", action="store_true", default=None, help="enable live free external signals")
    parser_strategy.add_argument("--no-external-signals", action="store_false", dest="external_signals", help="disable live free external signals")
    parser_strategy.add_argument("--external-signal-max-adjustment", type=float, default=None)
    parser_strategy.add_argument("--external-signal-min-providers", type=int, default=None)
    parser_strategy.add_argument("--external-signal-ttl", type=int, default=None)
    parser_strategy.add_argument("--external-signal-timeout", type=float, default=None)
    parser_strategy.add_argument("--external-news-ai", action="store_true", default=None)
    parser_strategy.add_argument("--no-external-news-ai", action="store_false", dest="external_news_ai")
    parser_strategy.add_argument("--external-news-ai-model", default=None)
    parser_strategy.add_argument("--external-news-ai-url", default=None)
    parser_strategy.add_argument("--external-news-ai-timeout", type=float, default=None)
    parser_strategy.add_argument("--external-news-provider-limit", type=int, default=None)
    parser_strategy.add_argument("--external-provider-parallelism", type=int, default=None)
    parser_strategy.add_argument("--external-provider-jitter", type=float, default=None)
    parser_strategy.add_argument("--external-poll-jitter", type=float, default=None)
    parser_strategy.add_argument("--telemetry-db", default=None)
    parser_strategy.add_argument("--no-telemetry", action="store_true", default=None)
    parser_strategy.add_argument("--source-grading", action="store_true", default=None)
    parser_strategy.add_argument("--no-source-grading", action="store_false", dest="source_grading")
    parser_strategy.add_argument("--source-grading-interval", type=int, default=None)
    parser_strategy.add_argument("--source-grading-window-hours", type=int, default=None)
    parser_strategy.add_argument("--source-grade-max-age-hours", type=float, default=None)
    parser_strategy.set_defaults(func=command_strategy)

    parser_shell = subparsers.add_parser("shell", help="launch the fallback-friendly slash-command shell")
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
    parser_train_suite.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="parallel candidate workers; defaults to available CPU cores",
    )
    parser_train_suite.add_argument(
        "--compute-backend",
        choices=_COMPUTE_BACKEND_CHOICES,
        default=None,
        help="training backend override; GPU backends run candidates sequentially to protect VRAM",
    )
    parser_train_suite.add_argument("--batch-size", type=int, default=8192, help="mini-batch size for GPU training")
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
        "autonomous", help="control the autonomous non-mainnet loop (start/pause/resume/stop/status)",
    )
    parser_autonomous.add_argument(
        "action", choices=["start", "pause", "resume", "stop", "status"],
        help="autonomous action to perform",
    )
    parser_autonomous.add_argument("--objective", default="default")
    parser_autonomous.add_argument("--model", default="data/model.json", help="model artifact used for autonomous decisions")
    parser_autonomous.add_argument("--poll-seconds", type=float, default=30.0, help="seconds between autonomous iterations")
    parser_autonomous.add_argument("--iterations", type=int, default=None, help="stop after N iterations; default runs until stopped")
    parser_autonomous.add_argument("--heartbeat-every", type=int, default=1, help="write heartbeat every N iterations")
    parser_autonomous.add_argument("--starting-cash", type=float, default=1000.0, help="reference cash for local autonomous risk stats")
    parser_autonomous.add_argument("--paper", action="store_true", default=False, help="force autonomous paper mode")
    parser_autonomous.add_argument("--live", action="store_true", default=False, help="force authenticated non-mainnet autonomous mode")
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
        demo=getattr(runtime, "demo", False),
        market_type=runtime.market_type,
        max_calls_per_minute=runtime.max_rate_calls_per_minute,
        recv_window_ms=getattr(runtime, "recv_window_ms", 5000),
    )


def _runtime_environment(runtime) -> str:
    if getattr(runtime, "demo", False):
        return "demo"
    return "testnet" if runtime.testnet else "mainnet"


def _allows_signed_execution(runtime) -> bool:
    return bool(runtime.testnet or getattr(runtime, "demo", False))


def _has_api_credentials(runtime) -> bool:
    api_key = str(getattr(runtime, "api_key", "") or "").strip()
    api_secret = str(getattr(runtime, "api_secret", "") or "").strip()
    return bool(api_key and api_secret)


def _credential_required_message(action: str) -> str:
    return f"{action} requires Binance API key and secret in Connection settings."


def _credential_failure_message(action: str, exc: Exception) -> str:
    return f"{action} requires valid Binance API credentials: {exc}"


def _credential_fingerprint(runtime) -> str:
    if not _has_api_credentials(runtime):
        return "missing"
    key = str(getattr(runtime, "api_key", "") or "")
    secret = str(getattr(runtime, "api_secret", "") or "")
    return hashlib.sha256(f"{key}\0{secret}".encode("utf-8")).hexdigest()


def _validate_runtime_connection(runtime, client) -> None:
    client.ping()
    client.ensure_btcusdc()
    if _has_api_credentials(runtime):
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
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite.")
    if minimum is not None and value < minimum:
        raise ValueError(f"{label} must be >= {minimum}.")
    if maximum is not None and value > maximum:
        raise ValueError(f"{label} must be <= {maximum}.")
    return value


def _unchanged_form_value(payload: dict[str, str], key: str, current: object) -> bool:
    return payload.get(key, "").strip() == str(current)


def _profile_field_value(
    profile: str,
    payload: dict[str, str],
    key: str,
    current: _T,
    parser: Callable[[str], _T],
) -> _T | None:
    if profile != "custom" and _unchanged_form_value(payload, key, current):
        return None
    return parser(payload[key])


async def _ui_edit_runtime(ui, current: RuntimeConfig) -> RuntimeConfig:
    from .tui import FormField

    payload = await ui.form(
        "Connection settings",
        [
            FormField("market_type", "Market type [spot/futures]", current.market_type),
            FormField("interval", "Kline interval", current.interval),
            FormField("testnet", "Use Binance testnet [yes/no]", "yes" if current.testnet else "no"),
            FormField(
                "demo",
                "Use Binance Demo Trading API [yes/no]",
                "yes" if getattr(current, "demo", False) else "no",
            ),
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
    demo = _parse_form_bool(payload.get("demo", ""), getattr(current, "demo", False))
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
        demo=demo,
        api_key=api_key,
        api_secret=api_secret,
        dry_run=dry_run,
        validate_account=validate_account,
        max_rate_calls_per_minute=max_rate,
        recv_window_ms=recv_window_ms,
        compute_backend=getattr(current, "compute_backend", "cpu"),
        managed_usdc=getattr(current, "managed_usdc", 0.0),
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
            external_signals=None,
            external_signal_max_adjustment=None,
            external_signal_min_providers=None,
            external_signal_ttl=None,
            external_signal_timeout=None,
            external_news_ai=None,
            external_news_ai_model=None,
            external_news_ai_url=None,
            external_news_ai_timeout=None,
            external_news_provider_limit=None,
            external_provider_parallelism=None,
            external_provider_jitter=None,
            external_poll_jitter=None,
            telemetry_db=None,
            no_telemetry=None,
            source_grading=None,
            source_grading_interval=None,
            source_grading_window_hours=None,
            source_grade_max_age_hours=None,
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
            FormField("external_signals", "External signals [yes/no]", str(cfg.external_signals_enabled)),
            FormField("external_signal_max_adjustment", "External max score adjustment", str(cfg.external_signal_max_adjustment)),
            FormField("external_signal_min_providers", "External min providers", str(cfg.external_signal_min_providers)),
            FormField("external_signal_ttl", "External cache TTL seconds", str(cfg.external_signal_ttl_seconds)),
            FormField("external_signal_timeout", "External timeout seconds", str(cfg.external_signal_timeout_seconds)),
            FormField("external_news_ai", "Ollama news AI [yes/no]", str(cfg.external_news_ai_enabled)),
            FormField("external_news_ai_model", "Ollama news model", str(cfg.external_news_ai_model)),
            FormField("external_news_provider_limit", "News provider limit", str(cfg.external_signal_news_provider_limit)),
            FormField("external_provider_parallelism", "News provider parallelism", str(cfg.external_signal_provider_parallelism)),
            FormField("external_provider_jitter", "Provider jitter seconds", str(cfg.external_signal_provider_jitter_seconds)),
            FormField("external_poll_jitter", "Poll jitter seconds", str(cfg.external_signal_poll_jitter_seconds)),
            FormField("telemetry_db", "Raw telemetry DB", str(cfg.telemetry_db_path)),
            FormField("source_grading", "Hourly source grading [yes/no]", str(cfg.source_grading_enabled)),
            FormField("source_grading_interval", "Source grading interval seconds", str(cfg.source_grading_interval_seconds)),
            FormField("source_grade_max_age_hours", "Source grade max age hours", str(cfg.source_grade_max_age_hours)),
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
            external_signals=None,
            external_signal_max_adjustment=None,
            external_signal_min_providers=None,
            external_signal_ttl=None,
            external_signal_timeout=None,
            external_news_ai=None,
            external_news_ai_model=None,
            external_news_ai_url=None,
            external_news_ai_timeout=None,
            external_news_provider_limit=None,
            external_provider_parallelism=None,
            external_provider_jitter=None,
            external_poll_jitter=None,
            telemetry_db=None,
            no_telemetry=None,
            source_grading=None,
            source_grading_interval=None,
            source_grading_window_hours=None,
            source_grade_max_age_hours=None,
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

    def field_bool(key: str, current: bool):
        return _profile_field_value(
            profile,
            payload,
            key,
            current,
            lambda raw: _parse_form_bool(raw, current),
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
        external_signals=field_bool("external_signals", cfg.external_signals_enabled),
        external_signal_max_adjustment=field_float(
            "external_signal_max_adjustment",
            cfg.external_signal_max_adjustment,
            "External max score adjustment",
            minimum=0.0,
            maximum=0.20,
        ),
        external_signal_min_providers=field_int(
            "external_signal_min_providers",
            cfg.external_signal_min_providers,
            "External min providers",
            minimum=0,
            maximum=120,
        ),
        external_signal_ttl=field_int(
            "external_signal_ttl",
            cfg.external_signal_ttl_seconds,
            "External cache TTL seconds",
            minimum=0,
        ),
        external_signal_timeout=field_float(
            "external_signal_timeout",
            cfg.external_signal_timeout_seconds,
            "External timeout seconds",
            minimum=0.1,
            maximum=30.0,
        ),
        external_news_ai=field_bool("external_news_ai", cfg.external_news_ai_enabled),
        external_news_ai_model=payload["external_news_ai_model"],
        external_news_ai_url=None,
        external_news_ai_timeout=None,
        external_news_provider_limit=field_int(
            "external_news_provider_limit",
            cfg.external_signal_news_provider_limit,
            "News provider limit",
            minimum=0,
            maximum=120,
        ),
        external_provider_parallelism=field_int(
            "external_provider_parallelism",
            cfg.external_signal_provider_parallelism,
            "News provider parallelism",
            minimum=1,
            maximum=64,
        ),
        external_provider_jitter=field_float(
            "external_provider_jitter",
            cfg.external_signal_provider_jitter_seconds,
            "Provider jitter seconds",
            minimum=0.0,
            maximum=30.0,
        ),
        external_poll_jitter=field_float(
            "external_poll_jitter",
            cfg.external_signal_poll_jitter_seconds,
            "Poll jitter seconds",
            minimum=0.0,
            maximum=60.0,
        ),
        telemetry_db=payload["telemetry_db"],
        no_telemetry=None,
        source_grading=field_bool("source_grading", cfg.source_grading_enabled),
        source_grading_interval=field_int(
            "source_grading_interval",
            cfg.source_grading_interval_seconds,
            "Source grading interval seconds",
            minimum=60,
        ),
        source_grading_window_hours=None,
        source_grade_max_age_hours=field_float(
            "source_grade_max_age_hours",
            cfg.source_grade_max_age_hours,
            "Source grade max age hours",
            minimum=0.0,
            maximum=8760.0,
        ),
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


def _account_overview_lines(runtime) -> list[str]:  # skipcq: PY-R1000
    if not _has_api_credentials(runtime):
        return [_credential_required_message("Account balances")]
    try:
        client = _build_client(runtime)
        account = client.get_account()
    except BinanceAPIError as exc:
        return [f"Account balances failed: {exc}"]
    balances_payload = account.get("balances", []) if isinstance(account, dict) else []
    assets_payload = account.get("assets", []) if isinstance(account, dict) else []
    positions_payload = account.get("positions", []) if isinstance(account, dict) else []
    balances = balances_payload if isinstance(balances_payload, list) else []
    assets = assets_payload if isinstance(assets_payload, list) else []
    positions = positions_payload if isinstance(positions_payload, list) else []
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
        return [
            f"market={runtime.market_type} environment={_runtime_environment(runtime)} testnet={runtime.testnet}",
            "No non-zero balances found.",
        ]
    return [
        f"market={runtime.market_type} environment={_runtime_environment(runtime)} testnet={runtime.testnet}",
        *interesting[:20],
    ]


def _show_account_overview() -> int:
    runtime = load_runtime()
    if not _has_api_credentials(runtime):
        print(_credential_required_message("Account balances"), file=sys.stderr)
        return 2
    lines = _account_overview_lines(runtime)
    print("Account balances")
    for line in lines:
        print(f"- {line}" if ":" in line and not line.startswith("market=") else line)
    if lines and lines[0].startswith("Account balances failed:"):
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
        account_lines=_account_overview_lines(runtime) if with_account else ["Run Account balances after Connect to fetch balances."],
        notes=notes,
    )


def _connection_status_line() -> str:
    runtime = load_runtime()
    environment = _runtime_environment(runtime)
    market = f"{runtime.market_type}/{environment}"
    mode = "paper-default" if runtime.dry_run else f"{environment}-live-default"
    checked_at = datetime.now(timezone.utc).strftime("%H:%M:%SZ")
    try:
        client = _build_client(runtime)
        client.ping()
        server_time = client.get_exchange_time()
    except BinanceAPIError as exc:
        return f"Connection {checked_at}: public endpoint unreachable {market}; {exc}"
    server_label = "server-time ok" if server_time is not None else "server-time response ok"
    if not _has_api_credentials(runtime):
        return f"Connection {checked_at}: public endpoint reachable {market} {mode}; credentials missing"
    if not _allows_signed_execution(runtime):
        return (
            f"Connection {checked_at}: public endpoint reachable {market} {mode}; "
            "credentials saved, signed validation locked until testnet or demo is enabled"
        )
    if not bool(getattr(runtime, "validate_account", True)):
        return f"Connection {checked_at}: public endpoint reachable {market} {mode}; credentials saved, not validated"
    try:
        account = client.get_account()
    except BinanceAPIError as exc:
        return f"Connection {checked_at}: authentication failed {market} {mode}; {exc}"
    auth_label = "authenticated" if isinstance(account, dict) else "auth response ok"
    return f"Connection {checked_at}: {auth_label} {market} {mode}; {server_label}"


def _readiness_report(*, input_path: str, model_path: str, online: bool = False) -> tuple[bool, list[str]]:  # skipcq: PY-R1000
    runtime = load_runtime()
    strategy = load_strategy()
    checks: list[tuple[bool, str, str]] = []

    def add(ok: bool, label: str, detail: str) -> None:
        checks.append((ok, label, detail))

    environment = _runtime_environment(runtime)
    safe_execution = _allows_signed_execution(runtime)
    mode = "paper" if runtime.dry_run else f"authenticated {environment}"
    add(
        safe_execution,
        "safety target",
        f"{environment} enabled" if safe_execution else "testnet/demo is disabled",
    )
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
    candles = None
    if data_file.exists():
        candles = _load_rows_for_command(str(data_file), label="Readiness data load failed")
        candle_count = len(candles) if candles is not None else 0
        add(candle_count > max(strategy.feature_windows), "training data", f"{candle_count} candles at {data_file}")
    else:
        add(False, "training data", f"missing {data_file}")

    model_file = Path(model_path)
    if model_file.exists():
        try:
            model, model_kind = _load_readiness_model(model_file, strategy)
        except (OSError, ValueError) as exc:
            add(False, "model artifact", f"{model_file} is not usable with current strategy ({exc})")
        else:
            add(True, "model artifact", f"{model_file} dim={model.feature_dim} kind={model_kind}")
            effective_strategy = apply_model_strategy_overrides(strategy, model)
            if effective_strategy.asdict() != strategy.asdict():
                add(
                    True,
                    "model strategy overlay",
                    (
                        f"threshold={effective_strategy.signal_threshold:.3f} "
                        f"risk={effective_strategy.risk_per_trade:.4f} "
                        f"stop={effective_strategy.stop_loss_pct:.4f} "
                        f"take={effective_strategy.take_profit_pct:.4f}"
                    ),
                )
            quality_score = getattr(model, "quality_score", None)
            if quality_score is not None:
                warnings = list(getattr(model, "quality_warnings", []) or [])
                warning_text = "; ".join(str(item) for item in warnings[:2]) or "none"
                add(
                    float(quality_score) >= 0.45,
                    "model quality",
                    f"score={float(quality_score):.2f} warnings={warning_text}",
                )
            probability_brier = getattr(model, "probability_brier_after", None)
            if probability_brier is not None:
                probability_temperature = float(getattr(model, "probability_temperature", 1.0) or 1.0)
                probability_ece = getattr(model, "probability_ece_after", None)
                probability_detail = (
                    f"temperature={probability_temperature:.2f} "
                    f"brier={float(probability_brier):.3f}"
                )
                if probability_ece is not None:
                    probability_detail += f" ece={float(probability_ece):.3f}"
                add(
                    float(probability_brier) <= 0.35,
                    "probability calibration",
                    probability_detail,
                )
            if candles is not None:
                rows = _readiness_model_rows(candles, effective_strategy, model)
                if rows:
                    drift = feature_drift_report(rows[-min(50, len(rows)):], model)
                    drift_warning = "; ".join(drift.warnings[:2]) or "none"
                    add(
                        drift.status != "fail",
                        "feature drift",
                        (
                            f"status={drift.status} max_z={drift.max_abs_z:.2f} "
                            f"outliers={drift.outlier_fraction:.1%} warnings={drift_warning}"
                        ),
                    )
    else:
        add(False, "model artifact", f"missing {model_file}")

    if online:
        line = _connection_status_line()
        authenticated = "authenticated" in line or "auth response ok" in line
        add(authenticated and "failed" not in line and "unreachable" not in line, "exchange connectivity", line)

    lines = [f"[{'ok' if ok else 'fix'}] {label}: {detail}" for ok, label, detail in checks]
    return all(ok for ok, _label, _detail in checks), lines


_COMPUTE_BACKEND_CHOICES = ("cpu", "cuda", "rocm", "directml", "mps", "auto")


def _account_free_balances(account: object) -> dict[str, float]:
    balances = {"USDC": 0.0, "BTC": 0.0}
    if not isinstance(account, dict):
        return balances
    spot_balances = account.get("balances", [])
    if isinstance(spot_balances, list):
        for item in spot_balances:
            if not isinstance(item, dict):
                continue
            asset = str(item.get("asset", "")).upper()
            if asset in balances:
                balances[asset] += max(0.0, _safe_float(item.get("free")))
    futures_assets = account.get("assets", [])
    if isinstance(futures_assets, list):
        for item in futures_assets:
            if not isinstance(item, dict):
                continue
            asset = str(item.get("asset", "")).upper()
            if asset in balances:
                value = item.get("availableBalance", item.get("walletBalance", 0.0))
                balances[asset] = max(balances[asset], max(0.0, _safe_float(value)))
    return balances


def _load_exchange_funds(runtime) -> dict[str, float]:
    if not _has_api_credentials(runtime):
        raise BinanceAPIError(_credential_required_message("Funds"))
    account = _build_client(runtime).get_account()
    return _account_free_balances(account)


def _funds_summary(runtime, balances: Mapping[str, float] | None = None) -> str:
    caps = f"Trading caps: USDC={runtime.managed_usdc:.4f} BTC={runtime.managed_btc:.8f}"
    if not _has_api_credentials(runtime):
        return f"API credentials missing. Exchange-backed Funds disabled. {caps}"
    if balances is None:
        return f"Exchange balance not loaded. {caps}"
    usdc = max(0.0, _safe_float(balances.get("USDC", 0.0)))
    btc = max(0.0, _safe_float(balances.get("BTC", 0.0)))
    return f"Exchange free: USDC={usdc:.4f} BTC={btc:.8f}. {caps}"


def _apply_funds_change(
    action: str,
    amount: float,
    *,
    balances: Mapping[str, float] | None = None,
) -> tuple[object, str]:
    """Apply an exchange-backed Funds-menu allocation cap mutation."""
    runtime = load_runtime()
    if action == "clear":
        runtime.managed_usdc = 0.0
        runtime.managed_btc = 0.0
        save_runtime(runtime)
        return runtime, "Cleared trading caps. No exchange balances were changed."
    if action in {"deposit_usdc", "withdraw_usdc", "deposit_btc", "withdraw_btc", "reset"}:
        return runtime, "Funds no longer supports local deposit, withdraw, or reset. Configure credentials and set exchange-backed caps."
    if action not in {"sync", "set_usdc", "set_btc"}:
        return runtime, f"Unknown funds action {action!r}."
    if balances is None:
        return runtime, _credential_required_message("Funds")
    if action == "sync":
        runtime.managed_usdc = max(0.0, _safe_float(balances.get("USDC", 0.0)))
        runtime.managed_btc = max(0.0, _safe_float(balances.get("BTC", 0.0)))
        msg = "Synced trading caps from exchange free balances."
    else:
        asset = "USDC" if action == "set_usdc" else "BTC"
        available = max(0.0, _safe_float(balances.get(asset, 0.0)))
        requested = max(0.0, float(amount))
        cap = min(requested, available)
        if asset == "USDC":
            runtime.managed_usdc = cap
            msg = f"Set USDC trading cap to {cap:.4f}."
        else:
            runtime.managed_btc = cap
            msg = f"Set BTC trading cap to {cap:.8f}."
        if requested > available:
            msg += f" Requested amount was capped to exchange free balance {available:.8f} {asset}."
    save_runtime(runtime)
    return runtime, msg


async def _ui_funds_menu(ui) -> int:
    from .tui import FormField

    while True:
        runtime = load_runtime()
        has_credentials = _has_api_credentials(runtime)
        options = (
            [
                ("sync", "Use exchange free balances as caps"),
                ("set_usdc", "Set USDC trading cap"),
                ("set_btc", "Set BTC trading cap"),
                ("clear", "Clear trading caps"),
                ("show", "Refresh exchange-backed allocation"),
                ("close", "Close"),
            ]
            if has_credentials
            else [
                ("show", "Show credential requirement"),
                ("close", "Close"),
            ]
        )
        choice = await ui.menu(
            "Trading caps - exchange-backed BTCUSDC limits",
            options,
            help_text=(
                f"{_funds_summary(runtime)}. Trading caps reads Binance balances and stores maximum "
                "strategy allocation caps; it never deposits, withdraws, or simulates money."
            ),
        )
        if choice in (None, "close"):
            return 0
        if not has_credentials:
            ui.append_log(_credential_required_message("Funds"))
            continue
        if choice == "clear":
            _, msg = _apply_funds_change("clear", 0.0)
            ui.append_log(msg)
            continue
        try:
            balances = await ui.run_blocking(_load_exchange_funds, runtime)
        except BinanceAPIError as exc:
            ui.append_log(_credential_failure_message("Funds", exc))
            continue
        if choice == "show":
            ui.append_log(_funds_summary(runtime, balances))
            continue
        if choice == "sync":
            _, msg = _apply_funds_change("sync", 0.0, balances=balances)
            ui.append_log(msg)
            continue
        is_btc = choice == "set_btc"
        unit = "BTC" if is_btc else "USDC"
        current_cap = runtime.managed_btc if is_btc else runtime.managed_usdc
        payload = await ui.form(
            f"Set {unit} trading cap",
            [
                FormField(
                    "amount",
                    f"Maximum {unit} the strategy may use",
                    f"{current_cap:.8f}" if is_btc else f"{current_cap:.4f}",
                ),
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
        _, msg = _apply_funds_change(choice, amount, balances=balances)
        ui.append_log(msg)


async def _ui_edit_execution(ui) -> int:
    from .tui import FormField

    cfg = load_strategy()
    payload = await ui.form(
        "Order settings",
        [
            FormField(
                "order_type",
                "Order type [MARKET]",
                str(getattr(cfg, "order_type", "MARKET")),
            ),
            FormField(
                "time_in_force",
                "Time in force [GTC; market orders ignore this]",
                str(getattr(cfg, "time_in_force", "GTC")),
            ),
            FormField(
                "post_only",
                "Post-only [no; unsupported for market orders]",
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
        ui.append_log("Order settings cancelled.")
        return 0
    order_type = payload["order_type"].strip().upper() or "MARKET"
    if order_type != "MARKET":
        ui.append_log(f"Unsupported order type {order_type!r}; using MARKET.")
        order_type = "MARKET"
    tif = payload["time_in_force"].strip().upper() or "GTC"
    if tif not in {"GTC", "IOC", "FOK"}:
        ui.append_log(f"Unsupported timeInForce {tif!r}; keeping {cfg.time_in_force!r}.")
        tif = cfg.time_in_force
    post_only = _parse_form_bool(payload["post_only"], getattr(cfg, "post_only", False))
    if post_only:
        ui.append_log("Post-only is not compatible with live market execution; using no.")
        post_only = False
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
                "Backend [cpu / cuda / rocm / directml / mps / auto]",
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


async def _ui_settings_menu(ui, mark_credentials: Callable[[str], None] | None = None) -> int:
    while True:
        choice = await ui.menu(
            "All settings",
            [
                ("runtime", "Connection - API, market, safety mode"),
                ("strategy", "Strategy - risk, signals, model behavior"),
                ("execution", "Orders - type and close behavior"),
                ("compute", "Compute - CPU / GPU / auto"),
                ("close", "Close"),
            ],
            help_text="Choose what to configure. Up/Down to choose, Enter to open, Escape to close.",
        )
        if choice in (None, "close"):
            return 0
        if choice == "runtime":
            current = load_runtime()
            try:
                next_runtime = await _ui_edit_runtime(ui, current)
            except ValueError as exc:
                ui.append_log(f"Connection settings invalid: {exc}")
                continue
            if next_runtime == current:
                ui.append_log("Connection settings cancelled.")
                continue
            save_runtime(next_runtime)
            if mark_credentials is not None:
                mark_credentials("unchecked" if _has_api_credentials(next_runtime) else "missing")
            ui.append_log("Connection settings saved.")
            if next_runtime.validate_account and _has_api_credentials(next_runtime):
                try:
                    client = _build_client(next_runtime)
                    await ui.run_blocking(_validate_runtime_connection, next_runtime, client)
                except BinanceAPIError as exc:
                    if mark_credentials is not None:
                        mark_credentials("invalid")
                    ui.append_log(f"Connection validation failed: {exc}")
                    continue
                if mark_credentials is not None:
                    mark_credentials("valid")
                ui.append_log("Connection credentials validated.")
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


def _tui_actions(credential_state: dict[str, str] | None = None):  # skipcq: PY-R1000
    from .tui import FormField, TUIAction

    def _credential_status() -> str:
        runtime = load_runtime()
        if not _has_api_credentials(runtime):
            return "missing"
        if credential_state is None:
            return "valid"
        fingerprint = _credential_fingerprint(runtime)
        if credential_state.get("fingerprint") != fingerprint:
            return "unchecked"
        return credential_state.get("status", "unchecked")

    def _mark_credentials(status: str) -> None:
        if credential_state is None:
            return
        runtime = load_runtime()
        credential_state["fingerprint"] = _credential_fingerprint(runtime)
        credential_state["status"] = "missing" if not _has_api_credentials(runtime) else status

    def _credential_lock_reason(action: str) -> str:
        runtime = load_runtime()
        if not _allows_signed_execution(runtime):
            return f"{action} is locked. Signed actions require testnet=true or demo=true."
        status = _credential_status()
        if status == "missing":
            return f"{action} is locked. Add Binance API key and secret in Connection settings first."
        if status == "invalid":
            return f"{action} is locked. The saved Binance credentials failed validation; replace them in Connection settings."
        if status == "unchecked":
            return f"{action} is locked. Run Connect after saving credentials, then try again."
        return f"{action} is locked until Binance credentials validate."

    def _connect_enabled() -> bool:
        status = _credential_status()
        return status in {"unchecked", "valid", "invalid", "unavailable"}

    def _signed_action_enabled() -> bool:
        return _credential_status() == "valid" and _allows_signed_execution(load_runtime())

    def _make_disabled_reason(action_title: str) -> Callable[[], str]:
        def disabled_reason() -> str:
            return _credential_lock_reason(action_title)

        return disabled_reason

    def _make_action(
        key: str,
        title: str,
        description: str,
        run,
        *,
        aliases: tuple[str, ...] = (),
        credentials: bool = False,
    ):
        if not credentials:
            return TUIAction(key, title, description, run, aliases=aliases)
        enabled = _connect_enabled if title == "Connect" else _signed_action_enabled
        return TUIAction(
            key,
            title,
            description,
            run,
            enabled=enabled,
            disabled_reason=_make_disabled_reason(title),
            aliases=aliases,
        )

    def _credentials_ready(ui, action: str) -> bool:
        if not _has_api_credentials(load_runtime()):
            ui.append_log(_credential_required_message(action))
            return False
        if action != "Connect" and not _allows_signed_execution(load_runtime()):
            ui.append_log(_credential_lock_reason(action))
            return False
        if credential_state is not None and action != "Connect" and _credential_status() != "valid":
            ui.append_log(_credential_lock_reason(action))
            return False
        return True

    async def _overview(ui):
        include_account = credential_state is None or _credential_status() == "valid"
        ui.append_log(await ui.run_blocking(lambda: render_dashboard(_dashboard_snapshot(with_account=include_account))))
        return 0

    async def _help(ui):
        ui.append_log(
            "\n".join(
                [
                    "Operator help - simple-ai-trading",
                    "==================================",
                    "",
                    "Scope: BTCUSDC spot trading on Binance testnet or Demo Trading only.",
                    "",
                    "First-time setup",
                    "----------------",
                    "  1. Connection settings: paste your Binance testnet API key and secret.",
                    "  2. Connect: validate those credentials.",
                    "  3. Trading caps: choose how much BTC / USDC the strategy may use.",
                    "  4. Safety check: confirm safety flags, data, model, and connectivity.",
                    "",
                    "End-to-end paper run",
                    "--------------------",
                    "  1. Build full setup: download data, train, evaluate, and backtest.",
                    "  2. Paper trading: run the strategy without placing real orders.",
                    "  3. Full report: print dashboard, artifacts, and safety summary.",
                    "",
                    "Manual pipeline (full control)",
                    "------------------------------",
                    "  Download market data -> Strategy settings -> Train AI model -> Evaluate model -> Backtest strategy -> Optimize strategy",
                    "",
                    "Authenticated testnet execution",
                    "-------------------------------",
                    "  * Always run Safety check first.",
                    "  * Test order is the smallest signed BUY/SELL check.",
                    "  * Testnet trading runs the strategy with signed orders against testnet/demo.",
                    "  * Trading caps can never exceed exchange free balances.",
                    "",
                    "Keyboard",
                    "--------",
                    "  Up / Down           always move the command or modal menu selection",
                    "  Enter               run the selected command",
                    "  r                   refresh the dashboard snapshot",
                    "  <  >                shrink / grow the command list",
                    "  -  +                shrink / grow the activity log",
                    "  Ctrl-L              clear the activity log",
                    "  q                   quit",
                    "  Inside forms: Tab cycles fields, Enter advances or saves, Escape cancels.",
                    "",
                    "Centralized configuration",
                    "-------------------------",
                    "  All settings opens: Connection, Strategy, Orders, Compute backend.",
                    "  Trading caps is exchange-backed: it reads balances and sets caps, never deposits or withdraws.",
                    "  Compute backend selects CPU (default), CUDA, ROCm, DirectML, MPS, or auto-detect.",
                    "",
                    "Safety",
                    "------",
                    "  testnet=true is the default; demo=true selects Binance Demo Trading endpoints.",
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
            print(f"Connection settings invalid: {exc}", file=sys.stderr)
            return 2
        if next_runtime == current:
            print("Connection settings cancelled.")
            return 0
        save_runtime(next_runtime)
        _mark_credentials("unchecked")
        if next_runtime.validate_account and _has_api_credentials(next_runtime):
            client = _build_client(next_runtime)
            try:
                await ui.run_blocking(_validate_runtime_connection, next_runtime, client)
            except BinanceAPIError as exc:
                _mark_credentials("invalid")
                print(f"Configuration saved, but validation failed: {exc}", file=sys.stderr)
                return 2
            _mark_credentials("valid")
        print("Runtime config saved to", config_paths()["runtime"])
        print(
            f"market={next_runtime.market_type} environment={_runtime_environment(next_runtime)} "
            f"testnet={next_runtime.testnet} demo={getattr(next_runtime, 'demo', False)} paper={next_runtime.dry_run}"
        )
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
        if not _credentials_ready(_ui, "Connect"):
            return 2
        result = await _ui.run_blocking(command_connect, argparse.Namespace())
        _mark_credentials("valid" if result == 0 else "invalid")
        return result

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
        if not _credentials_ready(ui, "Account balances"):
            return 2
        return await ui.run_blocking(_show_account_overview)

    async def _audit(ui):
        payload = await ui.form(
            "Data/model audit",
            [
                FormField("input", "Training input path", "data/historical_btcusdc.json"),
                FormField("model", "Model path", "data/model.json"),
            ],
        )
        if payload is None:
            print("Data/model audit cancelled.")
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
            "Download market data",
            [
                FormField("limit", "Fetch limit", "500"),
                FormField("batch_size", f"Klines per request [max {max_batch_size}]", "1000"),
                FormField("output", "Candle output path", "data/historical_btcusdc.json"),
            ],
        )
        if payload is None:
            print("Market data download cancelled.")
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
            "Train AI model",
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
            print("AI model training cancelled.")
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
            "Optimize strategy",
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
            print("Strategy optimization cancelled.")
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
            print(f"Optimization settings invalid: {exc}", file=sys.stderr)
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
            "Backtest strategy",
            [
                FormField("input", "Backtest input path", "data/historical_btcusdc.json"),
                FormField("model", "Model path", "data/model.json"),
                FormField("start_cash", "Starting cash", "1000"),
            ],
        )
        if payload is None:
            print("Strategy backtest cancelled.")
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
            "Build full setup",
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
            print("Build full setup cancelled.")
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
            calibrate_threshold_opt = _parse_optional_form_bool(payload.get("calibrate_threshold", ""))
            start_cash = _parse_form_float(payload["start_cash"], label="Backtest starting cash", default=1000.0, minimum=1.0)
        except ValueError as exc:
            print(f"Setup settings invalid: {exc}", file=sys.stderr)
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
                calibrate_threshold=calibrate_threshold_opt,
                start_cash=start_cash,
                online_doctor=_parse_form_bool(payload["online_doctor"], False),
            ),
        )

    async def _paper(ui):
        payload = await ui.form(
            "Paper trading",
            [
                FormField("model", "Model path", "data/model.json"),
                FormField("steps", "Paper trading steps", "20"),
                FormField("sleep", "Sleep seconds", "5"),
                FormField("retrain_interval", "Retrain interval", "0"),
                FormField("retrain_window", "Retrain window", "300"),
                FormField("retrain_min_rows", "Retrain minimum rows", "240"),
            ],
        )
        if payload is None:
            print("Paper trading cancelled.")
            return 0
        try:
            steps = _parse_form_int(payload["steps"], label="Paper trading steps", default=20, minimum=1)
            sleep = _parse_form_int(payload["sleep"], label="Sleep seconds", default=5, minimum=0)
            retrain_interval = _parse_form_int(payload["retrain_interval"], label="Retrain interval", default=0, minimum=0)
            retrain_window = _parse_form_int(payload["retrain_window"], label="Retrain window", default=300, minimum=1)
            retrain_min_rows = _parse_form_int(payload["retrain_min_rows"], label="Retrain minimum rows", default=240, minimum=1)
        except ValueError as exc:
            print(f"Paper trading settings invalid: {exc}", file=sys.stderr)
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
        if not _credentials_ready(ui, "Testnet trading"):
            return 2
        payload = await ui.form(
            "Testnet trading",
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
            print("Testnet trading cancelled.")
            return 0
        try:
            steps = _parse_form_int(payload["steps"], label="Live steps", default=1, minimum=1)
            sleep = _parse_form_int(payload["sleep"], label="Sleep seconds", default=5, minimum=0)
            retrain_interval = _parse_form_int(payload["retrain_interval"], label="Retrain interval", default=0, minimum=0)
            retrain_window = _parse_form_int(payload["retrain_window"], label="Retrain window", default=300, minimum=1)
            retrain_min_rows = _parse_form_int(payload["retrain_min_rows"], label="Retrain minimum rows", default=240, minimum=1)
        except ValueError as exc:
            print(f"Testnet trading settings invalid: {exc}", file=sys.stderr)
            return 2
        model = payload["model"].strip() or "data/model.json"
        runtime = load_runtime()
        environment = _runtime_environment(runtime)
        if not await ui.confirm(
            f"Run authenticated {environment} execution with model={model}, steps={steps}, sleep={sleep}s?"
        ):
            print(f"{environment.capitalize()} execution cancelled.")
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
        if not _credentials_ready(ui, "Test order"):
            return 2
        payload = await ui.form(
            "Test order",
            [
                FormField("quantity", "Order quantity", "0.00008"),
                FormField("mode", "Mode [auto/buy-sell/sell-buy]", "auto"),
            ],
        )
        if payload is None:
            print("Test order cancelled.")
            return 0
        try:
            quantity = _parse_form_float(payload["quantity"], label="Order quantity", default=0.00008, minimum=0.00001)
        except ValueError as exc:
            print(f"Test order settings invalid: {exc}", file=sys.stderr)
            return 2
        mode = payload["mode"].strip().lower() or "auto"
        if mode not in {"auto", "buy-sell", "sell-buy"}:
            print("Test order mode must be auto, buy-sell, or sell-buy.", file=sys.stderr)
            return 2
        runtime = load_runtime()
        if not await ui.confirm(
            f"Place {mode} spot {_runtime_environment(runtime)} roundtrip for quantity={quantity:.8f}?"
        ):
            print("Test order cancelled.")
            return 0
        return await ui.run_blocking(
            command_spot_roundtrip,
            argparse.Namespace(quantity=quantity, mode=mode, yes=True),
        )

    async def _report(ui):
        payload = await ui.form(
            "Full report",
            [
                FormField("input", "Training input path", "data/historical_btcusdc.json"),
                FormField("model", "Model path", "data/model.json"),
                FormField("readiness", "Include readiness report [yes/no]", "yes"),
                FormField("online", "Include exchange connectivity [yes/no]", "no"),
                FormField("account", "Include account state [yes/no]", "no"),
            ],
        )
        if payload is None:
            print("Full report cancelled.")
            return 0
        include_account = _parse_form_bool(payload["account"], False)
        if include_account and not _credentials_ready(ui, "Full report account section"):
            return 2
        return await ui.run_blocking(
            command_report,
            argparse.Namespace(
                input=payload["input"].strip() or "data/historical_btcusdc.json",
                model=payload["model"].strip() or "data/model.json",
                doctor=_parse_form_bool(payload["readiness"], True),
                online=_parse_form_bool(payload["online"], False),
                account=include_account,
            ),
        )

    async def _funds(ui):
        if not _credentials_ready(ui, "Trading caps"):
            return 2
        return await _ui_funds_menu(ui)

    async def _settings(ui):
        return await _ui_settings_menu(ui, mark_credentials=_mark_credentials)

    return [
        _make_action("1", "Connect", "Validate the saved Binance testnet credentials and unlock account-only actions.", _connect, credentials=True),
        _make_action("2", "Dashboard", "Show the current setup, strategy, model, and recent run artifacts in the activity log.", _overview, aliases=("Overview",)),
        _make_action("3", "Account balances", "Read authenticated BTC and USDC balances from Binance.", _account, credentials=True, aliases=("Account",)),
        _make_action("4", "Safety check", "Verify safety flags, training data, model compatibility, and optional exchange reachability.", _doctor, aliases=("Readiness check",)),
        _make_action("5", "Data/model audit", "Check candle quality, feature stability, model metadata, and risk posture without network calls.", _audit, aliases=("Local audit",)),
        _make_action("6", "Trading caps", "Read exchange balances and set the maximum BTC / USDC the strategy may use.", _funds, credentials=True, aliases=("Funds",)),
        _make_action("7", "Download market data", "Download fresh BTCUSDC candles into the local dataset.", _fetch, aliases=("Fetch candles",)),
        _make_action("8", "Train AI model", "Train or retrain the prediction model on cached market data.", _train, aliases=("Train model",)),
        _make_action("9", "Evaluate model", "Score the saved model on cached candles and inspect threshold quality.", _evaluate, aliases=("Evaluate",)),
        _make_action("10", "Backtest strategy", "Simulate trades on cached candles and report PnL, fees, and drawdown.", _backtest, aliases=("Backtest",)),
        _make_action("11", "Optimize strategy", "Search risk, threshold, take-profit, and stop-loss settings over a chosen history window.", _tune, aliases=("Tune strategy",)),
        _make_action("12", "Build full setup", "Download data, train, evaluate, backtest, audit, then run safety checks.", _prepare, aliases=("Prepare system",)),
        _make_action("13", "Paper trading", "Run the live loop without placing orders; useful before signed testnet trading.", _paper, aliases=("Paper loop",)),
        _make_action("14", "Testnet trading", "Run authenticated non-mainnet execution with signed testnet or demo orders.", _live, credentials=True, aliases=("Testnet loop",)),
        _make_action("15", "Test order", "Place a minimal BUY/SELL roundtrip on spot testnet or demo.", _roundtrip, credentials=True, aliases=("Spot roundtrip",)),
        _make_action("16", "Full report", "Print dashboard, recent artifacts, safety checks, and optional account state.", _report, aliases=("Operator report",)),
        # Backwards-compatible aliases for direct configuration shortcuts.
        _make_action("17", "Connection settings", "Edit API keys, market type, testnet/demo mode, paper mode, and request limits.", _runtime, aliases=("Runtime settings",)),
        _make_action("18", "Strategy settings", "Edit risk, thresholds, model windows, enabled features, and external signal behavior.", _strategy),
        _make_action("19", "All settings", "Open one settings hub for connection, strategy, order execution, and CPU/GPU backend.", _settings, aliases=("Settings",)),
        _make_action("20", "Help", "Show the plain-language workflow, keyboard shortcuts, safety notes, and setup guide.", _help),
    ]


def command_menu(_: argparse.Namespace) -> int:
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        print("Interactive console requires a real terminal (TTY).", file=sys.stderr)
        return 2
    if not supports_ansi_terminal(sys.stdout):
        print(
            "Interactive console needs ANSI/virtual-terminal support. "
            "Use Windows Terminal, or run `simple-ai-trading shell` for the plain fallback.",
            file=sys.stderr,
        )
        return 2

    from .tui import launch_tui

    initial_runtime = load_runtime()
    credential_state = {
        "fingerprint": _credential_fingerprint(initial_runtime),
        "status": "missing" if not _has_api_credentials(initial_runtime) else "unchecked",
    }

    def menu_connection_status_line() -> str:
        line = _connection_status_line()
        runtime = load_runtime()
        credential_state["fingerprint"] = _credential_fingerprint(runtime)
        if not _has_api_credentials(runtime):
            credential_state["status"] = "missing"
        elif "authentication failed" in line:
            credential_state["status"] = "invalid"
        elif "authenticated" in line or "auth response ok" in line:
            credential_state["status"] = "valid"
        else:
            credential_state["status"] = "unavailable"
        return line

    return launch_tui(
        title="simple-ai-trading interactive console",
        actions=_tui_actions(credential_state),
        snapshot_provider=_menu_dashboard_snapshot,
        connection_provider=menu_connection_status_line,
    )


def _menu_dashboard_snapshot(width: int = 72) -> str:
    return render_dashboard(_dashboard_snapshot(with_account=False), width=width)


def _load_json_candles(path: str) -> list[Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected candle list in JSON file: {path}")
    return payload


def _clamp(value: float, low: float, high: float) -> float:
    if not math.isfinite(float(value)):
        return low
    if value < low:
        return low
    if value > high:
        return high
    return value


def _jittered_seconds(base_seconds: float, jitter_seconds: float) -> float:
    base = max(0.0, float(base_seconds))
    jitter = max(0.0, float(jitter_seconds))
    return base + (_JITTER_RANDOM.uniform(0.0, jitter) if jitter else 0.0)


def _rows_from_json(path: str):
    candles_raw = _load_json_candles(path)

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
    except (OSError, ValueError) as exc:
        print(f"{label}: {exc}", file=sys.stderr)
        return None


def _parse_date_boundary(raw: str, *, end_of_day: bool) -> int:
    dt = datetime.strptime(raw, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    if end_of_day:
        dt = dt + timedelta(days=1) - timedelta(milliseconds=1)
    return int(dt.timestamp() * 1000)


def _filter_candles_for_time_window(  # skipcq: PY-R1000
    candles: Sequence[Candle],
    *,
    lookback_days: int | None = None,
    from_date: str | None = None,
    to_date: str | None = None,
) -> list[Candle]:
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
        latest_close = max(int(candle.close_time) for candle in filtered)
        start_ms = latest_close - (lookback_days * 24 * 60 * 60 * 1000)

    if start_ms is not None:
        filtered = [candle for candle in filtered if int(candle.open_time) >= start_ms]
    if end_ms is not None:
        filtered = [candle for candle in filtered if int(candle.open_time) <= end_ms]
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


def _build_model_rows(candles: Sequence[Candle], strategy: StrategyConfig) -> list[ModelRow]:
    candles = clean_candles(candles)
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
    leverage = float(cfg.leverage)
    if not math.isfinite(leverage):
        return 1.0
    return float(max(1.0, min(125.0, leverage)))


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


def _advanced_objective_for_model(
    model: TrainedModel,
    strategy: StrategyConfig,
) -> tuple[str, AdvancedFeatureConfig] | None:
    model_signature = getattr(model, "feature_signature", None)
    if not model_signature:
        return None
    for objective_name in available_objectives():
        feature_cfg = default_config_for(objective_name, strategy.enabled_features)
        if advanced_feature_signature(feature_cfg) == str(model_signature):
            return objective_name, feature_cfg
    return None


def _load_readiness_model(model_path: Path, strategy: StrategyConfig) -> tuple[TrainedModel, str]:
    try:
        return _load_runtime_model(model_path, strategy), "runtime"
    except ModelFeatureMismatchError as runtime_error:
        model = load_model(
            model_path,
            expected_feature_version=strategy.feature_version,
            expected_feature_dim=None,
        )
        advanced = _advanced_objective_for_model(model, strategy)
        if advanced is None:
            raise runtime_error
        objective_name, _feature_cfg = advanced
        return model, f"advanced:{objective_name}"


def _readiness_model_rows(
    candles: Sequence[Candle],
    strategy: StrategyConfig,
    model: TrainedModel,
) -> list[ModelRow]:
    advanced = _advanced_objective_for_model(model, strategy)
    if advanced is not None:
        _objective_name, feature_cfg = advanced
        return make_advanced_rows(candles, feature_cfg)
    return _build_model_rows(candles, strategy)


def _backtest_rows_for_model(
    candles: Sequence[Candle],
    strategy: StrategyConfig,
    model: TrainedModel,
) -> list[ModelRow]:
    advanced = _advanced_objective_for_model(model, strategy)
    if advanced is not None:
        _objective_name, feature_cfg = advanced
        rows = make_advanced_inference_rows(candles, feature_cfg)
        return rows if rows else _readiness_model_rows(candles, strategy, model)
    rows = make_inference_rows(
        candles,
        strategy.feature_windows[0],
        strategy.feature_windows[1],
        enabled_features=strategy.enabled_features,
    )
    return rows if rows else _readiness_model_rows(candles, strategy, model)


def _live_rows_for_model(
    candles: Sequence[Candle],
    strategy: StrategyConfig,
    model: TrainedModel | None,
) -> list[ModelRow]:
    if model is None:
        rows = make_inference_rows(
            candles,
            strategy.feature_windows[0],
            strategy.feature_windows[1],
            enabled_features=strategy.enabled_features,
        )
        return rows if rows else _build_model_rows(candles, strategy)
    advanced = _advanced_objective_for_model(model, strategy)
    if advanced is not None:
        _objective_name, feature_cfg = advanced
        rows = make_advanced_inference_rows(candles, feature_cfg)
        return rows if rows else _readiness_model_rows(candles, strategy, model)
    rows = make_inference_rows(
        candles,
        strategy.feature_windows[0],
        strategy.feature_windows[1],
        enabled_features=strategy.enabled_features,
    )
    return rows if rows else _readiness_model_rows(candles, strategy, model)


def _live_model_feature_signature(model: TrainedModel | None, strategy: StrategyConfig) -> str:
    if model is not None and _advanced_objective_for_model(model, strategy) is not None:
        return str(model.feature_signature)
    return _strategy_feature_signature(strategy)


def _load_runtime_model(model_path: Path, strategy: StrategyConfig) -> TrainedModel:
    strategy_signature = _strategy_feature_signature(strategy)
    return load_model(
        model_path,
        expected_feature_version=strategy.feature_version,
        expected_feature_signature=strategy_signature,
        expected_feature_dim=None,
    )


def _load_live_start_model(
    model_path: Path,
    strategy: StrategyConfig,
    *,
    effective_dry_run: bool,
) -> tuple[TrainedModel | None, str | None, str | None]:
    if model_path.exists():
        try:
            return _load_readiness_model(model_path, strategy)[0], None, None
        except ModelLoadError as exc:
            if not effective_dry_run:
                return None, f"Live mode requires a compatible model: {exc}", None
            return None, None, f"Model load failed; regenerating: {exc}"
        except Exception:
            if not effective_dry_run:
                return None, f"Live mode requires a readable model: {model_path}", None
            return None, None, None
    if not effective_dry_run:
        return None, f"Live mode needs model file: {model_path}", None
    return None, None, None


def _target_notional(
    cash: float,
    strategy: StrategyConfig,
    market_type: str,
    *,
    leverage: float | None = None,
) -> float:
    if not math.isfinite(float(cash)) or cash <= 0:
        return 0.0
    if leverage is None:
        leverage = _effective_leverage(strategy, market_type)
    if not math.isfinite(float(leverage)):
        return 0.0
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
    if not all(math.isfinite(float(value)) for value in (cash, price, leverage)):
        return 0.0, 0.0
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
    rows: list[ModelRow],
    *,
    retrain_window: int,
    retrain_min_rows: int,
) -> list[ModelRow]:
    if len(rows) < retrain_min_rows:
        return []
    if len(rows) <= retrain_window:
        return rows
    return rows[-retrain_window:]


def _build_live_model(
    rows: list[ModelRow],
    *,
    model: TrainedModel | None = None,
    retrain_every: int,
    step: int,
    cfg: StrategyConfig,
    retrain_window: int,
    retrain_min_rows: int,
    model_feature_signature: str | None = None,
) -> TrainedModel | None:
    if model is not None:
        if retrain_every <= 0:
            return model
        if retrain_every > 0 and step % retrain_every != 0:
            return model

    train_rows = _resolve_live_retrain_rows(rows, retrain_window=retrain_window, retrain_min_rows=retrain_min_rows)
    if not train_rows:
        return model

    epochs = max(20, int(cfg.training_epochs * 0.4))
    signature = model_feature_signature or _strategy_feature_signature(cfg)
    return train(train_rows, epochs=epochs, feature_signature=signature)


def _score_to_direction(score: float, cfg: StrategyConfig, market_type: str, threshold: float | None = None) -> int:
    threshold = cfg.signal_threshold if threshold is None else _clamp(float(threshold), 0.0, 1.0)
    if market_type == "futures":
        threshold = max(0.5, threshold)
        if score >= threshold:
            return 1
        if score <= (1.0 - threshold):
            return -1
        return 0
    return 1 if score >= threshold else 0


def _live_entry_risk_skip(
    *,
    step: int,
    day: int,
    score: float,
    entry_risk: EntryRiskDecision,
    max_daily_trades: int,
    max_open_positions: int,
) -> tuple[str, dict[str, object]]:
    if entry_risk.code == "trade_cap":
        message = f"step {step:>2}: trade cap reached ({max_daily_trades}/day), skipping entry"
        status = "skip_trade_cap"
        event = {"step": step, "status": status, "day": day, "score": float(score)}
    elif entry_risk.code == "max_open_positions":
        message = f"step {step:>2}: max open positions reached ({max_open_positions}), skipping entry"
        status = "skip_max_open_positions"
        event = {"step": step, "status": status, "score": float(score)}
    else:
        message = f"step {step:>2}: risk gate blocked entry ({entry_risk.code})"
        status = f"skip_risk_{entry_risk.code}"
        event = {"step": step, "status": status, "score": float(score)}
    event["risk_check"] = entry_risk.asdict()
    return message, event


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _detect_existing_position(
    runtime,
    client,
    *,
    leverage: float,
    reference_price: float | None = None,
    account: object | None = None,
) -> dict[str, float | int | str] | None:  # skipcq: PY-R1000
    if account is None:
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
            "entry_price_basis": "current_mark_price_not_cost_basis",
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
) -> dict[str, object]:
    if leverage is None:
        leverage = _effective_leverage(strategy, runtime.market_type)
    kwargs = {"dry_run": dry_run, "leverage": leverage}
    if reduce_only and not dry_run:
        kwargs["reduce_only"] = True
    response = client.place_order(runtime.symbol, side, size, **kwargs)
    if dry_run:
        print("paper order:", json.dumps(response, indent=2))
        return response
    print(f"live order: {side} {size:.8f} {runtime.symbol}")
    print(json.dumps(response, indent=2))
    return response


def _order_fill_details(
    order: object,
    *,
    fallback_qty: float,
    fallback_price: float,
) -> tuple[float, float, float]:
    qty = 0.0
    quote = 0.0
    average = 0.0
    if isinstance(order, dict):
        fills = order.get("fills")
        if isinstance(fills, list):
            for fill in fills:
                if not isinstance(fill, dict):
                    continue
                fill_qty = _safe_float(fill.get("qty"))
                fill_price = _safe_float(fill.get("price"))
                if fill_qty <= 0.0 or fill_price <= 0.0:
                    continue
                qty += fill_qty
                quote += fill_qty * fill_price
        if qty <= 0.0:
            qty = _safe_float(order.get("executedQty") or order.get("origQty"))
        quote = quote or _safe_float(
            order.get("cummulativeQuoteQty")
            or order.get("cumQuote")
            or order.get("cumBase")
            or order.get("notional")
        )
        average = _safe_float(order.get("avgPrice") or order.get("price"))
    if qty <= 0.0:
        qty = max(0.0, float(fallback_qty))
    if average <= 0.0:
        average = quote / qty if qty > 0.0 and quote > 0.0 else max(0.0, float(fallback_price))
    notional = quote if quote > 0.0 else qty * average
    return float(qty), float(average), float(notional)


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
        target = (min_notional / price) + (2.0 * step)
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


def command_spot_roundtrip(args: argparse.Namespace) -> int:  # skipcq: PY-R1000
    runtime = load_runtime()
    if not getattr(args, "yes", False):
        print("Pass --yes to confirm signed spot testnet/demo order placement.", file=sys.stderr)
        return 2
    if runtime.market_type != "spot":
        print("Test order requires market_type=spot in Connection settings.", file=sys.stderr)
        return 2
    if not _allows_signed_execution(runtime):
        print("Test order requires testnet=true or demo=true in Connection settings.", file=sys.stderr)
        return 2
    if not _has_api_credentials(runtime):
        print(_credential_required_message("Test order"), file=sys.stderr)
        return 2

    mode = str(getattr(args, "mode", "auto"))
    quantity_requested = float(getattr(args, "quantity", 0.0))
    client = _build_client(runtime)
    price = 0.0
    quantity = 0.0
    second_quantity = 0.0
    notional = 0.0
    before: object | None = None
    mid: object | None = None
    first: object | None = None
    first_side = ""
    second_side = ""
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
        if first is not None:
            partial_payload = {
                "command": "spot-roundtrip",
                "status": "partial_failed",
                "timestamp": int(time.time()),
                "runtime": _public_runtime_payload(runtime),
                "symbol": runtime.symbol,
                "mode": mode,
                "error": str(exc),
                "price_reference": float(price),
                "quantity_requested": float(quantity_requested),
                "quantity_first": float(quantity),
                "quantity_second_attempted": float(second_quantity),
                "notional_reference": float(notional),
                "balances_before": (
                    {
                        "BTC": _asset_free_balance(before, "BTC"),
                        "USDC": _asset_free_balance(before, "USDC"),
                    }
                    if before is not None
                    else None
                ),
                "balances_after_first": (
                    {
                        "BTC": _asset_free_balance(mid, "BTC"),
                        "USDC": _asset_free_balance(mid, "USDC"),
                    }
                    if mid is not None
                    else None
                ),
                "first_order": {
                    "side": first_side,
                    "status": first.get("status") if isinstance(first, dict) else None,
                    "orderId": first.get("orderId") if isinstance(first, dict) else None,
                    "executedQty": first.get("executedQty") if isinstance(first, dict) else None,
                },
                "second_order": {"side": second_side, "status": "not_completed"},
            }
            try:
                artifact_path = _persist_run_artifact("spot_roundtrip_partial", Path("data"), partial_payload)
                print(f"Partial spot roundtrip recorded at {artifact_path}", file=sys.stderr)
            except (OSError, RuntimeError, TypeError, ValueError) as persist_exc:
                print(f"Partial spot roundtrip could not be recorded: {persist_exc}", file=sys.stderr)
        print(_credential_failure_message("Test order", exc), file=sys.stderr)
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
    print(f"Spot {_runtime_environment(runtime)} roundtrip complete.")
    print(json.dumps({**payload, "artifact": str(artifact_path)}, indent=2))
    return 0


def command_configure(_: argparse.Namespace) -> int:
    current = load_runtime()
    next_runtime = prompt_runtime(current)
    save_runtime(next_runtime)

    if next_runtime.validate_account and _has_api_credentials(next_runtime):
        client = _build_client(next_runtime)
        try:
            _validate_runtime_connection(next_runtime, client)
        except BinanceAPIError as exc:
            print(f"Configuration saved, but validation failed: {exc}", file=sys.stderr)
            return 2

    print("Runtime config saved to", config_paths()["runtime"])
    print(
        f"market={next_runtime.market_type} environment={_runtime_environment(next_runtime)} "
        f"testnet={next_runtime.testnet} demo={getattr(next_runtime, 'demo', False)} paper={next_runtime.dry_run}"
    )
    if next_runtime.market_type == "futures":
        print("futures-mode enabled; leverage can be set via strategy.leverage")
    return 0


def command_connect(_: argparse.Namespace) -> int:
    runtime = load_runtime()
    if not _has_api_credentials(runtime):
        print(_credential_required_message("Connect"), file=sys.stderr)
        return 2
    if not _allows_signed_execution(runtime):
        print(
            "Connect validates signed credentials only on Binance testnet or demo. "
            "Enable testnet=true or demo=true before authenticating.",
            file=sys.stderr,
        )
        return 2
    try:
        client = _build_client(runtime)
        server_time = client.get_exchange_time()
        client.ensure_btcusdc()
    except BinanceAPIError as exc:
        print(f"Connection failed: {exc}", file=sys.stderr)
        return 2
    try:
        account = client.get_account()
    except BinanceAPIError as exc:
        print(_credential_failure_message("Connect", exc), file=sys.stderr)
        return 2
    if isinstance(account, dict):
        account = {
            "updateTime": account.get("updateTime"),
            "canTrade": account.get("canTrade"),
            "accountType": account.get("accountType"),
            "positions": account.get("positions"),
            "assets": account.get("assets"),
        }
    if runtime.market_type == "futures":
        try:
            max_leverage = client.get_max_leverage(runtime.symbol)
        except BinanceAPIError as exc:
            print(f"unable to fetch leverage bracket: {exc}", file=sys.stderr)
        else:
            print(f"max leverage on {runtime.symbol}: {max_leverage}x")

    print("exchange: connected")
    print("market:", runtime.market_type)
    print("environment:", _runtime_environment(runtime))
    print("testnet:", runtime.testnet)
    print("demo:", getattr(runtime, "demo", False))
    print("endpoint:", client.base_url)
    print("server_time:", server_time.get("serverTime") if isinstance(server_time, dict) else server_time)
    print("account:", json.dumps(account, indent=2))
    return 0


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


def command_risk(args: argparse.Namespace) -> int:
    return risk_workflows.command_risk(
        args,
        load_runtime_fn=load_runtime,
        load_strategy_fn=load_strategy,
    )


def command_report(args: argparse.Namespace) -> int:
    if bool(args.account) and not _has_api_credentials(load_runtime()):
        print(_credential_required_message("Full report account section"), file=sys.stderr)
        return 2
    report = _render_operator_report(
        with_account=bool(args.account),
        doctor=bool(args.doctor),
        online=bool(args.online),
        input_path=args.input,
        model_path=args.model,
    )
    print(report)
    if bool(args.account) and "Account balances failed:" in report:
        return 2
    return 0


def command_prepare(args: argparse.Namespace) -> int:  # skipcq: PY-R1000
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
        should_calibrate_threshold = (
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
            "Download market data",
            command_fetch,
            argparse.Namespace(symbol=runtime.symbol, interval=runtime.interval, limit=limit, batch_size=batch_size, output=historical),
        ),
        (
            "Train AI model",
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
                calibrate_threshold=should_calibrate_threshold,
            ),
        ),
        (
            "Evaluate model",
            command_evaluate,
            argparse.Namespace(input=historical, model=model, threshold=None, calibrate_threshold=should_calibrate_threshold),
        ),
        (
            "Backtest strategy",
            command_backtest,
            argparse.Namespace(input=historical, model=model, start_cash=start_cash),
        ),
        (
            "Data/model audit",
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


def command_compute(args: argparse.Namespace) -> int:
    from .compute import describe_backend, resolve_backend

    runtime = load_runtime()
    requested = str(getattr(args, "backend", None) or runtime.compute_backend or "cpu").lower()
    if requested not in _COMPUTE_BACKEND_CHOICES:
        print(f"Unknown compute backend {requested!r}.", file=sys.stderr)
        return 2
    info = resolve_backend(requested)
    if getattr(args, "backend", None) is not None:
        runtime.compute_backend = requested
        save_runtime(runtime)
    print(describe_backend(info))
    return 0


def command_strategy(args: argparse.Namespace) -> int:  # skipcq: PY-R1000
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
    if getattr(args, "external_signals", None) is not None:
        updates["external_signals_enabled"] = bool(args.external_signals)
    if getattr(args, "external_signal_max_adjustment", None) is not None:
        updates["external_signal_max_adjustment"] = _clamp(float(args.external_signal_max_adjustment), 0.0, 0.20)
    if getattr(args, "external_signal_min_providers", None) is not None:
        updates["external_signal_min_providers"] = max(0, min(120, int(args.external_signal_min_providers)))
    if getattr(args, "external_signal_ttl", None) is not None:
        updates["external_signal_ttl_seconds"] = max(0, int(args.external_signal_ttl))
    if getattr(args, "external_signal_timeout", None) is not None:
        updates["external_signal_timeout_seconds"] = _clamp(float(args.external_signal_timeout), 0.1, 30.0)
    if getattr(args, "external_news_ai", None) is not None:
        updates["external_news_ai_enabled"] = bool(args.external_news_ai)
    if getattr(args, "external_news_ai_model", None):
        updates["external_news_ai_model"] = str(args.external_news_ai_model)
    if getattr(args, "external_news_ai_url", None):
        updates["external_news_ai_url"] = str(args.external_news_ai_url)
    if getattr(args, "external_news_ai_timeout", None) is not None:
        updates["external_news_ai_timeout_seconds"] = _clamp(float(args.external_news_ai_timeout), 0.1, 30.0)
    if getattr(args, "external_news_provider_limit", None) is not None:
        updates["external_signal_news_provider_limit"] = max(0, min(120, int(args.external_news_provider_limit)))
    if getattr(args, "external_provider_parallelism", None) is not None:
        updates["external_signal_provider_parallelism"] = max(1, min(64, int(args.external_provider_parallelism)))
    if getattr(args, "external_provider_jitter", None) is not None:
        updates["external_signal_provider_jitter_seconds"] = _clamp(float(args.external_provider_jitter), 0.0, 30.0)
    if getattr(args, "external_poll_jitter", None) is not None:
        updates["external_signal_poll_jitter_seconds"] = _clamp(float(args.external_poll_jitter), 0.0, 60.0)
    if getattr(args, "telemetry_db", None):
        updates["telemetry_db_path"] = str(args.telemetry_db)
    if getattr(args, "no_telemetry", None) is not None:
        updates["telemetry_enabled"] = not bool(args.no_telemetry)
    if getattr(args, "source_grading", None) is not None:
        updates["source_grading_enabled"] = bool(args.source_grading)
    if getattr(args, "source_grading_interval", None) is not None:
        updates["source_grading_interval_seconds"] = max(60, int(args.source_grading_interval))
    if getattr(args, "source_grading_window_hours", None) is not None:
        updates["source_grading_window_hours"] = max(1, int(args.source_grading_window_hours))
    if getattr(args, "source_grade_max_age_hours", None) is not None:
        updates["source_grade_max_age_hours"] = _clamp(float(args.source_grade_max_age_hours), 0.0, 8760.0)
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


def _runtime_with_market(runtime: RuntimeConfig, market_type: str) -> RuntimeConfig:
    return data_workflows.runtime_with_market(runtime, market_type)


def _data_sync_config_from_args(args: argparse.Namespace, runtime: RuntimeConfig) -> MarketDataSyncConfig:
    return data_workflows.data_sync_config_from_args(args, runtime)


def _start_background_data_sync(args: argparse.Namespace) -> int:
    return data_workflows.start_background_data_sync(
        args,
        python_executable=sys.executable,
        popen=subprocess.Popen,
    )


def command_data_sync(args: argparse.Namespace) -> int:
    return data_workflows.command_data_sync(
        args,
        load_runtime_fn=load_runtime,
        build_client_fn=_build_client,
        sync_market_data_fn=sync_market_data,
        render_sync_result_fn=render_sync_result,
        sleep_fn=time.sleep,
        python_executable=sys.executable,
        popen=subprocess.Popen,
    )


def command_fetch(args: argparse.Namespace) -> int:
    return data_workflows.command_fetch(
        args,
        load_runtime_fn=load_runtime,
        build_client_fn=_build_client,
    )


def _load_training_candles_from_db(
    db_path: str | Path,
    runtime: RuntimeConfig,
    *,
    interval: str,
    market_type: str,
    min_rows: int,
) -> list | None:
    return data_workflows.load_training_candles_from_db(
        db_path,
        runtime,
        interval=interval,
        market_type=market_type,
        min_rows=min_rows,
    )


def _confirm_download_missing_training_data(
    *,
    symbol: str,
    market_type: str,
    interval: str,
    available: int,
    required: int,
) -> bool:
    return data_workflows.confirm_download_missing_training_data(
        symbol=symbol,
        market_type=market_type,
        interval=interval,
        available=available,
        required=required,
        stdin=sys.stdin,
        input_fn=builtins.input,
    )


def _download_training_candles(args: argparse.Namespace, runtime: RuntimeConfig, *, interval: str, market_type: str) -> bool:
    return data_workflows.download_training_candles(
        args,
        runtime,
        interval=interval,
        market_type=market_type,
        command_fn=command_data_sync,
    )


def _load_training_candles(args: argparse.Namespace, runtime: RuntimeConfig) -> tuple[list | None, str]:
    return data_workflows.load_training_candles(
        args,
        runtime,
        load_rows_fn=_load_rows_for_command,
        db_loader_fn=_load_training_candles_from_db,
        confirm_fn=_confirm_download_missing_training_data,
        download_fn=_download_training_candles,
    )


def _classification_payload(report: object) -> dict[str, float | int]:
    return {
        "accuracy": float(getattr(report, "accuracy", 0.0)),
        "precision": float(getattr(report, "precision", 0.0)),
        "recall": float(getattr(report, "recall", 0.0)),
        "f1": float(getattr(report, "f1", 0.0)),
        "threshold": float(getattr(report, "threshold", 0.5)),
        "true_positive": int(getattr(report, "true_positive", 0)),
        "false_positive": int(getattr(report, "false_positive", 0)),
        "true_negative": int(getattr(report, "true_negative", 0)),
        "false_negative": int(getattr(report, "false_negative", 0)),
    }


def _threshold_classification_guard(baseline: object, candidate: object) -> dict[str, float | str | bool]:
    baseline_accuracy = float(getattr(baseline, "accuracy", 0.0))
    baseline_f1 = float(getattr(baseline, "f1", 0.0))
    baseline_precision = float(getattr(baseline, "precision", 0.0))
    candidate_accuracy = float(getattr(candidate, "accuracy", 0.0))
    candidate_f1 = float(getattr(candidate, "f1", 0.0))
    candidate_precision = float(getattr(candidate, "precision", 0.0))
    candidate_true_positive = int(getattr(candidate, "true_positive", 0))
    candidate_false_negative = int(getattr(candidate, "false_negative", 0))
    accuracy_floor = max(0.0, baseline_accuracy - 0.03)
    f1_floor = max(0.0, baseline_f1 - 0.05)
    precision_floor = max(0.0, baseline_precision - 0.02)
    detects_positive_cases = not (candidate_true_positive <= 0 < candidate_false_negative)
    stable_f1 = detects_positive_cases and candidate_accuracy >= accuracy_floor and candidate_f1 >= f1_floor
    conservative_precision = (
        detects_positive_cases
        and candidate_f1 > 0.0
        and candidate_accuracy >= baseline_accuracy + 0.02
        and candidate_precision >= precision_floor
    )
    passed = stable_f1 or conservative_precision
    if not detects_positive_cases:
        mode = "zero_true_positive"
    else:
        mode = "f1_stable" if stable_f1 else ("accuracy_precision" if conservative_precision else "rejected")
    return {
        "passed": bool(passed),
        "mode": mode,
        "accuracy_floor": float(accuracy_floor),
        "f1_floor": float(f1_floor),
        "precision_floor": float(precision_floor),
    }


def _threshold_capital_preservation_guard(
    profit_calibration: object,
    candidate_report: object | None = None,
) -> dict[str, float | int | str | bool]:
    baseline_score = float(getattr(profit_calibration, "baseline_score", 0.0))
    best_score = float(getattr(profit_calibration, "best_score", baseline_score))
    baseline_pnl = float(getattr(profit_calibration, "baseline_realized_pnl", 0.0))
    realized_pnl = float(getattr(profit_calibration, "realized_pnl", baseline_pnl))
    closed_trades = max(0, int(getattr(profit_calibration, "closed_trades", 0)))
    baseline_closed_trades = max(0, int(getattr(profit_calibration, "baseline_closed_trades", 0)))
    candidate_true_positive = int(getattr(candidate_report, "true_positive", 0)) if candidate_report else 1
    candidate_false_negative = int(getattr(candidate_report, "false_negative", 0)) if candidate_report else 0
    detects_positive_cases = not (candidate_true_positive <= 0 < candidate_false_negative)
    score_delta = best_score - baseline_score
    pnl_delta = realized_pnl - baseline_pnl
    material_pnl_improvement = pnl_delta >= max(1e-9, abs(baseline_pnl) * 0.10)
    tolerated_loss = -abs(baseline_pnl) * 0.10 if baseline_pnl < 0.0 else baseline_pnl
    passed = (
        bool(getattr(profit_calibration, "accepted", False))
        and detects_positive_cases
        and closed_trades > 0
        and score_delta >= 0.05
        and material_pnl_improvement
        and realized_pnl >= max(0.0, tolerated_loss)
    )
    return {
        "passed": bool(passed),
        "mode": "capital_preservation" if passed else ("zero_true_positive" if not detects_positive_cases else "rejected"),
        "score_delta": float(score_delta),
        "pnl_delta": float(pnl_delta),
        "closed_trades": int(closed_trades),
        "baseline_closed_trades": int(baseline_closed_trades),
        "tolerated_loss": float(tolerated_loss),
    }


def command_train(args: argparse.Namespace) -> int:  # skipcq: PY-R1000
    from .compute import describe_backend, BackendInfo

    try:
        args = _apply_training_preset(args)
    except ValueError as exc:
        print(f"Training settings invalid: {exc}", file=sys.stderr)
        return 2
    cfg = load_strategy()
    runtime = load_runtime()
    candles, training_source = _load_training_candles(args, runtime)
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
    compute_backend = str(getattr(args, "compute_backend", None) or runtime.compute_backend or "cpu").lower()
    if compute_backend not in _COMPUTE_BACKEND_CHOICES:
        print(f"Training settings invalid: unknown compute backend {compute_backend!r}.", file=sys.stderr)
        return 2
    batch_size = max(1, int(getattr(args, "batch_size", 8192) or 8192))

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
            fold_values = wf["scores"]
            if isinstance(fold_values, list) and fold_values:
                print(
                    "walk-forward fold scores: "
                    + ", ".join(f"{float(v):.3f}" for v in fold_values)
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
        validation_rows=calibration_rows,
        early_stopping_rounds=max(5, min(25, int(args.epochs) // 5)) if calibration_rows else None,
        compute_backend=compute_backend,
        batch_size=batch_size,
    )
    backend_info = BackendInfo(
        requested=model.training_backend_requested,
        kind=cast(Any, model.training_backend_kind),
        device=model.training_backend_device,
        vendor=model.training_backend_vendor,
        reason=model.training_backend_reason,
    )
    probability_calibration = (
        calibrate_probability_temperature(calibration_rows, model)
        if calibration_rows
        else assess_probability_calibration([], model)
    )
    if probability_calibration.status != "fail":
        model.probability_temperature = float(probability_calibration.temperature)
        model.probability_calibration_size = int(probability_calibration.rows)
        model.probability_log_loss_before = float(probability_calibration.log_loss_before)
        model.probability_log_loss_after = float(probability_calibration.log_loss_after)
        model.probability_brier_before = float(probability_calibration.brier_before)
        model.probability_brier_after = float(probability_calibration.brier_after)
        model.probability_ece_before = float(probability_calibration.expected_calibration_error_before)
        model.probability_ece_after = float(probability_calibration.expected_calibration_error_after)
    threshold = cfg.signal_threshold
    threshold_source = "strategy"
    threshold_calibration: dict[str, object] | None = None
    if args.calibrate_threshold and calibration_rows:
        strategy_report = evaluate_classification(calibration_rows, model, threshold=cfg.signal_threshold)
        classification_threshold = calibrate_threshold(calibration_rows, model, start=0.05, end=0.95, steps=31)
        classification_report = evaluate_classification(calibration_rows, model, threshold=classification_threshold)
        profit_calibration = calibrate_threshold_for_backtest(
            calibration_rows,
            model,
            cfg,
            starting_cash=1000.0,
            market_type=runtime.market_type,
            baseline_threshold=cfg.signal_threshold,
            start=0.05,
            end=0.95,
            steps=181,
            compute_backend=compute_backend,
            score_batch_size=batch_size,
        )
        profit_report = evaluate_classification(calibration_rows, model, threshold=profit_calibration.best_threshold)
        classification_guard = _threshold_classification_guard(strategy_report, profit_report)
        classification_guard_passed = bool(classification_guard["passed"])
        capital_guard = _threshold_capital_preservation_guard(profit_calibration, profit_report)
        capital_guard_passed = bool(capital_guard["passed"])
        if profit_calibration.accepted and (classification_guard_passed or capital_guard_passed):
            threshold = profit_calibration.threshold
            threshold_source = "profit_backtest"
        threshold_calibration = {
            "source": threshold_source,
            "strategy_baseline": _classification_payload(strategy_report),
            "classification": _classification_payload(classification_report),
            "profit_candidate": _classification_payload(profit_report),
            "profit_backtest": profit_calibration.asdict(),
            "classification_guard": classification_guard,
            "capital_guard": capital_guard,
        }
    model.decision_threshold = float(threshold)
    model.calibration_size = len(calibration_rows) if args.calibrate_threshold else 0
    model.validation_size = len(validation_rows)
    model.training_cutoff_timestamp = train_rows[-1].timestamp if train_rows else None
    model.threshold_source = threshold_source
    if threshold_calibration:
        profit_backtest = cast(dict[str, object], threshold_calibration["profit_backtest"])
        threshold_score = cast(float | int | str, profit_backtest["score"])
        threshold_pnl = cast(float | int | str, profit_backtest["realized_pnl"])
        threshold_trades = cast(float | int | str, profit_backtest["closed_trades"])
        model.threshold_calibration_score = float(threshold_score)
        model.threshold_calibration_pnl = float(threshold_pnl)
        model.threshold_calibration_trades = int(threshold_trades)
    train_score = evaluate(train_rows, model, threshold=threshold)
    calibration_score = evaluate(calibration_rows, model, threshold=threshold) if calibration_rows else 0.0
    validation_score = evaluate(validation_rows, model, threshold=threshold) if validation_rows else 0.0
    quality = build_model_quality_report(train_rows, validation_rows, model, threshold)
    drift = feature_drift_report(validation_rows if validation_rows else rows[-min(50, len(rows)):], model)
    model.quality_score = float(quality.quality_score)
    model.quality_warnings = list(quality.warnings)
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
        print(f"validated threshold: {threshold:.3f} source={threshold_source}")
        print(f"calibration directional accuracy: {calibration_score:.3f}")
        threshold_calibration_payload = cast(dict[str, object], threshold_calibration)
        profit_backtest = cast(dict[str, object], threshold_calibration_payload["profit_backtest"])
        accepted = "accepted" if threshold_source == "profit_backtest" else "rejected"
        best_threshold = float(cast(float | int | str, profit_backtest["best_threshold"]))
        best_score = float(cast(float | int | str, profit_backtest["best_score"]))
        baseline_score = float(cast(float | int | str, profit_backtest["baseline_score"]))
        print(
            "profit threshold candidate: "
            f"{accepted} best={best_threshold:.3f} "
            f"score={best_score:.3f} "
            f"baseline={baseline_score:.3f}"
        )
    print(f"model quality: {quality.status} score={quality.quality_score:.2f}")
    for warning in quality.warnings[:3]:
        print(f"model quality warning: {warning}")
    print(
        f"probability calibration: {probability_calibration.status} "
        f"temperature={probability_calibration.temperature:.2f} "
        f"brier={probability_calibration.brier_before:.3f}->{probability_calibration.brier_after:.3f} "
        f"log_loss={probability_calibration.log_loss_before:.3f}->{probability_calibration.log_loss_after:.3f}"
    )
    for warning in probability_calibration.warnings[:3]:
        print(f"probability calibration warning: {warning}")
    print(
        f"feature drift: {drift.status} max_z={drift.max_abs_z:.2f} "
        f"outliers={drift.outlier_fraction:.1%}"
    )
    for warning in drift.warnings[:3]:
        print(f"feature drift warning: {warning}")
    artifact = {
        "command": "train",
        "timestamp": int(time.time()),
        "seed": int(seed),
        "runtime": _public_runtime_payload(runtime),
        "strategy": cfg.asdict(),
        "train": {
            "input": str(args.input),
            "data_source": training_source,
            "db": str(getattr(args, "db", "data/market_data.sqlite")),
            "interval": str(getattr(args, "interval", None) or runtime.interval),
            "output": str(args.output),
            "rows_total": len(rows),
            "rows_train": len(train_rows),
            "rows_calibration": len(calibration_rows),
            "rows_validation": len(validation_rows),
            "epochs": int(args.epochs),
            "learning_rate": float(learning_rate),
            "l2_penalty": float(l2_penalty),
            "compute_backend": compute_backend,
            "batch_size": int(batch_size),
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
            "threshold_source": threshold_source,
            "threshold_calibration": threshold_calibration,
            "calibrated_out_of_sample_accuracy": float(validation_score)
            if args.calibrate_threshold and validation_rows
            else None,
            "model_feature_version": model.feature_version,
            "model_feature_signature": model.feature_signature,
        },
        "model_quality": quality.asdict(),
        "probability_calibration": probability_calibration.asdict(),
        "feature_drift": drift.asdict(),
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
            "best_epoch": int(model.best_epoch)
            if model.best_epoch is not None
            else None,
            "training_loss": float(model.training_loss)
            if model.training_loss is not None
            else None,
            "validation_loss": float(model.validation_loss)
            if model.validation_loss is not None
            else None,
            "quality_score": float(model.quality_score)
            if model.quality_score is not None
            else None,
            "quality_warnings": list(model.quality_warnings),
            "probability_temperature": float(model.probability_temperature),
            "probability_calibration_size": int(model.probability_calibration_size),
            "probability_log_loss_before": float(model.probability_log_loss_before)
            if model.probability_log_loss_before is not None
            else None,
            "probability_log_loss_after": float(model.probability_log_loss_after)
            if model.probability_log_loss_after is not None
            else None,
            "probability_brier_before": float(model.probability_brier_before)
            if model.probability_brier_before is not None
            else None,
            "probability_brier_after": float(model.probability_brier_after)
            if model.probability_brier_after is not None
            else None,
            "probability_ece_before": float(model.probability_ece_before)
            if model.probability_ece_before is not None
            else None,
            "probability_ece_after": float(model.probability_ece_after)
            if model.probability_ece_after is not None
            else None,
            "threshold_source": model.threshold_source,
            "threshold_calibration_score": float(model.threshold_calibration_score)
            if model.threshold_calibration_score is not None
            else None,
            "threshold_calibration_pnl": float(model.threshold_calibration_pnl)
            if model.threshold_calibration_pnl is not None
            else None,
            "threshold_calibration_trades": int(model.threshold_calibration_trades),
            "training_backend_requested": model.training_backend_requested,
            "training_backend_kind": model.training_backend_kind,
            "training_backend_device": model.training_backend_device,
            "training_backend_vendor": model.training_backend_vendor,
            "training_backend_reason": model.training_backend_reason,
        },
        "market": runtime.market_type,
        "symbol": runtime.symbol,
    }
    resolved_leverage = _resolve_futures_leverage(runtime, cfg)
    print(f"training backend: {describe_backend(backend_info)}")
    print(f"market={runtime.market_type} leverage={resolved_leverage:.2f}")
    artifact_path = _persist_run_artifact("train", Path(args.output).parent, artifact)
    print(f"saved train artifact to {artifact_path}")
    return 0


def command_tune(args: argparse.Namespace) -> int:  # skipcq: PY-R1000
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
    try:
        model = _load_readiness_model(model_path, cfg)[0]
    except (OSError, json.JSONDecodeError, ModelLoadError) as exc:
        print(f"Model load failed: {exc}", file=sys.stderr)
        return 2
    cfg = apply_model_strategy_overrides(cfg, model)
    rows = _backtest_rows_for_model(candles, cfg, model)
    decision_threshold = model_decision_threshold(model, cfg.signal_threshold)
    compute_backend = str(getattr(args, "compute_backend", None) or runtime.compute_backend or "cpu")
    score_batch_size = max(1, int(getattr(args, "score_batch_size", 8192) or 8192))
    result = run_backtest(
        rows,
        model,
        cfg,
        starting_cash=args.start_cash,
        market_type=runtime.market_type,
        compute_backend=compute_backend,
        score_batch_size=score_batch_size,
    )
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
        "scoring_backend": {
            "requested": result.scoring_backend_requested,
            "kind": result.scoring_backend_kind,
            "device": result.scoring_backend_device,
            "reason": result.scoring_backend_reason,
            "score_batch_size": score_batch_size,
        },
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
    print(f"scoring_backend: {result.scoring_backend_kind} device={result.scoring_backend_device}")
    if result.scoring_backend_reason:
        print(f"scoring_backend_reason: {result.scoring_backend_reason}")
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
    return risk_adjusted_backtest_score(result, starting_cash=starting_cash)


def command_evaluate(args: argparse.Namespace) -> int:  # skipcq: PY-R1000
    cfg = load_strategy()
    runtime = load_runtime()
    candles = _load_rows_for_command(args.input, label="Evaluation data load failed")
    if candles is None:
        return 2
    base_rows = _build_model_rows(candles, cfg)
    if not base_rows:
        print("No rows available for evaluation. Fetch more data first.")
        return 2

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Model file not found: {model_path}")
        return 2

    try:
        model = _load_readiness_model(model_path, cfg)[0]
    except (OSError, json.JSONDecodeError, ModelLoadError) as exc:
        print(f"Model load failed: {exc}", file=sys.stderr)
        return 2
    cfg = apply_model_strategy_overrides(cfg, model)
    rows = _readiness_model_rows(candles, cfg, model)
    if not rows:
        print("No rows available for evaluation. Fetch more data first.")
        return 2

    split = temporal_validation_split(rows)
    train_rows = split.train_rows
    calibration_rows = split.calibration_rows
    test_rows = split.validation_rows

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
    quality = build_model_quality_report(train_rows, test_rows if test_rows else rows, model, threshold)
    probability_calibration = assess_probability_calibration(test_rows if test_rows else rows, model)
    drift = feature_drift_report(test_rows if test_rows else rows, model)
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
        "model_quality": quality.asdict(),
        "probability_calibration": probability_calibration.asdict(),
        "feature_drift": drift.asdict(),
    }
    _persist_run_artifact("evaluate", Path(args.model).parent, artifact)

    print(f"evaluate model={model_path}")
    print(f"threshold: {report.threshold:.3f}")
    train_accuracy = train_report.accuracy if train_report is not None else 0.0
    train_precision = train_report.precision if train_report is not None else 0.0
    train_recall = train_report.recall if train_report is not None else 0.0
    train_f1 = train_report.f1 if train_report is not None else 0.0
    print(
        "train_accuracy: "
        f"{train_accuracy:.3f} precision={train_precision:.3f} "
        f"recall={train_recall:.3f} f1={train_f1:.3f}"
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
    print(f"model_quality: {quality.status} score={quality.quality_score:.2f}")
    for warning in quality.warnings[:3]:
        print(f"quality_warning: {warning}")
    print(
        f"probability_calibration: {probability_calibration.status} "
        f"temperature={probability_calibration.temperature:.2f} "
        f"brier={probability_calibration.brier_after:.3f} "
        f"ece={probability_calibration.expected_calibration_error_after:.3f}"
    )
    for warning in probability_calibration.warnings[:3]:
        print(f"probability_warning: {warning}")
    print(
        f"feature_drift: {drift.status} max_z={drift.max_abs_z:.2f} "
        f"outliers={drift.outlier_fraction:.1%}"
    )
    for warning in drift.warnings[:3]:
        print(f"drift_warning: {warning}")
    return 0


def _external_signal_cache_path(model_path: Path) -> Path:
    return model_path.parent / "signals" / "external_signals.json"


def _strategy_with_external_risk(cfg: StrategyConfig, risk_multiplier: float) -> StrategyConfig:
    multiplier = _clamp(float(risk_multiplier), 0.0, 1.0)
    if multiplier >= 0.999:
        return cfg
    return StrategyConfig(
        **{
            **cfg.asdict(),
            "risk_per_trade": max(0.0001, cfg.risk_per_trade * multiplier),
            "max_position_pct": max(0.0, cfg.max_position_pct * multiplier),
        }
    )


def _apply_external_signal_to_score(
    score: float,
    cfg: StrategyConfig,
    report: ExternalSignalReport,
) -> tuple[float, StrategyConfig, float]:
    applied_adjustment = float(report.score_adjustment)
    if report.fresh_count < max(0, int(cfg.external_signal_min_providers)):
        applied_adjustment = min(0.0, applied_adjustment)
    max_adjustment = _clamp(float(cfg.external_signal_max_adjustment), 0.0, 0.20)
    applied_adjustment = _clamp(applied_adjustment, -max_adjustment, max_adjustment)
    adjusted_score = _clamp(float(score) + applied_adjustment, 0.0, 1.0)
    effective_cfg = _strategy_with_external_risk(cfg, report.risk_multiplier)
    return adjusted_score, effective_cfg, applied_adjustment


def _record_model_telemetry(
    cfg: StrategyConfig,
    *,
    step: int,
    row: ModelRow,
    raw_score: float,
    adjusted_score: float,
    threshold: float,
    model: object,
    runtime: RuntimeConfig,
) -> None:
    if not cfg.telemetry_enabled:
        return
    try:
        from .telemetry_store import TradingTelemetryStore

        payload = {
            "step": int(step),
            "timestamp": int(row.timestamp),
            "close": float(row.close),
            "features": [float(value) for value in row.features],
            "raw_score": float(raw_score),
            "adjusted_score": float(adjusted_score),
            "threshold": float(threshold),
            "model_type": model.__class__.__name__,
            "model_signature": str(getattr(model, "feature_signature", "") or ""),
            "training_backend_kind": str(getattr(model, "training_backend_kind", "") or ""),
            "runtime_compute_backend": str(runtime.compute_backend),
        }
        with TradingTelemetryStore(cfg.telemetry_db_path) as store:
            store.record_observation(
                kind="model_decision",
                source="internal_model",
                payload=payload,
                observed_at_ms=int(row.timestamp),
                symbol=runtime.symbol,
                horizon="short",
                score=float(adjusted_score),
                confidence=abs(float(adjusted_score) - float(threshold)),
            )
    except Exception:  # pragma: no cover - telemetry failures must not stop live scoring
        return


def command_signals(args: argparse.Namespace) -> int:  # skipcq: PY-R1000
    runtime = load_runtime()
    cfg = load_strategy()
    model_path = Path(getattr(args, "model", "data/model.json"))
    cache_path = Path(getattr(args, "cache", None) or _external_signal_cache_path(model_path))

    def run_once() -> tuple[int, ExternalSignalReport | None]:
        try:
            telemetry_path = None if getattr(args, "no_telemetry", False) or not cfg.telemetry_enabled else (
                getattr(args, "telemetry_db", None) or cfg.telemetry_db_path
            )
            report = collect_external_signals(
                symbol=runtime.symbol,
                cache_path=cache_path,
                ttl_seconds=max(0, int(getattr(args, "ttl", 300))),
                timeout_seconds=_clamp(float(getattr(args, "timeout", 3.0)), 0.1, 30.0),
                max_adjustment=_clamp(float(getattr(args, "max_adjustment", 0.04)), 0.0, 0.20),
                min_providers=max(0, min(120, int(getattr(args, "min_providers", 2)))),
                force_refresh=bool(getattr(args, "refresh", False)),
                compute_backend=str(getattr(args, "compute_backend", None) or runtime.compute_backend or "cpu"),
                short_reaction_refresh_seconds=max(1, int(getattr(args, "short_reaction_refresh", 30))),
                news_provider_limit=max(
                    0,
                    int(getattr(args, "news_provider_limit", None) or cfg.external_signal_news_provider_limit),
                ),
                news_items_per_provider=max(
                    1,
                    int(getattr(args, "news_items_per_provider", None) or cfg.external_signal_news_items_per_provider),
                ),
                news_provider_parallelism=max(
                    1,
                    int(getattr(args, "provider_parallelism", None) or cfg.external_signal_provider_parallelism),
                ),
                news_provider_jitter_seconds=max(
                    0.0,
                    float(
                        getattr(args, "provider_jitter", None)
                        if getattr(args, "provider_jitter", None) is not None
                        else cfg.external_signal_provider_jitter_seconds
                    ),
                ),
                ollama_news_enabled=(
                    bool(getattr(args, "ollama_news", None))
                    if getattr(args, "ollama_news", None) is not None
                    else cfg.external_news_ai_enabled
                ),
                ollama_model=str(getattr(args, "ollama_model", None) or cfg.external_news_ai_model),
                ollama_url=str(getattr(args, "ollama_url", None) or cfg.external_news_ai_url),
                ollama_timeout_seconds=_clamp(
                    float(getattr(args, "ollama_timeout", None) or cfg.external_news_ai_timeout_seconds),
                    0.1,
                    30.0,
                ),
                telemetry_path=telemetry_path,
                source_grade_max_age_hours=_clamp(
                    float(
                        cfg.source_grade_max_age_hours
                        if getattr(args, "source_grade_max_age_hours", None) is None
                        else getattr(args, "source_grade_max_age_hours", None)
                    ),
                    0.0,
                    8760.0,
                ),
            )
        except Exception as exc:
            print(f"External signal collection failed: {exc}", file=sys.stderr)
            return 2, None
        if getattr(args, "json", False):
            print(json.dumps(report.asdict(), indent=2, sort_keys=True))
        else:
            print(render_external_signal_report(report))
        return (0 if report.status != "fail" else 2), report

    if not getattr(args, "loop", False):
        code, _report = run_once()
        return code

    iterations = max(0, int(getattr(args, "iterations", 0) or 0))
    sleep_seconds = float(getattr(args, "sleep", None) if getattr(args, "sleep", None) is not None else 60.0)
    jitter_seconds = float(
        getattr(args, "jitter", None)
        if getattr(args, "jitter", None) is not None
        else cfg.external_signal_poll_jitter_seconds
    )
    code = 0
    completed = 0
    try:
        while iterations <= 0 or completed < iterations:  # pragma: no branch
            code, _report = run_once()
            completed += 1
            if iterations > 0 and completed >= iterations:
                break
            time.sleep(_jittered_seconds(sleep_seconds, jitter_seconds))
    except KeyboardInterrupt:  # pragma: no cover - manual operator stop path
        print("Signal loop stopped.")
    return code


def command_source_grades(args: argparse.Namespace) -> int:
    cfg = load_strategy()
    try:
        run = grade_sources(
            db_path=getattr(args, "db", None) or cfg.telemetry_db_path,
            window_hours=float(getattr(args, "window_hours", None) or cfg.source_grading_window_hours),
            model=str(getattr(args, "ollama_model", None) or cfg.external_news_ai_model),
            ollama_enabled=(
                bool(getattr(args, "ollama", None))
                if getattr(args, "ollama", None) is not None
                else cfg.source_grading_enabled
            ),
            ollama_url=str(getattr(args, "ollama_url", None) or cfg.external_news_ai_url),
            ollama_timeout_seconds=_clamp(
                float(getattr(args, "ollama_timeout", None) or cfg.external_news_ai_timeout_seconds),
                0.1,
                120.0,
            ),
        )
    except Exception as exc:
        print(f"Source grading failed: {exc}", file=sys.stderr)
        return 2
    if getattr(args, "json", False):
        print(json.dumps(run.asdict(), indent=2, sort_keys=True))
    else:
        print(render_source_grade_run(run))
    return 0 if run.status != "empty" else 2


def command_signals_benchmark(args: argparse.Namespace) -> int:  # skipcq: PY-R1000
    runtime = load_runtime()
    cfg = load_strategy()
    provider_limits = [max(0, int(value)) for value in (getattr(args, "provider_limit", None) or [30, 60])]
    parallelism_values = [max(1, int(value)) for value in (getattr(args, "parallelism", None) or [8, 16])]
    iterations = max(1, int(getattr(args, "iterations", 1) or 1))
    rows: list[dict[str, object]] = []
    worst_code = 0
    for provider_limit in provider_limits:
        for parallelism in parallelism_values:
            durations: list[float] = []
            fresh_counts: list[int] = []
            provider_counts: list[int] = []
            statuses: list[str] = []
            for _iteration in range(iterations):
                started = time.perf_counter()
                try:
                    report = collect_external_signals(
                        symbol=runtime.symbol,
                        cache_path=getattr(args, "cache", "data/signals/benchmark_external_signals.json"),
                        ttl_seconds=0,
                        timeout_seconds=_clamp(float(getattr(args, "timeout", 3.0)), 0.1, 30.0),
                        max_adjustment=cfg.external_signal_max_adjustment,
                        min_providers=cfg.external_signal_min_providers,
                        force_refresh=True,
                        compute_backend=runtime.compute_backend,
                        news_provider_limit=provider_limit,
                        news_provider_parallelism=parallelism,
                        news_provider_jitter_seconds=max(0.0, float(getattr(args, "provider_jitter", 0.0) or 0.0)),
                        ollama_news_enabled=(
                            bool(getattr(args, "ollama_news", None))
                            if getattr(args, "ollama_news", None) is not None
                            else cfg.external_news_ai_enabled
                        ),
                        ollama_model=str(getattr(args, "ollama_model", None) or cfg.external_news_ai_model),
                        ollama_url=str(getattr(args, "ollama_url", None) or cfg.external_news_ai_url),
                        ollama_timeout_seconds=_clamp(
                            float(getattr(args, "ollama_timeout", None) or cfg.external_news_ai_timeout_seconds),
                            0.1,
                            30.0,
                        ),
                        telemetry_path=None if getattr(args, "no_telemetry", False) or not cfg.telemetry_enabled else cfg.telemetry_db_path,
                        source_grade_max_age_hours=cfg.source_grade_max_age_hours,
                    )
                    status = report.status
                    fresh_counts.append(report.fresh_count)
                    provider_counts.append(report.provider_count)
                    if status == "fail":
                        worst_code = 2
                except Exception as exc:
                    status = "error"
                    worst_code = 2
                    fresh_counts.append(0)
                    provider_counts.append(provider_limit)
                    print(f"benchmark trial failed: {exc}", file=sys.stderr)
                durations.append((time.perf_counter() - started) * 1000.0)
                statuses.append(status)
            average_ms = sum(durations) / len(durations)
            row = {
                "provider_limit": provider_limit,
                "parallelism": parallelism,
                "iterations": iterations,
                "avg_ms": round(average_ms, 2),
                "max_ms": round(max(durations), 2),
                "avg_fresh": round(sum(fresh_counts) / len(fresh_counts), 2),
                "avg_providers": round(sum(provider_counts) / len(provider_counts), 2),
                "statuses": statuses,
            }
            rows.append(row)
    payload = {"command": "signals-benchmark", "trials": rows}
    if getattr(args, "json", False):
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print("Signal benchmark")
        for row in rows:
            print(
                f"- providers={row['provider_limit']} parallelism={row['parallelism']} "
                f"avg_ms={row['avg_ms']} max_ms={row['max_ms']} "
                f"fresh={row['avg_fresh']}/{row['avg_providers']} statuses={','.join(row['statuses'])}"
            )
    return worst_code


def command_live(args: argparse.Namespace) -> int:  # skipcq: PY-R1000
    runtime = load_runtime()
    cfg = load_strategy()
    if getattr(args, "paper", False) and getattr(args, "live", False):
        print("Choose either --paper or --live, not both.")
        return 2
    leverage_override = getattr(args, "leverage", None)
    if leverage_override is not None and runtime.market_type != "futures":
        print("Leverage override is spot-inactive; spot runs at 1x.")
    external_override = getattr(args, "external_signals", None)
    if external_override is not None:
        cfg = StrategyConfig(**{**cfg.asdict(), "external_signals_enabled": bool(external_override)})
    model_path = Path(getattr(args, "model", "data/model.json"))

    if getattr(args, "live", False):
        effective_dry_run = False
    else:
        effective_dry_run = runtime.dry_run or getattr(args, "paper", False)
    if not _allows_signed_execution(runtime):
        print("Real-money execution is disabled in this phase. Set testnet=true or demo=true to run.")
        return 2
    if not effective_dry_run and not _has_api_credentials(runtime):
        print(_credential_required_message("Authenticated live mode"), file=sys.stderr)
        return 2
    try:
        client = _build_client(runtime)
    except BinanceAPIError as exc:
        print(f"Live startup blocked: {exc}", file=sys.stderr)
        return 2
    sleep_seconds = max(0, int(getattr(args, "sleep", 0)))
    if not effective_dry_run and sleep_seconds < 1:
        print("Authenticated live mode uses minimum sleep=1s.")
        sleep_seconds = 1

    def live_sleep() -> None:
        delay = _jittered_seconds(sleep_seconds, cfg.external_signal_poll_jitter_seconds) if sleep_seconds > 0 else 0.0
        time.sleep(delay)

    model, model_error, model_notice = _load_live_start_model(model_path, cfg, effective_dry_run=effective_dry_run)
    if model_error is not None:
        print(model_error, file=sys.stderr)
        return 2
    if model is not None:
        protected = ("leverage",) if leverage_override is not None else ()
        cfg = apply_model_strategy_overrides(cfg, model, protected_keys=protected)
    if leverage_override is not None and runtime.market_type == "futures":
        cfg = StrategyConfig(**{**cfg.asdict(), "leverage": max(1.0, float(leverage_override))})
    if external_override is not None:
        cfg = StrategyConfig(**{**cfg.asdict(), "external_signals_enabled": bool(external_override)})
    if model_notice is not None:
        print(model_notice, file=sys.stderr)

    leverage = _resolve_futures_leverage(runtime, cfg)
    cash = max(0.0, _safe_float(getattr(runtime, "managed_usdc", 0.0)))
    exchange_account_snapshot: object | None = None
    if not effective_dry_run:
        try:
            exchange_account_snapshot = client.get_account()
        except BinanceAPIError as exc:
            print(f"Account balance check failed: {exc}", file=sys.stderr)
            return 2
        balances = _account_free_balances(exchange_account_snapshot)
        available_usdc = max(0.0, _safe_float(balances.get("USDC", 0.0)))
        if cash <= 0.0:
            print("Authenticated live mode requires a positive USDC trading cap from Funds.", file=sys.stderr)
            return 2
        if available_usdc <= 0.0:
            print("Authenticated live mode requires available USDC on the exchange account.", file=sys.stderr)
            return 2
        if cash > available_usdc:
            cash = available_usdc
            print(f"Authenticated live cash capped to exchange free USDC={cash:.2f}.")
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
    risk_policy = build_risk_policy_report(
        runtime,
        cfg,
        effective_dry_run=effective_dry_run,
        leverage=leverage,
    )
    if not risk_policy.allowed:
        print(render_risk_policy_report(risk_policy), file=sys.stderr)
        return 2
    if runtime.market_type == "futures" and not effective_dry_run:
        try:
            set_response = client.set_leverage(runtime.symbol, int(leverage))
            leverage_value = set_response.get("leverage") if isinstance(set_response, dict) else None
            if leverage_value is not None:
                leverage = _safe_float(leverage_value) or leverage
                post_set_risk_policy = build_risk_policy_report(
                    runtime,
                    cfg,
                    effective_dry_run=effective_dry_run,
                    leverage=leverage,
                )
                if not post_set_risk_policy.allowed:
                    print(render_risk_policy_report(post_set_risk_policy), file=sys.stderr)
                    return 2
        except BinanceAPIError as exc:
            print(f"Failed to set leverage: {exc}", file=sys.stderr)
            return 2

    mode_label = "paper" if effective_dry_run else "live"
    print(f"Starting {mode_label} loop for {args.steps} steps on {runtime.symbol} {runtime.interval} [{runtime.market_type}]")
    if runtime.market_type == "futures":
        print(f"effective leverage: {leverage:.1f}x")
    if cfg.external_signals_enabled:
        print(
            "external signals: enabled "
            f"max_adjust={cfg.external_signal_max_adjustment:.3f} "
            f"min_providers={cfg.external_signal_min_providers}"
        )

    fee_rate = max(0.0, cfg.taker_fee_bps) / 10_000.0
    slippage = max(0.0, cfg.slippage_bps) / 10_000.0
    constraints = _resolve_symbol_constraints(runtime, client)
    if constraints is None and not effective_dry_run:
        print("Authenticated live mode requires BTCUSDC exchange filters before any order is allowed.", file=sys.stderr)
        return 2
    max_daily_trades = int(cfg.max_trades_per_day)
    max_daily_trades = max(max_daily_trades, 0)
    daily_trade_count: dict[int, int] = {}
    max_open_positions = int(cfg.max_open_positions)
    live_run = build_live_run_payload(
        runtime_public=_public_runtime_payload(runtime),
        strategy=cfg,
        steps_total=int(args.steps),
        market=runtime.market_type,
        symbol=runtime.symbol,
        model_path=model_path,
        model=model,
        starting_cash=cash,
        external_signal_cache=_external_signal_cache_path(model_path),
        risk_policy=risk_policy,
    )
    live_events = cast(list[dict[str, object]], live_run["events"])
    if risk_policy.warning_count:
        print(f"risk policy warnings: {risk_policy.warning_count}")
    if not effective_dry_run:
        try:
            detected_position = _detect_existing_position(
                runtime,
                client,
                leverage=leverage,
                account=exchange_account_snapshot,
            )
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
            entry_basis = str(detected_position.get("entry_price_basis") or "exchange_entry_price")
            basis_note = " (mark-price basis; PnL estimate only)" if entry_basis != "exchange_entry_price" else ""
            print(
                "Detected existing exchange position: "
                f"{'long' if position_side > 0 else 'short'} qty={qty:.8f} entry={entry_price:.2f}{basis_note}"
            )
            live_events.append(
                {
                    "step": 0,
                    "status": "resume_exchange_position",
                    "market": str(detected_position["market"]),
                    "direction": int(position_side),
                    "qty": float(qty),
                    "entry_price": float(entry_price),
                    "entry_price_basis": entry_basis,
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
    def _next_source_grade_time_ms(after_ms: int) -> int:  # pragma: no cover - hourly live-loop scheduler
        return after_ms + int(_jittered_seconds(cfg.source_grading_interval_seconds, cfg.external_signal_poll_jitter_seconds) * 1000)

    next_source_grade_ms = _next_source_grade_time_ms(int(time.time() * 1000))

    def maybe_grade_sources(step: int) -> None:  # pragma: no cover - exercised by long-running live sessions
        nonlocal next_source_grade_ms
        if not cfg.telemetry_enabled or not cfg.source_grading_enabled:
            return
        current_ms = int(time.time() * 1000)
        if current_ms < next_source_grade_ms:
            return
        try:
            grade_run = grade_sources(
                db_path=cfg.telemetry_db_path,
                window_hours=cfg.source_grading_window_hours,
                model=cfg.external_news_ai_model,
                ollama_enabled=bool(cfg.source_grading_enabled),
                ollama_url=cfg.external_news_ai_url,
                ollama_timeout_seconds=cfg.external_news_ai_timeout_seconds,
            )
            live_events.append(
                {
                    "step": int(step),
                    "status": "source_grading",
                    "graded_sources": int(grade_run.graded_sources),
                    "ai_status": grade_run.ai_status,
                    "warnings": list(grade_run.warnings),
                }
            )
            print(
                f"step {step:>2}: source grades "
                f"graded={grade_run.graded_sources} ai={grade_run.ai_status}"
            )
        except Exception as exc:
            live_events.append({"step": int(step), "status": "source_grading_error", "error": str(exc)})
            print(f"step {step:>2}: source grading unavailable: {exc}", file=sys.stderr)
        finally:
            next_source_grade_ms = _next_source_grade_time_ms(current_ms)

    def record_order_error(step: int, side: str, size: float, exc: BinanceAPIError) -> None:
        nonlocal halt_reason, exit_code
        print(f"order error: {exc}", file=sys.stderr)
        halt_reason = "order_error"
        exit_code = 2
        live_events.append(
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
            live_events.append({"step": i + 1, "status": "market_error", "error": str(exc)})
            break

        steps_executed += 1

        rows = _live_rows_for_model(candles, cfg, model)
        training_rows = _readiness_model_rows(candles, cfg, model) if model is not None else _build_model_rows(candles, cfg)
        if not rows:
            print("not enough historical data for live signal")
            live_events.append({"step": i + 1, "status": "no_rows"})
            live_sleep()
            continue

        live_events.append({"step": i + 1, "status": "rows", "count": len(rows)})

        retrain_interval = getattr(args, "retrain_interval", 0)
        retrain_window = getattr(args, "retrain_window", 300)
        retrain_min_rows = getattr(args, "retrain_min_rows", 240)
        if retrain_min_rows <= 0:
            retrain_min_rows = max(1, 240)
        if retrain_window <= 0:
            retrain_window = max(1, 300)
        retrain_interval = max(retrain_interval, 0)

        previous_model = model
        model = _build_live_model(
            training_rows,
            model=model,
            retrain_every=retrain_interval,
            step=i + 1,
            cfg=cfg,
            retrain_window=retrain_window,
            retrain_min_rows=retrain_min_rows,
            model_feature_signature=_live_model_feature_signature(model, cfg),
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
            if not training_rows:
                print("not enough labeled historical data to train a live model")
                live_events.append({"step": i + 1, "status": "no_training_rows"})
                live_sleep()
                continue
            model = train(training_rows, epochs=40, feature_signature=_live_model_feature_signature(model, cfg))
            model_loads += 1
        latest = rows[-1]
        if hasattr(model, "feature_means") and hasattr(model, "feature_stds"):
            try:
                drift = feature_drift_report([latest], model)
            except ValueError as exc:
                if not effective_dry_run:
                    print(f"Live feature drift check failed: {exc}", file=sys.stderr)
                    halt_reason = "feature_drift_check_failed"
                    exit_code = 2
                    live_events.append({"step": i + 1, "status": halt_reason, "error": str(exc)})
                    break
                drift = None
            if drift is not None:
                live_events.append(
                    {
                        "step": i + 1,
                        "status": "feature_drift",
                        "drift_status": drift.status,
                        "max_abs_z": float(drift.max_abs_z),
                        "outlier_fraction": float(drift.outlier_fraction),
                        "warnings": list(drift.warnings),
                    }
                )
                if drift.status == "fail":
                    print(
                        f"step {i + 1:>2}: feature drift block "
                        f"max_z={drift.max_abs_z:.2f} outliers={drift.outlier_fraction:.1%}"
                    )
                    live_sleep()
                    continue
                if drift.status == "warn":
                    print(
                        f"step {i + 1:>2}: feature drift warning "
                        f"max_z={drift.max_abs_z:.2f} outliers={drift.outlier_fraction:.1%}"
                    )
        try:
            raw_score = model.predict_proba(latest.features)
            threshold = model_decision_threshold(model, cfg.signal_threshold)
            score = confidence_adjusted_probability(raw_score, cfg.confidence_beta)
        except ValueError as exc:
            if not effective_dry_run:
                print(f"Live model incompatible with current rows: {exc}", file=sys.stderr)
                halt_reason = "model_incompatible"
                exit_code = 2
                live_events.append({"step": i + 1, "status": "model_incompatible", "error": str(exc)})
                break
            print(f"paper model incompatible; retraining: {exc}", file=sys.stderr)
            if not training_rows:
                print("not enough labeled historical data to retrain a live model", file=sys.stderr)
                halt_reason = "model_incompatible"
                exit_code = 2
                live_events.append({"step": i + 1, "status": "model_incompatible", "error": "no labeled training rows"})
                break
            model = train(training_rows, epochs=40, feature_signature=_live_model_feature_signature(model, cfg))
            model_loads += 1
            raw_score = model.predict_proba(latest.features)
            threshold = model_decision_threshold(model, cfg.signal_threshold)
            score = confidence_adjusted_probability(raw_score, cfg.confidence_beta)
        decision_cfg = cfg
        if cfg.external_signals_enabled:
            score_before_external = score
            try:
                external_report = collect_external_signals(
                    symbol=runtime.symbol,
                    cache_path=_external_signal_cache_path(model_path),
                    ttl_seconds=cfg.external_signal_ttl_seconds,
                    timeout_seconds=cfg.external_signal_timeout_seconds,
                    max_adjustment=cfg.external_signal_max_adjustment,
                    min_providers=cfg.external_signal_min_providers,
                    compute_backend=runtime.compute_backend,
                    short_reaction_refresh_seconds=cfg.external_signal_short_reaction_refresh_seconds,
                    news_provider_limit=cfg.external_signal_news_provider_limit,
                    news_items_per_provider=cfg.external_signal_news_items_per_provider,
                    news_provider_parallelism=cfg.external_signal_provider_parallelism,
                    news_provider_jitter_seconds=cfg.external_signal_provider_jitter_seconds,
                    ollama_news_enabled=cfg.external_news_ai_enabled,
                    ollama_model=cfg.external_news_ai_model,
                    ollama_url=cfg.external_news_ai_url,
                    ollama_timeout_seconds=cfg.external_news_ai_timeout_seconds,
                    telemetry_path=cfg.telemetry_db_path if cfg.telemetry_enabled else None,
                    source_grade_max_age_hours=cfg.source_grade_max_age_hours,
                )
                score, decision_cfg, applied_adjustment = _apply_external_signal_to_score(
                    score,
                    cfg,
                    external_report,
                )
                live_events.append(
                    {
                        "step": i + 1,
                        "status": "external_signals",
                        "score_before": float(score_before_external),
                        "score_after": float(score),
                        "applied_adjustment": float(applied_adjustment),
                        "risk_multiplier": float(external_report.risk_multiplier),
                        "fresh_count": int(external_report.fresh_count),
                        "provider_count": int(external_report.provider_count),
                        "short_term_score": float(external_report.short_term_score),
                        "medium_term_score": float(external_report.medium_term_score),
                        "long_term_score": float(external_report.long_term_score),
                        "reaction_required": bool(external_report.reaction_required),
                        "reaction_reason": external_report.reaction_reason,
                        "news_backend_kind": external_report.news_backend_kind,
                        "news_ai_status": external_report.news_ai_status,
                        "news_ai_model": external_report.news_ai_model,
                        "news_ai_latency_ms": int(external_report.news_ai_latency_ms),
                        "report": external_report.asdict(),
                    }
                )
                print(
                    f"step {i + 1:>2}: external signals "
                    f"providers={external_report.fresh_count}/{external_report.provider_count} "
                    f"adj={applied_adjustment:+.4f} risk={external_report.risk_multiplier:.3f} "
                    f"short={external_report.short_term_score:+.3f} "
                    f"reaction={'yes' if external_report.reaction_required else 'no'}"
                )
            except Exception as exc:
                live_events.append(
                    {
                        "step": i + 1,
                        "status": "external_signals_error",
                        "error": str(exc),
                    }
                )
                print(f"step {i + 1:>2}: external signals unavailable: {exc}", file=sys.stderr)
        _record_model_telemetry(
            cfg,
            step=i + 1,
            row=latest,
            raw_score=raw_score,
            adjusted_score=score,
            threshold=threshold,
            model=model,
            runtime=runtime,
        )
        direction = _score_to_direction(score, decision_cfg, runtime.market_type, threshold)

        # cooldown reduces immediate flip-flopping in choppy conditions
        if cooldown_left > 0:
            if runtime.market_type == "spot":
                direction = 0
            cooldown_left -= 1

        price = latest.close
        day = _safe_day_ms(latest.timestamp)
        if position_side == 0 and direction != 0:
            current_drawdown = (equity_peak - cash) / equity_peak if equity_peak else 0.0
            entry_risk = assess_entry_risk(
                direction=direction,
                position_side=position_side,
                max_open_positions=max_open_positions,
                max_daily_trades=max_daily_trades,
                daily_trade_count=daily_trade_count.get(day, 0),
                cash=cash,
                price=price,
                drawdown=current_drawdown,
                drawdown_limit=drawdown_limit,
            )
            if not entry_risk.allowed:
                message, event = _live_entry_risk_skip(
                    step=i + 1,
                    day=day,
                    score=score,
                    entry_risk=entry_risk,
                    max_daily_trades=max_daily_trades,
                    max_open_positions=max_open_positions,
                )
                print(message)
                skipped_entries += 1
                live_events.append(event)
                live_sleep()
                continue

            notional, qty = _build_order_notional(
                cash,
                price,
                decision_cfg,
                runtime.market_type,
                leverage,
                client,
                constraints=constraints,
            )
            if notional <= 0:
                print(f"step {i + 1:>2}: skipped entry due to order constraints")
                skipped_entries += 1
                live_events.append(
                    {
                        "step": i + 1,
                        "status": "skip_constraints",
                        "score": float(score),
                        "notional": float(notional),
                    }
                )
                live_sleep()
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
                live_events.append(
                    {
                        "step": i + 1,
                        "status": "skip_insufficient_cash_pre_fill",
                        "score": float(score),
                    }
                )
                live_sleep()
                continue

            side_sign = 1 if direction > 0 else -1
            fill = price * (1.0 + side_sign * slippage)
            if fill <= 0:
                live_sleep()
                continue

            notional = qty * fill
            fee = abs(notional) * fee_rate
            total = margin + fee
            if total > cash:
                print(f"step {i + 1:>2}: insufficient cash after fill adjustment")
                skipped_entries += 1
                live_events.append(
                    {
                        "step": i + 1,
                        "status": "skip_insufficient_cash_after_fill",
                        "score": float(score),
                        "fill": float(fill),
                    }
                )
                live_sleep()
                continue

            side = "BUY" if side_sign > 0 else "SELL"
            try:
                order_response = _paper_or_live_order(
                    client,
                    runtime,
                    decision_cfg,
                    side=side,
                    size=qty,
                    dry_run=effective_dry_run,
                    leverage=leverage,
                )
            except BinanceAPIError as exc:
                record_order_error(i + 1, side, qty, exc)
                break

            filled_qty, fill, filled_notional = _order_fill_details(
                order_response,
                fallback_qty=qty,
                fallback_price=fill,
            )
            if filled_qty <= 0.0 or fill <= 0.0:
                record_order_error(i + 1, side, qty, BinanceAPIError("Order response did not include executed quantity"))
                break
            qty = abs(filled_qty)
            notional = filled_notional if filled_notional > 0.0 else qty * fill
            fee = abs(notional) * fee_rate
            margin = min(cash, abs(notional)) if runtime.market_type == "spot" else min(cash, abs(notional) / leverage)
            total = margin + fee

            cash = max(0.0, cash - total)
            position_side = direction
            position_notional = direction * notional
            qty = abs(qty)
            entry_price = fill
            margin_used = margin
            daily_trade_count[day] = daily_trade_count.get(day, 0) + 1

            print(f"step {i + 1:>2}: enter {'long' if position_side > 0 else 'short'} at {fill:.2f} qty={qty:.6f}")
            entries += 1
            live_events.append(
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

            opposite_signal = direction not in (0, position_side) if runtime.market_type == "futures" else direction == 0
            should_close = opposite_signal or pnl_pct >= cfg.take_profit_pct or pnl_pct <= -cfg.stop_loss_pct

            if should_close:
                fill = price * (1.0 - position_side * slippage)

                side_to_close = "SELL" if position_side > 0 else "BUY"
                try:
                    order_response = _paper_or_live_order(
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
                closed_qty, fill, closed_notional = _order_fill_details(
                    order_response,
                    fallback_qty=abs(qty),
                    fallback_price=fill,
                )
                if closed_qty <= 0.0 or fill <= 0.0:
                    record_order_error(i + 1, side_to_close, abs(qty), BinanceAPIError("Close order response did not include executed quantity"))
                    break
                closed_qty = min(abs(qty), abs(closed_qty))
                realized = position_side * (fill - entry_price) * closed_qty
                exit_fee = abs(closed_notional if closed_notional > 0.0 else fill * closed_qty) * fee_rate
                close_ratio = min(1.0, closed_qty / abs(qty)) if qty else 1.0
                cash += margin_used * close_ratio + realized - exit_fee
                print(
                    f"step {i + 1:>2}: close {'long' if position_side > 0 else 'short'} "
                    f"pnl={realized:.2f} cash={cash:.2f}"
                )
                closes += 1
                live_events.append(
                    {
                        "step": i + 1,
                        "status": "close",
                        "direction": int(position_side),
                        "score": float(score),
                        "price": float(fill),
                        "pnl": float(realized),
                        "qty_closed": float(closed_qty),
                        "cash_after": float(cash),
                    }
                )
                if realized > 0:
                    print("result: win")
                if close_ratio >= 0.999:
                    position_notional = 0.0
                    position_side = 0
                    qty = 0.0
                    margin_used = 0.0
                    entry_price = 0.0
                    cooldown_left = max(0, wait_ticks)
                else:
                    qty = max(0.0, abs(qty) - closed_qty)
                    margin_used = max(0.0, margin_used * (1.0 - close_ratio))
                    position_notional = position_side * qty * entry_price
            else:
                unrealized = margin_used + pnl
                print(f"step {i + 1:>2}: hold {'long' if position_side > 0 else 'short'} pnl={pnl_pct:.2%} cash={cash:.2f}")
                print(f"         unrealized equity={cash + unrealized:.2f}")
                live_events.append(
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

                    side_to_close = "SELL" if position_side > 0 else "BUY"
                    try:
                        order_response = _paper_or_live_order(
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
                    closed_qty, fill, closed_notional = _order_fill_details(
                        order_response,
                        fallback_qty=abs(qty),
                        fallback_price=fill,
                    )
                    if closed_qty <= 0.0 or fill <= 0.0:
                        record_order_error(i + 1, side_to_close, abs(qty), BinanceAPIError("Emergency close order response did not include executed quantity"))
                        break
                    closed_qty = min(abs(qty), abs(closed_qty))
                    realized = position_side * (fill - entry_price) * closed_qty
                    exit_fee = abs(closed_notional if closed_notional > 0.0 else fill * closed_qty) * fee_rate
                    close_ratio = min(1.0, closed_qty / abs(qty)) if qty else 1.0
                    cash += margin_used * close_ratio + realized - exit_fee
                    print(
                        f"step {i + 1:>2}: emergency close from drawdown "
                        f"{drawdown:.2%}; cash={cash:.2f}"
                    )
                    if close_ratio >= 0.999:
                        position_notional = 0.0
                        position_side = 0
                        qty = 0.0
                        margin_used = 0.0
                        entry_price = 0.0
                    else:
                        qty = max(0.0, abs(qty) - closed_qty)
                        margin_used = max(0.0, margin_used * (1.0 - close_ratio))
                        position_notional = position_side * qty * entry_price
                live_events.append(
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
            live_events.append(
                {
                    "step": i + 1,
                    "status": "signal_no_entry",
                    "score": float(score),
                    "direction": int(direction),
                }
            )

        maybe_grade_sources(i + 1)
        live_sleep()
    live_run["result"] = {
        "status": halt_reason,
        "steps_executed": steps_executed,
        "entries": entries,
        "closes": closes,
        "skipped_entries": skipped_entries,
        "model_loads": model_loads,
        "drawdown_seen": float(max_drawdown_seen),
        "ending_cash": float(cash),
        "ending_cash_is_estimate": bool(not effective_dry_run),
        "equity_peak": float(equity_peak),
        "drawdown_limit": drawdown_limit,
    }
    _persist_run_artifact("live", model_path.parent, live_run)
    if max_drawdown_seen > 0.0:
        print(f"max_drawdown observed: {max_drawdown_seen:.2%}")
    cash_label = "cash_estimate" if not effective_dry_run else "cash"
    print(f"finished loop market={runtime.market_type} {cash_label}={cash:.2f}")
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


def command_train_suite(args: argparse.Namespace) -> int:  # skipcq: PY-R1000
    from .objective import get_objective
    from .training_suite import describe_candidate_grid, run_training_suite

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
    try:
        objectives = (
            tuple(get_objective(name).name for name in args.objective)
            if args.objective
            else available_objectives()
        )
        grid_counts = {name: len(describe_candidate_grid(get_objective(name))) for name in objectives}
        total_candidates = sum(grid_counts.values())
        backend_label = str(getattr(args, "compute_backend", None) or runtime.compute_backend or "cpu")
        print(
            "training suite starting: "
            f"objectives={','.join(objectives)} candidates={total_candidates} "
            f"backend={backend_label} batch_size={getattr(args, 'batch_size', 8192)}",
            flush=True,
        )
        suite_kwargs: dict[str, object] = {
            "objectives": objectives,
            "market_type": runtime.market_type,
            "starting_cash": args.starting_cash,
            "output_dir": Path(args.output_dir),
            "max_workers": args.max_workers,
            "compute_backend": backend_label,
        }
        if getattr(args, "batch_size", 8192) != 8192:
            suite_kwargs["batch_size"] = max(1, int(args.batch_size))
        report = run_training_suite(candles, strategy, **suite_kwargs)
    except ValueError as err:
        print(f"training suite failed: {err}", file=sys.stderr)
        return 2
    print(f"training suite complete: {len(report.outcomes)} objective(s)")
    for outcome in report.outcomes:
        print(
            f"  {outcome.objective:<14} score={outcome.best_score:+.4f} "
            f"threshold={outcome.decision_threshold if outcome.decision_threshold is not None else 'n/a'} "
            f"source={outcome.threshold_source or 'n/a'} "
            f"validation={outcome.validation_score if outcome.validation_score is not None else 'n/a'} "
            f"full={outcome.full_sample_score if outcome.full_sample_score is not None else 'n/a'} "
            f"ensemble={'yes' if getattr(outcome, 'ensemble_refined', False) else 'no'} "
            f"backend={getattr(outcome, 'training_backend_kind', 'cpu')} "
            f"local_checks={getattr(outcome, 'local_refinement_candidates', 0)} "
            f"ensemble_checks={getattr(outcome, 'ensemble_refinement_candidates', 0)} "
            f"candidates={outcome.explored_candidates} model={outcome.model_path}"
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


def _build_autonomous_decision_fn(
    *,
    model_path: Path,
    strategy: StrategyConfig,
    effective_dry_run: bool,
):
    from .autonomous import Decision

    model, model_error, model_notice = _load_live_start_model(model_path, strategy, effective_dry_run=effective_dry_run)
    if model_error is not None:
        return None, model_error, model_notice
    if model is None and not effective_dry_run:
        return None, f"Autonomous live mode requires a compatible model: {model_path}", model_notice
    state: dict[str, object] = {"model": model, "step": 0}

    def decide(client, runtime: RuntimeConfig, current_strategy: StrategyConfig, _objective) -> Decision:
        state["step"] = int(state["step"]) + 1
        current_model = cast(TrainedModel | None, state["model"])
        candles = client.get_klines(runtime.symbol, runtime.interval, limit=max(current_strategy.model_lookback, 300))
        rows = _live_rows_for_model(candles, current_strategy, current_model)
        training_rows = (
            _readiness_model_rows(candles, current_strategy, current_model)
            if current_model is not None
            else _build_model_rows(candles, current_strategy)
        )
        if not rows:
            price, _ts = client.get_symbol_price(runtime.symbol)
            return Decision(side="FLAT", confidence=0.0, mark_price=float(price))
        if current_model is None:
            if not effective_dry_run:  # pragma: no cover - rejected before the decision function is returned
                raise RuntimeError(f"Autonomous live mode requires a compatible model: {model_path}")
            if not training_rows:
                price, _ts = client.get_symbol_price(runtime.symbol)
                return Decision(side="FLAT", confidence=0.0, mark_price=float(price))
            current_model = train(
                training_rows,
                epochs=40,
                feature_signature=_live_model_feature_signature(None, current_strategy),
            )
            state["model"] = current_model
        latest = rows[-1]
        raw_score = current_model.predict_proba(latest.features)
        threshold = model_decision_threshold(current_model, current_strategy.signal_threshold)
        score = confidence_adjusted_probability(raw_score, current_strategy.confidence_beta)
        decision_strategy = current_strategy
        if current_strategy.external_signals_enabled:
            external_report = collect_external_signals(
                symbol=runtime.symbol,
                cache_path=_external_signal_cache_path(model_path),
                ttl_seconds=current_strategy.external_signal_ttl_seconds,
                timeout_seconds=current_strategy.external_signal_timeout_seconds,
                max_adjustment=current_strategy.external_signal_max_adjustment,
                min_providers=current_strategy.external_signal_min_providers,
                compute_backend=runtime.compute_backend,
                short_reaction_refresh_seconds=current_strategy.external_signal_short_reaction_refresh_seconds,
                news_provider_limit=current_strategy.external_signal_news_provider_limit,
                news_items_per_provider=current_strategy.external_signal_news_items_per_provider,
                news_provider_parallelism=current_strategy.external_signal_provider_parallelism,
                news_provider_jitter_seconds=current_strategy.external_signal_provider_jitter_seconds,
                ollama_news_enabled=current_strategy.external_news_ai_enabled,
                ollama_model=current_strategy.external_news_ai_model,
                ollama_url=current_strategy.external_news_ai_url,
                ollama_timeout_seconds=current_strategy.external_news_ai_timeout_seconds,
                telemetry_path=current_strategy.telemetry_db_path if current_strategy.telemetry_enabled else None,
                source_grade_max_age_hours=current_strategy.source_grade_max_age_hours,
            )
            score, decision_strategy, _applied_adjustment = _apply_external_signal_to_score(score, current_strategy, external_report)
        _record_model_telemetry(
            current_strategy,
            step=int(state["step"]),
            row=latest,
            raw_score=raw_score,
            adjusted_score=score,
            threshold=threshold,
            model=current_model,
            runtime=runtime,
        )
        direction = _score_to_direction(score, decision_strategy, runtime.market_type, threshold)
        if direction > 0:
            side = "LONG"
        elif direction < 0:
            side = "SHORT"
        else:
            side = "FLAT"
        return Decision(side=side, confidence=float(score), mark_price=float(latest.close))

    return decide, None, model_notice


def command_autonomous(args: argparse.Namespace) -> int:
    from .autonomous import (
        AutonomousConfig,
        STATE_PAUSED,
        STATE_RUNNING,
        STATE_STOPPING,
        AutonomousControl,
        run_loop,
    )
    from .objective import get_objective

    control = AutonomousControl()
    action = args.action
    if action == "start":
        try:
            objective = get_objective(args.objective)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        runtime = load_runtime()
        strategy = load_strategy()
        if getattr(args, "live", False):
            effective_dry_run = False
        elif getattr(args, "paper", False):
            effective_dry_run = True
        else:
            effective_dry_run = bool(runtime.dry_run)
        if not _allows_signed_execution(runtime):
            print("Autonomous mode requires testnet=true or demo=true.", file=sys.stderr)
            return 2
        if not effective_dry_run and not _has_api_credentials(runtime):
            print(_credential_required_message("Autonomous live mode"), file=sys.stderr)
            return 2
        if not effective_dry_run:
            print(
                "Autonomous authenticated mode is disabled until exchange-order execution is wired into the autonomous loop. "
                "Use live --live for signed testnet/demo orders.",
                file=sys.stderr,
            )
            return 2
        model_path = Path(getattr(args, "model", "data/model.json"))
        decision_fn, model_error, model_notice = _build_autonomous_decision_fn(
            model_path=model_path,
            strategy=strategy,
            effective_dry_run=effective_dry_run,
        )
        if model_error is not None or decision_fn is None:
            print(model_error or f"Autonomous mode requires a readable model: {model_path}", file=sys.stderr)
            return 2
        if model_notice is not None:
            print(model_notice, file=sys.stderr)
        cfg = AutonomousConfig(
            objective=objective.name,
            poll_seconds=max(0.0, float(getattr(args, "poll_seconds", 30.0))),
            stop_after_iterations=getattr(args, "iterations", None),
            heartbeat_every=max(1, int(getattr(args, "heartbeat_every", 1))),
            dry_run=effective_dry_run,
            starting_reference_cash=max(0.0, float(getattr(args, "starting_cash", 1000.0))),
        )
        try:
            client = _build_client(runtime)
        except BinanceAPIError as exc:
            print(f"Autonomous startup blocked: {exc}", file=sys.stderr)
            return 2
        result = run_loop(
            client,
            runtime,
            strategy,
            cfg,
            decision_fn=decision_fn,
        )
        print(
            "autonomous: "
            f"{result.exit_reason} iterations={result.iterations} "
            f"opened={result.opened_trades} closed={result.closed_trades} skipped={result.skipped_entries}"
        )
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
