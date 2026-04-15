"""Entry point for the BTCUSDC test-trading CLI."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import sys
import time
from pathlib import Path
from typing import Callable, Iterable, Sequence

from .api import BinanceAPIError, BinanceClient
from .backtest import run_backtest
from .config import config_paths, load_runtime, load_strategy, prompt_runtime, save_runtime, save_strategy
from .dashboard import DashboardSnapshot, load_artifact_preview, render_dashboard
from .features import FEATURE_NAMES, feature_signature, make_rows, normalize_enabled_features
from .api import SymbolConstraints
from .model import (
    calibrate_threshold,
    evaluate_classification,
    evaluate,
    ModelFeatureMismatchError,
    ModelLoadError,
    load_model,
    serialize_model,
    train,
    walk_forward_report,
)
from .types import StrategyConfig


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

    parser_dashboard = subparsers.add_parser("dashboard", help="render operator dashboard snapshot")
    parser_dashboard.add_argument("--with-account", action="store_true", help="include authenticated account balances when credentials are present")
    parser_dashboard.set_defaults(func=command_dashboard)

    parser_menu = subparsers.add_parser("menu", help="launch interactive operator menu")
    parser_menu.set_defaults(func=command_menu)

    parser_fetch = subparsers.add_parser("fetch", help="download BTCUSDC klines")
    parser_fetch.add_argument("--symbol", default=None)
    parser_fetch.add_argument("--interval", default=None)
    parser_fetch.add_argument("--limit", type=int, default=500)
    parser_fetch.add_argument("--output", default="data/historical_btcusdc.json")
    parser_fetch.set_defaults(func=command_fetch)

    parser_train = subparsers.add_parser("train", help="train model from cached candles")
    parser_train.add_argument("--input", default="data/historical_btcusdc.json")
    parser_train.add_argument("--output", default="data/model.json")
    parser_train.add_argument("--epochs", type=int, default=250)
    parser_train.add_argument("--seed", type=int, default=7)
    parser_train.add_argument("--walk-forward", action="store_true", help="run walk-forward validation before final training")
    parser_train.add_argument("--walk-forward-train", type=int, default=300)
    parser_train.add_argument("--walk-forward-test", type=int, default=60)
    parser_train.add_argument("--walk-forward-step", type=int, default=30)
    parser_train.add_argument("--calibrate-threshold", action="store_true", help="optimize a probability threshold on validation split")
    parser_train.set_defaults(func=command_train)

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
    parser_strategy.add_argument("--set-features", default=None, help="comma-separated ordered feature list for retraining")
    parser_strategy.add_argument("--enable-feature", action="append", default=None, help="enable a feature by name")
    parser_strategy.add_argument("--disable-feature", action="append", default=None, help="disable a feature by name")
    parser_strategy.set_defaults(func=command_strategy)

    return parser.parse_args(argv)


def _build_client(runtime):
    return BinanceClient(
        api_key=runtime.api_key,
        api_secret=runtime.api_secret,
        testnet=runtime.testnet,
        market_type=runtime.market_type,
        max_calls_per_minute=runtime.max_rate_calls_per_minute,
    )


def _menu_prompt(text: str, *, input_fn: Callable[[str], str] = input) -> str:
    return input_fn(text).strip()


def _menu_prompt_default(text: str, current: str, *, input_fn: Callable[[str], str] = input) -> str:
    value = _menu_prompt(f"{text} [{current}]: ", input_fn=input_fn)
    return value or current


def _menu_prompt_int(text: str, current: int, *, minimum: int | None = None,
                     input_fn: Callable[[str], str] = input) -> int:
    while True:
        raw = _menu_prompt(f"{text} [{current}]: ", input_fn=input_fn)
        if not raw:
            return current
        try:
            value = int(raw)
        except ValueError:
            print("Enter a whole number.")
            continue
        if minimum is not None and value < minimum:
            print(f"Value must be >= {minimum}.")
            continue
        return value


def _menu_prompt_float(text: str, current: float, *, minimum: float | None = None,
                       maximum: float | None = None, input_fn: Callable[[str], str] = input) -> float:
    while True:
        raw = _menu_prompt(f"{text} [{current}]: ", input_fn=input_fn)
        if not raw:
            return current
        try:
            value = float(raw)
        except ValueError:
            print("Enter a numeric value.")
            continue
        if minimum is not None and value < minimum:
            print(f"Value must be >= {minimum}.")
            continue
        if maximum is not None and value > maximum:
            print(f"Value must be <= {maximum}.")
            continue
        return value


def _menu_prompt_bool(text: str, current: bool, *, input_fn: Callable[[str], str] = input) -> bool:
    default = "y" if current else "n"
    while True:
        raw = _menu_prompt(f"{text} (y/n) [{default}]: ", input_fn=input_fn).lower()
        if not raw:
            return current
        if raw in {"y", "yes"}:
            return True
        if raw in {"n", "no"}:
            return False
        print("Enter y or n.")


def _build_strategy_menu_args(cfg: StrategyConfig, *, input_fn: Callable[[str], str] = input) -> argparse.Namespace:
    return argparse.Namespace(
        leverage=_menu_prompt_float("Leverage", cfg.leverage, minimum=1.0, input_fn=input_fn),
        risk=_menu_prompt_float("Risk per trade", cfg.risk_per_trade, minimum=0.0001, input_fn=input_fn),
        max_position=_menu_prompt_float("Max position pct", cfg.max_position_pct, minimum=0.0, maximum=1.0, input_fn=input_fn),
        stop=_menu_prompt_float("Stop loss pct", cfg.stop_loss_pct, minimum=0.0, maximum=0.99, input_fn=input_fn),
        take=_menu_prompt_float("Take profit pct", cfg.take_profit_pct, minimum=0.0, maximum=0.99, input_fn=input_fn),
        cooldown=_menu_prompt_int("Cooldown minutes", cfg.cooldown_minutes, minimum=0, input_fn=input_fn),
        max_open=_menu_prompt_int("Max open positions", cfg.max_open_positions, minimum=0, input_fn=input_fn),
        max_trades_per_day=_menu_prompt_int("Max trades per day", cfg.max_trades_per_day, minimum=0, input_fn=input_fn),
        signal_threshold=_menu_prompt_float("Signal threshold", cfg.signal_threshold, minimum=0.01, maximum=0.99, input_fn=input_fn),
        max_drawdown=_menu_prompt_float("Max drawdown limit", cfg.max_drawdown_limit, minimum=0.0, input_fn=input_fn),
        taker_fee_bps=_menu_prompt_float("Taker fee bps", cfg.taker_fee_bps, minimum=0.0, input_fn=input_fn),
        slippage_bps=_menu_prompt_float("Slippage bps", cfg.slippage_bps, minimum=0.0, input_fn=input_fn),
        label_threshold=_menu_prompt_float("Label threshold", cfg.label_threshold, minimum=0.0001, input_fn=input_fn),
        set_features=",".join(_menu_feature_selection(cfg.enabled_features, input_fn=input_fn)),
        enable_feature=None,
        disable_feature=None,
    )


def _menu_feature_selection(current: Sequence[str], *, input_fn: Callable[[str], str] = input) -> tuple[str, ...]:
    selected = list(normalize_enabled_features(current))
    print("Model feature selection")
    for index, name in enumerate(FEATURE_NAMES, start=1):
        mark = "x" if name in selected else " "
        print(f"{index:>2}. [{mark}] {name}")
    raw = _menu_prompt("Toggle feature numbers (comma-separated), 'all', or blank to keep: ", input_fn=input_fn).lower()
    if not raw:
        return tuple(selected)
    if raw == "all":
        return tuple(FEATURE_NAMES)
    toggles = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            value = int(chunk)
        except ValueError:
            continue
        if 1 <= value <= len(FEATURE_NAMES):
            toggles.append(FEATURE_NAMES[value - 1])
    for name in toggles:
        if name in selected:
            selected.remove(name)
        else:
            selected.append(name)
    return normalize_enabled_features(selected)


def _menu_data_path(name: str, *, input_fn: Callable[[str], str] = input) -> str:
    default = f"data/{name}"
    return _menu_prompt_default(f"{name} path", default, input_fn=input_fn)


def _recent_artifacts(*, base_dir: Path = Path("data"), limit: int = 8) -> list[Path]:
    if not base_dir.exists():
        return []
    paths = [path for path in base_dir.glob("*.json") if path.is_file()]
    paths.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return paths[:limit]


def _artifact_summary(path: Path) -> str:
    return load_artifact_preview(path)


def _print_menu_header() -> None:
    runtime = load_runtime()
    strategy = load_strategy()
    print("\nOperator console")
    print(
        "runtime: "
        f"symbol={runtime.symbol} interval={runtime.interval} market={runtime.market_type} "
        f"testnet={runtime.testnet} paper_default={runtime.dry_run}"
    )
    print(
        "strategy: "
        f"threshold={strategy.signal_threshold:.3f} risk={strategy.risk_per_trade:.4f} "
        f"stop={strategy.stop_loss_pct:.4f} take={strategy.take_profit_pct:.4f} "
        f"cooldown={strategy.cooldown_minutes}m max_trades={strategy.max_trades_per_day}"
    )


def _show_recent_artifacts() -> int:
    artifacts = _recent_artifacts()
    if not artifacts:
        print("No recent artifacts under data/.")
        return 0
    print("Recent artifacts:")
    for path in artifacts:
        print(f"- {_artifact_summary(path)}")
    return 0


def _account_overview_lines(runtime) -> list[str]:
    if not runtime.api_key or not runtime.api_secret:
        return ["No API credentials configured."]
    client = _build_client(runtime)
    try:
        account = client.get_account()
    except BinanceAPIError as exc:
        return [f"Account overview failed: {exc}"]
    balances = account.get("balances", []) if isinstance(account, dict) else []
    interesting = []
    for item in balances:
        if not isinstance(item, dict):
            continue
        asset = str(item.get("asset", ""))
        free = str(item.get("free", "0"))
        locked = str(item.get("locked", "0"))
        if asset in {"BTC", "USDC"} or free not in {"0", "0.0", "0.00000000"} or locked not in {"0", "0.0", "0.00000000"}:
            interesting.append(f"{asset}: free={free} locked={locked}")
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
        "Primary entrypoints: menu for guided operation, dashboard for overview, subcommands for scripting.",
        "Use the explicit spot roundtrip or --live path only on testnet.",
    ]
    return DashboardSnapshot(
        runtime=runtime.public_dict(),
        strategy=strategy.asdict(),
        artifacts=[_artifact_summary(path) for path in _recent_artifacts()],
        account_lines=_account_overview_lines(runtime) if with_account else ["Account lookup skipped. Use --with-account to include it."],
        notes=notes,
    )


def command_dashboard(args: argparse.Namespace) -> int:
    snapshot = _dashboard_snapshot(with_account=bool(getattr(args, "with_account", False)))
    print(render_dashboard(snapshot))
    return 0


def _run_offline_pipeline(*, input_fn: Callable[[str], str] = input) -> int:
    historical = _menu_data_path("historical_btcusdc.json", input_fn=input_fn)
    model = _menu_data_path("model.json", input_fn=input_fn)
    fetch_args = argparse.Namespace(
        symbol=load_runtime().symbol,
        interval=load_runtime().interval,
        limit=_menu_prompt_int("Fetch limit", 500, minimum=1, input_fn=input_fn),
        output=historical,
    )
    train_args = argparse.Namespace(
        input=historical,
        output=model,
        epochs=_menu_prompt_int("Training epochs", 120, minimum=1, input_fn=input_fn),
        seed=_menu_prompt_int("Training seed", 7, minimum=0, input_fn=input_fn),
        walk_forward=_menu_prompt_bool("Run walk-forward validation", True, input_fn=input_fn),
        walk_forward_train=_menu_prompt_int("Walk-forward train window", 300, minimum=2, input_fn=input_fn),
        walk_forward_test=_menu_prompt_int("Walk-forward test window", 60, minimum=1, input_fn=input_fn),
        walk_forward_step=_menu_prompt_int("Walk-forward step", 30, minimum=1, input_fn=input_fn),
        calibrate_threshold=_menu_prompt_bool("Calibrate threshold", True, input_fn=input_fn),
    )
    backtest_args = argparse.Namespace(
        input=historical,
        model=model,
        start_cash=_menu_prompt_float("Backtest starting cash", 1000.0, minimum=1.0, input_fn=input_fn),
    )
    evaluate_args = argparse.Namespace(
        input=historical,
        model=model,
        threshold=None,
        calibrate_threshold=True,
    )
    for fn, args in (
        (command_fetch, fetch_args),
        (command_train, train_args),
        (command_backtest, backtest_args),
        (command_evaluate, evaluate_args),
    ):
        result = fn(args)
        if result != 0:
            return result
    return 0


def _run_spot_test_order_roundtrip(*, input_fn: Callable[[str], str] = input) -> int:
    runtime = load_runtime()
    if runtime.market_type != "spot":
        print("Spot test order is only available when runtime.market_type=spot.")
        return 2
    if not runtime.testnet:
        print("Spot test order requires testnet=true.")
        return 2
    if not runtime.api_key or not runtime.api_secret:
        print("Spot test order requires configured API credentials.")
        return 2
    confirm = _menu_prompt("Type ROUNDTRIP to place a minimal BUY then SELL on spot testnet: ", input_fn=input_fn)
    if confirm != "ROUNDTRIP":
        print("Spot test order cancelled.")
        return 0
    quantity = _menu_prompt_float("Order quantity", 0.00008, minimum=0.00001, input_fn=input_fn)
    client = _build_client(runtime)
    try:
        buy = client.place_order(runtime.symbol, "BUY", quantity, dry_run=False)
        sell = client.place_order(runtime.symbol, "SELL", quantity, dry_run=False)
    except BinanceAPIError as exc:
        print(f"Spot test order failed: {exc}", file=sys.stderr)
        return 2
    print("Spot test roundtrip complete.")
    print(json.dumps(
        {
            "buy_status": buy.get("status"),
            "buy_orderId": buy.get("orderId"),
            "buy_executedQty": buy.get("executedQty"),
            "sell_status": sell.get("status"),
            "sell_orderId": sell.get("orderId"),
            "sell_executedQty": sell.get("executedQty"),
        },
        indent=2,
    ))
    return 0


def _menu_tune_window_args(*, input_fn: Callable[[str], str] = input) -> tuple[int | None, str | None, str | None]:
    mode = _menu_prompt("Tune window mode [all/lookback/range] [all]: ", input_fn=input_fn).lower()
    if not mode or mode == "all":
        return None, None, None
    if mode == "lookback":
        return _menu_prompt_int("Tune lookback days", 30, minimum=1, input_fn=input_fn), None, None
    if mode == "range":
        from_date = _menu_prompt("Tune from date YYYY-MM-DD: ", input_fn=input_fn) or None
        to_date = _menu_prompt("Tune to date YYYY-MM-DD: ", input_fn=input_fn) or None
        return None, from_date, to_date
    print("Unknown tune window mode; using all data.")
    return None, None, None


def _run_menu(*, input_fn: Callable[[str], str] = input) -> int:
    while True:
        _print_menu_header()
        print("1. Status")
        print("2. Configure runtime")
        print("3. Configure strategy")
        print("4. Test connectivity")
        print("5. Account overview")
        print("6. Dashboard snapshot")
        print("7. Fetch market data")
        print("8. Train model")
        print("9. Tune strategy")
        print("10. Run backtest")
        print("11. Evaluate model")
        print("12. Run paper session")
        print("13. Run spot/futures testnet session")
        print("14. Run full offline pipeline")
        print("15. View recent artifacts")
        print("16. Spot testnet BUY+SELL roundtrip")
        print("17. Exit")

        choice = _menu_prompt("Select option: ", input_fn=input_fn)
        if choice == "1":
            command_status(argparse.Namespace())
        elif choice == "2":
            command_configure(argparse.Namespace())
        elif choice == "3":
            cfg = load_strategy()
            command_strategy(_build_strategy_menu_args(cfg, input_fn=input_fn))
        elif choice == "4":
            command_connect(argparse.Namespace())
        elif choice == "5":
            _show_account_overview()
        elif choice == "6":
            command_dashboard(argparse.Namespace(with_account=True))
        elif choice == "7":
            runtime = load_runtime()
            args = argparse.Namespace(
                symbol=runtime.symbol,
                interval=runtime.interval,
                limit=_menu_prompt_int("Fetch limit", 500, minimum=1, input_fn=input_fn),
                output=_menu_data_path("historical_btcusdc.json", input_fn=input_fn),
            )
            command_fetch(args)
        elif choice == "8":
            args = argparse.Namespace(
                input=_menu_data_path("historical_btcusdc.json", input_fn=input_fn),
                output=_menu_data_path("model.json", input_fn=input_fn),
                epochs=_menu_prompt_int("Training epochs", 250, minimum=1, input_fn=input_fn),
                seed=_menu_prompt_int("Training seed", 7, minimum=0, input_fn=input_fn),
                walk_forward=_menu_prompt_bool("Run walk-forward validation", False, input_fn=input_fn),
                walk_forward_train=_menu_prompt_int("Walk-forward train window", 300, minimum=2, input_fn=input_fn),
                walk_forward_test=_menu_prompt_int("Walk-forward test window", 60, minimum=1, input_fn=input_fn),
                walk_forward_step=_menu_prompt_int("Walk-forward step", 30, minimum=1, input_fn=input_fn),
                calibrate_threshold=_menu_prompt_bool("Calibrate threshold", True, input_fn=input_fn),
            )
            command_train(args)
        elif choice == "9":
            lookback_days, from_date, to_date = _menu_tune_window_args(input_fn=input_fn)
            args = argparse.Namespace(
                input=_menu_data_path("historical_btcusdc.json", input_fn=input_fn),
                save_best=_menu_prompt_bool("Save tuned strategy", False, input_fn=input_fn),
                min_risk=_menu_prompt_float("Min risk", 0.002, minimum=0.0001, input_fn=input_fn),
                max_risk=_menu_prompt_float("Max risk", 0.02, minimum=0.0001, input_fn=input_fn),
                steps=_menu_prompt_int("Tune steps", 5, minimum=1, input_fn=input_fn),
                min_leverage=_menu_prompt_float("Min leverage", 1.0, minimum=1.0, input_fn=input_fn),
                max_leverage=_menu_prompt_float("Max leverage", 20.0, minimum=1.0, input_fn=input_fn),
                min_threshold=_menu_prompt_float("Min threshold", 0.52, minimum=0.01, maximum=0.99, input_fn=input_fn),
                max_threshold=_menu_prompt_float("Max threshold", 0.88, minimum=0.01, maximum=0.99, input_fn=input_fn),
                min_take=_menu_prompt_float("Min take profit", 0.01, minimum=0.0, maximum=0.99, input_fn=input_fn),
                max_take=_menu_prompt_float("Max take profit", 0.06, minimum=0.0, maximum=0.99, input_fn=input_fn),
                min_stop=_menu_prompt_float("Min stop loss", 0.008, minimum=0.0, maximum=0.99, input_fn=input_fn),
                max_stop=_menu_prompt_float("Max stop loss", 0.04, minimum=0.0, maximum=0.99, input_fn=input_fn),
                lookback_days=lookback_days,
                from_date=from_date,
                to_date=to_date,
            )
            command_tune(args)
        elif choice == "10":
            args = argparse.Namespace(
                input=_menu_data_path("historical_btcusdc.json", input_fn=input_fn),
                model=_menu_data_path("model.json", input_fn=input_fn),
                start_cash=_menu_prompt_float("Starting cash", 1000.0, minimum=1.0, input_fn=input_fn),
            )
            command_backtest(args)
        elif choice == "11":
            threshold_raw = _menu_prompt("Evaluation threshold [strategy default]: ", input_fn=input_fn)
            args = argparse.Namespace(
                input=_menu_data_path("historical_btcusdc.json", input_fn=input_fn),
                model=_menu_data_path("model.json", input_fn=input_fn),
                threshold=float(threshold_raw) if threshold_raw else None,
                calibrate_threshold=_menu_prompt_bool("Calibrate threshold", False, input_fn=input_fn),
            )
            command_evaluate(args)
        elif choice == "12":
            args = argparse.Namespace(
                steps=_menu_prompt_int("Paper steps", 20, minimum=1, input_fn=input_fn),
                sleep=_menu_prompt_int("Paper sleep seconds", 5, minimum=0, input_fn=input_fn),
                leverage=None,
                retrain_interval=_menu_prompt_int("Retrain interval", 0, minimum=0, input_fn=input_fn),
                retrain_window=_menu_prompt_int("Retrain window", 300, minimum=1, input_fn=input_fn),
                retrain_min_rows=_menu_prompt_int("Retrain minimum rows", 240, minimum=1, input_fn=input_fn),
                paper=True,
                live=False,
            )
            command_live(args)
        elif choice == "13":
            confirm = _menu_prompt("Type TESTNET to allow authenticated testnet execution: ", input_fn=input_fn)
            if confirm != "TESTNET":
                print("Testnet execution cancelled.")
                continue
            args = argparse.Namespace(
                steps=_menu_prompt_int("Live testnet steps", 1, minimum=1, input_fn=input_fn),
                sleep=_menu_prompt_int("Live testnet sleep seconds", 5, minimum=0, input_fn=input_fn),
                leverage=None,
                retrain_interval=_menu_prompt_int("Retrain interval", 0, minimum=0, input_fn=input_fn),
                retrain_window=_menu_prompt_int("Retrain window", 300, minimum=1, input_fn=input_fn),
                retrain_min_rows=_menu_prompt_int("Retrain minimum rows", 240, minimum=1, input_fn=input_fn),
                paper=False,
                live=True,
            )
            command_live(args)
        elif choice == "14":
            _run_offline_pipeline(input_fn=input_fn)
        elif choice == "15":
            _show_recent_artifacts()
        elif choice == "16":
            _run_spot_test_order_roundtrip(input_fn=input_fn)
        elif choice in {"17", "q", "quit", "exit"}:
            print("Exiting menu.")
            return 0
        else:
            print("Invalid selection.")


def command_menu(_: argparse.Namespace) -> int:
    return _run_menu()


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
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _public_runtime_payload(runtime) -> dict[str, object]:
    return runtime.public_dict()


def _build_model_rows(candles: Sequence[object], strategy: StrategyConfig):
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


def _load_runtime_model(model_path: Path, strategy: StrategyConfig):
    strategy_signature = feature_signature(
        strategy.feature_windows[0],
        strategy.feature_windows[1],
        strategy.label_threshold,
        feature_version=strategy.feature_version,
        enabled_features=strategy.enabled_features,
    )
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
    return train(train_rows, epochs=epochs)


def _score_to_direction(score: float, cfg: StrategyConfig, market_type: str) -> int:
    if market_type == "futures":
        if score >= cfg.signal_threshold:
            return 1
        if score <= (1.0 - cfg.signal_threshold):
            return -1
        return 0
    return 1 if score >= cfg.signal_threshold else 0


def _paper_or_live_order(
    client: BinanceClient,
    runtime,
    strategy: StrategyConfig,
    *,
    side: str,
    size: float,
    dry_run: bool,
    leverage: float | None = None,
) -> None:
    if leverage is None:
        leverage = _effective_leverage(strategy, runtime.market_type)
    response = client.place_order(runtime.symbol, side, size, dry_run=dry_run, leverage=leverage)
    if dry_run:
        print("paper order:", json.dumps(response, indent=2))
        return
    print(f"live order: {side} {size:.8f} {runtime.symbol}")
    print(json.dumps(response, indent=2))


def command_configure(_: argparse.Namespace) -> int:
    current = load_runtime()
    next_runtime = prompt_runtime(current)
    save_runtime(next_runtime)

    if next_runtime.validate_account and next_runtime.api_key and next_runtime.api_secret:
        client = _build_client(next_runtime)
        try:
            client.ping()
            client.ensure_btcusdc()
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


def command_status(_: argparse.Namespace) -> int:
    runtime = load_runtime()
    strategy = load_strategy()
    print(json.dumps({"runtime": _public_runtime_payload(runtime), "strategy": strategy.asdict()}, indent=2))
    return 0


def command_strategy(args: argparse.Namespace) -> int:
    cfg = load_strategy()
    runtime = load_runtime()
    updates = {}
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

    client = _build_client(runtime)
    try:
        client.ensure_btcusdc()
        candles = client.get_klines(symbol, interval, limit=args.limit)
    except BinanceAPIError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

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
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"saved {len(payload)} candles to {output}")
    return 0


def command_train(args: argparse.Namespace) -> int:
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

    if args.walk_forward:
        try:
            wf = walk_forward_report(
                rows,
                train_window=args.walk_forward_train,
                test_window=args.walk_forward_test,
                step=args.walk_forward_step,
                epochs=max(50, args.epochs // 2),
                calibrate=args.calibrate_threshold,
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

    model_signature = feature_signature(
        cfg.feature_windows[0],
        cfg.feature_windows[1],
        cfg.label_threshold,
        feature_version=cfg.feature_version,
        enabled_features=cfg.enabled_features,
    )

    split = max(2, int(len(rows) * 0.8))
    train_rows = rows[:split]
    test_rows = rows[split:]

    model = train(
        train_rows,
        epochs=args.epochs,
        seed=seed,
        feature_signature=model_signature,
    )
    threshold = cfg.signal_threshold
    if args.calibrate_threshold and test_rows:
        threshold = calibrate_threshold(test_rows, model, start=0.05, end=0.95, steps=31)
    train_score = evaluate(train_rows, model, threshold=threshold)
    test_score = evaluate(test_rows, model, threshold=threshold) if test_rows else 0.0
    tuned_score = evaluate(test_rows, model, threshold=threshold) if test_rows else test_score
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    serialize_model(model, output)
    print(f"trained model saved to {output}")
    print(f"rows: {len(rows)} split train={len(train_rows)} test={len(test_rows)}")
    print(f"in-sample directional accuracy: {train_score:.3f}")
    print(f"out-of-sample directional accuracy: {test_score:.3f}")
    if args.calibrate_threshold and test_rows:
        print(f"validated threshold: {threshold:.3f}")
        print(f"out-of-sample directional accuracy (calibrated): {tuned_score:.3f}")
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
            "rows_test": len(test_rows),
            "epochs": int(args.epochs),
            "lookback_windows": list(cfg.feature_windows),
            "label_threshold": float(cfg.label_threshold),
            "walk_forward": bool(args.walk_forward),
        },
        "walk_forward": wf if wf is not None else None,
        "metrics": {
            "in_sample_accuracy": float(train_score),
            "out_of_sample_accuracy": float(test_score),
            "threshold": float(threshold),
            "tuned_threshold": float(threshold) if args.calibrate_threshold and test_rows else None,
            "calibrated_out_of_sample_accuracy": float(tuned_score) if args.calibrate_threshold and test_rows else None,
            "model_feature_version": model.feature_version,
            "model_feature_signature": model.feature_signature,
        },
        "model": {
            "path": str(args.output),
            "feature_dim": int(model.feature_dim),
            "feature_version": str(model.feature_version),
            "feature_signature": model.feature_signature,
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
    if not test_rows:
        print("Need post-split test data to tune parameters")
        return 2

    risks: Iterable[float] = [cfg.risk_per_trade + (args.max_risk - args.min_risk) * i / max(args.steps - 1, 1)
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
    result = run_backtest(rows, model, cfg, starting_cash=args.start_cash, market_type=runtime.market_type)
    artifact = {
        "command": "backtest",
        "timestamp": int(time.time()),
        "runtime": _public_runtime_payload(runtime),
        "strategy": cfg.asdict(),
        "input": str(args.input),
        "model": str(model_path),
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

    split = max(1, int(len(rows) * 0.8))
    train_rows = rows[:split]
    test_rows = rows[split:]

    try:
        model = _load_runtime_model(model_path, cfg)
    except (OSError, json.JSONDecodeError, ModelLoadError, ModelFeatureMismatchError) as exc:
        print(f"Model load failed: {exc}", file=sys.stderr)
        return 2
    threshold = cfg.signal_threshold
    if args.threshold is not None:
        threshold = float(args.threshold)
    elif test_rows:
        # make default threshold robust against short samples and class imbalance
        threshold = cfg.signal_threshold

    if args.calibrate_threshold and test_rows:
        threshold = calibrate_threshold(test_rows, model, start=0.05, end=0.95, steps=31)

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
    if train_report is None:
        print("train_size: 0")
    else:
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
    model_path = Path("data/model.json")

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

    leverage = _resolve_futures_leverage(runtime, cfg)
    if runtime.market_type == "futures" and not effective_dry_run:
        try:
            set_response = client.set_leverage(runtime.symbol, int(leverage))
            if isinstance(set_response, dict) and set_response.get("leverage"):
                leverage = float(set_response.get("leverage"))
        except BinanceAPIError as exc:
            print(f"Failed to set leverage: {exc}", file=sys.stderr)
            return 2

    cash = 1000.0
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
            print(f"Model load failed; regenerating: {exc}", file=sys.stderr)
            model = None
        except Exception:
            model = None
    else:
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
        "model_signature": None,
        "starting_cash": float(cash),
    }
    drawdown_limit = float(cfg.max_drawdown_limit)
    halt_reason = "completed"
    steps_executed = 0
    entries = 0
    closes = 0
    skipped_entries = 0
    model_loads = 0 if model is None else 1

    for i in range(args.steps):
        try:
            candles = client.get_klines(runtime.symbol, runtime.interval, limit=max(cfg.model_lookback, 300))
        except BinanceAPIError as exc:
            print(f"market error: {exc}", file=sys.stderr)
            halt_reason = "market_error"
            return 2

        steps_executed += 1

        rows = _build_model_rows(candles, cfg)
        if not rows:
            print("not enough historical data for live signal")
            live_run["events"].append({"step": i + 1, "status": "no_rows"})
            time.sleep(args.sleep)
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
            model = train(rows, epochs=40)
            model_loads += 1
        latest = rows[-1]
        score = model.predict_proba(latest.features)
        direction = _score_to_direction(score, cfg, runtime.market_type)

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
                time.sleep(max(1, args.sleep))
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
                time.sleep(max(1, args.sleep))
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
                time.sleep(max(1, args.sleep))
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
                time.sleep(max(1, args.sleep))
                continue

            side_sign = 1 if direction > 0 else -1
            fill = price * (1.0 + side_sign * slippage)
            if fill <= 0:
                time.sleep(max(1, args.sleep))
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
                time.sleep(max(1, args.sleep))
                continue

            cash -= total
            position_side = direction
            position_notional = direction * notional
            qty = abs(qty)
            entry_price = fill
            margin_used = margin
            daily_trade_count[day] = daily_trade_count.get(day, 0) + 1

            side = "BUY" if side_sign > 0 else "SELL"
            _paper_or_live_order(
                client,
                runtime,
                cfg,
                side=side,
                size=qty,
                dry_run=effective_dry_run,
                leverage=leverage,
            )
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
                cash += margin_used + realized - exit_fee

                side_to_close = "SELL" if position_side > 0 else "BUY"
                _paper_or_live_order(
                    client,
                    runtime,
                    cfg,
                    side=side_to_close,
                    size=abs(qty),
                    dry_run=effective_dry_run,
                    leverage=leverage,
                )
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
                    cash += margin_used + realized - exit_fee

                    side_to_close = "SELL" if position_side > 0 else "BUY"
                    _paper_or_live_order(
                        client,
                        runtime,
                        cfg,
                        side=side_to_close,
                        size=abs(qty),
                        dry_run=effective_dry_run,
                        leverage=leverage,
                    )
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

        time.sleep(max(1, args.sleep))
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
    return 0


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if not argv:
        return command_menu(argparse.Namespace())
    args = _parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
