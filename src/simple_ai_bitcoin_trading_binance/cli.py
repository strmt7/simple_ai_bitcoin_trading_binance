"""Entry point for the BTCUSDC test-trading CLI."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Iterable, Sequence

from .api import BinanceAPIError, BinanceClient
from .backtest import run_backtest
from .config import config_paths, load_runtime, load_strategy, prompt_runtime, save_runtime, save_strategy
from .features import make_rows
from .api import SymbolConstraints
from .model import (
    calibrate_threshold,
    evaluate_classification,
    evaluate,
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
        "--paper",
        action="store_true",
        help="force paper mode for this run even when runtime.dry_run is false",
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


def _build_model_rows(candles: Sequence[object], strategy: StrategyConfig):
    return make_rows(
        candles,
        strategy.feature_windows[0],
        strategy.feature_windows[1],
        lookahead=1,
        label_threshold=strategy.label_threshold,
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
    payload = runtime.asdict()
    if payload.get("api_secret"):
        payload["api_secret"] = "*" * min(4, len(payload["api_secret"]))
    print(json.dumps({"runtime": payload, "strategy": strategy.asdict()}, indent=2))
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
    candles = _rows_from_json(args.input)
    rows = _build_model_rows(candles, cfg)
    if not rows:
        print("No rows produced. Fetch more data or increase lookback.")
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

    split = max(2, int(len(rows) * 0.8))
    train_rows = rows[:split]
    test_rows = rows[split:]

    model = train(train_rows, epochs=args.epochs)
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
    resolved_leverage = _resolve_futures_leverage(runtime, cfg)
    print(f"market={runtime.market_type} leverage={resolved_leverage:.2f}")
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
    candles = _rows_from_json(args.input)
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
                        if getattr(candidate_result, "stopped_by_drawdown", False):
                            continue
                        if score > best_score:
                            best_score = score
                            best = candidate

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

    rows = _build_model_rows(_rows_from_json(args.input), cfg)
    model = load_model(model_path)
    result = run_backtest(rows, model, cfg, starting_cash=args.start_cash, market_type=runtime.market_type)
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
    rows = _build_model_rows(_rows_from_json(args.input), cfg)
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

    model = load_model(model_path)
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
    leverage_override = getattr(args, "leverage", None)
    if leverage_override is not None:
        requested = max(1.0, leverage_override)
        if runtime.market_type == "futures":
            cfg = StrategyConfig(**{**cfg.asdict(), "leverage": requested})
        else:
            print("Leverage override is spot-inactive; spot runs at 1x.")
    client = _build_client(runtime)
    model_path = Path("data/model.json")

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
            model = load_model(model_path)
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

    for i in range(args.steps):
        try:
            candles = client.get_klines(runtime.symbol, runtime.interval, limit=max(cfg.model_lookback, 300))
        except BinanceAPIError as exc:
            print(f"market error: {exc}", file=sys.stderr)
            return 2

        rows = _build_model_rows(candles, cfg)
        if not rows:
            print("not enough historical data for live signal")
            time.sleep(args.sleep)
            continue

        if model is None:
            model = train(rows, epochs=40)
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
                time.sleep(max(1, args.sleep))
                continue
            if max_open_positions <= 0:
                print(f"step {i + 1:>2}: max open positions reached ({max_open_positions}), skipping entry")
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
                time.sleep(max(1, args.sleep))
                continue

            cash -= total
            position_side = direction
            position_notional = direction * notional
            qty = abs(qty)
            entry_price = fill
            margin_used = margin

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
            cooldown_left = 0

        elif position_side != 0:
            pnl = position_side * (price - entry_price) * qty
            pnl_pct = ((price - entry_price) / entry_price) if position_side > 0 else ((entry_price - price) / entry_price)

            opposite_signal = direction != 0 and direction != position_side if runtime.market_type == "futures" else direction == 0
            should_close = opposite_signal or pnl_pct >= cfg.take_profit_pct or pnl_pct <= -cfg.stop_loss_pct

            if should_close:
                fill = price * (1.0 - position_side * slippage)
                realized = position_side * (fill - entry_price) * qty
                exit_fee = abs(position_notional) * fee_rate
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
                daily_trade_count[day] = daily_trade_count.get(day, 0) + 1
                print(
                    f"step {i + 1:>2}: close {'long' if position_side > 0 else 'short'} "
                    f"pnl={pnl:.2f} cash={cash:.2f}"
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
                    exit_fee = abs(position_notional) * fee_rate
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
                print(f"step {i + 1:>2}: drawdown limit reached ({cfg.max_drawdown_limit:.1%}), stopping loop")
                break

        time.sleep(max(1, args.sleep))

    if max_drawdown_seen > 0.0:
        print(f"max_drawdown observed: {max_drawdown_seen:.2%}")
    print(f"finished loop market={runtime.market_type} cash={cash:.2f}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
