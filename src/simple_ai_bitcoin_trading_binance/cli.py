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
from .model import evaluate, load_model, serialize_model, train
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
    parser_train.set_defaults(func=command_train)

    parser_tune = subparsers.add_parser("tune", help="perform a focused walk-forward tune over few risk parameters")
    parser_tune.add_argument("--input", default="data/historical_btcusdc.json")
    parser_tune.add_argument("--save-best", action="store_true")
    parser_tune.add_argument("--min-risk", type=float, default=0.002)
    parser_tune.add_argument("--max-risk", type=float, default=0.02)
    parser_tune.add_argument("--steps", type=int, default=5)
    parser_tune.set_defaults(func=command_tune)

    parser_backtest = subparsers.add_parser("backtest", help="run backtest against cached data")
    parser_backtest.add_argument("--input", default="data/historical_btcusdc.json")
    parser_backtest.add_argument("--model", default="data/model.json")
    parser_backtest.add_argument("--start-cash", type=float, default=1000.0)
    parser_backtest.set_defaults(func=command_backtest)

    parser_live = subparsers.add_parser("live", help="run a conservative live loop on testnet or paper mode")
    parser_live.add_argument("--steps", type=int, default=20)
    parser_live.add_argument("--sleep", type=int, default=5)
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


def _rows_from_json(path: str):
    candles_raw = _load_json_candles(path)
    from .api import Candle

    rows: list[Candle] = []
    for item in candles_raw:
        if not isinstance(item, dict):
            continue
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
    return rows


def _build_model_rows(candles: Sequence[object], strategy: StrategyConfig):
    return make_rows(candles, strategy.feature_windows[0], strategy.feature_windows[1])


def _effective_leverage(cfg: StrategyConfig, market_type: str) -> float:
    if market_type != "futures":
        return 1.0
    return float(max(1.0, min(125.0, cfg.leverage)))


def _target_notional(cash: float, strategy: StrategyConfig, market_type: str) -> float:
    if cash <= 0:
        return 0.0
    leverage = _effective_leverage(strategy, market_type)
    risk_exposure = strategy.risk_per_trade * leverage
    risk_exposure = min(risk_exposure, strategy.max_position_pct * leverage, 1.0)
    return cash * risk_exposure


def _direction_from_signal(signal: int, market_type: str) -> int:
    if market_type == "futures":
        return 1 if signal == 1 else -1
    return 1 if signal == 1 else 0


def _paper_or_live_order(client: BinanceClient, runtime, strategy: StrategyConfig, *, side: str, size: float, dry_run: bool) -> None:
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
    updates = {}
    if args.leverage is not None:
        updates["leverage"] = max(1.0, args.leverage)
    if args.risk is not None:
        updates["risk_per_trade"] = max(0.0001, args.risk)
    if args.max_position is not None:
        updates["max_position_pct"] = max(0.0, min(1.0, args.max_position))
    if args.stop is not None:
        updates["stop_loss_pct"] = max(0.0, min(0.99, args.stop))
    if args.take is not None:
        updates["take_profit_pct"] = max(0.0, min(0.99, args.take))
    if args.cooldown is not None:
        updates["cooldown_minutes"] = max(0, args.cooldown)

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

    split = max(2, int(len(rows) * 0.8))
    train_rows = rows[:split]
    test_rows = rows[split:]

    model = train(train_rows, epochs=args.epochs)
    train_score = evaluate(train_rows, model, threshold=cfg.signal_threshold)
    test_score = evaluate(test_rows, model, threshold=cfg.signal_threshold) if test_rows else 0.0
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    serialize_model(model, output)
    print(f"trained model saved to {output}")
    print(f"rows: {len(rows)} split train={len(train_rows)} test={len(test_rows)}")
    print(f"in-sample directional accuracy: {train_score:.3f}")
    print(f"out-of-sample directional accuracy: {test_score:.3f}")
    print(f"market={runtime.market_type} leverage={cfg.leverage}")
    return 0


def command_tune(args: argparse.Namespace) -> int:
    cfg = load_strategy()
    runtime = load_runtime()
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
    best: StrategyConfig = cfg
    best_score = float("-inf")

    for risk in risks:
        candidate = StrategyConfig(**{**cfg.asdict(), "risk_per_trade": max(0.0001, risk)})
        model = train(train_rows, epochs=max(50, candidate.training_epochs // 2))
        candidate_result = run_backtest(test_rows, model, candidate, market_type=runtime.market_type, starting_cash=1000.0)
        score = candidate_result.realized_pnl
        if score > best_score:
            best_score = score
            best = candidate

    print(f"tune best score: {best_score:.4f}")
    print(f"tune best config risk={best.risk_per_trade:.4f} take={best.take_profit_pct:.4f} stop={best.stop_loss_pct:.4f}")
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
    print(f"starting_cash: {result.starting_cash:.2f}")
    print(f"ending_cash: {result.ending_cash:.2f}")
    print(f"max_drawdown: {result.max_drawdown:.2%}")
    return 0


def command_live(args: argparse.Namespace) -> int:
    runtime = load_runtime()
    cfg = load_strategy()
    client = _build_client(runtime)

    effective_dry_run = runtime.dry_run or args.paper
    if not runtime.testnet:
        print("Real-money execution is disabled in this phase. Set testnet=true to run.")
        return 2
    if not effective_dry_run and (not runtime.api_key or not runtime.api_secret):
        print("Live mode needs API key and secret. Run configure first or run with --paper.")
        return 2

    leverage = _effective_leverage(cfg, runtime.market_type)
    if runtime.market_type == "futures" and not effective_dry_run:
        try:
            client.set_leverage(runtime.symbol, int(leverage))
        except BinanceAPIError as exc:
            print(f"Failed to set leverage: {exc}", file=sys.stderr)
            return 2

    cash = 1000.0
    position_notional = 0.0
    position_side = 0
    entry_price = 0.0
    wait_ticks = cfg.cooldown_minutes
    cooldown_left = 0

    mode_label = "paper" if effective_dry_run else "live"
    print(f"Starting {mode_label} loop for {args.steps} steps on {runtime.symbol} {runtime.interval} [{runtime.market_type}]")
    if runtime.market_type == "futures":
        print(f"effective leverage: {leverage:.1f}x")

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

        model = train(rows, epochs=40)
        latest = rows[-1]
        signal = model.predict(latest.features, cfg.signal_threshold)
        direction = _direction_from_signal(signal, runtime.market_type)

        # cooldown reduces immediate flip-flopping in choppy conditions
        if cooldown_left > 0:
            direction = 0 if runtime.market_type == "spot" else direction
            cooldown_left -= 1

        price = latest.close
        if position_side == 0 and direction != 0:
            direction_to_open = direction
            notional = _target_notional(cash, cfg, runtime.market_type)
            if notional <= 0:
                time.sleep(args.sleep)
                continue
            size = abs(notional / price)
            position_notional = direction_to_open * notional
            position_side = direction_to_open
            entry_price = price
            side = "BUY" if direction_to_open > 0 else "SELL"
            _paper_or_live_order(
                client,
                runtime,
                cfg,
                side=side,
                size=size,
                dry_run=effective_dry_run,
            )
            print(f"step {i + 1:>2}: enter {'long' if position_side > 0 else 'short'} at {price:.2f} qty={size:.6f}")
            cooldown_left = 0

        elif position_side != 0:
            pnl_pct = (price - entry_price) / entry_price * (1 if position_side > 0 else -1)

            opposite_signal = direction != 0 and direction != position_side if runtime.market_type == "futures" else direction == 0
            should_close = opposite_signal or pnl_pct >= cfg.take_profit_pct or pnl_pct <= -cfg.stop_loss_pct

            if should_close:
                cash += abs(position_notional) * pnl_pct
                side_to_close = "SELL" if position_side > 0 else "BUY"
                _paper_or_live_order(
                    client,
                    runtime,
                    cfg,
                    side=side_to_close,
                    size=abs(position_notional / entry_price),
                    dry_run=effective_dry_run,
                )
                print(
                    f"step {i + 1:>2}: close {'long' if position_side > 0 else 'short'} "
                    f"pnl={pnl_pct:.2%} cash={cash:.2f}"
                )
                position_notional = 0.0
                position_side = 0
                entry_price = 0.0
                cooldown_left = max(0, wait_ticks)
            else:
                print(f"step {i + 1:>2}: hold {'long' if position_side > 0 else 'short'} pnl={pnl_pct:.2%} cash={cash:.2f}")

                if direction == 0:
                    cooldown_left = max(0, wait_ticks)

        time.sleep(max(1, args.sleep))

    print(f"finished loop market={runtime.market_type} cash={cash:.2f}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
