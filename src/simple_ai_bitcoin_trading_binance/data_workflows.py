"""CLI-facing data download and training-data loading workflows."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

from .api import BinanceAPIError
from .data_downloader import MarketDataSyncConfig, render_sync_result, sync_market_data
from .intervals import interval_milliseconds
from .market_data import clean_candles
from .market_store import MarketDataStore
from .storage import write_json_atomic
from .types import RuntimeConfig


def runtime_with_market(runtime: RuntimeConfig, market_type: str) -> RuntimeConfig:
    return RuntimeConfig(**{**runtime.asdict(), "market_type": market_type})


def data_sync_config_from_args(args: argparse.Namespace, runtime: RuntimeConfig) -> MarketDataSyncConfig:
    market_type = getattr(args, "market", None) or runtime.market_type
    return MarketDataSyncConfig(
        symbol=(getattr(args, "symbol", None) or runtime.symbol).upper(),
        interval=getattr(args, "interval", None) or runtime.interval,
        market_type=market_type,
        db_path=getattr(args, "db", "data/market_data.sqlite"),
        rows=max(0, int(getattr(args, "rows", 500))),
        batch_size=max(1, int(getattr(args, "batch_size", 1000))),
        include_futures_metrics=bool(getattr(args, "include_futures_metrics", True)),
    )


def start_background_data_sync(
    args: argparse.Namespace,
    *,
    python_executable: str = sys.executable,
    popen: Callable[..., object] = subprocess.Popen,
) -> int:
    pid_file = Path(getattr(args, "pid_file", "data/market_data_sync.pid"))
    log_file = Path(getattr(args, "log_file", "data/market_data_sync.log"))
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    command = [
        python_executable,
        "-m",
        "simple_ai_bitcoin_trading_binance",
        "data-sync",
        "--db",
        str(getattr(args, "db", "data/market_data.sqlite")),
        "--rows",
        str(max(0, int(getattr(args, "rows", 500)))),
        "--batch-size",
        str(max(1, int(getattr(args, "batch_size", 1000)))),
        "--loop",
        "--iterations",
        str(max(0, int(getattr(args, "iterations", 0)))),
        "--sleep",
        str(max(1, int(getattr(args, "sleep", 300)))),
    ]
    for option, value in (
        ("--symbol", getattr(args, "symbol", None)),
        ("--interval", getattr(args, "interval", None)),
        ("--market", getattr(args, "market", None)),
    ):
        if value:
            command.extend([option, str(value)])
    if not bool(getattr(args, "include_futures_metrics", True)):
        command.append("--no-include-futures-metrics")
    with log_file.open("ab") as log_handle:
        process = popen(command, stdout=log_handle, stderr=subprocess.STDOUT, start_new_session=True)
    pid_file.write_text(f"{process.pid}\n", encoding="utf-8")
    print(f"started market data downloader pid={process.pid}")
    print(f"pid_file={pid_file}")
    print(f"log_file={log_file}")
    return 0


def command_data_sync(
    args: argparse.Namespace,
    *,
    load_runtime_fn: Callable[[], RuntimeConfig],
    build_client_fn: Callable[[RuntimeConfig], object],
    sync_market_data_fn: Callable[..., object] = sync_market_data,
    render_sync_result_fn: Callable[[object], str] = render_sync_result,
    sleep_fn: Callable[[float], None] = time.sleep,
    python_executable: str = sys.executable,
    popen: Callable[..., object] = subprocess.Popen,
) -> int:
    if getattr(args, "background", False):
        return start_background_data_sync(args, python_executable=python_executable, popen=popen)
    runtime = load_runtime_fn()
    config = data_sync_config_from_args(args, runtime)
    client = build_client_fn(runtime_with_market(runtime, config.market_type))
    futures_client = None
    if config.include_futures_metrics and config.market_type != "futures":
        futures_client = build_client_fn(runtime_with_market(runtime, "futures"))

    loop = bool(getattr(args, "loop", False))
    iterations = max(0, int(getattr(args, "iterations", 1)))
    completed = 0
    exit_code = 0
    while True:  # pragma: no branch - loop exits through the iteration guard below
        try:
            result = sync_market_data_fn(client, config, futures_client=futures_client)
        except (BinanceAPIError, ValueError, OSError) as exc:
            print(f"Market data sync failed: {exc}", file=sys.stderr)
            return 2
        if getattr(args, "json", False):
            print(json.dumps(result.asdict(), indent=2, sort_keys=True))
        else:
            print(render_sync_result_fn(result))
        if result.status == "fail":
            exit_code = 2
        completed += 1
        if not loop or (iterations > 0 and completed >= iterations):
            break
        sleep_fn(max(0, int(getattr(args, "sleep", 300))))
    return exit_code


def command_fetch(
    args: argparse.Namespace,
    *,
    load_runtime_fn: Callable[[], RuntimeConfig],
    build_client_fn: Callable[[RuntimeConfig], object],
) -> int:
    runtime = load_runtime_fn()
    symbol = (args.symbol or runtime.symbol).upper()
    interval = args.interval or runtime.interval
    output = Path(args.output)
    if symbol != "BTCUSDC":
        print("Error: this CLI supports BTCUSDC only", file=sys.stderr)
        return 2
    limit = max(1, int(args.limit))
    max_batch_size = 1500 if runtime.market_type == "futures" else 1000
    batch_size = max(1, min(max_batch_size, int(getattr(args, "batch_size", 1000))))

    client = build_client_fn(runtime)
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


def load_training_candles_from_db(
    db_path: str | Path,
    runtime: RuntimeConfig,
    *,
    interval: str,
    market_type: str,
    min_rows: int,
) -> list | None:
    with MarketDataStore(db_path) as store:
        candles = store.fetch_candles(runtime.symbol, market_type, interval)
        quality = store.coverage_quality(runtime.symbol, market_type, interval, interval_milliseconds(interval))
    if len(candles) >= min_rows and quality.gap_count:
        print(
            f"warning: {runtime.symbol} {market_type} {interval} database coverage has "
            f"{quality.gap_count} missing intervals ({quality.coverage_ratio:.1%} coverage)",
            file=sys.stderr,
        )
    return candles if len(candles) >= min_rows else None


def confirm_download_missing_training_data(
    *,
    symbol: str,
    market_type: str,
    interval: str,
    available: int,
    required: int,
    stdin=sys.stdin,
    input_fn: Callable[[str], str] = input,
) -> bool:
    if not stdin.isatty():
        return False
    answer = input_fn(
        f"Only {available}/{required} {symbol} {market_type} {interval} rows are available. "
        "Download missing data now? [y/N] "
    )
    return answer.strip().lower() in {"y", "yes"}


def download_training_candles(
    args: argparse.Namespace,
    runtime: RuntimeConfig,
    *,
    interval: str,
    market_type: str,
    command_fn: Callable[[argparse.Namespace], int],
) -> bool:
    min_rows = int(getattr(args, "min_rows", 120))
    sync_args = argparse.Namespace(
        db=getattr(args, "db", "data/market_data.sqlite"),
        symbol=runtime.symbol,
        interval=interval,
        market=market_type,
        rows=max(min_rows, min_rows + 50),
        batch_size=1000,
        include_futures_metrics=True,
        loop=False,
        iterations=1,
        sleep=0,
        background=False,
        json=False,
    )
    return command_fn(sync_args) == 0


def load_training_candles(
    args: argparse.Namespace,
    runtime: RuntimeConfig,
    *,
    load_rows_fn: Callable[..., list | None],
    db_loader_fn: Callable[..., list | None],
    confirm_fn: Callable[..., bool],
    download_fn: Callable[..., bool],
) -> tuple[list | None, str]:
    source = str(getattr(args, "source", "file") or "file")
    input_path = Path(getattr(args, "input", "data/historical_btcusdc.json"))
    interval = getattr(args, "interval", None) or runtime.interval
    market_type = getattr(args, "market", None) or runtime.market_type
    min_rows = max(1, int(getattr(args, "min_rows", 120)))
    if source in {"auto", "file"} and input_path.exists():
        candles = load_rows_fn(str(input_path), label="Training data load failed")
        return candles, "file" if candles is not None else "missing"
    if source == "file":
        candles = load_rows_fn(str(input_path), label="Training data load failed")
        return candles, "file" if candles is not None else "missing"

    candles = db_loader_fn(
        getattr(args, "db", "data/market_data.sqlite"),
        runtime,
        interval=interval,
        market_type=market_type,
        min_rows=min_rows,
    )
    if candles is not None:
        print(f"loaded {len(candles)} candles from market database for {runtime.symbol} {market_type} {interval}")
        return candles, "db"

    with MarketDataStore(getattr(args, "db", "data/market_data.sqlite")) as store:
        available = store.coverage(runtime.symbol, market_type, interval).count
    should_download = bool(getattr(args, "download_missing", False)) or confirm_fn(
        symbol=runtime.symbol,
        market_type=market_type,
        interval=interval,
        available=available,
        required=min_rows,
    )
    if should_download and download_fn(args, runtime, interval=interval, market_type=market_type):
        candles = db_loader_fn(
            getattr(args, "db", "data/market_data.sqlite"),
            runtime,
            interval=interval,
            market_type=market_type,
            min_rows=min_rows,
        )
        if candles is not None:
            return candles, "db_downloaded"
    print(
        f"Training data unavailable for {runtime.symbol} {market_type} {interval}: "
        f"{available}/{min_rows} rows in {getattr(args, 'db', 'data/market_data.sqlite')}. "
        "Run data-sync or use --download-missing.",
        file=sys.stderr,
    )
    return None, "missing"
