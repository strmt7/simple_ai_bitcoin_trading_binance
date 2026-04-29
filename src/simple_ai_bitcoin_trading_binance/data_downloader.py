"""Rate-limited Binance market-data downloader backed by SQLite."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from .api import BinanceAPIError, BinanceClient
from .intervals import max_limit, validate_interval
from .market_data import clean_candles
from .market_store import MarketDataStore


@dataclass(frozen=True)
class MarketDataSyncConfig:
    symbol: str = "BTCUSDC"
    interval: str = "15m"
    market_type: str = "spot"
    db_path: str | Path = "data/market_data.sqlite"
    rows: int = 500
    batch_size: int = 1000
    include_futures_metrics: bool = True
    now_ms: int | None = None


@dataclass(frozen=True)
class MarketDataSyncResult:
    status: str
    db_path: str
    symbol: str
    interval: str
    market_type: str
    candles_inserted: int
    candles_available: int
    latest_open_time: int | None
    snapshots_inserted: int
    errors: list[str]
    request_info: dict[str, object]

    def asdict(self) -> dict[str, object]:
        return asdict(self)


def _snapshot_time(payload: object, fallback_ms: int | None) -> int | None:
    if not isinstance(payload, dict):
        return fallback_ms
    for key in ("time", "closeTime", "openTime", "fundingTime"):
        if key in payload:
            try:
                return int(float(payload[key]))
            except (TypeError, ValueError):
                return fallback_ms
    return fallback_ms


def _capture_snapshot(
    store: MarketDataStore,
    client: object,
    symbol: str,
    market_type: str,
    kind: str,
    fetcher: Callable[[], object],
    errors: list[str],
    now_ms: int | None,
) -> int:
    try:
        payload = fetcher()
        if not isinstance(payload, (dict, list)):
            raise BinanceAPIError(f"Unexpected {kind} payload")
        return store.insert_snapshot(
            "binance",
            symbol,
            market_type,
            kind,
            payload,
            ts_ms=_snapshot_time(payload, now_ms),
        )
    except (BinanceAPIError, OSError, ValueError) as exc:
        errors.append(f"{kind}: {exc}")
        return 0


def sync_market_data(
    client: BinanceClient,
    config: MarketDataSyncConfig,
    *,
    futures_client: BinanceClient | None = None,
) -> MarketDataSyncResult:
    symbol = config.symbol.upper()
    if symbol != "BTCUSDC":
        raise BinanceAPIError("This downloader supports BTCUSDC only")
    interval = validate_interval(config.interval, config.market_type)
    batch_size = max(1, min(max_limit(config.market_type), int(config.batch_size)))
    rows_requested = max(0, int(config.rows))
    errors: list[str] = []
    candles_inserted = 0
    snapshots_inserted = 0

    with MarketDataStore(config.db_path) as store:
        end_time = None
        remaining = rows_requested
        while remaining > 0:
            request_limit = min(batch_size, remaining)
            try:
                chunk = client.get_klines(symbol, interval, limit=request_limit, end_time=end_time)
            except BinanceAPIError as exc:
                errors.append(f"klines: {exc}")
                break
            if not chunk:
                break
            cleaned = clean_candles(chunk, now_ms=config.now_ms)
            candles_inserted += store.upsert_candles(
                symbol,
                config.market_type,
                interval,
                cleaned,
                ingested_at_ms=config.now_ms,
            )
            remaining -= len(chunk)
            earliest_open = min(candle.open_time for candle in chunk)
            next_end = earliest_open - 1
            if len(chunk) < request_limit or next_end == end_time:
                break
            end_time = next_end

        snapshots_inserted += _capture_snapshot(
            store,
            client,
            symbol,
            config.market_type,
            "ticker_24h",
            lambda: client.get_ticker_24h(symbol),
            errors,
            config.now_ms,
        )
        snapshots_inserted += _capture_snapshot(
            store,
            client,
            symbol,
            config.market_type,
            "book_ticker",
            lambda: client.get_book_ticker(symbol),
            errors,
            config.now_ms,
        )

        fclient = futures_client if futures_client is not None else client
        if config.include_futures_metrics and getattr(fclient, "market_type", "") == "futures":
            snapshots_inserted += _capture_snapshot(
                store,
                fclient,
                symbol,
                "futures",
                "premium_index",
                lambda: fclient.get_futures_premium_index(symbol),
                errors,
                config.now_ms,
            )
            snapshots_inserted += _capture_snapshot(
                store,
                fclient,
                symbol,
                "futures",
                "open_interest",
                lambda: fclient.get_futures_open_interest(symbol),
                errors,
                config.now_ms,
            )
            snapshots_inserted += _capture_snapshot(
                store,
                fclient,
                symbol,
                "futures",
                "funding_rate_history",
                lambda: fclient.get_futures_funding_rate(symbol, limit=100),
                errors,
                config.now_ms,
            )
        elif config.include_futures_metrics:
            errors.append("futures_metrics: futures client unavailable")

        coverage = store.coverage(symbol, config.market_type, interval)
        status = "ok" if candles_inserted > 0 and not errors else ("warn" if candles_inserted > 0 else "fail")
        result = MarketDataSyncResult(
            status=status,
            db_path=str(config.db_path),
            symbol=symbol,
            interval=interval,
            market_type=config.market_type,
            candles_inserted=candles_inserted,
            candles_available=coverage.count,
            latest_open_time=coverage.last_open_time,
            snapshots_inserted=snapshots_inserted,
            errors=errors,
            request_info=dict(getattr(client, "last_request_info", {})),
        )
        store.insert_sync_run(result.asdict())
        return result


def render_sync_result(result: MarketDataSyncResult) -> str:
    lines = [
        "Market data sync",
        (
            f"status={result.status} symbol={result.symbol} market={result.market_type} "
            f"interval={result.interval} candles_inserted={result.candles_inserted} "
            f"candles_available={result.candles_available} snapshots={result.snapshots_inserted}"
        ),
        f"db={result.db_path}",
    ]
    if result.latest_open_time is not None:
        lines.append(f"latest_open_time={result.latest_open_time}")
    for error in result.errors:
        lines.append(f"warning: {error}")
    return "\n".join(lines)
