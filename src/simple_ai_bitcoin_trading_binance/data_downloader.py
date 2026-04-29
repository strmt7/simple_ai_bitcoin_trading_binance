"""Rate-limited Binance market-data downloader backed by SQLite."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from .api import BinanceAPIError, BinanceClient, Candle
from .intervals import interval_milliseconds, max_limit, validate_interval
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
    sync_mode: str = "backfill"
    candles_added: int = 0
    gap_count: int = 0
    coverage_ratio: float = 0.0
    kline_requests: int = 0
    kline_rows_received: int = 0

    def asdict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class CandleSyncStats:
    rows_changed: int = 0
    requests: int = 0
    rows_received: int = 0


@dataclass(frozen=True)
class CandleChunkStats:
    rows_changed: int
    closed_rows: int


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


def _store_candle_chunk(
    store: MarketDataStore,
    symbol: str,
    market_type: str,
    interval: str,
    chunk: list[Candle],
    now_ms: int | None,
) -> CandleChunkStats:
    cleaned = clean_candles(chunk, now_ms=now_ms)
    rows_changed = store.upsert_candles(
        symbol,
        market_type,
        interval,
        cleaned,
        ingested_at_ms=now_ms,
    )
    return CandleChunkStats(rows_changed=rows_changed, closed_rows=len(cleaned))


def _sync_incremental_candles(
    store: MarketDataStore,
    client: BinanceClient,
    symbol: str,
    market_type: str,
    interval: str,
    *,
    batch_size: int,
    start_time: int,
    now_ms: int | None,
    errors: list[str],
) -> CandleSyncStats:
    try:
        chunk = client.get_klines(symbol, interval, limit=batch_size, start_time=start_time)
    except BinanceAPIError as exc:
        errors.append(f"klines: {exc}")
        return CandleSyncStats(requests=1)
    chunk_stats = _store_candle_chunk(store, symbol, market_type, interval, chunk, now_ms)
    return CandleSyncStats(
        rows_changed=chunk_stats.rows_changed,
        requests=1,
        rows_received=len(chunk),
    )


def _sync_backfill_candles(
    store: MarketDataStore,
    client: BinanceClient,
    symbol: str,
    market_type: str,
    interval: str,
    *,
    batch_size: int,
    rows_requested: int,
    now_ms: int | None,
    errors: list[str],
) -> CandleSyncStats:
    rows_changed = 0
    requests = 0
    rows_received = 0
    end_time = None
    remaining_closed = rows_requested
    while remaining_closed > 0:
        latest_page_extra = 1 if end_time is None else 0
        request_limit = min(batch_size, remaining_closed + latest_page_extra)
        requests += 1
        try:
            chunk = client.get_klines(symbol, interval, limit=request_limit, end_time=end_time)
        except BinanceAPIError as exc:
            errors.append(f"klines: {exc}")
            break
        if not chunk:
            break
        rows_received += len(chunk)
        chunk_stats = _store_candle_chunk(store, symbol, market_type, interval, chunk, now_ms)
        rows_changed += chunk_stats.rows_changed
        remaining_closed -= chunk_stats.closed_rows
        earliest_open = min(candle.open_time for candle in chunk)
        next_end = earliest_open - 1
        if len(chunk) < request_limit or next_end == end_time:
            break
        end_time = next_end
    return CandleSyncStats(rows_changed=rows_changed, requests=requests, rows_received=rows_received)


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
    step_ms = interval_milliseconds(interval)
    batch_size = max(1, min(max_limit(config.market_type), int(config.batch_size)))
    rows_requested = max(0, int(config.rows))
    errors: list[str] = []
    candles_inserted = 0
    kline_requests = 0
    kline_rows_received = 0
    snapshots_inserted = 0
    sync_mode = "backfill"

    with MarketDataStore(config.db_path) as store:
        coverage_before = store.coverage(symbol, config.market_type, interval)
        latest_open_time = coverage_before.last_open_time
        should_increment = rows_requested > 0 and coverage_before.count >= rows_requested and latest_open_time is not None
        if should_increment:
            sync_mode = "incremental"
            candle_stats = _sync_incremental_candles(
                store,
                client,
                symbol,
                config.market_type,
                interval,
                batch_size=batch_size,
                start_time=int(latest_open_time) + step_ms,
                now_ms=config.now_ms,
                errors=errors,
            )
        else:
            candle_stats = _sync_backfill_candles(
                store,
                client,
                symbol,
                config.market_type,
                interval,
                batch_size=batch_size,
                rows_requested=rows_requested,
                now_ms=config.now_ms,
                errors=errors,
            )
        candles_inserted += candle_stats.rows_changed
        kline_requests += candle_stats.requests
        kline_rows_received += candle_stats.rows_received

        coverage_after_candles = store.coverage(symbol, config.market_type, interval)
        candles_added = max(0, coverage_after_candles.count - coverage_before.count)

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

        coverage_quality = store.coverage_quality(symbol, config.market_type, interval, step_ms)
        coverage = coverage_quality.coverage
        has_candles = coverage.count > 0
        status = "ok" if has_candles and not errors else ("warn" if has_candles else "fail")
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
            sync_mode=sync_mode,
            candles_added=candles_added,
            gap_count=coverage_quality.gap_count,
            coverage_ratio=coverage_quality.coverage_ratio,
            kline_requests=kline_requests,
            kline_rows_received=kline_rows_received,
        )
        store.insert_sync_run(result.asdict())
        return result


def render_sync_result(result: MarketDataSyncResult) -> str:
    lines = [
        "Market data sync",
        (
            f"status={result.status} symbol={result.symbol} market={result.market_type} "
            f"interval={result.interval} mode={result.sync_mode} "
            f"candles_inserted={result.candles_inserted} candles_added={result.candles_added} "
            f"candles_available={result.candles_available} snapshots={result.snapshots_inserted} "
            f"kline_requests={result.kline_requests} kline_rows={result.kline_rows_received}"
        ),
        f"db={result.db_path}",
    ]
    if result.latest_open_time is not None:
        lines.append(f"latest_open_time={result.latest_open_time}")
    lines.append(f"coverage_ratio={result.coverage_ratio:.4f} gap_count={result.gap_count}")
    for error in result.errors:
        lines.append(f"warning: {error}")
    return "\n".join(lines)
