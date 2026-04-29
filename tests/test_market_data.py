from __future__ import annotations

import json

import pytest

from simple_ai_bitcoin_trading_binance.api import Candle
from simple_ai_bitcoin_trading_binance.data_downloader import (
    MarketDataSyncConfig,
    _snapshot_time,
    render_sync_result,
    sync_market_data,
)
import simple_ai_bitcoin_trading_binance.storage as storage
from simple_ai_bitcoin_trading_binance.market_data import clean_candles
from simple_ai_bitcoin_trading_binance.market_store import MarketDataStore
from simple_ai_bitcoin_trading_binance.storage import write_json_atomic


def _candle(open_time: int, close: float = 100.0, close_time: int | None = None) -> Candle:
    return Candle(
        open_time=open_time,
        open=close,
        high=close + 1.0,
        low=close - 1.0,
        close=close,
        volume=1.0,
        close_time=open_time + 60_000 if close_time is None else close_time,
    )


def test_clean_candles_sorts_dedupes_invalid_and_unclosed() -> None:
    first = _candle(60_000, close=100.0)
    replacement = _candle(60_000, close=101.0)
    valid_early = _candle(0, close=99.0)
    invalid = Candle(120_000, open=5.0, high=4.0, low=6.0, close=5.0, volume=1.0, close_time=180_000)
    unclosed = _candle(180_000, close=102.0, close_time=10_000_000)

    cleaned = clean_candles([first, invalid, unclosed, valid_early, replacement], now_ms=240_000)

    assert [c.open_time for c in cleaned] == [0, 60_000]
    assert cleaned[-1].close == 101.0


def test_clean_candles_ignores_non_candles_and_can_keep_unclosed_rows() -> None:
    future = _candle(120_000, close=102.0, close_time=10_000_000)

    cleaned = clean_candles([object(), future], now_ms=0, drop_unclosed=False)  # type: ignore[list-item]

    assert cleaned == [future]


def test_write_json_atomic_replaces_payload_and_applies_mode(tmp_path) -> None:
    target = tmp_path / "nested" / "payload.json"
    write_json_atomic(target, {"b": 2}, sort_keys=True, mode=0o600)
    write_json_atomic(target, {"a": 1}, sort_keys=True, mode=0o600)

    assert json.loads(target.read_text(encoding="utf-8")) == {"a": 1}
    assert target.stat().st_mode & 0o777 == 0o600


def test_write_json_atomic_removes_temp_file_on_failed_write(tmp_path, monkeypatch) -> None:
    def fail_dump(*_args, **_kwargs) -> None:
        raise ValueError("boom")

    target = tmp_path / "payload.json"
    monkeypatch.setattr(storage.json, "dump", fail_dump)

    with pytest.raises(ValueError, match="boom"):
        write_json_atomic(target, {"x": 1})

    assert not target.exists()
    assert list(tmp_path.glob(".payload.json.*.tmp")) == []


def test_write_json_atomic_preserves_original_error_when_cleanup_fails(tmp_path, monkeypatch) -> None:
    def fail_dump(*_args, **_kwargs) -> None:
        raise ValueError("boom")

    def fail_unlink(_path) -> None:
        raise OSError("locked")

    target = tmp_path / "payload.json"
    with monkeypatch.context() as context:
        context.setattr(storage.json, "dump", fail_dump)
        context.setattr(storage.Path, "unlink", fail_unlink)
        with pytest.raises(ValueError, match="boom"):
            write_json_atomic(target, {"x": 1})

    for leftover in tmp_path.glob(".payload.json.*.tmp"):
        leftover.unlink()


def test_market_data_store_roundtrip_snapshots_and_sync_runs(tmp_path) -> None:
    db = tmp_path / "market.sqlite"
    store = MarketDataStore(db)
    assert store.connect() is store.connect()
    assert store.upsert_candles("BTCUSDC", "spot", "15m", []) == 0
    candles = [
        Candle(0, 100, 101, 99, 100, 1, 59_000, quote_volume=100, trade_count=3),
        Candle(60_000, 101, 102, 100, 101, 2, 119_000, quote_volume=202, trade_count=4),
    ]
    assert store.upsert_candles("BTCUSDC", "spot", "15m", candles, ingested_at_ms=123) == 2
    assert store.upsert_candles("BTCUSDC", "spot", "15m", candles, ingested_at_ms=456) == 0
    changed = [
        Candle(0, 100, 101, 99, 100, 1, 59_000, quote_volume=100, trade_count=3),
        Candle(60_000, 101, 103, 100, 102, 2, 119_000, quote_volume=204, trade_count=5),
    ]
    assert store.upsert_candles("BTCUSDC", "spot", "15m", changed, ingested_at_ms=789) == 1
    assert [c.open_time for c in store.fetch_candles("btcusdc", "spot", "15m")] == [0, 60_000]
    latest = store.fetch_candles("BTCUSDC", "spot", "15m", limit=1)
    assert latest[0].quote_volume == 204
    coverage = store.coverage("BTCUSDC", "spot", "15m")
    assert coverage.asdict()["count"] == 2
    quality = store.coverage_quality("BTCUSDC", "spot", "15m", 60_000)
    assert quality.expected_count == 2
    assert quality.gap_count == 0
    assert quality.coverage_ratio == 1.0
    assert quality.asdict()["coverage"]["count"] == 2
    assert store.coverage_quality("BTCUSDC", "spot", "15m", 0).coverage_ratio == 1.0
    assert store.latest_open_time("BTCUSDC", "spot", "15m") == 60_000
    assert store.insert_snapshot("binance", "btcusdc", "spot", "ticker_24h", {"closeTime": 5}, ts_ms=5) == 1
    assert store.latest_snapshot("BTCUSDC", "spot", "ticker_24h") == {"closeTime": 5}
    assert store.latest_snapshot("BTCUSDC", "spot", "missing") is None
    store.connect().execute(
        "INSERT OR REPLACE INTO market_snapshots VALUES (?, ?, ?, ?, ?, ?)",
        ("binance", "BTCUSDC", "spot", "scalar", 1, "3"),
    )
    store.connect().commit()
    assert store.latest_snapshot("BTCUSDC", "spot", "scalar") is None
    assert store.insert_sync_run({"ok": True}) >= 1
    store.close()
    store.close()
    with MarketDataStore(db) as reopened:
        assert reopened.coverage("BTCUSDC", "spot", "15m").count == 2
    with MarketDataStore(tmp_path / "empty.sqlite") as empty_store:
        empty = empty_store.coverage_quality("BTCUSDC", "spot", "15m", 60_000)
        assert empty.expected_count == 0
        assert empty.coverage_ratio == 0.0


def test_market_data_store_reports_coverage_gaps(tmp_path) -> None:
    with MarketDataStore(tmp_path / "gaps.sqlite") as store:
        store.upsert_candles("BTCUSDC", "spot", "1m", [_candle(0), _candle(180_000)])
        quality = store.coverage_quality("BTCUSDC", "spot", "1m", 60_000)

    assert quality.expected_count == 4
    assert quality.gap_count == 2
    assert quality.coverage_ratio == 0.5


class _SyncClient:
    def __init__(self, *, market_type: str = "spot", fail_snapshot: bool = False, bad_book: bool = False) -> None:
        self.market_type = market_type
        self.fail_snapshot = fail_snapshot
        self.bad_book = bad_book
        self.calls = 0
        self.last_request_info = {"path": "fake"}

    def get_klines(self, _symbol, _interval, *, limit, end_time=None):
        self.calls += 1
        if self.calls == 1:
            return [_candle(60_000), _candle(120_000)][:limit]
        return [_candle(0)][:limit]

    def get_ticker_24h(self, _symbol):
        if self.fail_snapshot:
            from simple_ai_bitcoin_trading_binance.api import BinanceAPIError

            raise BinanceAPIError("ticker down")
        return {"closeTime": 120_000, "priceChangePercent": "1.5"}

    def get_book_ticker(self, _symbol):
        return "bad" if self.bad_book else {"time": 120_001, "bidPrice": "99", "askPrice": "101"}

    def get_futures_premium_index(self, _symbol):
        return {"time": 120_002, "lastFundingRate": "0.0001"}

    def get_futures_open_interest(self, _symbol):
        return {"time": 120_003, "openInterest": "100"}

    def get_futures_funding_rate(self, _symbol, *, limit):
        assert limit == 100
        return [{"fundingTime": 120_004, "fundingRate": "0.0001"}]


def test_sync_market_data_paginates_and_stores_metrics(tmp_path) -> None:
    spot = _SyncClient()
    futures = _SyncClient(market_type="futures")
    result = sync_market_data(
        spot,  # type: ignore[arg-type]
        MarketDataSyncConfig(db_path=tmp_path / "m.sqlite", rows=3, batch_size=2, now_ms=999_999),
        futures_client=futures,  # type: ignore[arg-type]
    )
    assert result.status == "ok"
    assert result.candles_inserted == 3
    assert result.candles_added == 3
    assert result.sync_mode == "backfill"
    assert result.gap_count == 0
    assert result.coverage_ratio == 1.0
    assert result.kline_requests == 2
    assert result.kline_rows_received == 3
    assert result.snapshots_inserted == 5
    assert "candles_available=3" in render_sync_result(result)
    assert "kline_requests=2 kline_rows=3" in render_sync_result(result)


def test_sync_market_data_incremental_mode_skips_duplicate_candle_writes(tmp_path) -> None:
    class IncrementalClient(_SyncClient):
        def __init__(self, *, fail_incremental: bool = False) -> None:
            super().__init__()
            self.fail_incremental = fail_incremental
            self.requests: list[dict[str, int | None]] = []

        def get_klines(self, _symbol, _interval, *, limit, end_time=None, start_time=None):
            self.requests.append({"limit": limit, "end_time": end_time, "start_time": start_time})
            if start_time is not None:
                if self.fail_incremental:
                    from simple_ai_bitcoin_trading_binance.api import BinanceAPIError

                    raise BinanceAPIError("incremental down")
                assert start_time == 180_000
                return []
            return [_candle(0), _candle(60_000), _candle(120_000)][:limit]

    client = IncrementalClient()
    config = MarketDataSyncConfig(
        db_path=tmp_path / "incremental.sqlite",
        interval="1m",
        rows=3,
        batch_size=3,
        include_futures_metrics=False,
        now_ms=999_999,
    )
    first = sync_market_data(client, config)  # type: ignore[arg-type]
    second = sync_market_data(client, config)  # type: ignore[arg-type]

    assert first.sync_mode == "backfill"
    assert first.candles_inserted == 3
    assert first.candles_added == 3
    assert second.status == "ok"
    assert second.sync_mode == "incremental"
    assert second.candles_inserted == 0
    assert second.candles_added == 0
    assert second.coverage_ratio == 1.0
    assert second.kline_requests == 1
    assert second.kline_rows_received == 0
    assert client.requests[-1] == {"limit": 3, "end_time": None, "start_time": 180_000}
    rendered = render_sync_result(second)
    assert "mode=incremental" in rendered
    assert "candles_added=0" in rendered
    assert "coverage_ratio=1.0000 gap_count=0" in rendered

    failing_client = IncrementalClient(fail_incremental=True)
    failing_config = MarketDataSyncConfig(
        db_path=tmp_path / "incremental-error.sqlite",
        interval="1m",
        rows=3,
        batch_size=3,
        include_futures_metrics=False,
    )
    assert sync_market_data(failing_client, failing_config).status == "ok"  # type: ignore[arg-type]
    warned = sync_market_data(failing_client, failing_config)  # type: ignore[arg-type]
    assert warned.status == "warn"
    assert warned.kline_requests == 1
    assert warned.kline_rows_received == 0
    assert any("incremental down" in error for error in warned.errors)


def test_sync_market_data_backfill_counts_closed_rows_not_open_latest_page(tmp_path) -> None:
    class LatestOpenClient(_SyncClient):
        def __init__(self) -> None:
            super().__init__()
            self.requests: list[dict[str, int | None]] = []

        def get_klines(self, _symbol, _interval, *, limit, end_time=None, start_time=None):
            self.requests.append({"limit": limit, "end_time": end_time, "start_time": start_time})
            if end_time is None:
                assert limit == 2
                return [
                    _candle(60_000, close_time=119_000),
                    _candle(120_000, close_time=999_999),
                ]
            assert limit == 1
            return [_candle(0, close_time=59_000)]

    client = LatestOpenClient()
    result = sync_market_data(
        client,  # type: ignore[arg-type]
        MarketDataSyncConfig(
            db_path=tmp_path / "latest-open.sqlite",
            interval="1m",
            rows=2,
            batch_size=2,
            include_futures_metrics=False,
            now_ms=120_000,
        ),
    )

    assert result.candles_inserted == 2
    assert result.candles_available == 2
    assert result.kline_requests == 2
    assert result.kline_rows_received == 3
    assert client.requests == [
        {"limit": 2, "end_time": None, "start_time": None},
        {"limit": 1, "end_time": 59_999, "start_time": None},
    ]


def test_sync_market_data_reports_snapshot_warnings_and_failures(tmp_path) -> None:
    warn = sync_market_data(
        _SyncClient(fail_snapshot=True, bad_book=True),  # type: ignore[arg-type]
        MarketDataSyncConfig(db_path=tmp_path / "warn.sqlite", rows=1, include_futures_metrics=False),
    )
    assert warn.status == "warn"
    assert any("ticker_24h" in error for error in warn.errors)
    assert any("book_ticker" in error for error in warn.errors)

    fail = sync_market_data(
        _SyncClient(),  # type: ignore[arg-type]
        MarketDataSyncConfig(db_path=tmp_path / "fail.sqlite", rows=0, include_futures_metrics=True),
    )
    assert fail.status == "fail"
    assert any("futures client unavailable" in error for error in fail.errors)
    rendered = render_sync_result(fail)
    assert "latest_open_time" not in rendered
    assert "warning:" in rendered


def test_sync_market_data_validates_symbol_interval_and_snapshot_time(tmp_path) -> None:
    from simple_ai_bitcoin_trading_binance.api import BinanceAPIError

    assert _snapshot_time([{"x": 1}], 7) == 7
    assert _snapshot_time({"time": "bad"}, 8) == 8
    assert _snapshot_time({"fundingTime": "9"}, None) == 9
    assert _snapshot_time({"nothing": "9"}, 10) == 10
    with pytest.raises(BinanceAPIError, match="BTCUSDC only"):
        sync_market_data(
            _SyncClient(),  # type: ignore[arg-type]
            MarketDataSyncConfig(symbol="ETHUSDC", db_path=tmp_path / "x.sqlite"),
        )
    with pytest.raises(ValueError, match="not supported"):
        sync_market_data(
            _SyncClient(),  # type: ignore[arg-type]
            MarketDataSyncConfig(interval="1s", market_type="futures", db_path=tmp_path / "x.sqlite"),
        )


def test_sync_market_data_handles_kline_error_empty_and_short_page(tmp_path) -> None:
    from simple_ai_bitcoin_trading_binance.api import BinanceAPIError

    class ErrorClient(_SyncClient):
        def get_klines(self, *_args, **_kwargs):
            raise BinanceAPIError("rate limited")

    errored = sync_market_data(
        ErrorClient(),  # type: ignore[arg-type]
        MarketDataSyncConfig(db_path=tmp_path / "error.sqlite", rows=2, include_futures_metrics=False),
    )
    assert errored.status == "fail"
    assert errored.kline_requests == 1
    assert errored.kline_rows_received == 0
    assert any("klines" in error for error in errored.errors)

    class EmptyClient(_SyncClient):
        def get_klines(self, *_args, **_kwargs):
            return []

    empty = sync_market_data(
        EmptyClient(),  # type: ignore[arg-type]
        MarketDataSyncConfig(db_path=tmp_path / "empty.sqlite", rows=2, include_futures_metrics=False),
    )
    assert empty.candles_inserted == 0
    assert empty.kline_requests == 1
    assert empty.kline_rows_received == 0

    class ShortClient(_SyncClient):
        def get_klines(self, _symbol, _interval, *, limit, end_time=None):
            assert limit == 3
            return [_candle(0)]

    short = sync_market_data(
        ShortClient(),  # type: ignore[arg-type]
        MarketDataSyncConfig(db_path=tmp_path / "short.sqlite", rows=3, batch_size=3, include_futures_metrics=False),
    )
    assert short.candles_inserted == 1
    assert short.kline_requests == 1
    assert short.kline_rows_received == 1
