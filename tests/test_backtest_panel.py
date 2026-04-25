"""Comprehensive unit tests for the backtest panel module."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Sequence

import pytest

from simple_ai_bitcoin_trading_binance import backtest_panel
from simple_ai_bitcoin_trading_binance.api import Candle
from simple_ai_bitcoin_trading_binance.backtest_panel import (
    BacktestRequest,
    PanelListing,
    _load_model_or_baseline,
    _zero_model,
    build_report_filename,
    describe_supported_intervals,
    estimated_candle_count,
    filter_candles,
    list_reports,
    load_candles_from_json,
    parse_date_ms,
    run_panel,
)
from simple_ai_bitcoin_trading_binance.features import ModelRow
from simple_ai_bitcoin_trading_binance.model import ModelLoadError, TrainedModel
from simple_ai_bitcoin_trading_binance.types import StrategyConfig


# ----- BacktestRequest.validated_interval ----------------------------------


def test_validated_interval_spot_ok() -> None:
    req = BacktestRequest(interval="5m", market_type="spot")
    assert req.validated_interval() == "5m"


def test_validated_interval_futures_ok() -> None:
    req = BacktestRequest(interval="15m", market_type="futures")
    assert req.validated_interval() == "15m"


def test_validated_interval_spot_rejects_bad() -> None:
    req = BacktestRequest(interval="7m", market_type="spot")
    with pytest.raises(ValueError):
        req.validated_interval()


def test_validated_interval_futures_rejects_spot_only() -> None:
    # "1s" is spot only, futures should reject it
    req = BacktestRequest(interval="1s", market_type="futures")
    with pytest.raises(ValueError):
        req.validated_interval()


# ----- parse_date_ms --------------------------------------------------------


def test_parse_date_ms_none_returns_none() -> None:
    assert parse_date_ms(None) is None


def test_parse_date_ms_empty_returns_none() -> None:
    assert parse_date_ms("") is None


def test_parse_date_ms_ymd_start_of_day() -> None:
    ms = parse_date_ms("2026-01-01")
    # 2026-01-01T00:00:00Z
    assert ms == 1767225600000


def test_parse_date_ms_ymd_end_of_day() -> None:
    ms = parse_date_ms("2026-01-01", end_of_day=True)
    # 2026-01-01T23:59:59Z
    assert ms == 1767225600000 + (23 * 3600 + 59 * 60 + 59) * 1000


def test_parse_date_ms_iso_with_seconds() -> None:
    ms = parse_date_ms("2026-01-02T03:04:05")
    assert isinstance(ms, int)
    assert ms > 0


def test_parse_date_ms_iso_without_seconds() -> None:
    ms = parse_date_ms("2026-01-02T03:04")
    assert isinstance(ms, int)


def test_parse_date_ms_invalid_raises() -> None:
    with pytest.raises(ValueError):
        parse_date_ms("not a date")


# ----- filter_candles -------------------------------------------------------


def _make_candles(close_times: list[int]) -> list[Candle]:
    return [
        Candle(
            open_time=t - 1000,
            open=100.0,
            high=105.0,
            low=95.0,
            close=100.0,
            volume=1.0,
            close_time=t,
        )
        for t in close_times
    ]


def test_filter_candles_no_bounds_returns_all() -> None:
    candles = _make_candles([1, 2, 3])
    assert filter_candles(candles, start_ms=None, end_ms=None) == candles


def test_filter_candles_with_bounds_filters() -> None:
    candles = _make_candles([10, 20, 30, 40])
    filtered = filter_candles(candles, start_ms=15, end_ms=35)
    assert [c.close_time for c in filtered] == [20, 30]


# ----- build_report_filename -----------------------------------------------


def test_build_report_filename_with_tag() -> None:
    req = BacktestRequest(interval="5m", market_type="spot", tag="MyTag!")
    name = build_report_filename(req, ts_ms=0)
    assert name.startswith("backtest_MyTag_spot_5m_")
    assert name.endswith(".json")


def test_build_report_filename_falls_back_to_objective() -> None:
    req = BacktestRequest(interval="5m", market_type="spot", tag="", objective="default")
    name = build_report_filename(req, ts_ms=0)
    assert "default" in name


def test_build_report_filename_falls_back_to_untagged() -> None:
    req = BacktestRequest(interval="5m", market_type="spot", tag="", objective=None)
    name = build_report_filename(req, ts_ms=0)
    assert "untagged" in name


def test_build_report_filename_sanitizes_weird_chars() -> None:
    req = BacktestRequest(interval="5m", market_type="spot", tag="!!!@@@$$$")
    name = build_report_filename(req, ts_ms=0)
    # the sanitized tag is empty after stripping, so it falls back to untagged
    assert "untagged" in name


def test_build_report_filename_truncates_long_tags() -> None:
    req = BacktestRequest(
        interval="5m", market_type="spot",
        tag="abcdefghij" * 10,  # 100 chars, must truncate to 40
    )
    name = build_report_filename(req, ts_ms=1_600_000_000_000)
    # tag portion bounded to <= 40 chars
    tag_part = name.split("_")[1]
    assert len(tag_part) <= 40


# ----- load_candles_from_json ----------------------------------------------


def test_load_candles_from_json_happy(tmp_path: Path) -> None:
    path = tmp_path / "candles.json"
    payload = [
        {
            "open_time": 1, "close_time": 60,
            "open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 10.0,
        },
        {
            "open_time": 61, "close_time": 120,
            "open": 1.5, "high": 2.5, "low": 1.0, "close": 2.0, "volume": 20.0,
        },
    ]
    path.write_text(json.dumps(payload), encoding="utf-8")
    candles = load_candles_from_json(str(path))
    assert len(candles) == 2
    assert candles[0].close == 1.5


def test_load_candles_from_json_not_a_list_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"not": "a list"}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_candles_from_json(str(path))


def test_load_candles_from_json_skips_malformed(tmp_path: Path) -> None:
    path = tmp_path / "mixed.json"
    payload = [
        "not a dict",
        {"open_time": 1, "close_time": 60, "open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5},
        # volume defaults; missing high key => skipped
        {"open_time": 2, "close_time": 61, "open": 1.0, "low": 0.5, "close": 1.2},
        # bad float value
        {"open_time": "x", "close_time": 62, "open": 1.0, "high": 2.0, "low": 0.5, "close": 1.3},
    ]
    path.write_text(json.dumps(payload), encoding="utf-8")
    candles = load_candles_from_json(str(path))
    assert len(candles) == 1
    assert candles[0].close == 1.5


# ----- _zero_model ----------------------------------------------------------


def test_zero_model_shape() -> None:
    model = _zero_model(4)
    assert model.feature_dim == 4
    assert model.weights == [0.0] * 4
    assert model.bias == 0.0
    assert model.feature_means == [0.0] * 4
    assert model.feature_stds == [1.0] * 4


# ----- _load_model_or_baseline ---------------------------------------------


def test_load_model_or_baseline_no_path() -> None:
    rows: list[ModelRow] = [
        ModelRow(timestamp=1, close=1.0, features=(0.1, 0.2, 0.3), label=0),
    ]
    model, loaded, resolved = _load_model_or_baseline(None, rows, loader=lambda p: None)
    assert loaded is False
    assert resolved is None
    assert model.feature_dim == 3


def test_load_model_or_baseline_no_path_no_rows() -> None:
    model, loaded, resolved = _load_model_or_baseline(None, [], loader=lambda p: None)
    assert loaded is False
    assert resolved is None
    assert model.feature_dim == 1


def test_load_model_or_baseline_loader_success() -> None:
    expected = _zero_model(2)

    def loader(path: Path) -> TrainedModel:
        return expected

    rows = [ModelRow(timestamp=1, close=1.0, features=(0.0, 0.0), label=0)]
    model, loaded, resolved = _load_model_or_baseline("some/path.json", rows, loader=loader)
    assert model is expected
    assert loaded is True
    assert resolved == "some/path.json"


def test_load_model_or_baseline_file_not_found() -> None:
    def loader(path: Path) -> TrainedModel:
        raise FileNotFoundError(path)

    rows = [ModelRow(timestamp=1, close=1.0, features=(0.1, 0.2), label=0)]
    model, loaded, resolved = _load_model_or_baseline("missing.json", rows, loader=loader)
    assert loaded is False
    assert resolved == "missing.json"
    assert model.feature_dim == 2


def test_load_model_or_baseline_model_load_error_no_rows() -> None:
    def loader(path: Path) -> TrainedModel:
        raise ModelLoadError("bad model")

    model, loaded, resolved = _load_model_or_baseline("bad.json", [], loader=loader)
    assert loaded is False
    assert resolved == "bad.json"
    assert model.feature_dim == 1


# ----- run_panel end-to-end ------------------------------------------------


def _synthetic_candles(n: int = 250, base: float = 100.0) -> list[Candle]:
    # Gentle upward walk so make_rows has enough data.
    candles: list[Candle] = []
    price = base
    for i in range(n):
        open_ = price
        close = price * (1.0 + 0.0005 * math.sin(i / 7.0) + 0.0001)
        high = max(open_, close) * 1.001
        low = min(open_, close) * 0.999
        candles.append(Candle(
            open_time=i * 60_000,
            open=open_,
            high=high,
            low=low,
            close=close,
            volume=1.0,
            close_time=i * 60_000 + 60_000,
        ))
        price = close
    return candles


def test_run_panel_end_to_end_without_objective(tmp_path: Path) -> None:
    req = BacktestRequest(
        interval="5m",
        market_type="spot",
        start_ms=None,
        end_ms=None,
        model_path=None,
        data_path="ignored",
        starting_cash=1000.0,
        objective=None,
        tag="runpanel",
    )
    strat = StrategyConfig()
    candles = _synthetic_candles(n=300)

    def candles_loader(path: str) -> Sequence[Candle]:
        return candles

    def clock():
        return 1_700_000_000.0

    report = run_panel(
        req,
        strat,
        candles_loader=candles_loader,
        model_loader=lambda p: None,
        report_dir=tmp_path / "reports",
        clock=clock,
    )
    assert report.filename.startswith("backtest_runpanel_spot_5m_")
    report_file = (tmp_path / "reports") / report.filename
    assert report_file.exists()
    data = json.loads(report_file.read_text(encoding="utf-8"))
    assert data["tag"] == "runpanel"
    assert data["objective"]["name"] is None
    assert data["objective"]["score"] is None


def test_run_panel_end_to_end_with_objective(tmp_path: Path) -> None:
    req = BacktestRequest(
        interval="5m",
        market_type="spot",
        start_ms=None,
        end_ms=None,
        model_path=None,
        data_path="ignored",
        starting_cash=1000.0,
        objective="default",
        tag="withobj",
    )
    strat = StrategyConfig()

    def candles_loader(path: str) -> Sequence[Candle]:
        return _synthetic_candles(n=300)

    report = run_panel(
        req,
        strat,
        candles_loader=candles_loader,
        model_loader=lambda p: None,
        report_dir=tmp_path / "reports",
        clock=lambda: 1_700_000_000.0,
    )
    report_path = (tmp_path / "reports") / report.filename
    data = json.loads(report_path.read_text(encoding="utf-8"))
    assert data["objective"]["name"] == "default"
    # score is a float, accepted is bool
    assert isinstance(data["objective"]["score"], (int, float))
    assert isinstance(data["objective"]["accepted"], bool)


# ----- list_reports --------------------------------------------------------


def test_list_reports_empty_dir(tmp_path: Path) -> None:
    assert list_reports(tmp_path / "does-not-exist") == []
    # existing but empty dir
    empty = tmp_path / "empty"
    empty.mkdir()
    assert list_reports(empty) == []


def test_list_reports_valid_and_malformed(tmp_path: Path) -> None:
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    valid = {
        "request": {"tag": "foo", "interval": "5m", "market_type": "spot"},
        "finished_at_ms": 1_700_000_000_000,
    }
    (report_dir / "backtest_foo_spot_5m_20260101.json").write_text(
        json.dumps(valid), encoding="utf-8",
    )
    (report_dir / "backtest_bad.json").write_text("not json", encoding="utf-8")
    listings = list_reports(report_dir)
    assert len(listings) == 1
    assert isinstance(listings[0], PanelListing)
    assert listings[0].tag == "foo"
    # asdict coverage
    d = listings[0].asdict()
    assert d["tag"] == "foo"
    assert d["market"] == "spot"


# ----- describe_supported_intervals ----------------------------------------


def test_describe_supported_intervals_spot() -> None:
    text = describe_supported_intervals("spot")
    assert "5m" in text
    assert "1s" in text


def test_describe_supported_intervals_futures() -> None:
    text = describe_supported_intervals("futures")
    assert "5m" in text
    assert "1s" not in text


# ----- estimated_candle_count ----------------------------------------------


def test_estimated_candle_count_open_bounds_returns_zero() -> None:
    req = BacktestRequest(interval="5m", start_ms=None, end_ms=1_000)
    assert estimated_candle_count(req) == 0
    req2 = BacktestRequest(interval="5m", start_ms=1_000, end_ms=None)
    assert estimated_candle_count(req2) == 0


def test_estimated_candle_count_zero_minute_window() -> None:
    req = BacktestRequest(interval="5m", start_ms=1_000, end_ms=1_000)
    assert estimated_candle_count(req) == 0


def test_estimated_candle_count_realistic_window() -> None:
    # 1 hour window at 5m should produce 12 candles
    req = BacktestRequest(interval="5m", start_ms=0, end_ms=3600 * 1000)
    assert estimated_candle_count(req) == 12
