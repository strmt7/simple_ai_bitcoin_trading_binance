from __future__ import annotations

import math
import pytest
from simple_ai_bitcoin_trading_binance.api import Candle
from simple_ai_bitcoin_trading_binance.features import (
    FEATURE_NAMES,
    feature_signature,
    make_rows,
    make_rows_legacy,
    normalize_enabled_features,
    _safe_div,
    _sma,
    _ema,
    _rsi,
    _true_range,
    _valid_ohlcv,
)
from simple_ai_bitcoin_trading_binance.features import _safe_features


def _fake_candles() -> list[Candle]:
    data = [
        (10000 + i * 2 + (i % 3) * 0.2, 10000 + i * 2 + (i % 3) * 0.2)
        for i in range(120)
    ]
    rows: list[Candle] = []
    for i, (close, o) in enumerate(data):
        rows.append(
            Candle(
                open_time=i * 60000,
                open=o,
                high=o * 1.001,
                low=o * 0.999,
                close=close,
                volume=1.0,
                close_time=i * 60000 + 60000,
            )
        )
    return rows


def test_make_rows_shapes() -> None:
    rows = make_rows(_fake_candles(), short_window=10, long_window=30)
    assert rows
    first = rows[0]
    assert len(first.features) == 13
    assert first.label in (0, 1)


def test_make_rows_respects_enabled_feature_subset() -> None:
    selected = ("momentum_1", "rsi", "volume_ratio")
    rows = make_rows(_fake_candles(), short_window=10, long_window=30, enabled_features=selected)
    assert rows
    assert len(rows[0].features) == 3


def test_normalize_enabled_features_validates_input() -> None:
    assert normalize_enabled_features(["momentum_1", "rsi"]) == ("momentum_1", "rsi")
    assert normalize_enabled_features() == FEATURE_NAMES
    with pytest.raises(ValueError, match="Unknown feature"):
        normalize_enabled_features(["not-real"])
    with pytest.raises(ValueError, match="At least one feature"):
        normalize_enabled_features([])


def test_feature_utilities() -> None:
    assert _safe_div(10.0, 0.0) == 0.0
    assert math.isnan(_sma([1.0, 2.0], 3))
    assert _sma([1.0, 2.0], 2) == 1.5
    assert _ema([1.0, 2.0, 3.0], 3) == 2.25
    assert _rsi([1.0, 1.0], 1) == 100.0
    assert _rsi([1.0], 5) != _rsi([1.0, 2.0, 3.0], 1)


def test_make_rows_returns_empty_without_data() -> None:
    assert make_rows([], short_window=5, long_window=10) == []


def test_make_rows_filters_invalid_candles_and_stable_signature() -> None:
    candles = [
        Candle(
            open_time=0,
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.0,
            volume=1.0,
            close_time=60_000,
        ),
        Candle(
            open_time=60_000,
            open=100.0,
            high=0.0,
            low=99.0,
            close=100.0,
            volume=1.0,
            close_time=120_000,
        ),
    ]
    rows = make_rows(candles, short_window=5, long_window=10)
    assert rows == []
    assert feature_signature(10, 20, 0.001) == "feature_version=v1|feature_count=13|feature_names=momentum_1,momentum_3,momentum_10,momentum_20,ema_spread,rsi,ema_gap,relative_atr,volatility_20,volume_ratio,trend_acceleration,gap_to_vwap,volume_trend|short_window=10|long_window=20|label_threshold=0.001"


def test_make_rows_rejects_invalid_windows() -> None:
    candle = Candle(open_time=0, open=100.0, high=101.0, low=99.0, close=100.0, volume=1.0, close_time=60_000)
    with pytest.raises(ValueError, match="short_window"):
        make_rows([candle], short_window=0, long_window=10)
    with pytest.raises(ValueError, match="long_window"):
        make_rows([candle], short_window=20, long_window=10)


def test_true_range_with_non_positive_prev_close() -> None:
    candles = [
        Candle(
            open_time=0,
            open=10.0,
            high=11.0,
            low=9.0,
            close=0.0,
            volume=1.0,
            close_time=60_000,
        ),
        Candle(
            open_time=60_000,
            open=10.0,
            high=12.0,
            low=8.0,
            close=10.0,
            volume=1.0,
            close_time=120_000,
        ),
    ]
    assert _true_range(candles, 1) == 0.0


def test_feature_edge_helpers_cover_short_and_nonfinite_inputs() -> None:
    assert math.isnan(_ema([1.0], 3))
    assert _safe_features([1.0, float("nan"), float("inf"), 2.0]) == [1.0, 0.0, 0.0, 2.0]


def test_valid_ohlcv_rejects_invalid_shapes() -> None:
    assert not _valid_ohlcv(Candle(0, float("nan"), 101.0, 99.0, 100.0, 1.0, 60_000))
    assert not _valid_ohlcv(Candle(0, 100.0, 101.0, 102.0, 100.5, 1.0, 60_000))
    assert not _valid_ohlcv(Candle(0, -1.0, 101.0, 99.0, 100.0, 1.0, 60_000))
    assert not _valid_ohlcv(Candle(0, 100.0, 101.0, 99.0, 100.0, -1.0, 60_000))
    assert not _valid_ohlcv(Candle(0, 98.0, 101.0, 99.0, 100.0, 1.0, 60_000))
    assert not _valid_ohlcv(Candle(0, 100.0, 101.0, 99.0, 102.0, 1.0, 60_000))
    assert not _valid_ohlcv(Candle(60_000, 100.0, 101.0, 99.0, 100.0, 1.0, 0))


def test_make_rows_sorts_input_and_applies_label_threshold() -> None:
    candles = list(reversed(_fake_candles()))
    rows_low_threshold = make_rows(candles, short_window=10, long_window=30, label_threshold=0.0)
    rows_high_threshold = make_rows(candles, short_window=10, long_window=30, label_threshold=0.5)
    assert rows_low_threshold
    assert rows_low_threshold == sorted(rows_low_threshold, key=lambda row: row.timestamp)
    assert sum(row.label for row in rows_low_threshold) >= sum(row.label for row in rows_high_threshold)


def test_make_rows_legacy_matches_default_feature_shape() -> None:
    candles = _fake_candles()
    legacy_rows = make_rows_legacy(candles, short_window=10, long_window=30)
    default_rows = make_rows(candles, short_window=10, long_window=30, label_threshold=0.001)
    assert len(legacy_rows) == len(default_rows)
    assert legacy_rows[0].features == default_rows[0].features
    assert legacy_rows[0].label == default_rows[0].label
