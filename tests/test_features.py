from __future__ import annotations

import math
from simple_ai_bitcoin_trading_binance.api import Candle
from simple_ai_bitcoin_trading_binance.features import make_rows, _safe_div, _sma, _ema, _rsi, _true_range
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


def test_feature_utilities() -> None:
    assert _safe_div(10.0, 0.0) == 0.0
    assert math.isnan(_sma([1.0, 2.0], 3))
    assert _sma([1.0, 2.0], 2) == 1.5
    assert _ema([1.0, 2.0, 3.0], 3) == 2.25
    assert _rsi([1.0, 1.0], 1) == 100.0
    assert _rsi([1.0], 5) != _rsi([1.0, 2.0, 3.0], 1)


def test_make_rows_returns_empty_without_data() -> None:
    assert make_rows([], short_window=5, long_window=10) == []


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
