from __future__ import annotations

from simple_ai_bitcoin_trading_binance.api import Candle
from simple_ai_bitcoin_trading_binance.features import make_rows


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
