from __future__ import annotations



from simple_ai_bitcoin_trading_binance import chart


_SPARK_RAMP = "▁▂▃▄▅▆▇█"


def test_sparkline_empty_returns_empty():
    assert chart.sparkline([]) == ""


def test_sparkline_all_non_finite_returns_empty():
    # NaN and inf are non-finite, and strings are filtered out by _finite
    assert chart.sparkline([float("nan"), float("inf"), float("-inf")]) == ""
    assert chart.sparkline(["not a number"]) == ""  # type: ignore[list-item]


def test_sparkline_all_equal_collapses_to_mid_glyph():
    mid = _SPARK_RAMP[len(_SPARK_RAMP) // 2]
    out = chart.sparkline([3.0, 3.0, 3.0, 3.0])
    assert out == mid * 4


def test_sparkline_ascending_range_first_lowest_last_highest():
    out = chart.sparkline([1, 2, 3, 4, 5, 6, 7, 8])
    assert out[0] == _SPARK_RAMP[0]
    assert out[-1] == _SPARK_RAMP[-1]


def test_sparkline_negative_and_positive_mix():
    out = chart.sparkline([-5, 0, 5])
    # first is min -> lowest, last is max -> highest
    assert out[0] == _SPARK_RAMP[0]
    assert out[-1] == _SPARK_RAMP[-1]
    assert len(out) == 3


def test_equity_curve_empty_returns_no_data():
    assert chart.equity_curve([]) == ["(no data)"]


def test_equity_curve_width_leq_zero_clamped():
    # width is clamped to 4 minimum so output still rendered
    rows = chart.equity_curve([1.0, 2.0, 3.0], width=0, height=4)
    assert len(rows) == 4
    for row in rows:
        assert len(row) == 4


def test_equity_curve_height_leq_zero_clamped():
    rows = chart.equity_curve([1.0, 2.0, 3.0], width=4, height=-1)
    assert len(rows) == 4


def test_equity_curve_baseline_none():
    rows = chart.equity_curve([1.0, 2.0, 3.0], width=6, height=4, baseline=None)
    assert len(rows) == 4
    joined = "\n".join(rows)
    assert "*" in joined
    assert "·" not in joined


def test_equity_curve_baseline_inside_range():
    rows = chart.equity_curve([1.0, 2.0, 3.0], width=6, height=5, baseline=2.0)
    joined = "\n".join(rows)
    assert "·" in joined


def test_equity_curve_baseline_below_range():
    rows = chart.equity_curve([10.0, 11.0, 12.0], width=6, height=5, baseline=0.0)
    # baseline is at low -> should appear on the bottom row
    joined = "\n".join(rows)
    assert "·" in joined


def test_equity_curve_baseline_above_range():
    rows = chart.equity_curve([10.0, 11.0, 12.0], width=6, height=5, baseline=100.0)
    # baseline at top -> should draw baseline at top row
    joined = "\n".join(rows)
    assert "·" in joined


def test_equity_curve_decimation_when_values_exceed_width():
    values = list(range(100))
    rows = chart.equity_curve(values, width=10, height=5)
    assert len(rows) == 5
    for row in rows:
        assert len(row) == 10
    # should have plotted values
    assert "*" in "\n".join(rows)


def test_equity_curve_all_equal_values_without_baseline():
    # exercises the `span <= 0` fallback inside equity_curve itself
    rows = chart.equity_curve([5.0, 5.0, 5.0, 5.0], width=6, height=4, baseline=None)
    assert len(rows) == 4
    assert "*" in "\n".join(rows)


def test_resample_width_zero_returns_empty_list():
    # directly drive the private helper to hit the ``width <= 0`` branch which
    # public callers clamp away.
    assert chart._resample([1.0, 2.0, 3.0], 0) == []


def test_resample_multiple_full_buckets():
    # With len > width, every bucket is guaranteed to span at least one
    # sample, exercising the normal path of the decimation loop where
    # ``end > start`` (the else-side of the defensive ``end <= start`` guard).
    out = chart._resample(list(range(20)), 5)
    assert len(out) == 5
    # values should be monotonically non-decreasing since inputs are ascending
    assert out == sorted(out)




def test_mini_candles_out_of_band_values_trigger_bounds_guard():
    # An inverted candle (high < low) paired with a normal one pushes the wick
    # iteration past the grid -- exercises the defensive y-bounds `continue`.
    ok = chart.MiniCandle(open=10.0, high=11.0, low=9.0, close=10.5)
    weird = chart.MiniCandle(open=5.0, high=-100.0, low=100.0, close=5.0)
    rows = chart.mini_candles([ok, weird], width=4, height=4)
    assert len(rows) == 4


def test_mini_candles_empty_returns_sentinel():
    assert chart.mini_candles([]) == ["(no candles)"]


def test_mini_candles_single_candle():
    c = chart.MiniCandle(open=10.0, high=11.0, low=9.0, close=10.5)
    rows = chart.mini_candles([c], width=4, height=4)
    assert len(rows) == 4
    # bullish body
    joined = "\n".join(rows)
    assert "█" in joined


def test_mini_candles_bullish_and_bearish_mix():
    bull = chart.MiniCandle(open=10.0, high=11.0, low=9.5, close=10.8)
    bear = chart.MiniCandle(open=10.0, high=10.5, low=9.0, close=9.2)
    rows = chart.mini_candles([bull, bear], width=4, height=6)
    joined = "\n".join(rows)
    assert "█" in joined
    assert "▓" in joined


def test_mini_candles_span_zero_all_equal_ohlc():
    flat = chart.MiniCandle(open=5.0, high=5.0, low=5.0, close=5.0)
    rows = chart.mini_candles([flat, flat, flat], width=4, height=4)
    assert len(rows) == 4


def test_mini_candles_output_dimensions_respected():
    candles = [chart.MiniCandle(open=i, high=i + 1, low=i - 1, close=i + 0.5)
               for i in range(1, 7)]
    rows = chart.mini_candles(candles, width=6, height=5)
    assert len(rows) == 5
    for row in rows:
        # each column is one candle (char)
        assert len(row) == 6


def test_format_equity_footer_positive_pnl():
    out = chart.format_equity_footer(1000.0, 1250.5, 250.5, 0.6)
    assert "start 1,000.00" in out
    assert "end 1,250.50" in out
    assert "pnl +250.50" in out
    assert "win 60%" in out


def test_format_equity_footer_negative_pnl():
    out = chart.format_equity_footer(1000.0, 900.0, -100.0, 0.25)
    assert "pnl -100.00" in out
    assert "win 25%" in out
