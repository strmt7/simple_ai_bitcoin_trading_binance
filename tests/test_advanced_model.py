"""Branch-coverage tests for the advanced model / feature-expansion module."""

from __future__ import annotations

import math

import pytest

from simple_ai_bitcoin_trading_binance import advanced_model as am
from simple_ai_bitcoin_trading_binance.api import Candle
from simple_ai_bitcoin_trading_binance.features import FEATURE_NAMES, ModelRow


def _candles(n: int = 220) -> list[Candle]:
    out = []
    for i in range(n):
        price = 100.0 + (i % 7) * 0.5 + (i * 0.01)
        out.append(Candle(
            open_time=i * 60_000,
            open=price,
            high=price + 0.5,
            low=price - 0.5,
            close=price + 0.1,
            volume=1.0 + (i % 3),
            close_time=i * 60_000 + 59_000,
        ))
    return out


def test_tanh_overflow_branch_positive():
    # math.tanh raises OverflowError for very large inputs on some platforms; the
    # helper must handle it by clamping to ±1.0.
    # We monkeypatch math.tanh via the module to force the overflow branch.
    original = am.math.tanh
    try:
        am.math.tanh = lambda _x: (_ for _ in ()).throw(OverflowError)
        assert am._tanh(1.0) == 1.0
        assert am._tanh(-1.0) == -1.0
    finally:
        am.math.tanh = original


def test_tanh_happy_path():
    assert am._tanh(0.0) == 0.0
    assert -1.0 < am._tanh(0.5) < 1.0


def test_log1p_signed_zero_and_values():
    assert am._log1p_signed(0.0) == 0.0
    assert am._log1p_signed(1.0) > 0.0
    assert am._log1p_signed(-1.0) < 0.0


def test_sma_and_rsi_guards():
    assert math.isnan(am._sma([1.0, 2.0], 0))
    assert math.isnan(am._sma([1.0], 5))
    assert am._sma([1.0, 2.0, 3.0], 3) == 2.0
    assert math.isnan(am._rsi([1.0], 0))
    assert math.isnan(am._rsi([1.0], 5))
    # all losses zero → 100
    assert am._rsi([1.0, 1.0, 1.0, 1.0, 1.0], 3) == 100.0
    # alternating produces a number strictly between 0 and 100
    val = am._rsi([1.0, 2.0, 1.0, 2.0, 1.0], 3)
    assert math.isfinite(val)


def test_volatility_guards():
    assert math.isnan(am._volatility([1.0], 0))
    assert math.isnan(am._volatility([1.0], 5))
    assert am._volatility([1.0, 2.0, 3.0, 4.0], 3) > 0.0


def test_safe_helper_and_extras():
    assert am._safe(float("inf")) == 0.0
    assert am._safe(1.2) == pytest.approx(1.2)
    closes = [100.0 + i for i in range(40)]
    features = am._extra_window_features(closes, (5, 10))
    assert len(features) == 6


def test_nonlinear_expand_unknown_raises():
    with pytest.raises(ValueError):
        am._nonlinear_expand([0.1], ["unknown"])


def test_polynomial_pairs_branches():
    assert am._polynomial_pairs([1.0, 2.0], top_k=1, degree=2) == []
    assert am._polynomial_pairs([1.0, 2.0], top_k=2, degree=1) == []
    pairs = am._polynomial_pairs([1.0, 2.0, 3.0], top_k=3, degree=2)
    # upper triangle: (i*i pairs): 3+2+1 = 6 values
    assert len(pairs) == 6
    cubes = am._polynomial_pairs([1.0, 2.0, 3.0], top_k=3, degree=3)
    assert len(cubes) == 10  # 6 pairs + 1 triple + 3 cubes


def test_advanced_feature_dimension_skips_polynomial_when_disabled():
    cfg = am.AdvancedFeatureConfig(
        base_features=tuple(FEATURE_NAMES[:4]),
        polynomial_degree=1,
        polynomial_top_features=4,
        extra_lookback_windows=(5,),
    )
    dim = am.advanced_feature_dimension(cfg)
    # polynomial disabled → base(4) + extras(3) + transforms(4*2=8) = 15
    assert dim == 15


def test_advanced_feature_dimension_degree3_branch():
    cfg = am.AdvancedFeatureConfig(
        base_features=tuple(FEATURE_NAMES[:6]),
        polynomial_degree=3,
        polynomial_top_features=5,
        extra_lookback_windows=(5,),
    )
    dim = am.advanced_feature_dimension(cfg)
    # base(6) + extras(3) + transforms(6*2=12) + pairs (k=5 → 15) + cube bonus (4) = 40
    assert dim == 6 + 3 + 12 + 15 + 4


def test_advanced_feature_dimension_matches_expand():
    cfg = am.AdvancedFeatureConfig(
        base_features=tuple(FEATURE_NAMES[:6]),
        polynomial_degree=2,
        polynomial_top_features=4,
        extra_lookback_windows=(5, 20),
    )
    dim = am.advanced_feature_dimension(cfg)
    row = ModelRow(timestamp=0, close=100.0, features=(0.1,) * 6, label=1)
    candles = _candles(60)
    expanded = am.expand_row(row, candles, cfg, at_index=59)
    assert len(expanded.features) == dim


def test_advanced_feature_signature_stable():
    cfg = am.default_config_for("default", FEATURE_NAMES)
    assert am.advanced_feature_signature(cfg) == am.advanced_feature_signature(cfg)


def test_default_config_for_branches():
    a = am.default_config_for("conservative", FEATURE_NAMES)
    b = am.default_config_for("risky", FEATURE_NAMES)
    c = am.default_config_for("default", FEATURE_NAMES)
    d = am.default_config_for("nothing", ())
    assert a.polynomial_top_features == 5
    assert b.polynomial_degree == 3
    assert c.polynomial_top_features == 7
    assert d.polynomial_top_features == 7


def test_make_advanced_rows_happy_path():
    cfg = am.default_config_for("default", FEATURE_NAMES)
    rows = am.make_advanced_rows(_candles(250), cfg)
    assert rows
    assert len(rows[0].features) == am.advanced_feature_dimension(cfg)


def test_make_advanced_rows_handles_missing_index(monkeypatch):
    cfg = am.default_config_for("default", FEATURE_NAMES)
    # Force every base row to carry a timestamp not present in index_by_time
    from simple_ai_bitcoin_trading_binance import advanced_model as mod

    def fake_base(*_args, **_kwargs):
        return [ModelRow(timestamp=-999, close=100.0, features=(0.0,) * len(FEATURE_NAMES), label=0)]

    monkeypatch.setattr(mod, "make_base_rows", fake_base)
    rows = am.make_advanced_rows(_candles(60), cfg)
    assert rows == []


def test_make_advanced_rows_empty_input():
    cfg = am.default_config_for("default", FEATURE_NAMES)
    assert am.make_advanced_rows([], cfg) == []


def test_filter_valid_rejects_bad_candles():
    bad = [
        Candle(open_time=0, open=float("nan"), high=1, low=1, close=1, volume=1, close_time=1),
        Candle(open_time=0, open=-1, high=1, low=1, close=1, volume=1, close_time=1),
        Candle(open_time=0, open=1, high=1, low=2, close=1, volume=1, close_time=1),  # low>high
        Candle(open_time=0, open=5, high=2, low=1, close=1, volume=1, close_time=1),  # open out of band
        Candle(open_time=0, open=1, high=2, low=1, close=3, volume=1, close_time=1),  # close out of band
        Candle(open_time=10, open=1, high=2, low=1, close=1, volume=1, close_time=5),  # close_time<open_time
        Candle(open_time=0, open=1, high=2, low=1, close=1, volume=-1, close_time=1),  # negative volume
        Candle(open_time=-1, open=1, high=2, low=1, close=1, volume=1, close_time=1),
    ]
    assert am._filter_valid(bad) == []


def test_train_advanced_empty_rows():
    cfg = am.default_config_for("default", FEATURE_NAMES)
    with pytest.raises(ValueError):
        am.train_advanced([], cfg, epochs=10, learning_rate=0.05, l2_penalty=1e-3)


def test_train_advanced_produces_matching_dim():
    cfg = am.default_config_for("default", FEATURE_NAMES)
    rows = am.make_advanced_rows(_candles(260), cfg)
    assert rows
    model, report = am.train_advanced(rows, cfg, epochs=3, learning_rate=0.05, l2_penalty=1e-3)
    assert model.feature_dim == len(rows[0].features)
    assert report.row_count == len(rows)
    assert 0.0 <= report.positive_rate <= 1.0
    assert report.feature_signature == am.advanced_feature_signature(cfg)
