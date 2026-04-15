from __future__ import annotations

from types import SimpleNamespace
import math

import pytest

from simple_ai_bitcoin_trading_binance.features import make_rows, make_rows_legacy, _safe_div, _sma, _ema, _rsi
from simple_ai_bitcoin_trading_binance.features import _rsi as rsi_fn, _true_range
from simple_ai_bitcoin_trading_binance.model import (
    TrainedModel,
    evaluate,
    evaluate_classification,
    _collect_feature_stats,
    _normalize_rows,
    _sigmoid,
    _f1,
    _confusion,
    _class_weights,
    train,
    calibrate_threshold,
    evaluate_confusion,
    ModelLoadError,
    load_model,
    walk_forward_report,
)
from simple_ai_bitcoin_trading_binance.api import Candle


def test_feature_helpers_cover_edge_cases() -> None:
    assert _safe_div(1.0, 0.0) == 0.0
    assert math.isnan(_sma([1, 2], 3))
    assert _sma([1, 2], 2) == 1.5

    candles = [Candle(0, 1, 2, 1, 1.5, 1, 60), Candle(60_000, 1.5, 2.0, 1.0, 2.0, 1, 120_000)]
    assert _true_range(candles, 1) >= 0.0

    assert rsi_fn([1.0], 2) != rsi_fn([1.0, 2.0, 3.0], 1)


def test_collect_feature_stats_and_normalize_rows() -> None:
    rows = [
        SimpleNamespace(features=(1.0, 2.0), label=1),
        SimpleNamespace(features=(3.0, 4.0), label=0),
        SimpleNamespace(features=(5.0, 6.0), label=1),
    ]
    means, stds = _collect_feature_stats(rows)
    assert means == [3.0, 4.0]
    assert len(stds) == 2
    normalized = _normalize_rows(rows, means, stds)
    assert normalized[0][0] == -1.224744871391589
    assert normalized[1][1] == 0.0

    with pytest.raises(ValueError, match="No rows to collect statistics"):
        _collect_feature_stats([])


def test_normalization_and_training_edge_cases() -> None:
    model = TrainedModel(
        weights=[0.0],
        bias=0.0,
        feature_dim=1,
        epochs=1,
        feature_means=[1.0],
        feature_stds=[1.0],
    )
    with pytest.raises(ValueError, match="Feature dimension"):
        model._normalize((1.0, 2.0))

    rows = [
        SimpleNamespace(features=(1.0,), label=1),
        SimpleNamespace(features=(1.0,), label=1),
        SimpleNamespace(features=(1.0,), label=1),
    ]
    means, stds = _collect_feature_stats(rows)
    assert means == [1.0]
    assert stds == [1.0]

    model2 = TrainedModel(
        weights=[1.0],
        bias=0.0,
        feature_dim=1,
        epochs=1,
        feature_means=means,
        feature_stds=stds,
    )
    pred = model2.predict_proba((1.0,))
    assert 0.0 <= pred <= 1.0

    with pytest.raises(ValueError, match="No training rows"):
        train([])


def test_model_class_weights_and_metrics() -> None:
    rows = [
        SimpleNamespace(features=(1.0,), label=1),
        SimpleNamespace(features=(0.0,), label=0),
        SimpleNamespace(features=(0.0,), label=0),
    ]
    pos, neg = _class_weights(rows)
    assert pos == 2 / 3
    assert neg == 1 / 3
    assert _f1(0, 0, 0) == 0.0

    model = TrainedModel(
        weights=[1.0],
        bias=0.0,
        feature_dim=1,
        epochs=1,
        feature_means=[0.0],
        feature_stds=[1.0],
    )
    conf = _confusion(rows, model, threshold=0.5)
    assert len(conf) == 4
    assert evaluate_confusion(rows, model, threshold=0.5) == conf
    assert _sigmoid(1000.0) >= 0.999999999


def test_evaluate_classification_with_empty_rows():
    rows = []
    model = TrainedModel(weights=[0.0], bias=0.0, feature_dim=1, epochs=1, feature_means=[0.0], feature_stds=[1.0])
    report = evaluate_classification(rows, model, threshold=0.5)
    assert report.accuracy == 0.0
    assert report.true_positive == 0
    assert report.false_positive == 0
    assert report.false_negative == 0
    assert report.true_negative == 0


def test_model_class_weights_handles_all_one_or_zero_labels() -> None:
    rows = [
        SimpleNamespace(features=(1.0,), label=1),
        SimpleNamespace(features=(1.0,), label=1),
    ]
    pos, neg = _class_weights(rows)
    assert pos == 1.0
    assert neg == 1.0

    rows = [
        SimpleNamespace(features=(0.0,), label=0),
        SimpleNamespace(features=(0.0,), label=0),
    ]
    pos, neg = _class_weights(rows)
    assert pos == 1.0
    assert neg == 1.0


def test_model_train_and_calibrate_edges() -> None:
    empty = []
    with pytest.raises(ValueError, match="No training rows"):
        train([], epochs=1)  # type: ignore[arg-type]

    rows = [SimpleNamespace(features=(1.0,), label=0), SimpleNamespace(features=(1.0,), label=0)]
    trained = train(rows, epochs=1, learning_rate=0.01, seed=1)
    assert trained.feature_dim == 1

    calibrated = calibrate_threshold(rows, trained, start=-1.0, end=2.0, steps=3)
    assert 0.0 <= calibrated <= 1.0

    report = walk_forward_report(rows * 80, train_window=2, test_window=2, step=1, epochs=1, calibrate=True)
    assert report["folds"] == len(report["scores"])


def test_load_model_rejects_missing_feature_version(tmp_path) -> None:
    payload = tmp_path / "legacy_model.json"
    payload.write_text(
        """
        {
          "weights": [0.1, 0.2],
          "bias": 0.0,
          "feature_dim": 2,
          "epochs": 5,
          "feature_means": [1.0, 2.0],
          "feature_stds": [1.0, 1.0]
        }
        """.strip(),
        encoding="utf-8",
    )
    with pytest.raises(ModelLoadError, match="missing `feature_version`"):
        load_model(payload)


def test_evaluate_clamp_and_empty_inputs() -> None:
    model = TrainedModel(weights=[0.0], bias=0.0, feature_dim=1, epochs=1, feature_means=[0.0], feature_stds=[1.0])
    assert evaluate([], model) == 0.0
    assert evaluate([SimpleNamespace(features=(0.0,), label=0)], model, threshold=-1.0) == 0.0


def test_calibrate_threshold_handles_short_inputs() -> None:
    rows = [
        SimpleNamespace(features=(1.0,), label=1),
        SimpleNamespace(features=(0.0,), label=0),
    ]
    model = TrainedModel(weights=[0.0], bias=0.0, feature_dim=1, epochs=1, feature_means=[0.0], feature_stds=[1.0])
    assert calibrate_threshold([], model) == 0.5
    assert calibrate_threshold(rows, model, steps=1) == 0.5


def test_walk_forward_report_validates_inputs() -> None:
    rows = [
        SimpleNamespace(features=(1.0, 0.0), label=1),
        SimpleNamespace(features=(0.0, 1.0), label=0),
    ]
    with pytest.raises(ValueError, match="Not enough rows for walk-forward evaluation"):
        walk_forward_report(rows, train_window=5, test_window=5, step=1, epochs=10)

    with pytest.raises(ValueError, match="train_window, test_window, and step must be positive"):
        from simple_ai_bitcoin_trading_binance.features import ModelRow
        rows = [
            ModelRow(timestamp=0, close=1.0, features=(0.0, 0.0), label=0),
            ModelRow(timestamp=1, close=2.0, features=(0.0, 0.0), label=0),
        ] * 200
        walk_forward_report(rows, train_window=10, test_window=10, step=0, epochs=1)
