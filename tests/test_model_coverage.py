from __future__ import annotations

from types import SimpleNamespace
import math

import pytest

from simple_ai_bitcoin_trading_binance.features import make_rows, make_rows_legacy, _safe_div, _sma, _ema, _rsi
from simple_ai_bitcoin_trading_binance.features import _rsi as rsi_fn, _true_range
from simple_ai_bitcoin_trading_binance.model import (
    TrainedModel,
    build_model_quality_report,
    evaluate,
    evaluate_classification,
    _collect_feature_stats,
    _log_loss,
    _majority_baseline,
    _normalize_rows,
    _positive_rate,
    _probability_stats,
    _sigmoid,
    _f1,
    _confusion,
    _class_weights,
    train,
    calibrate_threshold,
    evaluate_confusion,
    ModelLoadError,
    load_model,
    validate_model_rows,
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


def test_validate_model_rows_rejects_malformed_inputs() -> None:
    good = [SimpleNamespace(features=(1.0,), label=1)]
    assert validate_model_rows(good) == 1

    bad_cases = [
        ([SimpleNamespace(label=1)], "missing features"),
        ([SimpleNamespace(features=(), label=1)], "at least one feature"),
        ([SimpleNamespace(features=(1.0,), label=1)], "dimension mismatch"),
        ([SimpleNamespace(features=(1.0,), label=1), SimpleNamespace(label=0)], "missing features"),
        (
            [SimpleNamespace(features=(1.0,), label=1), SimpleNamespace(features=(1.0, 2.0), label=0)],
            "dimension mismatch",
        ),
        ([SimpleNamespace(features=("bad",), label=1)], "not numeric"),
        ([SimpleNamespace(features=(float("inf"),), label=1)], "not finite"),
        ([SimpleNamespace(features=(1.0,))], "label is not binary"),
        ([SimpleNamespace(features=(1.0,), label=2)], "label is not binary"),
    ]
    for rows, message in bad_cases:
        with pytest.raises(ValueError, match=message):
            validate_model_rows(rows, expected_feature_dim=2 if rows is bad_cases[2][0] else None)


def test_probability_and_quality_helpers_cover_edges() -> None:
    model = TrainedModel(
        weights=[0.0],
        bias=0.0,
        feature_dim=1,
        epochs=1,
        feature_means=[0.0],
        feature_stds=[1.0],
    )
    assert _log_loss([], [0.0], 0.0, [0.0], [1.0]) == 0.0
    assert _positive_rate([]) == 0.0
    assert _probability_stats([], model).asdict() == {"minimum": 0.0, "maximum": 0.0, "mean": 0.0, "std": 0.0}
    assert _majority_baseline([]) == 0.0

    validation = [SimpleNamespace(features=(1.0,), label=1)] * 5
    weak = build_model_quality_report([], validation, model, threshold=0.9)
    assert weak.status == "fail"
    assert "validation labels contain only one class" in weak.warnings
    assert weak.asdict()["probability_stats"]["std"] == 0.0

    no_validation = build_model_quality_report(validation, [], model, threshold=0.5)
    assert no_validation.validation_rows == 0

    strong_model = TrainedModel(
        weights=[8.0],
        bias=-4.0,
        feature_dim=1,
        epochs=1,
        feature_means=[0.0],
        feature_stds=[1.0],
    )
    mixed = [
        SimpleNamespace(features=(0.0,), label=0),
        SimpleNamespace(features=(1.0,), label=1),
    ] * 20
    strong = build_model_quality_report(mixed, mixed, strong_model, threshold=0.5)
    assert strong.status == "ok"
    assert strong.quality_score == 1.0

    overfit_train = [SimpleNamespace(features=(1.0,), label=1)] * 30
    mixed_constant = [
        SimpleNamespace(features=(0.0,), label=0),
        SimpleNamespace(features=(0.0,), label=1),
    ] * 5
    overfit = build_model_quality_report(overfit_train, mixed_constant, strong_model, threshold=0.5)
    assert overfit.status == "fail"
    assert any("overfitting" in warning for warning in overfit.warnings)
    assert any("F1 is zero" in warning for warning in overfit.warnings)


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

    early = train(
        rows,
        epochs=5,
        learning_rate=0.0,
        validation_rows=rows,
        early_stopping_rounds=1,
    )
    assert early.best_epoch == 1
    assert early.validation_loss is not None

    no_patience = train(
        rows,
        epochs=2,
        learning_rate=0.0,
        validation_rows=rows,
        early_stopping_rounds=None,
    )
    assert no_patience.best_epoch == 1

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


def test_load_model_rejects_missing_feature_stats_and_bad_lengths(tmp_path) -> None:
    missing_means = tmp_path / "missing_means.json"
    missing_means.write_text(
        """
        {
          "weights": [0.1, 0.2],
          "feature_version": "v1",
          "bias": 0.0,
          "feature_dim": 2,
          "epochs": 5,
          "feature_stds": [1.0, 1.0]
        }
        """.strip(),
        encoding="utf-8",
    )
    with pytest.raises(ModelLoadError, match="missing feature_means"):
        load_model(missing_means)

    bad_means_length = tmp_path / "bad_means_length.json"
    bad_means_length.write_text(
        """
        {
          "weights": [0.1, 0.2],
          "feature_version": "v1",
          "bias": 0.0,
          "feature_dim": 2,
          "epochs": 5,
          "feature_means": [1.0],
          "feature_stds": [1.0, 1.0]
        }
        """.strip(),
        encoding="utf-8",
    )
    with pytest.raises(ModelLoadError, match="feature_dim does not match feature_means"):
        load_model(bad_means_length)

    bad_weights_length = tmp_path / "bad_weights_length.json"
    bad_weights_length.write_text(
        """
        {
          "weights": [0.1],
          "feature_version": "v1",
          "bias": 0.0,
          "feature_dim": 2,
          "epochs": 5,
          "feature_means": [1.0, 1.0],
          "feature_stds": [1.0, 1.0]
        }
        """.strip(),
        encoding="utf-8",
    )
    with pytest.raises(ModelLoadError, match="weights length does not match feature_dim"):
        load_model(bad_weights_length, expected_feature_dim=2)


def test_load_model_rejects_non_array_weights_and_feature_stats(tmp_path) -> None:
    payload = tmp_path / "bad_arrays.json"
    payload.write_text(
        """
        {
          "weights": "bad",
          "feature_version": "v1",
          "bias": 0.0,
          "feature_dim": 2,
          "epochs": 5,
          "feature_means": "bad",
          "feature_stds": [1.0, 1.0]
        }
        """.strip(),
        encoding="utf-8",
    )
    with pytest.raises(ModelLoadError, match="feature stats must be arrays"):
        load_model(payload)

    payload.write_text(
        """
        {
          "weights": "bad",
          "feature_version": "v1",
          "bias": 0.0,
          "feature_dim": 2,
          "epochs": 5,
          "feature_means": [1.0, 1.0],
          "feature_stds": [1.0, 1.0]
        }
        """.strip(),
        encoding="utf-8",
    )
    with pytest.raises(ModelLoadError, match="missing weights"):
        load_model(payload, expected_feature_dim=2)
