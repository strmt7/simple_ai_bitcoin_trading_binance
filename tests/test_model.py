from __future__ import annotations

import json
import pytest

from simple_ai_bitcoin_trading_binance.features import ModelRow
from pathlib import Path

from simple_ai_bitcoin_trading_binance.model import (
    TrainedModel,
    ClassificationReport,
    feature_dimension,
    ModelFeatureMismatchError,
    load_model,
    evaluate_classification,
    evaluate,
    train,
    walk_forward_report,
)


def _rows() -> list[ModelRow]:
    out: list[ModelRow] = []
    for i in range(120):
        features = (float(i), float(i * 0.1), 0.5, float(i % 2), 0.01, float(i) / 10.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
        label = 1 if i % 2 == 0 else 0
        out.append(ModelRow(timestamp=1000 + i, close=20000.0 + i, features=features, label=label))
    return out


def test_train_and_evaluate() -> None:
    model = train(_rows(), epochs=5)
    assert isinstance(model, TrainedModel)
    assert model.feature_dim == 13
    score = evaluate(_rows(), model, threshold=0.5)
    assert 0.0 <= score <= 1.0


def test_load_model_backwards_compatibility(tmp_path: Path) -> None:
    model_path = tmp_path / "legacy_model.json"
    model_path.write_text(
        """
        {
          "weights": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3],
          "feature_version": "v1",
          "bias": 0.01,
          "feature_dim": 13,
          "epochs": 10,
          "feature_means": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
          "feature_stds": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        }
        """.strip(),
        encoding="utf-8",
    )
    model = load_model(model_path)
    assert isinstance(model, TrainedModel)
    assert model.learning_rate == 0.05
    assert model.l2_penalty == 1e-4
    assert model.class_weight_pos == 1.0
    assert model.class_weight_neg == 1.0


def test_load_model_rejects_mismatched_version(tmp_path: Path) -> None:
    model_path = tmp_path / "bad_model.json"
    model_path.write_text(
        """
        {
          "weights": [0.1, 0.2, 0.3],
          "feature_version": "v0",
          "bias": 0.01,
          "feature_dim": 3,
          "epochs": 10,
          "feature_means": [1.0, 1.0, 1.0],
          "feature_stds": [1.0, 1.0, 1.0]
        }
        """.strip(),
        encoding="utf-8",
    )
    with pytest.raises(ModelFeatureMismatchError, match="Feature version mismatch"):
        load_model(model_path)


def test_load_model_rejects_signature_mismatch(tmp_path: Path) -> None:
    model_path = tmp_path / "sig_mismatch.json"
    model_payload = {
        "weights": [0.1] * feature_dimension(),
        "feature_version": "v1",
        "bias": 0.01,
        "feature_dim": feature_dimension(),
        "epochs": 3,
        "feature_means": [1.0] * feature_dimension(),
        "feature_stds": [1.0] * feature_dimension(),
        "feature_signature": "feature_version=v1|feature_count=13|feature_names=momentum_1,momentum_3,momentum_10,momentum_20,ema_spread,rsi,ema_gap,relative_atr,volatility_20,volume_ratio,trend_acceleration,gap_to_vwap,volume_trend|short_window=6|long_window=24|label_threshold=0.001",
    }
    model_path.write_text(json.dumps(model_payload), encoding="utf-8")
    with pytest.raises(ModelFeatureMismatchError, match="Feature signature mismatch"):
        load_model(
            model_path,
            expected_feature_signature="feature_version=v1|feature_count=13|feature_names=momentum_1,momentum_3,momentum_10,momentum_20,ema_spread,rsi,ema_gap,relative_atr,volatility_20,volume_ratio,trend_acceleration,gap_to_vwap,volume_trend|short_window=4|long_window=8|label_threshold=0.001",
        )


def test_load_model_rejects_missing_signature_when_expected(tmp_path: Path) -> None:
    model_path = tmp_path / "missing_signature.json"
    model_payload = {
        "weights": [0.1] * feature_dimension(),
        "feature_version": "v1",
        "bias": 0.01,
        "feature_dim": feature_dimension(),
        "epochs": 3,
        "feature_means": [1.0] * feature_dimension(),
        "feature_stds": [1.0] * feature_dimension(),
    }
    model_path.write_text(json.dumps(model_payload), encoding="utf-8")
    with pytest.raises(ModelFeatureMismatchError, match="missing `feature_signature`"):
        load_model(model_path, expected_feature_signature="runtime-signature")


def test_load_model_allows_subset_feature_dim_when_signature_matches(tmp_path: Path) -> None:
    model_path = tmp_path / "subset_model.json"
    model_payload = {
        "weights": [0.1, 0.2, 0.3],
        "feature_version": "v1",
        "bias": 0.01,
        "feature_dim": 3,
        "epochs": 3,
        "feature_means": [1.0, 1.0, 1.0],
        "feature_stds": [1.0, 1.0, 1.0],
        "feature_signature": "feature_version=v1|feature_count=3|feature_names=momentum_1,rsi,volume_ratio|short_window=10|long_window=40|label_threshold=0.001",
    }
    model_path.write_text(json.dumps(model_payload), encoding="utf-8")
    model = load_model(
        model_path,
        expected_feature_signature="feature_version=v1|feature_count=3|feature_names=momentum_1,rsi,volume_ratio|short_window=10|long_window=40|label_threshold=0.001",
        expected_feature_dim=None,
    )
    assert model.feature_dim == 3


def test_evaluate_classification_report() -> None:
    rows = [
        ModelRow(timestamp=0, close=100.0, features=(1.0, 0.0), label=1),
        ModelRow(timestamp=1, close=101.0, features=(0.0, 0.0), label=0),
        ModelRow(timestamp=2, close=102.0, features=(1.0, 0.0), label=1),
    ]
    model = TrainedModel(
        weights=[1.0, 1.0],
        bias=-0.1,
        feature_dim=2,
        epochs=1,
        feature_means=[0.0, 0.0],
        feature_stds=[1.0, 1.0],
    )
    report = evaluate_classification(rows, model, threshold=0.5)
    assert isinstance(report, ClassificationReport)
    assert report.true_positive + report.false_positive + report.true_negative + report.false_negative == len(rows)
    assert 0.0 <= report.accuracy <= 1.0


def test_walk_forward_report_runs() -> None:
    rows = _rows()
    report = walk_forward_report(rows, train_window=60, test_window=20, step=20, epochs=5, calibrate=False)
    assert report["folds"] == 3
    assert report["train_window"] == 60
    assert report["test_window"] == 20
    assert report["step"] == 20
    assert report["average_score"] >= 0.0
