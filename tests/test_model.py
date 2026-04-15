from __future__ import annotations

from simple_ai_bitcoin_trading_binance.features import ModelRow
from pathlib import Path

from simple_ai_bitcoin_trading_binance.model import (
    TrainedModel,
    ClassificationReport,
    evaluate_classification,
    evaluate,
    load_model,
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
