from __future__ import annotations

import json
import pytest

from simple_ai_bitcoin_trading_binance.features import ModelRow
from pathlib import Path

from simple_ai_bitcoin_trading_binance.model import (
    TrainedModel,
    EnsembleMember,
    ClassificationReport,
    calibrate_probability_temperature,
    confidence_adjusted_probability,
    feature_dimension,
    ModelFeatureMismatchError,
    ModelLoadError,
    load_model,
    model_decision_threshold,
    evaluate_classification,
    evaluate,
    temporal_validation_split,
    train,
    walk_forward_report,
)
from simple_ai_bitcoin_trading_binance.compute import BackendInfo
from simple_ai_bitcoin_trading_binance.strategy_overrides import clean_strategy_overrides


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
    assert model.training_backend_kind == "cpu"
    assert model.training_backend_vendor == "Python stdlib"
    score = evaluate(_rows(), model, threshold=0.5)
    assert 0.0 <= score <= 1.0


def test_train_records_requested_backend_fallback_when_unavailable() -> None:
    model = train(_rows(), epochs=2, compute_backend="directml")
    if model.training_backend_kind == "cpu":
        assert model.training_backend_requested == "directml"
        assert "DirectML" in model.training_backend_reason


def test_train_falls_back_when_resolved_gpu_training_errors(monkeypatch) -> None:
    from simple_ai_bitcoin_trading_binance import model as model_mod

    monkeypatch.setattr(
        model_mod,
        "resolve_backend",
        lambda _backend: BackendInfo("cuda", "cuda", "cuda:0", "Fake GPU", ""),
    )

    def fail_gpu(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(model_mod, "_train_torch", fail_gpu)
    trained = model_mod.train(_rows(), epochs=2, compute_backend="cuda")
    assert trained.training_backend_kind == "cpu"
    assert trained.training_backend_requested == "cuda"
    assert "training failed" in trained.training_backend_reason


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
    assert model.decision_threshold is None
    assert model.calibration_size == 0
    assert model.validation_size == 0
    assert model.strategy_overrides == {}


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
    assert report["calibration_sizes"] == [0, 0, 0]


def test_walk_forward_calibrates_inside_training_window() -> None:
    rows = _rows()
    report = walk_forward_report(rows, train_window=50, test_window=10, step=30, epochs=3, calibrate=True)
    assert report["folds"] == 3
    assert report["calibration_sizes"] == [10, 10, 10]
    assert len(report["thresholds"]) == 3


def test_walk_forward_skips_calibration_when_train_window_too_small() -> None:
    rows = _rows()[:20]
    report = walk_forward_report(rows, train_window=8, test_window=3, step=3, epochs=2, calibrate=True)
    assert report["calibration_sizes"]
    assert all(size == 0 for size in report["calibration_sizes"])


def test_temporal_validation_split_keeps_calibration_out_of_validation() -> None:
    rows = list(reversed(_rows()))
    split = temporal_validation_split(rows, calibration_ratio=0.2, validation_ratio=0.2)
    assert len(split.train_rows) == 72
    assert len(split.calibration_rows) == 24
    assert len(split.validation_rows) == 24
    assert split.train_rows[-1].timestamp < split.calibration_rows[0].timestamp
    assert split.calibration_rows[-1].timestamp < split.validation_rows[0].timestamp


def test_temporal_validation_split_caps_oversized_holdouts() -> None:
    rows = [
        ModelRow(timestamp=2, close=102.0, features=(1.0,), label=1),
        ModelRow(timestamp=0, close=100.0, features=(0.0,), label=0),
        ModelRow(timestamp=1, close=101.0, features=(0.5,), label=1),
    ]

    split = temporal_validation_split(rows, calibration_ratio=1.0, validation_ratio=1.0)

    assert [row.timestamp for row in split.train_rows] == [0]
    assert [row.timestamp for row in split.calibration_rows] == [1]
    assert [row.timestamp for row in split.validation_rows] == [2]


def test_decision_threshold_metadata_and_confidence_adjustment(tmp_path: Path) -> None:
    model = TrainedModel(
        weights=[0.0],
        bias=0.0,
        feature_dim=1,
        epochs=1,
        feature_means=[0.0],
        feature_stds=[1.0],
        decision_threshold=0.65,
        calibration_size=12,
        validation_size=8,
        training_cutoff_timestamp=123,
        probability_temperature=2.0,
        probability_calibration_size=6,
        probability_log_loss_before=0.8,
        probability_log_loss_after=0.7,
        probability_brier_before=0.25,
        probability_brier_after=0.22,
        probability_ece_before=0.20,
        probability_ece_after=0.15,
        threshold_source="profit_backtest",
        threshold_calibration_score=12.3,
        threshold_calibration_pnl=4.5,
        threshold_calibration_trades=6,
        training_backend_requested="directml",
        training_backend_kind="directml",
        training_backend_device="privateuseone:0",
        training_backend_vendor="DirectML",
        training_backend_reason="",
        strategy_overrides={
            "risk_per_trade": 0.005,
            "signal_threshold": 0.63,
            "take_profit_pct": 0.04,
        },
        ensemble_members=[
            EnsembleMember(
                weights=[0.0],
                bias=2.0,
                feature_means=[0.0],
                feature_stds=[1.0],
                seed=7,
                epochs=3,
                training_loss=0.4,
                validation_loss=0.5,
            ),
            EnsembleMember(
                weights=[0.0],
                bias=-2.0,
                feature_means=[0.0],
                feature_stds=[1.0],
                seed=11,
                epochs=3,
            ),
        ],
    )
    from simple_ai_bitcoin_trading_binance.model import serialize_model

    path = tmp_path / "model.json"
    serialize_model(model, path)
    loaded = load_model(path, expected_feature_dim=1)
    assert model_decision_threshold(loaded, 0.55) == 0.65
    assert loaded.calibration_size == 12
    assert loaded.validation_size == 8
    assert loaded.training_cutoff_timestamp == 123
    assert loaded.probability_temperature == 2.0
    assert loaded.probability_calibration_size == 6
    assert loaded.probability_brier_after == 0.22
    assert loaded.threshold_source == "profit_backtest"
    assert loaded.threshold_calibration_score == 12.3
    assert loaded.threshold_calibration_pnl == 4.5
    assert loaded.threshold_calibration_trades == 6
    assert loaded.training_backend_requested == "directml"
    assert loaded.training_backend_kind == "directml"
    assert loaded.training_backend_device == "privateuseone:0"
    assert loaded.training_backend_vendor == "DirectML"
    assert loaded.strategy_overrides == {
        "risk_per_trade": 0.005,
        "signal_threshold": 0.63,
        "take_profit_pct": 0.04,
    }
    assert len(loaded.ensemble_members) == 2
    assert loaded.ensemble_members[0].seed == 7
    assert loaded.ensemble_members[0].training_loss == 0.4
    assert loaded.predict_proba((0.0,)) == pytest.approx(0.5)
    with pytest.raises(ValueError, match="Feature dimension"):
        loaded.predict_proba(())
    assert confidence_adjusted_probability(0.9, 0.5) == 0.7
    assert confidence_adjusted_probability(0.1, 0.5) == 0.3
    assert confidence_adjusted_probability("bad", 0.5) == 0.5
    assert confidence_adjusted_probability(0.8, None) == 0.8
    assert confidence_adjusted_probability(0.8, "bad") == 0.8
    assert confidence_adjusted_probability(float("nan"), 0.5) == 0.5
    assert confidence_adjusted_probability(0.8, float("nan")) == 0.8
    loaded.decision_threshold = float("nan")
    assert model_decision_threshold(loaded, 0.55) == 0.55
    assert model_decision_threshold(loaded, float("nan")) == 0.5
    loaded.decision_threshold = "bad"
    assert model_decision_threshold(loaded, 0.55) == 0.55


def test_load_model_sanitizes_strategy_overrides(tmp_path: Path) -> None:
    model_path = tmp_path / "model.json"
    model_payload = {
        "weights": [0.1],
        "feature_version": "v1",
        "bias": 0.01,
        "feature_dim": 1,
        "epochs": 3,
        "feature_means": [1.0],
        "feature_stds": [1.0],
        "strategy_overrides": {
            "risk_per_trade": 0.004,
            "signal_threshold": "0.7",
            "feature_windows": [1, 2],
            "take_profit_pct": float("nan"),
            "cooldown_minutes": 3,
            "max_trades_per_day": True,
        },
    }
    model_path.write_text(json.dumps(model_payload), encoding="utf-8")

    loaded = load_model(model_path, expected_feature_dim=1)

    assert loaded.strategy_overrides == {
        "risk_per_trade": 0.004,
        "cooldown_minutes": 3,
    }
    assert clean_strategy_overrides(["not", "a", "dict"]) == {}


def test_load_model_rejects_invalid_ensemble_members(tmp_path: Path) -> None:
    model_path = tmp_path / "bad_ensemble.json"
    payload: dict[str, object] = {
        "weights": [0.1],
        "feature_version": "v1",
        "bias": 0.01,
        "feature_dim": 1,
        "epochs": 3,
        "feature_means": [1.0],
        "feature_stds": [1.0],
        "ensemble_members": [
            {
                "weights": [0.1, 0.2],
                "bias": 0.0,
                "feature_means": [0.0],
                "feature_stds": [1.0],
            }
        ],
    }
    model_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ModelFeatureMismatchError):
        load_model(model_path, expected_feature_dim=2)
    with pytest.raises(ModelLoadError, match="ensemble member"):
        load_model(model_path, expected_feature_dim=1)

    for ensemble_members, match in [
        ({}, "must be an array"),
        ([1], "is not an object"),
        (
            [
                {
                    "weights": [0.1],
                    "bias": 0.0,
                    "feature_means": [0.0],
                    "feature_stds": [1.0],
                    "training_loss": "nan",
                }
            ],
            "is invalid",
        ),
    ]:
        payload["ensemble_members"] = ensemble_members
        model_path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ModelLoadError, match=match):
            load_model(model_path, expected_feature_dim=1)


def test_temperature_calibration_softens_overconfident_probabilities() -> None:
    rows = [
        ModelRow(timestamp=1, close=1.0, features=(1.0,), label=0),
        ModelRow(timestamp=2, close=1.0, features=(1.0,), label=0),
        ModelRow(timestamp=3, close=1.0, features=(1.0,), label=1),
        ModelRow(timestamp=4, close=1.0, features=(1.0,), label=1),
    ] * 10
    model = TrainedModel(
        weights=[8.0],
        bias=0.0,
        feature_dim=1,
        epochs=1,
        feature_means=[0.0],
        feature_stds=[1.0],
    )
    before = model.predict_proba((1.0,))
    report = calibrate_probability_temperature(rows, model, min_temperature=1.0, max_temperature=6.0, steps=26)
    model.probability_temperature = report.temperature
    after = model.predict_proba((1.0,))
    assert report.improved is True
    assert report.temperature > 1.0
    assert report.log_loss_after < report.log_loss_before
    assert abs(after - 0.5) < abs(before - 0.5)


def test_temperature_calibration_uses_ensemble_predictions() -> None:
    rows = [
        ModelRow(timestamp=index, close=1.0, features=(1.0,), label=index % 2)
        for index in range(40)
    ]
    model = TrainedModel(
        weights=[0.0],
        bias=0.0,
        feature_dim=1,
        epochs=1,
        feature_means=[0.0],
        feature_stds=[1.0],
        ensemble_members=[
            EnsembleMember(weights=[0.0], bias=8.0, feature_means=[0.0], feature_stds=[1.0]),
            EnsembleMember(weights=[0.0], bias=8.0, feature_means=[0.0], feature_stds=[1.0]),
        ],
    )

    before = model.predict_proba((1.0,))
    report = calibrate_probability_temperature(rows, model, min_temperature=1.0, max_temperature=8.0, steps=29)
    model.probability_temperature = report.temperature
    after = model.predict_proba((1.0,))

    assert report.improved is True
    assert report.temperature > 1.0
    assert report.log_loss_before > 1.0
    assert report.log_loss_after < report.log_loss_before
    assert abs(after - 0.5) < abs(before - 0.5)
