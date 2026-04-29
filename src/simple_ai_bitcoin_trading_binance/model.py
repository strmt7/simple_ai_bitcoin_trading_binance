"""Pure-stdlib training and inference utilities."""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass, field
from statistics import mean, pstdev
from typing import Any, Iterable, List, Sequence, Tuple

from .features import FEATURE_VERSION, ModelRow, feature_dimension as _feature_dimension
from .storage import write_json_atomic


def feature_dimension(enabled_features: Sequence[str] | None = None) -> int:
    return _feature_dimension(enabled_features)


def _clamp(x: float, low: float, high: float) -> float:
    if not math.isfinite(x):
        return low
    return low if x < low else (high if x > high else x)


class ModelLoadError(ValueError):
    """Raised when a model artifact is invalid or incompatible."""


class ModelFeatureMismatchError(ModelLoadError):
    """Raised when feature metadata diverges between model and current runtime."""


@dataclass
class TrainedModel:
    weights: List[float]
    bias: float
    feature_dim: int
    epochs: int
    feature_means: List[float]
    feature_stds: List[float]
    feature_version: str = FEATURE_VERSION
    feature_signature: str | None = None
    learning_rate: float = 0.05
    l2_penalty: float = 1e-4
    seed: int = 7
    class_weight_pos: float = 1.0
    class_weight_neg: float = 1.0
    decision_threshold: float | None = None
    calibration_size: int = 0
    validation_size: int = 0
    training_cutoff_timestamp: int | None = None
    best_epoch: int | None = None
    training_loss: float | None = None
    validation_loss: float | None = None
    quality_score: float | None = None
    quality_warnings: List[str] = field(default_factory=list)
    probability_temperature: float = 1.0
    probability_calibration_size: int = 0
    probability_log_loss_before: float | None = None
    probability_log_loss_after: float | None = None
    probability_brier_before: float | None = None
    probability_brier_after: float | None = None
    probability_ece_before: float | None = None
    probability_ece_after: float | None = None

    def _normalize(self, features: Tuple[float, ...]) -> Tuple[float, ...]:
        if len(features) != self.feature_dim:
            raise ValueError("Feature dimension does not match this model")
        if not self.feature_stds:
            return features
        return tuple(
            (x - mean_) / std_ if std_ != 0 else (x - mean_)
            for x, mean_, std_ in zip(features, self.feature_means, self.feature_stds, strict=True)
        )

    def _linear_score(self, features: Tuple[float, ...]) -> float:
        score = self.bias
        normed = self._normalize(features)
        for w, x in zip(self.weights, normed, strict=True):
            score += w * x
        score = max(-50.0, min(50.0, score))
        return score

    def predict_proba(self, features: Tuple[float, ...]) -> float:
        score = _temperature_scaled_score(self._linear_score(features), self.probability_temperature)
        return _sigmoid(score)

    def predict(self, features: Tuple[float, ...], threshold: float) -> int:
        threshold = _clamp(threshold, 0.0, 1.0)
        return int(self.predict_proba(features) >= threshold)


def _collect_feature_stats(rows: Iterable[ModelRow]) -> tuple[List[float], List[float]]:
    rows_list = list(rows)
    if not rows_list:
        raise ValueError("No rows to collect statistics")
    dim = len(rows_list[0].features)
    means = [0.0] * dim
    stds = [1.0] * dim

    columns = list(zip(*[r.features for r in rows_list], strict=True))
    for i, col in enumerate(columns):
        m = mean(col)
        s = pstdev(col)
        means[i] = float(m)
        stds[i] = float(s if s and abs(s) > 1e-12 else 1.0)
    return means, stds


@dataclass(frozen=True)
class ClassificationReport:
    accuracy: float
    precision: float
    recall: float
    f1: float
    threshold: float
    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int


@dataclass(frozen=True)
class ProbabilityStats:
    minimum: float
    maximum: float
    mean: float
    std: float

    def asdict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class ModelQualityReport:
    quality_score: float
    status: str
    warnings: List[str]
    train_accuracy: float
    validation_accuracy: float
    validation_f1: float
    validation_majority_baseline: float
    train_validation_gap: float
    validation_rows: int
    validation_positive_rate: float
    probability_stats: ProbabilityStats

    def asdict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["probability_stats"] = self.probability_stats.asdict()
        return payload


@dataclass(frozen=True)
class ProbabilityCalibrationReport:
    status: str
    warnings: List[str]
    rows: int
    temperature: float
    log_loss_before: float
    log_loss_after: float
    brier_before: float
    brier_after: float
    expected_calibration_error_before: float
    expected_calibration_error_after: float
    improved: bool

    def asdict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class FeatureDriftReport:
    status: str
    warnings: List[str]
    rows: int
    feature_dim: int
    max_abs_z: float
    mean_abs_z: float
    outlier_fraction: float
    warn_threshold: float
    fail_threshold: float

    def asdict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class TemporalValidationSplit:
    train_rows: List[ModelRow]
    calibration_rows: List[ModelRow]
    validation_rows: List[ModelRow]


def _safe_division(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def validate_model_rows(
    rows: List[ModelRow],
    *,
    label: str = "training rows",
    expected_feature_dim: int | None = None,
) -> int:
    """Validate feature rows before fitting or scoring.

    The training code intentionally accepts light duck-typed row objects in
    tests and integrations, so this guard focuses on the contract actually used
    by the model: fixed finite feature vectors plus binary labels.
    """

    if not rows:
        raise ValueError(f"No {label} available")
    try:
        feature_dim = len(rows[0].features)
    except (AttributeError, TypeError):
        raise ValueError(f"{label} row 0 is missing features") from None
    if feature_dim == 0:
        raise ValueError("Rows must contain at least one feature")
    if expected_feature_dim is not None and feature_dim != expected_feature_dim:
        raise ValueError(
            f"{label} feature dimension mismatch: row 0 has {feature_dim}, expected {expected_feature_dim}"
        )

    for index, row in enumerate(rows):
        try:
            features = tuple(row.features)
        except (AttributeError, TypeError):
            raise ValueError(f"{label} row {index} is missing features") from None
        if len(features) != feature_dim:
            raise ValueError(
                f"{label} row {index} feature dimension mismatch: "
                f"got {len(features)}, expected {feature_dim}"
            )
        for value_index, value in enumerate(features):
            try:
                feature_value = float(value)
            except (TypeError, ValueError):
                raise ValueError(f"{label} row {index} feature {value_index} is not numeric") from None
            if not math.isfinite(feature_value):
                raise ValueError(f"{label} row {index} feature {value_index} is not finite")
        try:
            row_label = int(row.label)
        except (AttributeError, TypeError, ValueError):
            raise ValueError(f"{label} row {index} label is not binary") from None
        if row_label not in {0, 1}:
            raise ValueError(f"{label} row {index} label is not binary")
    return feature_dim


def _normalize_rows(rows: List[ModelRow], means: List[float], stds: List[float]) -> List[Tuple[float, ...]]:
    return [tuple((x - m) / s for x, m, s in zip(r.features, means, stds, strict=True)) for r in rows]


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-50.0, min(50.0, x))))


def _class_weights(rows: List[ModelRow]) -> tuple[float, float]:
    positives = sum(1 for row in rows if row.label == 1)
    negatives = len(rows) - positives
    if positives == 0:
        return 1.0, 1.0
    if negatives == 0:
        return 1.0, 1.0
    total = len(rows)
    pos_weight = float(negatives) / float(total)
    neg_weight = float(positives) / float(total)
    return pos_weight, neg_weight


def _f1(tp: int, fp: int, fn: int) -> float:
    denom = (2 * tp + fp + fn)
    if denom <= 0:
        return 0.0
    return (2.0 * tp) / denom


def _confusion(rows: List[ModelRow], model: TrainedModel, threshold: float) -> tuple[int, int, int, int]:
    tp = fp = tn = fn = 0
    for row in rows:
        pred = model.predict(row.features, threshold)
        label = row.label
        if pred == 1 and label == 1:
            tp += 1
        elif pred == 1 and label == 0:
            fp += 1
        elif pred == 0 and label == 0:
            tn += 1
        else:
            fn += 1
    return tp, fp, tn, fn


def _log_loss(
    rows: List[ModelRow],
    weights: List[float],
    bias: float,
    means: List[float],
    stds: List[float],
) -> float:
    if not rows:
        return 0.0
    total = 0.0
    for row in rows:
        score = bias
        for weight, value, mean_, std_ in zip(weights, row.features, means, stds, strict=True):
            normalized = (value - mean_) / std_ if std_ != 0 else (value - mean_)
            score += weight * normalized
        probability = _clamp(_sigmoid(score), 1e-12, 1.0 - 1e-12)
        if int(row.label) == 1:
            total -= math.log(probability)
        else:
            total -= math.log(1.0 - probability)
    return total / len(rows)


def _coerce_temperature(temperature: Any) -> float:
    try:
        value = float(temperature)
    except (TypeError, ValueError):
        return 1.0
    if not math.isfinite(value) or value <= 1e-6:
        return 1.0
    return value


def _temperature_scaled_score(score: float, temperature: object) -> float:
    value = _coerce_temperature(temperature)
    return max(-50.0, min(50.0, score / value))


def _model_probability(model: TrainedModel, features: Tuple[float, ...], *, temperature: float | None = None) -> float:
    chosen_temperature = model.probability_temperature if temperature is None else temperature
    return _sigmoid(_temperature_scaled_score(model._linear_score(features), chosen_temperature))


def _model_log_loss(rows: List[ModelRow], model: TrainedModel, *, temperature: float | None = None) -> float:
    if not rows:
        return 0.0
    total = 0.0
    for row in rows:
        probability = _clamp(_model_probability(model, row.features, temperature=temperature), 1e-12, 1.0 - 1e-12)
        if int(row.label) == 1:
            total -= math.log(probability)
        else:
            total -= math.log(1.0 - probability)
    return total / len(rows)


def _brier_score(rows: List[ModelRow], model: TrainedModel, *, temperature: float | None = None) -> float:
    if not rows:
        return 0.0
    total = 0.0
    for row in rows:
        probability = _model_probability(model, row.features, temperature=temperature)
        total += (probability - float(int(row.label))) ** 2
    return total / len(rows)


def _expected_calibration_error(
    rows: List[ModelRow],
    model: TrainedModel,
    *,
    temperature: float | None = None,
    bins: int = 5,
) -> float:
    if not rows:
        return 0.0
    bins = max(1, int(bins))
    buckets: list[list[tuple[float, int]]] = [[] for _ in range(bins)]
    for row in rows:
        probability = _clamp(_model_probability(model, row.features, temperature=temperature), 0.0, 1.0)
        index = min(bins - 1, int(probability * bins))
        buckets[index].append((probability, int(row.label)))
    error = 0.0
    for bucket in buckets:
        if not bucket:
            continue
        avg_probability = mean(probability for probability, _label in bucket)
        positive_rate = mean(label for _probability, label in bucket)
        error += (len(bucket) / len(rows)) * abs(avg_probability - positive_rate)
    return float(error)


def assess_probability_calibration(rows: List[ModelRow], model: TrainedModel) -> ProbabilityCalibrationReport:
    """Measure current probability reliability without changing the model."""

    if not rows:
        return ProbabilityCalibrationReport(
            status="fail",
            warnings=["no rows available for probability calibration report"],
            rows=0,
            temperature=_coerce_temperature(getattr(model, "probability_temperature", 1.0)),
            log_loss_before=0.0,
            log_loss_after=0.0,
            brier_before=0.0,
            brier_after=0.0,
            expected_calibration_error_before=0.0,
            expected_calibration_error_after=0.0,
            improved=False,
        )
    validate_model_rows(rows, label="probability calibration rows", expected_feature_dim=model.feature_dim)
    temperature = _coerce_temperature(getattr(model, "probability_temperature", 1.0))
    log_loss_before = _model_log_loss(rows, model, temperature=1.0)
    log_loss_after = _model_log_loss(rows, model, temperature=temperature)
    brier_before = _brier_score(rows, model, temperature=1.0)
    brier_after = _brier_score(rows, model, temperature=temperature)
    ece_before = _expected_calibration_error(rows, model, temperature=1.0)
    ece_after = _expected_calibration_error(rows, model, temperature=temperature)
    improved = (
        abs(temperature - 1.0) > 1e-12
        and (
            log_loss_after < log_loss_before - 1e-6
            or brier_after < brier_before - 1e-6
        )
    )
    warnings: List[str] = []
    if len(rows) < 20:
        warnings.append("probability calibration sample has fewer than 20 rows")
    if rows and _positive_rate(rows) in {0.0, 1.0}:
        warnings.append("probability calibration labels contain only one class")
    return ProbabilityCalibrationReport(
        status="warn" if warnings else "ok",
        warnings=warnings,
        rows=len(rows),
        temperature=temperature,
        log_loss_before=float(log_loss_before),
        log_loss_after=float(log_loss_after),
        brier_before=float(brier_before),
        brier_after=float(brier_after),
        expected_calibration_error_before=float(ece_before),
        expected_calibration_error_after=float(ece_after),
        improved=bool(improved),
    )


def calibrate_probability_temperature(
    rows: List[ModelRow],
    model: TrainedModel,
    *,
    min_temperature: float = 0.5,
    max_temperature: float = 5.0,
    steps: int = 46,
) -> ProbabilityCalibrationReport:
    """Fit a one-parameter temperature calibrator on held-out rows."""

    base = assess_probability_calibration(rows, model)
    if base.status == "fail":
        return base
    warnings = list(base.warnings)
    if _positive_rate(rows) in {0.0, 1.0}:
        return ProbabilityCalibrationReport(
            status="warn",
            warnings=warnings,
            rows=base.rows,
            temperature=base.temperature,
            log_loss_before=base.log_loss_before,
            log_loss_after=base.log_loss_after,
            brier_before=base.brier_before,
            brier_after=base.brier_after,
            expected_calibration_error_before=base.expected_calibration_error_before,
            expected_calibration_error_after=base.expected_calibration_error_after,
            improved=False,
        )

    low = max(0.05, float(min_temperature))
    high = max(low, float(max_temperature))
    steps = max(2, int(steps))
    candidates = [low + (high - low) * i / (steps - 1) for i in range(steps)]
    current = _coerce_temperature(getattr(model, "probability_temperature", 1.0))
    if current not in candidates:
        candidates.append(current)
    best_temperature = current
    best_log_loss = base.log_loss_before
    best_brier = base.brier_before
    best_ece = base.expected_calibration_error_before
    for temperature in candidates:
        log_loss = _model_log_loss(rows, model, temperature=temperature)
        brier = _brier_score(rows, model, temperature=temperature)
        if log_loss < best_log_loss - 1e-12 or (abs(log_loss - best_log_loss) <= 1e-12 and brier < best_brier):
            best_temperature = float(temperature)
            best_log_loss = float(log_loss)
            best_brier = float(brier)
            best_ece = _expected_calibration_error(rows, model, temperature=temperature)

    improved = best_log_loss < base.log_loss_before - 1e-6 or best_brier < base.brier_before - 1e-6
    if not improved:
        warnings.append("temperature calibration did not improve held-out probability loss")
        best_temperature = current
        best_log_loss = base.log_loss_before
        best_brier = base.brier_before
        best_ece = base.expected_calibration_error_before
    return ProbabilityCalibrationReport(
        status="warn" if warnings else "ok",
        warnings=warnings,
        rows=base.rows,
        temperature=float(best_temperature),
        log_loss_before=base.log_loss_before,
        log_loss_after=float(best_log_loss),
        brier_before=base.brier_before,
        brier_after=float(best_brier),
        expected_calibration_error_before=base.expected_calibration_error_before,
        expected_calibration_error_after=float(best_ece),
        improved=bool(improved),
    )


def _positive_rate(rows: List[ModelRow]) -> float:
    if not rows:
        return 0.0
    return sum(1 for row in rows if int(row.label) == 1) / len(rows)


def _probability_stats(rows: List[ModelRow], model: TrainedModel) -> ProbabilityStats:
    if not rows:
        return ProbabilityStats(0.0, 0.0, 0.0, 0.0)
    probabilities = [model.predict_proba(row.features) for row in rows]
    return ProbabilityStats(
        minimum=min(probabilities),
        maximum=max(probabilities),
        mean=mean(probabilities),
        std=pstdev(probabilities) if len(probabilities) > 1 else 0.0,
    )


def _majority_baseline(rows: List[ModelRow]) -> float:
    if not rows:
        return 0.0
    positives = sum(1 for row in rows if int(row.label) == 1)
    negatives = len(rows) - positives
    return max(positives, negatives) / len(rows)


def build_model_quality_report(
    train_rows: List[ModelRow],
    validation_rows: List[ModelRow],
    model: TrainedModel,
    threshold: float,
) -> ModelQualityReport:
    """Summarize whether a fitted model is credible enough for operator review."""

    if train_rows:
        validate_model_rows(train_rows, label="quality train rows", expected_feature_dim=model.feature_dim)
    if validation_rows:
        validate_model_rows(validation_rows, label="quality validation rows", expected_feature_dim=model.feature_dim)

    train_report = evaluate_classification(train_rows, model, threshold=threshold) if train_rows else None
    validation_report = evaluate_classification(validation_rows, model, threshold=threshold)
    baseline = _majority_baseline(validation_rows)
    probability_stats = _probability_stats(validation_rows, model)
    validation_positive_rate = _positive_rate(validation_rows)
    train_accuracy = float(train_report.accuracy) if train_report is not None else 0.0
    validation_accuracy = float(validation_report.accuracy)
    gap = max(0.0, train_accuracy - validation_accuracy)
    warnings: List[str] = []
    penalty = 0.0

    if len(validation_rows) < 20:
        warnings.append("validation sample has fewer than 20 rows")
        penalty += 0.20
    if validation_rows and validation_positive_rate in {0.0, 1.0}:
        warnings.append("validation labels contain only one class")
        penalty += 0.20
    if validation_rows and validation_accuracy + 0.02 < baseline:
        warnings.append("validation accuracy is below majority-class baseline")
        penalty += 0.25
    if gap > 0.25:
        warnings.append("train/validation accuracy gap suggests overfitting")
        penalty += min(0.25, gap - 0.25)
    if validation_rows and probability_stats.std < 0.01:
        warnings.append("validation probabilities are nearly constant")
        penalty += 0.15
    if validation_rows and validation_report.f1 == 0.0 and 0.0 < validation_positive_rate < 1.0:
        warnings.append("validation F1 is zero despite mixed labels")
        penalty += 0.20

    quality_score = _clamp(1.0 - penalty, 0.0, 1.0)
    if quality_score >= 0.75 and not warnings:
        status = "ok"
    elif quality_score >= 0.45:
        status = "warn"
    else:
        status = "fail"
    return ModelQualityReport(
        quality_score=quality_score,
        status=status,
        warnings=warnings,
        train_accuracy=train_accuracy,
        validation_accuracy=validation_accuracy,
        validation_f1=float(validation_report.f1),
        validation_majority_baseline=float(baseline),
        train_validation_gap=float(gap),
        validation_rows=len(validation_rows),
        validation_positive_rate=float(validation_positive_rate),
        probability_stats=probability_stats,
    )


def feature_drift_report(
    rows: List[ModelRow],
    model: TrainedModel,
    *,
    warn_z: float = 4.0,
    fail_z: float = 8.0,
    outlier_warn_fraction: float = 0.10,
    outlier_fail_fraction: float = 0.25,
) -> FeatureDriftReport:
    """Compare feature rows against the model's fitted normalization stats."""

    warn_z = max(0.1, float(warn_z))
    fail_z = max(warn_z, float(fail_z))
    outlier_warn_fraction = _clamp(float(outlier_warn_fraction), 0.0, 1.0)
    outlier_fail_fraction = _clamp(float(outlier_fail_fraction), outlier_warn_fraction, 1.0)
    feature_dim = int(getattr(model, "feature_dim", 0) or 0)
    warnings: List[str] = []
    if not rows:
        return FeatureDriftReport(
            status="fail",
            warnings=["no feature rows available for drift check"],
            rows=0,
            feature_dim=feature_dim,
            max_abs_z=0.0,
            mean_abs_z=0.0,
            outlier_fraction=0.0,
            warn_threshold=warn_z,
            fail_threshold=fail_z,
        )

    validate_model_rows(rows, label="drift rows", expected_feature_dim=feature_dim)
    means = list(getattr(model, "feature_means", []) or [])
    stds = list(getattr(model, "feature_stds", []) or [])
    if len(means) != feature_dim or len(stds) != feature_dim:
        return FeatureDriftReport(
            status="fail",
            warnings=["model feature statistics are incomplete"],
            rows=len(rows),
            feature_dim=feature_dim,
            max_abs_z=0.0,
            mean_abs_z=0.0,
            outlier_fraction=1.0,
            warn_threshold=warn_z,
            fail_threshold=fail_z,
        )

    z_values: List[float] = []
    for row in rows:
        for value, mean_, std_ in zip(row.features, means, stds, strict=True):
            denominator = float(std_) if abs(float(std_)) > 1e-12 else 1.0
            z_values.append(abs((float(value) - float(mean_)) / denominator))
    max_abs_z = max(z_values) if z_values else 0.0
    mean_abs_z = mean(z_values) if z_values else 0.0
    outliers = sum(1 for value in z_values if value >= warn_z)
    outlier_fraction = outliers / len(z_values) if z_values else 0.0

    status = "ok"
    if max_abs_z >= fail_z:
        warnings.append("feature drift exceeds hard threshold")
        status = "fail"
    elif max_abs_z >= warn_z:
        warnings.append("feature drift exceeds warning threshold")
        status = "warn"
    if outliers > 0 and outlier_fraction >= outlier_fail_fraction:
        warnings.append("too many feature values are out-of-distribution")
        status = "fail"
    elif outliers > 0 and outlier_fraction >= outlier_warn_fraction:
        warnings.append("elevated out-of-distribution feature fraction")
        if status != "fail":
            status = "warn"

    return FeatureDriftReport(
        status=status,
        warnings=warnings,
        rows=len(rows),
        feature_dim=feature_dim,
        max_abs_z=float(max_abs_z),
        mean_abs_z=float(mean_abs_z),
        outlier_fraction=float(outlier_fraction),
        warn_threshold=warn_z,
        fail_threshold=fail_z,
    )


def confidence_adjusted_probability(probability: float, beta: float | None) -> float:
    """Shrink a probability toward neutral confidence without changing order."""

    try:
        value = float(probability)
    except (TypeError, ValueError):
        value = 0.5
    if not math.isfinite(value):
        value = 0.5
    value = _clamp(value, 0.0, 1.0)
    if beta is None:
        return value
    try:
        shrink = float(beta)
    except (TypeError, ValueError):
        shrink = 1.0
    if not math.isfinite(shrink):
        shrink = 1.0
    shrink = _clamp(shrink, 0.0, 1.0)
    return 0.5 + (value - 0.5) * shrink


def model_decision_threshold(model: TrainedModel, fallback: float) -> float:
    threshold = getattr(model, "decision_threshold", None)
    if threshold is None:
        threshold = fallback
    try:
        parsed = float(threshold)
    except (TypeError, ValueError):
        parsed = float(fallback)
    if not math.isfinite(parsed):
        parsed = float(fallback)
    if not math.isfinite(parsed):
        parsed = 0.5
    return _clamp(parsed, 0.0, 1.0)


def temporal_validation_split(
    rows: List[ModelRow],
    *,
    calibration_ratio: float = 0.15,
    validation_ratio: float = 0.15,
) -> TemporalValidationSplit:
    """Chronologically split rows so calibration and reported validation differ."""

    ordered = sorted(rows, key=lambda row: row.timestamp)
    total = len(ordered)
    if total < 3:
        return TemporalValidationSplit(ordered, [], [])

    calibration_ratio = _clamp(float(calibration_ratio), 0.0, 0.8)
    validation_ratio = _clamp(float(validation_ratio), 0.0, 0.8)
    calibration_size = max(1, int(total * calibration_ratio)) if calibration_ratio > 0.0 else 0
    validation_size = max(1, int(total * validation_ratio)) if validation_ratio > 0.0 else 0
    while calibration_size + validation_size >= total and (calibration_size > 0 or validation_size > 0):
        if validation_size >= calibration_size and validation_size > 0:
            validation_size -= 1
        else:
            calibration_size -= 1

    train_end = total - calibration_size - validation_size
    calibration_end = total - validation_size
    return TemporalValidationSplit(
        train_rows=ordered[:train_end],
        calibration_rows=ordered[train_end:calibration_end],
        validation_rows=ordered[calibration_end:],
    )


def calibrate_threshold(rows: List[ModelRow], model: TrainedModel, *, start: float = 0.1, end: float = 0.9,
                       steps: int = 17) -> float:
    """Pick a threshold that balances precision and recall for the current model."""
    if not rows:
        return 0.5
    if steps <= 1:
        return _clamp(0.5, 0.0, 1.0)

    best_threshold = 0.5
    best_f1 = -1.0
    if start < 0.0:
        start = 0.0
    if end > 1.0:
        end = 1.0
    if end <= start:
        end = min(1.0, start + 0.01)

    for i in range(steps):
        threshold = start + (end - start) * i / (steps - 1)
        tp, fp, _, fn = _confusion(rows, model, threshold)
        score = _f1(tp, fp, fn)
        if score > best_f1:
            best_f1 = score
            best_threshold = threshold

    return best_threshold


def train(rows: List[ModelRow], *, epochs: int = 200, learning_rate: float = 0.05,
          seed: int = 7, l2_penalty: float = 1e-4,
          feature_signature: str | None = None,
          validation_rows: List[ModelRow] | None = None,
          early_stopping_rounds: int | None = None,
          min_delta: float = 1e-6) -> TrainedModel:
    feature_dim = validate_model_rows(rows)
    validation_rows = list(validation_rows or [])
    if validation_rows:
        validate_model_rows(
            validation_rows,
            label="validation rows",
            expected_feature_dim=feature_dim,
        )

    means, stds = _collect_feature_stats(rows)
    normalized = _normalize_rows(rows, means, stds)

    rng = random.Random(seed)  # nosec B311
    weights = [rng.uniform(-0.05, 0.05) for _ in range(feature_dim)]
    bias = 0.0
    class_weight_pos, class_weight_neg = _class_weights(rows)
    if class_weight_pos <= 0.0 or class_weight_neg <= 0.0:
        class_weight_pos = 1.0
        class_weight_neg = 1.0

    indices = list(range(len(rows)))
    best_weights = list(weights)
    best_bias = bias
    best_epoch: int | None = None
    best_validation_loss = float("inf")
    rounds_without_improvement = 0
    patience = int(early_stopping_rounds or 0)

    for epoch_index in range(1, int(epochs) + 1):
        rng.shuffle(indices)
        for idx in indices:
            row = rows[idx]
            x = normalized[idx]
            y = row.label
            score = bias + sum(w * xi for w, xi in zip(weights, x, strict=True))
            pred = _sigmoid(score)
            weight = class_weight_pos if y == 1 else class_weight_neg
            error = (pred - y) * weight
            for i, xi in enumerate(x):
                # L2 penalty and signed-gradient update
                grad = error * xi + l2_penalty * weights[i]
                weights[i] -= learning_rate * grad
            bias -= learning_rate * error
        if validation_rows:
            current_loss = _log_loss(validation_rows, weights, bias, means, stds)
            if current_loss < best_validation_loss - float(min_delta):
                best_validation_loss = current_loss
                best_weights = list(weights)
                best_bias = bias
                best_epoch = epoch_index
                rounds_without_improvement = 0
            else:
                rounds_without_improvement += 1
                if patience > 0 and rounds_without_improvement >= patience:
                    break

    if validation_rows and best_epoch is not None:
        weights = best_weights
        bias = best_bias

    training_loss = _log_loss(rows, weights, bias, means, stds)
    validation_loss = _log_loss(validation_rows, weights, bias, means, stds) if validation_rows else None

    return TrainedModel(
        weights=weights,
        bias=bias,
        feature_dim=feature_dim,
        epochs=epochs,
        feature_means=means,
        feature_stds=stds,
        feature_signature=feature_signature,
        learning_rate=float(learning_rate),
        l2_penalty=float(l2_penalty),
        seed=int(seed),
        class_weight_pos=float(class_weight_pos),
        class_weight_neg=float(class_weight_neg),
        best_epoch=best_epoch,
        training_loss=float(training_loss),
        validation_loss=float(validation_loss) if validation_loss is not None else None,
    )


def evaluate(rows: List[ModelRow], model: TrainedModel, threshold: float = 0.5) -> float:
    if not rows:
        return 0.0
    correct = 0
    threshold = _clamp(threshold, 0.0, 1.0)
    for row in rows:
        pred = model.predict(row.features, threshold)
        if pred == row.label:
            correct += 1
    return correct / len(rows)


def evaluate_confusion(rows: List[ModelRow], model: TrainedModel, threshold: float = 0.5) -> tuple[int, int, int, int]:
    return _confusion(rows, model, threshold)


def evaluate_classification(
    rows: List[ModelRow],
    model: TrainedModel,
    threshold: float = 0.5,
) -> ClassificationReport:
    if not rows:
        return ClassificationReport(
            accuracy=0.0,
            precision=0.0,
            recall=0.0,
            f1=0.0,
            threshold=_clamp(threshold, 0.0, 1.0),
            true_positive=0,
            false_positive=0,
            true_negative=0,
            false_negative=0,
        )
    tp, fp, tn, fn = _confusion(rows, model, threshold)
    total = tp + fp + tn + fn
    precision = _safe_division(tp, tp + fp)
    recall = _safe_division(tp, tp + fn)
    f1 = _f1(tp, fp, fn)
    accuracy = (tp + tn) / total
    return ClassificationReport(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        threshold=_clamp(threshold, 0.0, 1.0),
        true_positive=tp,
        false_positive=fp,
        true_negative=tn,
        false_negative=fn,
    )


def walk_forward_report(
    rows: List[ModelRow],
    *,
    train_window: int = 300,
    test_window: int = 60,
    step: int = 20,
    epochs: int = 80,
    calibrate: bool = False,
    learning_rate: float = 0.05,
    l2_penalty: float = 1e-4,
) -> dict[str, object]:
    if len(rows) <= train_window + test_window:
        raise ValueError("Not enough rows for walk-forward evaluation")
    if train_window <= 0 or test_window <= 0 or step <= 0:
        raise ValueError("train_window, test_window, and step must be positive")

    scores: List[float] = []
    thresholds: List[float] = []
    calibration_sizes: List[int] = []
    for start in range(0, len(rows) - train_window - test_window + 1, step):
        train_rows = rows[start : start + train_window]
        test_rows = rows[start + train_window : start + train_window + test_window]
        fit_rows = train_rows
        calibration_rows: List[ModelRow] = []
        if calibrate and len(train_rows) >= 10:
            calibration_size = max(1, int(len(train_rows) * 0.2))
            fit_rows = train_rows[:-calibration_size]
            calibration_rows = train_rows[-calibration_size:]
        model = train(fit_rows, epochs=epochs, learning_rate=learning_rate, l2_penalty=l2_penalty)
        threshold = 0.5
        if calibration_rows:
            threshold = calibrate_threshold(calibration_rows, model, start=0.05, end=0.95, steps=31)
        score = evaluate(test_rows, model, threshold=threshold)
        scores.append(score)
        thresholds.append(threshold)
        calibration_sizes.append(len(calibration_rows))

    return {
        "folds": len(scores),
        "average_score": mean(scores) if scores else 0.0,
        "scores": scores,
        "thresholds": thresholds,
        "calibration_sizes": calibration_sizes,
        "train_window": train_window,
        "test_window": test_window,
        "step": step,
        "learning_rate": float(learning_rate),
        "l2_penalty": float(l2_penalty),
    }


def serialize_model(model: TrainedModel, path) -> None:
    write_json_atomic(path, asdict(model), indent=2)


def load_model(
    path,
    *,
    expected_feature_version: str | None = FEATURE_VERSION,
    expected_feature_dim: int | None = None,
    expected_feature_signature: str | None = None,
) -> TrainedModel:
    payload = json.loads(path.read_text(encoding="utf-8"))
    model_version = payload.get("feature_version")
    if not isinstance(model_version, str) or not model_version:
        raise ModelLoadError("Model metadata is missing `feature_version`; please retrain the model")

    if expected_feature_version is not None and model_version != expected_feature_version:
        raise ModelFeatureMismatchError(
            f"Feature version mismatch: model={model_version} runtime={expected_feature_version}"
        )
    payload_signature = payload.get("feature_signature")
    if expected_feature_signature is not None:
        if payload_signature is None:
            raise ModelFeatureMismatchError("Model metadata is missing `feature_signature`; please retrain the model")
        if str(payload_signature) != expected_feature_signature:
            raise ModelFeatureMismatchError(
                f"Feature signature mismatch: model={payload_signature} runtime={expected_feature_signature}"
            )

    dim = int(payload["feature_dim"])
    means = payload.get("feature_means")
    stds = payload.get("feature_stds")
    if means is None:
        raise ModelLoadError("Model payload missing feature_means")
    if stds is None:
        raise ModelLoadError("Model payload missing feature_stds")
    if not isinstance(means, list) or not isinstance(stds, list):
        raise ModelLoadError("Model payload feature stats must be arrays")

    if len(means) != dim:
        raise ModelLoadError("Model payload feature_dim does not match feature_means length")
    if len(stds) != dim:
        raise ModelLoadError("Model payload feature_dim does not match feature_stds length")

    if expected_feature_dim is not None and dim != expected_feature_dim:
        raise ModelFeatureMismatchError(
            f"Feature dimension mismatch: model={dim} expected={expected_feature_dim}"
        )

    weights = payload.get("weights")
    if not isinstance(weights, list):
        raise ModelLoadError("Model payload missing weights")
    if len(weights) != dim:
        raise ModelLoadError("Model payload weights length does not match feature_dim")

    return TrainedModel(
        weights=list(float(w) for w in weights),
        bias=float(payload["bias"]),
        feature_dim=dim,
        epochs=int(payload["epochs"]),
        feature_version=str(model_version),
        feature_signature=str(payload_signature) if payload_signature is not None else None,
        feature_means=list(float(x) for x in means),
        feature_stds=list(float(x) for x in stds),
        learning_rate=float(payload.get("learning_rate", 0.05)),
        l2_penalty=float(payload.get("l2_penalty", 1e-4)),
        seed=int(payload.get("seed", 7)),
        class_weight_pos=float(payload.get("class_weight_pos", 1.0)),
        class_weight_neg=float(payload.get("class_weight_neg", 1.0)),
        decision_threshold=(
            float(payload["decision_threshold"])
            if payload.get("decision_threshold") is not None
            else None
        ),
        calibration_size=int(payload.get("calibration_size", 0)),
        validation_size=int(payload.get("validation_size", 0)),
        training_cutoff_timestamp=(
            int(payload["training_cutoff_timestamp"])
            if payload.get("training_cutoff_timestamp") is not None
            else None
        ),
        best_epoch=(
            int(payload["best_epoch"])
            if payload.get("best_epoch") is not None
            else None
        ),
        training_loss=(
            float(payload["training_loss"])
            if payload.get("training_loss") is not None
            else None
        ),
        validation_loss=(
            float(payload["validation_loss"])
            if payload.get("validation_loss") is not None
            else None
        ),
        quality_score=(
            float(payload["quality_score"])
            if payload.get("quality_score") is not None
            else None
        ),
        quality_warnings=[
            str(value)
            for value in payload.get("quality_warnings", [])
            if isinstance(value, str)
        ],
        probability_temperature=float(payload.get("probability_temperature", 1.0)),
        probability_calibration_size=int(payload.get("probability_calibration_size", 0)),
        probability_log_loss_before=(
            float(payload["probability_log_loss_before"])
            if payload.get("probability_log_loss_before") is not None
            else None
        ),
        probability_log_loss_after=(
            float(payload["probability_log_loss_after"])
            if payload.get("probability_log_loss_after") is not None
            else None
        ),
        probability_brier_before=(
            float(payload["probability_brier_before"])
            if payload.get("probability_brier_before") is not None
            else None
        ),
        probability_brier_after=(
            float(payload["probability_brier_after"])
            if payload.get("probability_brier_after") is not None
            else None
        ),
        probability_ece_before=(
            float(payload["probability_ece_before"])
            if payload.get("probability_ece_before") is not None
            else None
        ),
        probability_ece_after=(
            float(payload["probability_ece_after"])
            if payload.get("probability_ece_after") is not None
            else None
        ),
    )
