"""Pure-stdlib training and inference utilities."""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from statistics import mean, pstdev
from typing import Iterable, List, Tuple

from .features import ModelRow


def _clamp(x: float, low: float, high: float) -> float:
    return low if x < low else (high if x > high else x)


@dataclass
class TrainedModel:
    weights: List[float]
    bias: float
    feature_dim: int
    epochs: int
    feature_means: List[float]
    feature_stds: List[float]
    learning_rate: float = 0.05
    l2_penalty: float = 1e-4
    seed: int = 7
    class_weight_pos: float = 1.0
    class_weight_neg: float = 1.0

    def _normalize(self, features: Tuple[float, ...]) -> Tuple[float, ...]:
        if len(features) != self.feature_dim:
            raise ValueError("Feature dimension does not match this model")
        if not self.feature_stds:
            return features
        return tuple(
            (x - mean_) / std_ if std_ != 0 else (x - mean_)
            for x, mean_, std_ in zip(features, self.feature_means, self.feature_stds)
        )

    def predict_proba(self, features: Tuple[float, ...]) -> float:
        score = self.bias
        normed = self._normalize(features)
        for w, x in zip(self.weights, normed):
            score += w * x
        score = max(-50.0, min(50.0, score))
        return 1.0 / (1.0 + math.exp(-score))

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

    columns = list(zip(*[r.features for r in rows_list]))
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


def _safe_division(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _normalize_rows(rows: List[ModelRow], means: List[float], stds: List[float]) -> List[Tuple[float, ...]]:
    return [tuple((x - m) / s for x, m, s in zip(r.features, means, stds)) for r in rows]


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
          seed: int = 7, l2_penalty: float = 1e-4) -> TrainedModel:
    if not rows:
        raise ValueError("No training rows available")
    feature_dim = len(rows[0].features)
    if feature_dim == 0:
        raise ValueError("Rows must contain at least one feature")

    means, stds = _collect_feature_stats(rows)
    normalized = _normalize_rows(rows, means, stds)

    rng = random.Random(seed)
    weights = [rng.uniform(-0.05, 0.05) for _ in range(feature_dim)]
    bias = 0.0
    class_weight_pos, class_weight_neg = _class_weights(rows)
    if class_weight_pos <= 0.0 or class_weight_neg <= 0.0:
        class_weight_pos = 1.0
        class_weight_neg = 1.0

    indices = list(range(len(rows)))
    for _ in range(epochs):
        rng.shuffle(indices)
        for idx in indices:
            row = rows[idx]
            x = normalized[idx]
            y = row.label
            score = bias + sum(w * xi for w, xi in zip(weights, x))
            pred = _sigmoid(score)
            weight = class_weight_pos if y == 1 else class_weight_neg
            error = (pred - y) * weight
            for i, xi in enumerate(x):
                # L2 penalty and signed-gradient update
                grad = error * xi + l2_penalty * weights[i]
                weights[i] -= learning_rate * grad
            bias -= learning_rate * error

    return TrainedModel(
        weights=weights,
        bias=bias,
        feature_dim=feature_dim,
        epochs=epochs,
        feature_means=means,
        feature_stds=stds,
        learning_rate=float(learning_rate),
        l2_penalty=float(l2_penalty),
        seed=int(seed),
        class_weight_pos=float(class_weight_pos),
        class_weight_neg=float(class_weight_neg),
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
) -> dict[str, object]:
    if len(rows) <= train_window + test_window:
        raise ValueError("Not enough rows for walk-forward evaluation")
    if train_window <= 0 or test_window <= 0 or step <= 0:
        raise ValueError("train_window, test_window, and step must be positive")

    scores: List[float] = []
    thresholds: List[float] = []
    for start in range(0, len(rows) - train_window - test_window + 1, step):
        train_rows = rows[start : start + train_window]
        test_rows = rows[start + train_window : start + train_window + test_window]
        model = train(train_rows, epochs=epochs)
        threshold = 0.5
        if calibrate and len(test_rows) >= 10:
            threshold = calibrate_threshold(test_rows, model, start=0.05, end=0.95, steps=31)
        score = evaluate(test_rows, model, threshold=threshold)
        scores.append(score)
        thresholds.append(threshold)

    return {
        "folds": len(scores),
        "average_score": mean(scores) if scores else 0.0,
        "scores": scores,
        "thresholds": thresholds,
        "train_window": train_window,
        "test_window": test_window,
        "step": step,
    }


def serialize_model(model: TrainedModel, path) -> None:
    path.write_text(json.dumps(asdict(model), indent=2), encoding="utf-8")


def load_model(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    dim = int(payload["feature_dim"])
    means = payload.get("feature_means")
    stds = payload.get("feature_stds")
    if not means:
        means = [0.0] * dim
    if not stds:
        stds = [1.0] * dim
    return TrainedModel(
        weights=list(payload["weights"]),
        bias=float(payload["bias"]),
        feature_dim=dim,
        epochs=int(payload["epochs"]),
        feature_means=list(float(x) for x in means),
        feature_stds=list(float(x) for x in stds),
        learning_rate=float(payload.get("learning_rate", 0.05)),
        l2_penalty=float(payload.get("l2_penalty", 1e-4)),
        seed=int(payload.get("seed", 7)),
        class_weight_pos=float(payload.get("class_weight_pos", 1.0)),
        class_weight_neg=float(payload.get("class_weight_neg", 1.0)),
    )
