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


def _normalize_rows(rows: List[ModelRow], means: List[float], stds: List[float]) -> List[Tuple[float, ...]]:
    return [tuple((x - m) / s for x, m, s in zip(r.features, means, stds)) for r in rows]


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-50.0, min(50.0, x))))


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

    indices = list(range(len(rows)))
    for _ in range(epochs):
        rng.shuffle(indices)
        for idx in indices:
            row = rows[idx]
            x = normalized[idx]
            y = row.label
            score = bias + sum(w * xi for w, xi in zip(weights, x))
            pred = _sigmoid(score)
            error = pred - y
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
    )

