"""Pure-stdlib training and inference utilities."""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from typing import List, Tuple

from .features import ModelRow


@dataclass
class TrainedModel:
    weights: List[float]
    bias: float
    feature_dim: int
    epochs: int

    def predict_proba(self, features: Tuple[float, ...]) -> float:
        score = self.bias
        for w, x in zip(self.weights, features):
            score += w * x
        return 1.0 / (1.0 + math.exp(-score))

    def predict(self, features: Tuple[float, ...], threshold: float) -> int:
        return int(self.predict_proba(features) >= threshold)


def _sigmoid_gradient(pred: float, target: int, features: Tuple[float, ...], lr: float,
                      weights: List[float], bias: float) -> Tuple[List[float], float]:
    error = pred - target
    for i, x in enumerate(features):
        weights[i] -= lr * error * x
    bias -= lr * error
    return weights, bias


def train(rows: List[ModelRow], *, epochs: int = 200, learning_rate: float = 0.05,
          seed: int = 7) -> TrainedModel:
    if not rows:
        raise ValueError("No training rows available")
    feature_dim = len(rows[0].features)
    if feature_dim == 0:
        raise ValueError("Rows must contain at least one feature")

    rng = random.Random(seed)
    weights = [rng.uniform(-0.02, 0.02) for _ in range(feature_dim)]
    bias = 0.0

    for _ in range(epochs):
        for row in rows:
            logit = sum(w * x for w, x in zip(weights, row.features)) + bias
            pred = 1.0 / (1.0 + math.exp(-max(-50.0, min(50.0, logit))))

            # very small form of regularization to avoid runaway confidence in tiny datasets
            weights = [w - 1e-4 * w for w in weights]
            weights, bias = _sigmoid_gradient(pred, row.label, row.features, learning_rate, weights, bias)

    return TrainedModel(weights=weights, bias=bias, feature_dim=feature_dim, epochs=epochs)


def evaluate(rows: List[ModelRow], model: TrainedModel, threshold: float = 0.5) -> float:
    if not rows:
        return 0.0
    correct = 0
    for row in rows:
        pred = model.predict(row.features, threshold)
        if pred == row.label:
            correct += 1
    return correct / len(rows)


def serialize_model(model: TrainedModel, path) -> None:
    payload = asdict(model)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_model(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    return TrainedModel(
        weights=list(payload["weights"]),
        bias=float(payload["bias"]),
        feature_dim=int(payload["feature_dim"]),
        epochs=int(payload["epochs"]),
    )
