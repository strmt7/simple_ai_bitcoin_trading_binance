from __future__ import annotations

from simple_ai_bitcoin_trading_binance.features import ModelRow
from simple_ai_bitcoin_trading_binance.model import TrainedModel, evaluate, train


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
