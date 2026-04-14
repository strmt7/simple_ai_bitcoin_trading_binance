from __future__ import annotations

from simple_ai_bitcoin_trading_binance.backtest import run_backtest
from simple_ai_bitcoin_trading_binance.features import ModelRow
from simple_ai_bitcoin_trading_binance.model import TrainedModel
from simple_ai_bitcoin_trading_binance.types import StrategyConfig


def test_backtest_runs() -> None:
    rows = [
        ModelRow(timestamp=i, close=100 + i, features=(0.1, 0.0, 0.5, 0.0, 0.0), label=1)
        for i in range(20)
    ]
    model = TrainedModel(weights=[0.0, 0.0, 1.0, 0.0, 0.0], bias=10.0, feature_dim=5, epochs=1)
    cfg = StrategyConfig()
    result = run_backtest(rows, model, cfg, starting_cash=1000.0)
    assert result.trades >= 0
    assert result.starting_cash == 1000.0
