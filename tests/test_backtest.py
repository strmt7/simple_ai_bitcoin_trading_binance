from __future__ import annotations

from types import SimpleNamespace

from simple_ai_bitcoin_trading_binance.backtest import (
    calibrate_threshold_for_backtest,
    risk_adjusted_backtest_score,
    run_backtest,
)
from simple_ai_bitcoin_trading_binance.features import ModelRow
from simple_ai_bitcoin_trading_binance.model import TrainedModel
from simple_ai_bitcoin_trading_binance.types import StrategyConfig


def test_backtest_runs() -> None:
    rows = [
        ModelRow(
            timestamp=i,
            close=100 + i,
            features=(0.1, 0.0, 0.01, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            label=1,
        )
        for i in range(20)
    ]
    model = TrainedModel(
        weights=[0.0] * 13,
        bias=10.0,
        feature_dim=13,
        epochs=1,
        feature_means=[0.0] * 13,
        feature_stds=[1.0] * 13,
    )
    cfg = StrategyConfig()
    result = run_backtest(rows, model, cfg, starting_cash=1000.0)
    assert result.trades >= 0
    assert result.starting_cash == 1000.0
    assert result.buy_hold_pnl > 0.0
    assert result.edge_vs_buy_hold == result.realized_pnl - result.buy_hold_pnl


def test_backtest_tracks_fees_and_cap_hits() -> None:
    rows = [
        ModelRow(
            timestamp=i * 60_000,
            close=1000.0 + (50 if i % 2 == 0 else -20),
            features=(1.0 if i % 2 == 0 else -1.0, *[0.0] * 12),
            label=1 if i % 2 == 0 else 0,
        )
        for i in range(20)
    ]
    # alternating long/short signal, forcing multiple close/open cycles in futures mode
    model = TrainedModel(
        weights=[10.0] + [0.0] * 12,
        bias=0.0,
        feature_dim=13,
        epochs=1,
        feature_means=[0.0] * 13,
        feature_stds=[1.0] * 13,
    )
    cfg = StrategyConfig(
        leverage=5.0,
        risk_per_trade=0.15,
        max_position_pct=0.5,
        signal_threshold=0.6,
        take_profit_pct=0.02,
        stop_loss_pct=0.02,
        max_trades_per_day=1,
        taker_fee_bps=10.0,
    )
    result = run_backtest(rows, model, cfg, starting_cash=10_000.0, market_type="futures")
    assert result.total_fees >= 0.0
    assert result.trades_per_day_cap_hit >= 1
    assert result.gross_exposure >= 0.0


def test_backtest_unlimited_trades_when_disabled() -> None:
    rows = [
        ModelRow(
            timestamp=i * 60_000,
            close=1000.0 + (10 if i % 2 == 0 else -10),
            features=(1.0 if i % 2 == 0 else -1.0, *[0.0] * 12),
            label=1,
        )
        for i in range(30)
    ]
    model = TrainedModel(
        weights=[10.0] + [0.0] * 12,
        bias=0.0,
        feature_dim=13,
        epochs=1,
        feature_means=[0.0] * 13,
        feature_stds=[1.0] * 13,
    )
    cfg = StrategyConfig(
        leverage=2.0,
        risk_per_trade=0.05,
        max_position_pct=0.5,
        signal_threshold=0.6,
        take_profit_pct=0.01,
        stop_loss_pct=0.01,
        max_trades_per_day=0,
    )
    result = run_backtest(rows, model, cfg, starting_cash=5000.0, market_type="futures")
    assert result.trades_per_day_cap_hit == 0


def test_backtest_exit_fee_uses_exit_notional() -> None:
    rows = [
        ModelRow(
            timestamp=0,
            close=100.0,
            features=(10.0, *[0.0] * 12),
            label=1,
        ),
        ModelRow(
            timestamp=60_000,
            close=120.0,
            features=(0.0, *[0.0] * 12),
            label=0,
        ),
    ]
    model = TrainedModel(
        weights=[1.0] + [0.0] * 12,
        bias=0.0,
        feature_dim=13,
        epochs=1,
        feature_means=[0.0] * 13,
        feature_stds=[1.0] * 13,
    )
    cfg = StrategyConfig(
        risk_per_trade=0.1,
        max_position_pct=0.5,
        taker_fee_bps=100.0,
        slippage_bps=0.0,
        signal_threshold=0.55,
        take_profit_pct=0.5,
        stop_loss_pct=0.5,
    )
    result = run_backtest(rows, model, cfg, starting_cash=1000.0, market_type="spot")
    assert result.closed_trades == 1
    assert result.total_fees == 2.2


def test_backtest_uses_model_threshold_and_confidence_shrinkage() -> None:
    rows = [
        ModelRow(timestamp=0, close=100.0, features=(1.0,), label=1),
        ModelRow(timestamp=60_000, close=105.0, features=(1.0,), label=1),
    ]
    model = TrainedModel(
        weights=[0.0],
        bias=0.5,
        feature_dim=1,
        epochs=1,
        feature_means=[0.0],
        feature_stds=[1.0],
        decision_threshold=0.60,
    )
    active = StrategyConfig(signal_threshold=0.90, confidence_beta=1.0, risk_per_trade=0.1)
    conservative = StrategyConfig(signal_threshold=0.90, confidence_beta=0.1, risk_per_trade=0.1)

    assert run_backtest(rows, model, active, starting_cash=1000.0).closed_trades == 1
    assert run_backtest(rows, model, conservative, starting_cash=1000.0).closed_trades == 0


def test_backtest_buy_hold_baseline_handles_invalid_inputs() -> None:
    model = TrainedModel(
        weights=[0.0],
        bias=0.0,
        feature_dim=1,
        epochs=1,
        feature_means=[0.0],
        feature_stds=[1.0],
    )
    cfg = StrategyConfig()
    zero_cash = run_backtest(
        [ModelRow(timestamp=0, close=100.0, features=(0.0,), label=0)],
        model,
        cfg,
        starting_cash=0.0,
    )
    zero_price = run_backtest(
        [
            ModelRow(timestamp=0, close=0.0, features=(0.0,), label=0),
            ModelRow(timestamp=1, close=100.0, features=(0.0,), label=0),
        ],
        model,
        cfg,
        starting_cash=1000.0,
    )
    impossible_exit = run_backtest(
        [
            ModelRow(timestamp=0, close=100.0, features=(0.0,), label=0),
            ModelRow(timestamp=1, close=100.0, features=(0.0,), label=0),
        ],
        model,
        StrategyConfig(slippage_bps=20_000.0),
        starting_cash=1000.0,
    )

    assert zero_cash.buy_hold_pnl == 0.0
    assert zero_price.buy_hold_pnl == 0.0
    assert impossible_exit.buy_hold_pnl == 0.0


def test_profit_threshold_calibration_accepts_profitable_threshold() -> None:
    rows = [
        ModelRow(timestamp=0, close=100.0, features=(0.02,), label=1),
        ModelRow(timestamp=60_000, close=90.0, features=(-0.02,), label=0),
        ModelRow(timestamp=120_000, close=90.0, features=(0.14,), label=1),
        ModelRow(timestamp=180_000, close=105.0, features=(0.14,), label=1),
    ]
    model = TrainedModel(
        weights=[10.0],
        bias=0.0,
        feature_dim=1,
        epochs=1,
        feature_means=[0.0],
        feature_stds=[1.0],
    )
    cfg = StrategyConfig(
        risk_per_trade=0.5,
        max_position_pct=0.5,
        taker_fee_bps=0.0,
        slippage_bps=0.0,
        signal_threshold=0.5,
        take_profit_pct=0.5,
        stop_loss_pct=0.5,
    )

    baseline = run_backtest(
        rows,
        TrainedModel(**{**model.__dict__, "decision_threshold": 0.5}),
        cfg,
        starting_cash=1000.0,
    )
    report = calibrate_threshold_for_backtest(
        rows,
        model,
        cfg,
        baseline_threshold=0.5,
        start=0.45,
        end=0.85,
        steps=9,
        starting_cash=1000.0,
    )

    assert report.accepted is True
    assert report.threshold > 0.5
    assert report.realized_pnl > baseline.realized_pnl
    assert report.baseline_realized_pnl == baseline.realized_pnl
    assert report.asdict()["evaluated_thresholds"] == 9


def test_profit_threshold_calibration_rejects_non_improvement_and_score_edges() -> None:
    rows = [
        ModelRow(timestamp=0, close=100.0, features=(1.0,), label=1),
        ModelRow(timestamp=60_000, close=105.0, features=(1.0,), label=1),
    ]
    model = TrainedModel(
        weights=[0.0],
        bias=2.0,
        feature_dim=1,
        epochs=1,
        feature_means=[0.0],
        feature_stds=[1.0],
    )
    cfg = StrategyConfig(risk_per_trade=0.1, max_position_pct=0.1, signal_threshold=0.5)

    report = calibrate_threshold_for_backtest(
        rows,
        model,
        cfg,
        baseline_threshold=float("nan"),
        start=0.9,
        end=0.1,
        steps=1,
        min_score_delta=float("nan"),
    )
    assert report.accepted is False
    assert report.threshold == 0.5
    assert report.evaluated_thresholds == 1
    nan_grid = calibrate_threshold_for_backtest(
        rows,
        model,
        cfg,
        baseline_threshold=0.5,
        start=float("nan"),
        end=0.1,
        steps=2,
    )
    assert nan_grid.evaluated_thresholds == 2
    reversed_grid = calibrate_threshold_for_backtest(
        rows,
        model,
        cfg,
        baseline_threshold=0.5,
        start=0.9,
        end=0.1,
        steps=2,
    )
    assert reversed_grid.evaluated_thresholds == 3

    stopped = SimpleNamespace(
        realized_pnl=10.0,
        total_fees=1.0,
        max_drawdown=0.5,
        stopped_by_drawdown=True,
        closed_trades=0,
    )
    stable = SimpleNamespace(
        realized_pnl=10.0,
        total_fees=1.0,
        max_drawdown=0.0,
        stopped_by_drawdown=False,
        closed_trades=1,
    )
    assert risk_adjusted_backtest_score(stopped, starting_cash=1000.0) < risk_adjusted_backtest_score(
        stable,
        starting_cash=1000.0,
    )
    assert risk_adjusted_backtest_score(SimpleNamespace(realized_pnl="bad"), starting_cash=float("nan")) < 0.0
    assert risk_adjusted_backtest_score(SimpleNamespace(realized_pnl=object()), starting_cash=1000.0) < 0.0
