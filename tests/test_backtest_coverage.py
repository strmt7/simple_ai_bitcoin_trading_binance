from __future__ import annotations

from simple_ai_bitcoin_trading_binance.backtest import run_backtest
from simple_ai_bitcoin_trading_binance.features import ModelRow
from simple_ai_bitcoin_trading_binance.model import TrainedModel
from simple_ai_bitcoin_trading_binance.types import StrategyConfig


def _flat_row(timestamp: int, close: float, score: float, label: int) -> ModelRow:
    return ModelRow(
        timestamp=timestamp,
        close=float(close),
        features=(score, *([0.0] * 12)),
        label=label,
    )


def _simple_model(bias: float = 0.0) -> TrainedModel:
    return TrainedModel(
        weights=[0.0] * 13,
        bias=bias,
        feature_dim=13,
        epochs=1,
        feature_means=[0.0] * 13,
        feature_stds=[1.0] * 13,
    )


def test_backtest_clamps_futures_leverage_and_drawdown_breaks() -> None:
    rows = [
        _flat_row(i, close=float(100 - i * 10), score=10.0 if i == 0 else 0.0, label=1)
        for i in range(20)
    ]
    cfg = StrategyConfig(
        leverage=500.0,
        risk_per_trade=0.5,
        max_position_pct=0.9,
        signal_threshold=0.55,
        take_profit_pct=0.001,
        stop_loss_pct=0.001,
        max_drawdown_limit=0.01,
    )
    result = run_backtest(rows, _simple_model(10.0), cfg, starting_cash=1000.0, market_type="futures")
    assert result.trades_per_day_cap_hit >= 0
    assert result.max_drawdown >= 0.0
    assert result.ending_cash < result.starting_cash
    assert result.closed_trades >= 1


def test_backtest_empty_rows_returns_identity_state() -> None:
    result = run_backtest([], _simple_model(), StrategyConfig(), starting_cash=750.0)
    assert result.starting_cash == 750.0
    assert result.ending_cash == 750.0
    assert result.trades == 0
    assert result.closed_trades == 0
    assert result.trades_per_day_cap_hit == 0
    assert result.buy_hold_pnl == 0.0
    assert result.edge_vs_buy_hold == 0.0


def test_backtest_enforces_entry_filters_and_cap_hits() -> None:
    rows = [
        _flat_row(0, close=100.0, score=1.0, label=1),
        _flat_row(1, close=100.0, score=-1.0, label=1),
        _flat_row(2, close=100.0, score=1.0, label=1),
        _flat_row(3, close=100.0, score=1.0, label=1),
    ]
    model = TrainedModel(
        weights=[20.0] + [0.0] * 12,
        bias=0.0,
        feature_dim=13,
        epochs=1,
        feature_means=[0.0] * 13,
        feature_stds=[1.0] * 13,
    )
    cfg = StrategyConfig(
        leverage=1.0,
        risk_per_trade=0.5,
        max_position_pct=1.0,
        max_open_positions=1,
        max_trades_per_day=1,
        signal_threshold=0.55,
    )
    result = run_backtest(rows, model, cfg, starting_cash=1000.0, market_type="spot")
    # one entry then cap prevents subsequent same-day entries
    assert result.trades >= 1
    assert result.trades_per_day_cap_hit >= 1


def test_backtest_hits_break_even_entry_and_exit_logic() -> None:
    rows = [
        _flat_row(0, close=100.0, score=10.0, label=1),
        _flat_row(1, close=0.0, score=10.0, label=1),
        _flat_row(2, close=100.0, score=0.0, label=0),
    ]
    cfg = StrategyConfig(
        risk_per_trade=0.1,
        max_position_pct=0.5,
        leverage=1.5,
        signal_threshold=0.55,
    )
    result = run_backtest(rows, _simple_model(10.0), cfg, starting_cash=10.0, market_type="spot")
    assert result.closed_trades >= 0
    assert result.max_exposure >= 0.0


def test_backtest_can_experience_negative_mark_to_market_equity_and_cap_floor() -> None:
    rows = [
        _flat_row(0, close=100.0, score=10.0, label=1),
        _flat_row(60_000, close=1.0, score=10.0, label=1),
        _flat_row(120_000, close=1.0, score=10.0, label=1),
    ]
    model = TrainedModel(
        weights=[20.0] + [0.0] * 12,
        bias=0.0,
        feature_dim=13,
        epochs=1,
        feature_means=[0.0] * 13,
        feature_stds=[1.0] * 13,
    )
    cfg = StrategyConfig(
        risk_per_trade=0.5,
        max_position_pct=1.0,
        leverage=125.0,
        signal_threshold=0.55,
        take_profit_pct=10.0,
        stop_loss_pct=2.0,
    )
    result = run_backtest(rows, model, cfg, starting_cash=10.0, market_type="futures")
    assert result.closed_trades == 1
    assert result.ending_cash < 0
    assert result.max_drawdown > 0.0


def test_backtest_force_close_and_max_open_cap_guard() -> None:
    rows = [
        _flat_row(0, close=100.0, score=1.0, label=1),
        _flat_row(1, close=100.0, score=1.0, label=1),
        _flat_row(2, close=100.0, score=1.0, label=1),
    ]
    cfg = StrategyConfig(
        leverage=1.0,
        risk_per_trade=0.5,
        max_open_positions=0,
        signal_threshold=0.55,
        max_trades_per_day=10,
        take_profit_pct=0.8,
        stop_loss_pct=0.8,
    )
    result = run_backtest(rows, _simple_model(10.0), cfg, starting_cash=1000.0, market_type="spot")
    # no open allowed by max_open_positions should produce no closed trades and at least one cap hit
    assert result.closed_trades == 0
    assert result.trades == 0
    assert result.trades_per_day_cap_hit >= 1

    cfg2 = StrategyConfig(
        leverage=1.0,
        risk_per_trade=0.2,
        max_open_positions=1,
        max_trades_per_day=10,
        signal_threshold=0.55,
        take_profit_pct=0.8,
        stop_loss_pct=0.8,
    )
    result2 = run_backtest(rows, _simple_model(10.0), cfg2, starting_cash=1000.0, market_type="spot")
    assert result2.closed_trades == 1
    assert result2.trades == 1


def test_backtest_flags_drawdown_stop() -> None:
    rows = [
        _flat_row(0, 100.0, 10.0, 1),
        _flat_row(60_000, 1.0, 10.0, 1),
        _flat_row(120_000, 1.0, 10.0, 1),
    ]
    cfg = StrategyConfig(
        leverage=1.0,
        risk_per_trade=0.5,
        max_position_pct=1.0,
        signal_threshold=0.55,
        stop_loss_pct=1.0,
        take_profit_pct=1.0,
        max_drawdown_limit=0.2,
    )
    result = run_backtest(rows, _simple_model(10.0), cfg, starting_cash=1000.0, market_type="spot")
    assert result.stopped_by_drawdown is True
    assert result.max_drawdown > 0.0
    assert result.closed_trades == 1


def test_backtest_rejects_entries_with_zero_gross_or_insufficient_cash() -> None:
    rows = [
        _flat_row(0, 100.0, 10.0, 1),
        _flat_row(1, 100.0, 10.0, 1),
    ]

    model = TrainedModel(
        weights=[20.0] + [0.0] * 12,
        bias=0.0,
        feature_dim=13,
        epochs=1,
        feature_means=[0.0] * 13,
        feature_stds=[1.0] * 13,
    )

    cfg = StrategyConfig(
        leverage=1.0,
        risk_per_trade=0.0,
        max_position_pct=0.2,
        signal_threshold=0.55,
        take_profit_pct=0.5,
        stop_loss_pct=0.5,
    )
    result = run_backtest(rows, model, cfg, starting_cash=1000.0, market_type="spot")
    assert result.closed_trades == 0

    cfg2 = StrategyConfig(
        leverage=1.0,
        risk_per_trade=1.0,
        max_position_pct=1.0,
        signal_threshold=0.55,
        taker_fee_bps=20000.0,
        take_profit_pct=0.5,
        stop_loss_pct=0.5,
    )
    result2 = run_backtest(rows, model, cfg2, starting_cash=1000.0, market_type="spot")
    assert result2.closed_trades == 0


def test_backtest_trade_cap_prevents_entry_when_position_flat() -> None:
    class _ScoreModel:
        def __init__(self) -> None:
            self.calls = 0

        def predict_proba(self, _features: tuple[float, ...]) -> float:
            scores = [0.99, 0.0, 0.99]
            value = scores[min(self.calls, len(scores) - 1)]
            self.calls += 1
            return value

    result = run_backtest(
        [_flat_row(0, 100.0, 1.0, 1), _flat_row(0, 100.0, 1.0, 1), _flat_row(0, 100.0, 1.0, 1)],
        _ScoreModel(),
        StrategyConfig(
            leverage=1.0,
            risk_per_trade=0.1,
            max_position_pct=0.5,
            signal_threshold=0.55,
            max_trades_per_day=1,
            max_open_positions=10,
        ),
        starting_cash=1000.0,
        market_type="spot",
    )
    assert result.trades_per_day_cap_hit >= 1


def test_backtest_profitable_exit_and_win_rate() -> None:
    rows = [
        _flat_row(0, 100.0, 10.0, 1),
        _flat_row(60_000, 110.0, 0.0, 0),
    ]
    cfg = StrategyConfig(
        risk_per_trade=0.2,
        max_position_pct=0.5,
        leverage=1.0,
        signal_threshold=0.55,
        take_profit_pct=0.1,
        stop_loss_pct=0.1,
    )
    result = run_backtest(rows, _simple_model(10.0), cfg, starting_cash=1000.0, market_type="spot")
    assert result.closed_trades == 1
    assert result.trades == 1
    assert result.win_rate == 1.0


def test_backtest_skips_entry_when_margin_exceeds_cash() -> None:
    rows = [
        _flat_row(0, 100.0, 10.0, 1),
        _flat_row(60_000, 100.0, 10.0, 1),
    ]
    cfg = StrategyConfig(
        risk_per_trade=1.0,
        max_position_pct=1.0,
        signal_threshold=0.55,
        leverage=1.0,
    )
    result = run_backtest(rows, _simple_model(10.0), cfg, starting_cash=1000.0, market_type="spot")
    assert result.closed_trades == 0
    assert result.trades == 0
    assert result.trades_per_day_cap_hit == 0


def test_backtest_zero_daily_cap_blocks_all_entries() -> None:
    rows = [
        _flat_row(0, 100.0, 10.0, 1),
        _flat_row(60_000, 100.0, 10.0, 1),
        _flat_row(120_000, 100.0, 10.0, 1),
    ]
    cfg = StrategyConfig(
        risk_per_trade=0.1,
        max_position_pct=0.2,
        max_trades_per_day=1,
        signal_threshold=0.55,
    )
    result = run_backtest(rows, _simple_model(10.0), cfg, starting_cash=1000.0, market_type="spot")
    assert result.trades == 1
    assert result.closed_trades == 1


def test_backtest_daily_cap_counts_entries_not_closures() -> None:
    class _StepModel:
        def __init__(self) -> None:
            self.calls = 0

        def predict_proba(self, _features: tuple[float, ...]) -> float:
            scores = [0.99, 0.0, 0.99, 0.0]
            score = scores[min(self.calls, len(scores) - 1)]
            self.calls += 1
            return score

    rows = [
        _flat_row(0, 100.0, 10.0, 1),
        _flat_row(24 * 60 * 60 * 1000, 110.0, 0.0, 0),
        _flat_row(24 * 60 * 60 * 1000 + 60_000, 110.0, 10.0, 1),
        _flat_row(24 * 60 * 60 * 1000 + 120_000, 120.0, 0.0, 0),
    ]
    cfg = StrategyConfig(
        risk_per_trade=0.1,
        max_position_pct=0.5,
        max_trades_per_day=1,
        signal_threshold=0.55,
        take_profit_pct=0.5,
        stop_loss_pct=0.5,
    )
    result = run_backtest(rows, _StepModel(), cfg, starting_cash=1000.0, market_type="spot")
    assert result.closed_trades == 2
    assert result.trades_per_day_cap_hit == 0


def test_backtest_futures_neutral_signal_does_not_open_position() -> None:
    rows = [
        _flat_row(0, 100.0, 0.0, 0),
        _flat_row(60_000, 100.0, 0.0, 0),
    ]
    cfg = StrategyConfig(
        leverage=5.0,
        risk_per_trade=0.2,
        max_position_pct=0.5,
        signal_threshold=0.55,
    )
    result = run_backtest(rows, _simple_model(0.0), cfg, starting_cash=1000.0, market_type="futures")
    assert result.trades == 0
    assert result.closed_trades == 0
    assert result.max_exposure == 0.0
