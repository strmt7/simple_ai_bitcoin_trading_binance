from __future__ import annotations

import argparse

from simple_ai_bitcoin_trading_binance.cli import (
    _build_order_notional,
    _build_live_model,
    _effective_leverage,
    _resolve_live_retrain_rows,
    _target_notional,
    command_strategy,
)
from simple_ai_bitcoin_trading_binance.config import load_strategy
from simple_ai_bitcoin_trading_binance.api import SymbolConstraints
from simple_ai_bitcoin_trading_binance.types import StrategyConfig


def test_effective_leverage_clamps_by_market() -> None:
    cfg = StrategyConfig(leverage=250.0)
    assert _effective_leverage(cfg, "spot") == 1.0
    assert _effective_leverage(cfg, "futures") == 125.0


def test_target_notional_scales_with_futures_leverage() -> None:
    cfg = StrategyConfig(leverage=20.0, risk_per_trade=0.01, max_position_pct=0.2)
    spot_notional = _target_notional(1000.0, cfg, "spot")
    futures_notional = _target_notional(1000.0, cfg, "futures")
    assert spot_notional == 10.0
    assert futures_notional == 200.0


class _ConstraintClient:
    def __init__(self, constraints: SymbolConstraints) -> None:
        self.constraints = constraints

    def normalize_quantity(self, symbol: str, quantity: float):
        if symbol != self.constraints.symbol:
            return 0.0, self.constraints
        if quantity <= 0:
            return 0.0, self.constraints
        if quantity < self.constraints.min_qty:
            return 0.0, self.constraints
        step = self.constraints.step_size
        quantized = int(quantity / step) * step
        if quantized > self.constraints.max_qty > 0:
            quantized = self.constraints.max_qty
        return quantized, self.constraints


def test_build_order_notional_respects_symbol_constraints() -> None:
    cfg = StrategyConfig(leverage=2.0, risk_per_trade=0.5, max_position_pct=0.75)
    constraints = SymbolConstraints(
        symbol="BTCUSDC",
        min_qty=0.5,
        max_qty=2.0,
        step_size=0.5,
        min_notional=300.0,
        max_notional=700.0,
    )
    client = _ConstraintClient(constraints)

    notional, qty = _build_order_notional(
        cash=1000.0,
        price=500.0,
        cfg=cfg,
        market_type="futures",
        leverage=2.0,
        client=client,
        constraints=constraints,
    )
    assert notional == 500.0
    assert qty == 1.0

    constraints = SymbolConstraints(
        symbol="BTCUSDC",
        min_qty=2.0,
        max_qty=5.0,
        step_size=0.5,
        min_notional=1200.0,
        max_notional=3000.0,
    )
    client = _ConstraintClient(constraints)

    notional, qty = _build_order_notional(
        cash=1000.0,
        price=500.0,
        cfg=cfg,
        market_type="futures",
        leverage=2.0,
        client=client,
        constraints=constraints,
    )
    assert notional == 0.0
    assert qty == 0.0


def test_command_strategy_updates_risk_and_rate_limits(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    cfg = StrategyConfig()
    # ensure baseline config exists
    from simple_ai_bitcoin_trading_binance.config import save_strategy

    save_strategy(cfg)

    args = argparse.Namespace(
        leverage=None,
        risk=0.003,
        max_position=None,
        stop=None,
        take=None,
        cooldown=None,
        max_open=4,
        signal_threshold=None,
        max_drawdown=None,
        taker_fee_bps=None,
        slippage_bps=None,
        label_threshold=None,
        max_trades_per_day=7,
    )
    result = command_strategy(args)
    assert result == 0
    updated = load_strategy()
    assert updated.risk_per_trade == 0.003
    assert updated.max_open_positions == 4
    assert updated.max_trades_per_day == 7


def test_resolve_live_retrain_rows_handles_short_full_and_tail_windows() -> None:
    rows = list(range(10))
    assert _resolve_live_retrain_rows(rows[:3], retrain_window=5, retrain_min_rows=4) == []
    assert _resolve_live_retrain_rows(rows[:5], retrain_window=5, retrain_min_rows=4) == rows[:5]
    assert _resolve_live_retrain_rows(rows, retrain_window=4, retrain_min_rows=4) == rows[-4:]


def test_build_live_model_respects_existing_model_and_retrain_cadence(monkeypatch) -> None:
    cfg = StrategyConfig(training_epochs=100)
    rows = list(range(12))
    existing = object()

    assert _build_live_model(
        rows,
        model=existing,
        retrain_every=0,
        step=5,
        cfg=cfg,
        retrain_window=10,
        retrain_min_rows=4,
    ) is existing

    assert _build_live_model(
        rows,
        model=existing,
        retrain_every=3,
        step=4,
        cfg=cfg,
        retrain_window=10,
        retrain_min_rows=4,
    ) is existing

    calls: list[tuple[list[int], int]] = []

    def fake_train(train_rows, *, epochs: int):
        calls.append((list(train_rows), epochs))
        return {"trained": True}

    monkeypatch.setattr("simple_ai_bitcoin_trading_binance.cli.train", fake_train)
    rebuilt = _build_live_model(
        rows,
        model=existing,
        retrain_every=2,
        step=4,
        cfg=cfg,
        retrain_window=5,
        retrain_min_rows=4,
    )
    assert rebuilt == {"trained": True}
    assert calls == [(rows[-5:], 40)]


def test_build_live_model_returns_existing_when_rows_insufficient(monkeypatch) -> None:
    cfg = StrategyConfig(training_epochs=50)
    existing = object()
    called = {"value": False}

    def fake_train(*_args, **_kwargs):
        called["value"] = True
        return {"trained": True}

    monkeypatch.setattr("simple_ai_bitcoin_trading_binance.cli.train", fake_train)
    rebuilt = _build_live_model(
        [1, 2, 3],
        model=existing,
        retrain_every=1,
        step=1,
        cfg=cfg,
        retrain_window=10,
        retrain_min_rows=5,
    )
    assert rebuilt is existing
    assert called["value"] is False
