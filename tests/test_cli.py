from __future__ import annotations

import argparse

from simple_ai_bitcoin_trading_binance.cli import _build_order_notional, _effective_leverage, _target_notional, command_strategy
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
