from __future__ import annotations

from simple_ai_bitcoin_trading_binance.cli import _effective_leverage, _target_notional
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
