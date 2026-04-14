"""Backtesting engine for BTCUSDC model-driven strategies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .features import ModelRow
from .model import TrainedModel
from .types import StrategyConfig


@dataclass
class BacktestResult:
    starting_cash: float
    ending_cash: float
    realized_pnl: float
    win_rate: float
    trades: int
    max_drawdown: float


def run_backtest(rows: List[ModelRow], model: TrainedModel, cfg: StrategyConfig,
                *, starting_cash: float = 1000.0, market_type: str = "spot") -> BacktestResult:
    if not rows:
        return BacktestResult(starting_cash=starting_cash, ending_cash=starting_cash,
                             realized_pnl=0.0, win_rate=0.0, trades=0, max_drawdown=0.0)

    cash = starting_cash
    peak = starting_cash
    max_drawdown = 0.0
    wins = 0
    trades = 0
    position_notional = 0.0
    entry_price = 0.0

    for row in rows:
        price = row.close
        signal = model.predict(row.features, cfg.signal_threshold)
        if market_type == "futures":
            direction = 1 if signal == 1 else -1
        else:
            # spot markets cannot short in this CLI; we model long-only execution
            direction = 1 if signal == 1 else 0

        if position_notional == 0 and direction != 0:
            max_exposure = min(1.0, cfg.risk_per_trade * cfg.leverage)
            notional = cash * max_exposure
            if notional <= 0:
                continue
            entry_price = price
            position_notional = direction * min(notional, cash * cfg.max_position_pct)
            continue

        if position_notional == 0:
            continue

        # fixed stop loss / take profit exits in terms of PnL
        pnl_pct = (price - entry_price) / entry_price * (1 if position_notional > 0 else -1)
        take_profit = cfg.take_profit_pct
        stop_loss = -cfg.stop_loss_pct
        if pnl_pct >= take_profit or pnl_pct <= stop_loss:
            pnl = abs(position_notional) * pnl_pct
            cash += pnl
            trades += 1
            if pnl > 0:
                wins += 1
            position_notional = 0.0
            entry_price = 0.0

        peak = max(peak, cash)
        max_drawdown = max(max_drawdown, (peak - cash) / peak if peak else 0.0)

    realized_pnl = cash - starting_cash
    win_rate = wins / trades if trades else 0.0
    return BacktestResult(
        starting_cash=starting_cash,
        ending_cash=cash,
        realized_pnl=realized_pnl,
        win_rate=win_rate,
        trades=trades,
        max_drawdown=max_drawdown,
    )
