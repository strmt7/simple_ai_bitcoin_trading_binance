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
    closed_trades: int
    gross_exposure: float


def _bps_to_rate(bps: float) -> float:
    return max(0.0, bps) / 10_000.0


def _fill_price(price: float, side_sign: int, slippage_bps: float) -> float:
    slippage = _bps_to_rate(slippage_bps)
    return price * (1.0 + side_sign * slippage)


def _normalize_market_direction(signal_score: float, cfg: StrategyConfig, market_type: str) -> int:
    if market_type == "futures":
        if signal_score >= cfg.signal_threshold:
            return 1
        if signal_score <= (1.0 - cfg.signal_threshold):
            return -1
        return 0
    return 1 if signal_score >= cfg.signal_threshold else 0


def run_backtest(rows: List[ModelRow], model: TrainedModel, cfg: StrategyConfig,
                *, starting_cash: float = 1000.0, market_type: str = "spot") -> BacktestResult:
    if not rows:
        return BacktestResult(
            starting_cash=starting_cash,
            ending_cash=starting_cash,
            realized_pnl=0.0,
            win_rate=0.0,
            trades=0,
            max_drawdown=0.0,
            closed_trades=0,
            gross_exposure=0.0,
        )

    cash = float(starting_cash)
    equity_peak = cash
    max_drawdown = 0.0
    wins = 0
    closed_trades = 0

    position_side = 0
    notional = 0.0
    qty = 0.0
    entry_price = 0.0
    margin_used = 0.0

    fee_rate = _bps_to_rate(cfg.taker_fee_bps)
    leverage = 1.0 if market_type == "spot" else cfg.leverage
    if leverage < 1:
        leverage = 1.0
    if market_type == "futures" and leverage > 125:
        leverage = 125.0

    for row in rows:
        score = model.predict_proba(row.features)
        signal = _normalize_market_direction(score, cfg, market_type)
        price = row.close

        if position_side == 0 and signal != 0:
            gross = cash * cfg.risk_per_trade * leverage
            if market_type == "spot":
                gross = min(gross, cash * cfg.max_position_pct)
                effective_margin = gross
            else:
                gross = min(gross, cash * cfg.max_position_pct * leverage, cash * leverage)
                gross = max(0.0, gross)
                effective_margin = gross / leverage

            if gross <= 0 or effective_margin >= cash:
                continue

            side_sign = 1 if signal > 0 else -1
            entry = _fill_price(price, side_sign, cfg.slippage_bps)
            if entry <= 0:
                continue

            fee = gross * fee_rate
            total_cost = effective_margin + fee
            if cash < total_cost:
                continue

            cash -= total_cost
            position_side = side_sign
            notional = side_sign * gross
            qty = abs(gross / entry)
            entry_price = entry
            margin_used = effective_margin

        elif position_side != 0:
            current_pnl = position_side * (price - entry_price) * qty
            current_pnl_pct = (price - entry_price) / entry_price if position_side > 0 else (entry_price - price) / entry_price
            should_close = (
                current_pnl_pct >= cfg.take_profit_pct
                or current_pnl_pct <= -cfg.stop_loss_pct
                or signal == 0
                or signal == (-position_side)
            )

            if should_close:
                exit_price = _fill_price(price, -position_side, cfg.slippage_bps)
                realized = position_side * (exit_price - entry_price) * qty
                exit_fee = abs(notional) * fee_rate
                cash += margin_used + realized - exit_fee
                closed_trades += 1
                if realized > 0:
                    wins += 1

                if position_side != 0:
                    notional = 0.0
                    qty = 0.0
                    entry_price = 0.0
                    margin_used = 0.0
                    position_side = 0

        # mark-to-market drawdown control with unrealized exposure
        if position_side != 0:
            unrealized = position_side * (price - entry_price) * qty
            equity = cash + margin_used + unrealized
        else:
            equity = cash

        if equity < 0:
            equity = cash

        if equity > equity_peak:
            equity_peak = equity
        dd = (equity_peak - equity) / equity_peak if equity_peak else 0.0
        if dd > max_drawdown:
            max_drawdown = dd

        if cfg.max_drawdown_limit > 0.0 and dd >= cfg.max_drawdown_limit:
            break

    # force close residual position at final mark
    if position_side != 0:
        final_price = rows[-1].close
        final_exit = _fill_price(final_price, -position_side, cfg.slippage_bps)
        final_realized = position_side * (final_exit - entry_price) * qty
        final_fee = abs(notional) * fee_rate
        cash += margin_used + final_realized - final_fee
        closed_trades += 1
        if final_realized > 0:
            wins += 1

    realized_pnl = cash - starting_cash
    win_rate = wins / closed_trades if closed_trades else 0.0

    trades = closed_trades
    if position_side != 0:
        trades += 1

    return BacktestResult(
        starting_cash=starting_cash,
        ending_cash=cash,
        realized_pnl=realized_pnl,
        win_rate=win_rate,
        trades=trades,
        max_drawdown=max_drawdown,
        closed_trades=closed_trades,
        gross_exposure=abs(notional),
    )
