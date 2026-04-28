# Design Research Notes - 2026-04-28

Scope: BTCUSDC-only, Binance testnet-first prototype. These notes summarize
the source-backed design pass used for the 2026-04-28 hardening work.

## High-confidence Findings

1. **Bias diagnostics must be first-class.**
   Freqtrade documents separate lookahead and recursive-analysis workflows
   because vectorized backtests can silently see future data, and recursive
   indicators can differ between full-history backtests and shorter dry/live
   caches. This repo now has a local `audit` command that reports candle
   quality, feature-row generation, and latest-row feature stability across
   different cache depths.

2. **Autonomous mode needs controller-style gates.**
   Hummingbot Strategy V2 separates long-running strategy logic from
   self-contained execution components and emphasizes configurable controllers,
   executors, lifecycle management, and performance reporting. This repo keeps
   the simpler single-process prototype, but the autonomous loop now has an
   explicit pre-entry gate for confidence, max open positions, daily caps,
   cooldown, and drawdown lockout.

3. **Protections should block entries, not only close bad trades.**
   Freqtrade protections include max drawdown and cooldown rules that stop
   trading after poor recent behavior or after an exit. The autonomous loop now
   enforces equivalent local protections before opening a new position.

4. **Order sizing must remain exchange-filter-aware.**
   Binance symbol filters define quantity, notional, market lot size, max
   orders, and max position rules. The current client already normalizes order
   quantity and notional against exchange info; the audit/reporting layer now
   makes these safety assumptions visible before longer testnet sessions.

5. **Backtests need a baseline.**
   A raw positive P&L is not enough. Every backtest now reports a
   fee/slippage-aware buy-and-hold BTCUSDC baseline and `edge_vs_buy_hold`, so
   operators can see whether the strategy added value versus passive exposure.

6. **Threshold calibration must stay inside the training side of each fold.**
   The walk-forward helper now calibrates thresholds on a held-out slice inside
   the training window, then reports performance on the untouched forward test
   window.

## Implemented In This Pass

- Added `simple-ai-trading audit` and a matching TUI action.
- Added `src/simple_ai_bitcoin_trading_binance/audit.py` for no-network local
  diagnostics.
- Bounded EMA history in the feature pipeline so latest features match across
  full and live-sized caches.
- Added autonomous pre-entry risk gates with durable skip reasons in the
  heartbeat.
- Added buy-and-hold baseline P&L and edge reporting to backtests.
- Moved walk-forward threshold calibration off the reported test window.

## Sources

- Freqtrade lookahead analysis:
  https://docs.freqtrade.io/en/stable/lookahead-analysis/
- Freqtrade recursive analysis:
  https://docs.freqtrade.io/en/stable/recursive-analysis/
- Freqtrade protections:
  https://docs.freqtrade.io/en/2026.1/plugins/
- Hummingbot Strategy V2 architecture:
  https://hummingbot.org/strategies/v2-strategies/
- Hummingbot controllers:
  https://hummingbot.org/strategies/v2-strategies/controllers/
- Hummingbot executors:
  https://hummingbot.org/strategies/v2-strategies/executors/
- scikit-learn time-series split:
  https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html
- Binance spot filters:
  https://github.com/binance/binance-spot-api-docs/blob/master/filters.md
- Binance spot WebSocket streams:
  https://github.com/binance/binance-spot-api-docs/blob/master/web-socket-streams.md
