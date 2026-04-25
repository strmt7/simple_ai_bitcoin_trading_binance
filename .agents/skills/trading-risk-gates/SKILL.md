---
name: trading-risk-gates
description: Risk guardrails that must survive any CLI, strategy, or live-loop change — drawdown caps, daily trade caps, position sanity, leverage clamps.
origin: repo-local skill for simple_ai_bitcoin_trading_binance
---

# Trading Risk Gates

Use this skill whenever you touch the backtest engine, the live loop, the strategy profile table, or anything downstream of `_build_order_notional` / `_paper_or_live_order`.

## Invariants that must hold

1. **Leverage clamp.** Futures paths clamp effective leverage to `min(strategy.leverage, client.get_max_leverage(symbol), 125)` before any order submission. A bug that submits uncapped leverage is a security-class incident — add a regression test.
2. **Drawdown circuit breaker.** Both backtest and live loops honor `cfg.max_drawdown_limit`. When `0 < limit ≤ drawdown`, stop entering new positions and emit a `stopped_by_drawdown` marker. Do not swallow this signal silently.
3. **Daily trade cap.** `cfg.max_trades_per_day > 0` is enforced per-UTC-day. A value of `0` is an explicit opt-out (no cap) and must be tested as such.
4. **Max open positions.** `cfg.max_open_positions` defaults to `1`. Any path that would exceed it without the user changing the setting is a bug.
5. **No live real-money execution this phase.** `testnet=True` is required before any authenticated live run. Add a preflight that fails loud if someone disables it.
6. **Position resume parity.** Authenticated live runs read exchange account state before the loop and hydrate the run state from it. Do not assume the previous process exited flat.

## When adding a new strategy profile

- Declare all risk fields explicitly. Missing fields silently inherit from the previous profile and that has bitten us before.
- Add a test that asserts the exact profile table for the new name, including leverage, risk_per_trade, max_position_pct, signal_threshold, and max_drawdown_limit.
- Document operator intent in one sentence in the README profile table.

## When adding a new order path

- Use `reduceOnly=true` for any close / emergency-close on futures.
- Use `newOrderRespType=RESULT` on futures so the response includes a fill price we can persist.
- Wrap exchange rejections as `order_error` artifacts; do not let raw `BinanceAPIError` escape the loop.

## Don't

- Don't compute notional in the CLI layer bypassing `_build_order_notional`. That function is the single source of truth for sizing.
- Don't introduce asynchronous order submission. The live loop is intentionally sequential so rate-limit and position accounting stay correct.
- Don't let a tuning search propose a profile that violates any invariant above. The tune scorer must reject those candidates explicitly.
