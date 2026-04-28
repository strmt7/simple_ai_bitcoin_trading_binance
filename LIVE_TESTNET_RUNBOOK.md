# Live Testnet Runbook

This runbook is for the next iteration, when the interactive console will be exercised on Binance testnet.

## Scope

- `BTCUSDC` only
- `testnet=true` only
- Prefer `dry_run=true` first, then controlled testnet order placement
- No real-money execution

## Preconditions

1. Local branch is green:
   - `python3 -m pytest -q`
   - `python3 -m coverage run --source=src/simple_ai_bitcoin_trading_binance -m pytest -q`
2. GitHub PR workflow is green on the exact commit under test.
3. Binance testnet credentials exist and are valid.
4. Interactive console starts cleanly in a real terminal.
5. Runtime config is explicitly checked before any live step.
6. Cached historical data and a model artifact can be regenerated locally if needed.
7. No command output or artifact should contain raw API keys or secrets.

## Launch

```bash
cd /opt/trader/simple_ai_bitcoin_trading_binance
PYTHONPATH=src python3 -m simple_ai_bitcoin_trading_binance.cli
```

## Required runtime settings

From the console, open `Runtime settings` and verify:

- `runtime.symbol == BTCUSDC`
- `runtime.testnet == true`
- `runtime.market_type` matches the intended session (`spot` or `futures`)
- `runtime.dry_run == true` for the first live validation pass
- `strategy.max_drawdown_limit` is conservative
- `strategy.max_trades_per_day` is not unintentionally disabled
- `strategy.max_open_positions >= 1`

UI notes:

- use the left action list for navigation
- `Action` shows the selected operation and when to use it
- `Overview` shows current runtime, strategy, and account context
- `Activity` shows command output and failures

Optional host override checks:

- if a proxy or alternate host is required, set `BINANCE_BASE_URL`, `BINANCE_SPOT_BASE_URL`, or `BINANCE_FUTURES_BASE_URL` explicitly
- confirm the chosen host is still a testnet-compatible environment before any run

## Session order inside the interactive console

1. Connectivity check
   - Run `Connect`
   - Confirm endpoint is testnet
   - Confirm BTCUSDC availability

2. Market data sanity
   - Run `Fetch candles`
   - Confirm the dataset is updated and non-empty

3. Model sanity
   - Run `Train model`
   - Run `Evaluate`
   - If model load fails, regenerate before continuing

4. Backtest sanity
   - Run `Backtest`
   - Confirm no obvious instability or immediate drawdown-limit termination
   - Compare strategy P&L with `buy_hold_pnl` and `edge_vs_buy_hold`

5. Local audit
   - Run `Local audit`
   - Resolve every `[fix]` item before moving to paper or testnet execution
   - Investigate `[warn]` items before increasing risk or loop length

6. Dry-run live session
   - Run `Paper loop`
   - Verify:
     - no runtime exceptions
     - expected event logging
     - generated live artifact under `data/`
     - entries/closes/skips are plausible

7. Controlled testnet order session
   - Only after dry-run behavior is understood
   - Use the smallest reasonable exposure
   - Prefer a short `Testnet loop`
   - For spot, use `Spot roundtrip` first
   - For futures, confirm effective leverage printed by the console log

## Abort conditions

Stop immediately if any of the following occur:

- endpoint is not testnet
- symbol is not `BTCUSDC`
- credentials appear to target a live environment
- leverage or notional is higher than expected
- repeated market/API errors occur
- model artifact mismatch appears unexpectedly
- drawdown emergency-close triggers unexpectedly on the first controlled run
- order responses differ materially from expected testnet behavior

## Expected artifacts to inspect

- `data/model.json`
- `data/*train*_run_*.json`
- `data/*evaluate*_run_*.json`
- `data/*backtest*_run_*.json`
- `data/*live*_run_*.json`

## During the supervised session

- change one variable at a time
- keep `steps` low
- prefer deterministic re-runs over long sessions
- inspect printed leverage, side, quantity, cash, and drawdown after each run
- if futures are used, verify bracket-clamped leverage before trusting execution size

## First console actions for next iteration

- `Connect`
- `Fetch candles`
- `Train model`
- `Evaluate`
- `Backtest`
- `Local audit`
- `Paper loop`

## Decision gate before any non-paper testnet order

Proceed only if:

- local tests are green
- PR workflow is green
- `connect` confirms testnet
- `audit` has no `[fix]` findings
- dry-run live loop behaves as expected
- generated artifacts are internally consistent
