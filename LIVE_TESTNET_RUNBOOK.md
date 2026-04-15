# Live Testnet Runbook

This runbook is for the next iteration, when the app will be exercised interactively on Binance testnet.

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
4. Runtime config is explicitly checked before any live step.
5. Cached historical data and a model artifact can be regenerated locally if needed.

## Required runtime settings

Run `simple-ai-trading status` and verify:

- `runtime.symbol == BTCUSDC`
- `runtime.testnet == true`
- `runtime.market_type` matches the intended session (`spot` or `futures`)
- `runtime.dry_run == true` for the first live validation pass
- `strategy.max_drawdown_limit` is conservative
- `strategy.max_trades_per_day` is not unintentionally disabled
- `strategy.max_open_positions >= 1`

## Session order

1. Connectivity check
   - `simple-ai-trading connect`
   - Confirm endpoint is testnet
   - Confirm BTCUSDC availability

2. Market data sanity
   - `simple-ai-trading fetch --limit 500`
   - Confirm the file is updated and non-empty

3. Model sanity
   - `simple-ai-trading train --epochs 250`
   - `simple-ai-trading evaluate --input data/historical_btcusdc.json --model data/model.json`
   - If model load fails, regenerate before continuing

4. Backtest sanity
   - `simple-ai-trading backtest`
   - Confirm no obvious instability or immediate drawdown-limit termination

5. Dry-run live session
   - `simple-ai-trading live --steps 3 --sleep 5 --paper`
   - Verify:
     - no runtime exceptions
     - expected event logging
     - generated live artifact under `data/`
     - entries/closes/skips are plausible

6. Controlled testnet order session
   - Only after dry-run behavior is understood
   - Use the smallest reasonable exposure
   - Prefer a short `--steps` run
   - For futures, confirm effective leverage printed by the CLI

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

## First commands for next iteration

```bash
simple-ai-trading status
simple-ai-trading connect
simple-ai-trading fetch --limit 500
simple-ai-trading train --epochs 250
simple-ai-trading evaluate --input data/historical_btcusdc.json --model data/model.json
simple-ai-trading backtest
simple-ai-trading live --steps 3 --sleep 5 --paper
```

## Decision gate before any non-paper testnet order

Proceed only if:

- local tests are green
- PR workflow is green
- `connect` confirms testnet
- dry-run live loop behaves as expected
- generated artifacts are internally consistent
