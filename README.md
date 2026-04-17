# simple_ai_bitcoin_trading_binance

Interactive BTCUSDC testnet trading console for Binance.

The project is intentionally narrow and operator-focused:

- `BTCUSDC` only for data, training, backtesting, and execution
- Binance spot and futures testnet support
- one primary interface: the interactive terminal console
- guided runtime editing, strategy editing, feature selection, training, tuning, backtesting, and live-loop control
- local credential storage with `600` permissions
- credential redaction in visible status output and generated JSON artifacts

## Safety defaults

- `testnet` defaults to `true`.
- Strategy can run in paper mode (`dry_run=true`) and is intended to be that way by default.
- Real order execution happens only when `dry_run=false` in `live`.
- This phase blocks real-money execution; live trading requires `testnet=true`.
- Authenticated testnet live runs require API credentials and a readable model that matches the current strategy feature signature.
- `max_trades_per_day` can be set to `0` to disable daily caps.

## Quick start

```bash
python3 -m pip install -e .
simple-ai-trading
```

If your shell does not expose the console entrypoint:

```bash
PYTHONPATH=src python3 -m simple_ai_bitcoin_trading_binance.cli
```

This opens the interactive console inside the current terminal window. The console is the primary operator workflow.

The layout is intentionally simple:

- a single action list in operational order
- a selected-action detail panel
- a live runtime/strategy/artifact snapshot
- an activity log
- modal forms for editing runtime, strategy, tuning, and execution parameters
- password-masked API key and secret fields inside `Runtime settings`
- a bottom bar with real exchange connection status and keyboard hints

Use the left action list to:

- read `Help` for the recommended operator sequence
- update `Runtime settings`, then run `Connect`
- run `Readiness check` before paper or authenticated testnet execution
- edit runtime settings
- edit strategy and feature selection
- run `Prepare system` for the fetch, train, evaluate, backtest, readiness sequence
- fetch candles
- train or retrain the model with `custom`, `quick`, `balanced`, or `thorough` presets
- tune over all data, a lookback window, or an explicit date range
- run backtests and evaluation
- run paper or authenticated testnet live loops
- inspect recent artifacts, account state, and `Operator report`

By default data is written to `data/historical_btcusdc.json` and `data/model.json`.

Useful direct commands:

```bash
simple-ai-trading menu
simple-ai-trading prepare --preset balanced --epochs 180 --learning-rate 0.05 --l2-penalty 0.0001 --batch-size 1000 --online-doctor
simple-ai-trading report
simple-ai-trading doctor --online
simple-ai-trading train --preset balanced
simple-ai-trading strategy --profile conservative
simple-ai-trading live --paper --model data/model.json --steps 20 --sleep 0
```

## Host overrides

If you need to point the client at a compatible proxy or alternate host without code changes, use environment overrides:

```bash
BINANCE_BASE_URL=https://example-proxy.local simple-ai-trading connect
BINANCE_SPOT_BASE_URL=https://spot-proxy.local simple-ai-trading connect
BINANCE_FUTURES_BASE_URL=https://futures-proxy.local simple-ai-trading connect
```

## Interactive console capabilities

### Runtime settings

The console edits:

- interval
- market type
- testnet flag
- API key and secret through password-masked fields
- paper/live default
- startup validation
- max REST calls per minute

### Strategy settings

The console edits:

- leverage
- risk per trade
- max position percent
- stop loss and take profit
- cooldown
- max open positions
- max trades per day
- signal threshold
- max drawdown
- taker fee and slippage
- label threshold
- model lookback
- training epochs
- confidence beta
- short and long feature windows
- enabled model features

### Tuning windows

The console supports:

- all available data
- a recent lookback window in days
- an explicit inclusive date range

### Model and execution workflow

- training uses the current feature selection from strategy settings
- training supports `custom`, `quick`, `balanced`, and `thorough` presets
- `fetch --batch-size N` pages kline downloads into request sizes up to Binance's spot 1000-candle limit or USD-M futures 1500-candle limit; live and signed order calls stay sequential to preserve exchange state and rate-limit safety
- `prepare` runs the normal offline sequence: fetch candles, train, evaluate, backtest, then readiness checks; it stops at the first failed step
- `prepare` exposes fetch batch size, preset, epochs, learning rate, L2 penalty, seed, walk-forward windows, threshold calibration, and backtest starting cash so one command can reproduce many training configurations
- `report` prints the current dashboard, recent artifacts, and readiness checks by default; add `--no-doctor`, `--online`, or `--account` when you need to omit readiness, check connectivity, or include authenticated account state
- evaluation and backtesting use the current saved model artifact
- the live loop supports paper mode and explicit authenticated testnet execution; `--paper` forces paper mode, while `--live` forces authenticated testnet execution
- `live --model PATH` loads that model before the loop; paper runs can regenerate a missing or incompatible model from current rows, but authenticated live runs fail fast instead
- `live --sleep 0` is preserved as a real zero-delay loop for scripted paper/test runs; authenticated `--live` mode clamps this to a one-second minimum
- authenticated live runs inspect exchange account state before the loop and resume a detected BTCUSDC spot or futures position into the local run state instead of assuming the previous process exited flat
- futures close and emergency-close orders use reduce-only market orders with result responses requested, so a close path cannot intentionally increase exposure
- exchange order rejections during entry, close, or emergency close are captured as `order_error` live artifacts instead of crashing the process
- request telemetry redacts signed query fields such as timestamps and signatures before storing `last_request_info`
- the `doctor` command checks safety flags, training data, model compatibility, risk settings, and optional exchange connectivity
- the interactive bottom bar refreshes the exchange connection status automatically while the console is open
- spot roundtrip execution is an explicit console action, not an automatic side effect

### Strategy risk profiles

Use `simple-ai-trading strategy --profile NAME` to apply a saved profile, then add explicit flags such as `--risk` or `--signal-threshold` to override individual fields.

| Profile | Operator intent | Key defaults |
|---|---|---|
| `custom` | Keep current settings unless explicit flags are supplied. | no profile changes |
| `conservative` | Smaller sizing, higher signal threshold, slower cadence. | 1x, 0.5% risk, 10% max position, 6 trades/day, 12% drawdown cap |
| `balanced` | Middle default for testnet iteration. | 2x, 1% risk, 20% max position, 12 trades/day, 20% drawdown cap |
| `active` | Faster and larger testnet experiments. | 3x, 1.5% risk, 25% max position, 24 trades/day, 25% drawdown cap |

### Live artifacts

Run artifacts are JSON files written next to the model path, normally under `data/`. `train`, `evaluate`, `backtest`, and started `live` loops persist run context with redacted runtime credentials. A started live loop still writes a `live_run_*.json` when it halts from market errors, exchange order rejections, drawdown limits, or model incompatibility during signal generation. Preflight rejections before the loop starts, such as missing credentials or an invalid model for authenticated live mode, return an error without a loop artifact.

## Research reference

For verified design comparisons against high-status trading bots, exchange SDKs, and backtesting frameworks, see `docs/SIMILAR_TRADING_REPOS_REVIEW.md`. Agent instructions in `AGENTS.md` require reading it before broad product, architecture, CLI, or workflow redesigns.

## Development

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m coverage run --source=src/simple_ai_bitcoin_trading_binance -m pytest -q
.venv/bin/python -m coverage report --fail-under=100
```

## Limitations

- The current model backend is still intentionally lightweight and conservative; it is configurable and retrainable, but it is not a large deep-learning stack.
- This is not production trading software; behavior is intentionally conservative and constrained to test-phase workflows.
- API key security depends on file-system permissions and host security; do not commit secrets to version control.
- Host selection is configurable via environment overrides, but execution scope remains BTCUSDC-only and testnet-first.
