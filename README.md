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
- fetch candles
- train or retrain the model with `custom`, `quick`, `balanced`, or `thorough` presets
- tune over all data, a lookback window, or an explicit date range
- run backtests and evaluation
- run paper or authenticated testnet live loops
- inspect recent artifacts and account state

By default data is written to `data/historical_btcusdc.json` and `data/model.json`.

Useful direct commands:

```bash
simple-ai-trading menu
simple-ai-trading doctor --online
simple-ai-trading train --preset balanced
simple-ai-trading live --paper --steps 20
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
- evaluation and backtesting use the current saved model artifact
- the live loop supports paper mode and explicit authenticated testnet execution
- the `doctor` command checks safety flags, training data, model compatibility, risk settings, and optional exchange connectivity
- the interactive bottom bar refreshes the exchange connection status automatically while the console is open
- spot roundtrip execution is an explicit console action, not an automatic side effect

## Research reference

For verified design comparisons against high-status trading bots, exchange SDKs, and backtesting frameworks, see `docs/SIMILAR_TRADING_REPOS_REVIEW.md`. Agent instructions in `AGENTS.md` require reading it before broad product, architecture, CLI, or workflow redesigns.

## Development

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m coverage run --source=src/simple_ai_bitcoin_trading_binance -m pytest -q
```

## Limitations

- The current model backend is still intentionally lightweight and conservative; it is configurable and retrainable, but it is not a large deep-learning stack.
- This is not production trading software; behavior is intentionally conservative and constrained to test-phase workflows.
- API key security depends on file-system permissions and host security; do not commit secrets to version control.
- Host selection is configurable via environment overrides, but execution scope remains BTCUSDC-only and testnet-first.
