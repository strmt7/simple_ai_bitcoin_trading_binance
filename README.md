# simple_ai_bitcoin_trading_binance

Simple AI Bitcoin trading CLI (BTCUSDC test-trade focused)

This repository is a compact, testnet-first trading assistant that keeps scope intentionally narrow:

- Supports only `BTCUSDC` for both market data and signal execution.
- Supports Binance spot and futures endpoints.
- Uses only free/public Binance REST endpoints for discovery, market data, and order simulation/execution.
- Supports model training, tuning, backtesting, and live loop execution.
- Stores credentials locally in a user config file with restrictive permissions (`600`).

## Safety defaults

- `testnet` defaults to `true`.
- Strategy can run in paper mode (`dry_run=true`) and is intended to be that way by default.
- Real order execution happens only when `dry_run=false` in `live`.
- This phase blocks real-money execution; live trading requires `testnet=true`.

## Quick start

```bash
python3 -m pip install -e .
simple-ai-trading configure
simple-ai-trading fetch --limit 500
simple-ai-trading train --epochs 250
simple-ai-trading backtest
simple-ai-trading live --steps 10 --sleep 5
```

By default data is stored under `data/historical_btcusdc.json` and `data/model.json`.

If your shell does not source the console entrypoint, run with `PYTHONPATH=src`:

```bash
PYTHONPATH=src python3 -m simple_ai_bitcoin_trading_binance.cli status
PYTHONPATH=src python3 -m simple_ai_bitcoin_trading_binance.cli fetch
```

## Commands

### `configure`

Prompts for:

- symbol (kept at `BTCUSDC` by default)
- interval
- market mode (`spot` or `futures`)
- testnet mode
- Binance API key and secret
- paper mode toggle

### `fetch`

Fetches market bars from Binance and stores them as JSON in `data/historical_btcusdc.json` by default.

### `train`

Builds feature rows from historical candles and trains a pure-stdlib logistic model. Saved to `data/model.json`.

### `backtest`

Runs a conservative backtest against cached data with stop-loss / take-profit risk limits.

### `connect`

Checks exchange connectivity, validates BTCUSDC availability on the selected market, and optionally prints account metadata.

### `live`

Runs a real-time loop that continuously re-trains on a rolling window and executes paper or live orders.
- `--paper` forces dry-run execution even when runtime is set to live.
- Spot mode always executes at 1x. Futures mode uses configured leverage.

### `strategy`

Adjusts risk controls and model-related parameters (`risk_per_trade`, `stop_loss_pct`, leverage, and more).

- use `--leverage` to set intended futures leverage (clamped to exchange limits).

### `status`

Prints current runtime and strategy settings.

## Development

```bash
python3 -m pytest
```

## Limitations

- No deep-learning frameworks are used currently to keep bootstrap dependencies minimal.
- This is not production trading software; behavior is intentionally conservative and constrained to test-phase workflows.
- API key security depends on file-system permissions and host security; do not commit secrets to version control.
