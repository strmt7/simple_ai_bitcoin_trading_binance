# simple_ai_bitcoin_trading_binance

<!-- BEGIN GENERATED BADGES -->
[![andrej-karpathy-skills](https://img.shields.io/static/v1?label=&message=andrej-karpathy-skills&color=555&logo=github&logoColor=white)](https://github.com/forrestchang/andrej-karpathy-skills)
<!-- END GENERATED BADGES -->

> **Early alpha test software.** This project is experimental and incomplete. Many or most features, workflows, trading assumptions, integrations, outputs, and safeguards may not work correctly or as intended. Use it only for testing and review, preferably on Binance testnet or Binance Demo Trading with paper/dry-run behavior. The project authors and contributors take no responsibility for losses, incorrect behavior, missed trades, API issues, data errors, or any other consequences from using this software.

Interactive BTCUSDC non-mainnet trading console for Binance.

The project is intentionally narrow and operator-focused:

- `BTCUSDC` only for data, training, backtesting, and execution
- Binance spot and futures testnet support, plus explicit Binance Demo Trading endpoint selection
- one primary interface: the interactive terminal console
- guided runtime editing, strategy editing, feature selection, training, tuning, backtesting, and live-loop control
- local credential storage with `600` permissions
- credential redaction in visible status output and generated JSON artifacts

## Safety defaults

- `testnet` defaults to `true`.
- `demo` defaults to `false`; when set to `true`, Binance Demo Trading endpoints are used.
- Strategy can run in paper mode (`dry_run=true`) and is intended to be that way by default.
- Real order execution happens only when `dry_run=false` in `live`.
- This phase blocks real-money execution; signed live execution requires `testnet=true` or `demo=true`.
- Authenticated non-mainnet live runs require API credentials and a readable model that matches the current strategy feature signature.
- `max_trades_per_day` can be set to `0` to disable daily caps.

## Quick start

```bash
python3 -m pip install -e .
simple-ai-trading shell      # Claude-Code-style interactive shell
simple-ai-trading menu       # legacy textual operator console
simple-ai-trading objectives # preview Conservative / Default / Risky presets
```

If your shell does not expose the console entrypoint:

```bash
PYTHONPATH=src python3 -m simple_ai_bitcoin_trading_binance.cli shell
```

The `shell` command opens a slash-command REPL inspired by Claude Code — tab
completion, a muted-gradient palette, live status bar, and fall-through to the
rest of the CLI (type `status` or `/status`; both work).  The `menu` command
launches the legacy full-screen Textual console.

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
simple-ai-trading shell                     # Claude-Code-style interactive REPL
simple-ai-trading objectives                # list Conservative / Default / Risky
simple-ai-trading train-suite               # parallel: one model per objective
simple-ai-trading backtest-panel --interval 5m --tag week --objective default \
    --from-date 2026-04-20 --to-date 2026-04-25 --model data/model_default.json
simple-ai-trading autonomous start --objective default
simple-ai-trading autonomous pause          # or: resume, stop, status
simple-ai-trading positions --stats         # open positions + realized/unrealized P&L
simple-ai-trading close <id|all>            # local ledger close (no exchange order)
simple-ai-trading prepare --preset balanced --epochs 180 --learning-rate 0.05 --l2-penalty 0.0001 --batch-size 1000 --online-doctor
simple-ai-trading report
simple-ai-trading risk --paper
simple-ai-trading doctor --online
simple-ai-trading audit
simple-ai-trading data-sync --rows 1000 --db data/market_data.sqlite
simple-ai-trading data-sync --background --rows 1000 --sleep 300
simple-ai-trading signals --refresh
python tools/quality_metrics.py --compare-ref HEAD
simple-ai-trading spot-roundtrip --mode auto --quantity 0.00008 --yes
simple-ai-trading train --source auto --preset balanced --download-missing
simple-ai-trading strategy --profile conservative --external-signals
simple-ai-trading live --paper --model data/model.json --steps 20 --sleep 0 --external-signals
```

### Market data database and external signals

`data-sync` is the durable downloader. It writes closed BTCUSDC candles and
auxiliary Binance metrics to SQLite (`data/market_data.sqlite` by default):

- kline OHLCV plus quote volume, trade count, and taker-buy volumes
- Binance 24h ticker and L1 book ticker snapshots
- Binance USD-M futures premium index, open interest, and funding-rate history
  when a futures public client is available
- per-run sync summaries and warnings for later inspection

The downloader uses the same client throttle/backoff path as the rest of the
app, so `max_rate_calls_per_minute`, retry handling, and `Retry-After` support
remain centralized. Run it once for a bounded backfill or with `--background`
to start a detached loop that writes a PID and log file. Once the requested
history window is present, later runs switch to incremental mode and request
only candles after the latest stored open time. Sync output includes the mode,
net new candle count, coverage ratio, and gap count so repeated no-new-data
runs are visible instead of silently rewriting the same rows.

`python tools/quality_metrics.py --compare-ref <sha>` prints deterministic
before/after metrics for source lines, test lines, CLI command count, function
length, and approximate cyclomatic complexity. Use it before accepting a broad
refinement pass so changes can be judged by measured behavior, not intuition.

Training can now use `--source auto|file|db`. In `auto` mode the CLI trains
from the JSON file when present, otherwise it checks the SQLite store for the
chosen `--market` and `--interval`. If not enough rows exist, an interactive
terminal prompts to download the missing data; non-interactive runs can pass
`--download-missing`.

`signals` fetches the live external confirmation layer used by `live
--external-signals`. It currently blends Alternative.me Fear and Greed,
CoinGecko BTC 24h change, Binance futures positioning, and mempool.space fee
pressure behind a cached report. Positive boosts require the configured minimum
number of fresh providers; negative signals can reduce score and risk sizing.

### Objectives (risk-adjusted scorers)

`train-suite` trains one advanced model per registered objective and writes
`data/model_<objective>.json` plus a `training_suite_summary.json`:

| Objective | Intent | Notable defaults |
|---|---|---|
| `conservative` | Capital preservation, rejects > 15% drawdown, fewer trades. | 1x leverage, signal ≥ 0.66, stop 1.0%, take 2.2%, 400 epochs, poly deg 2 |
| `default` | Balanced risk-adjusted return — the middle preset. | 1.5x, signal ≥ 0.58, stop 1.8%, take 3.0%, 600 epochs, poly deg 2 |
| `risky` | Chase return, tolerate bigger drawdowns + more trades. | 2.5x, signal ≥ 0.53, stop 2.8%, take 4.5%, 900 epochs, poly deg 3 |

Training is parallelized across a `ProcessPoolExecutor` (defaults to
`os.cpu_count()`).  Feature expansion is computed once per objective and
shared across every candidate in the hyperparameter grid; the full 3-objective
suite on ~500 candles runs in seconds on a modern laptop.

### Backtest panel (independent)

`backtest-panel` is a standalone surface: pick any Binance-supported interval,
any time window, any saved model — training is never forced.  Each run writes
a timestamped, tagged JSON under `data/backtests/`:

```
data/backtests/backtest_<tag>_<market>_<interval>_<YYYYMMDDHHMMSS>.json
```

Interval strings are validated against Binance's published enums (1s, 1m, 3m,
5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M; spot-only adds `1s`).
A typo is rejected with a clear error listing every allowed value.

### Autonomous non-mainnet loop

`autonomous start` drives an indefinite, pause-able live loop on Binance
testnet or Binance Demo Trading.  Control is a tiny state file under `data/autonomous/state.json`
(`RUNNING` / `PAUSED` / `STOPPING` / `STOPPED`) so a second shell can pause or
stop the loop without signals.  Every iteration writes a heartbeat to
`data/autonomous/heartbeat.json`; every fill + close updates
`data/autonomous/open_positions.json` and `data/autonomous/ledger.json`.  The
loop refuses to start when both `testnet=False` and `demo=False` — real-money execution stays
blocked in this phase.

### Positions + P&L stats

```
simple-ai-trading positions --stats
#  id           side    qty       entry       mark        pnl$   pnl%
# 1 ab3f2e…     LONG    0.012345  78200.00   78450.50   +3.09   +0.32%
# Closed trades  : 12  (wins 7, losses 5)
# Realized P&L   : +18.42 USDC  (+1.84%)
# Unrealized P&L : +3.09 USDC  (+0.32%)
```

Inside the shell, the same data is reachable via `/positions`, `/stats`,
`/close <id|all>`.

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
- cached external signal toggle, max score adjustment, provider quorum, TTL,
  and timeout

### Tuning windows

The console supports:

- all available data
- a recent lookback window in days
- an explicit inclusive date range

### Model and execution workflow

- training uses the current feature selection from strategy settings
- training supports `custom`, `quick`, `balanced`, and `thorough` presets
- `fetch --batch-size N` pages kline downloads into request sizes up to Binance's spot 1000-candle limit or USD-M futures 1500-candle limit; live and signed order calls stay sequential to preserve exchange state and rate-limit safety
- `data-sync` persists closed candles and auxiliary market metrics into SQLite,
  skips duplicate latest-window rewrites after the requested window is present,
  and reports coverage quality for repeatable training and future enrichment
- `train --source auto` falls back to the SQLite store for the selected
  market/interval and can prompt or `--download-missing` when history is absent
- `signals` and `live --external-signals` add cached free-provider confirmation
  without blocking the exchange loop on slow non-Binance APIs
- `prepare` runs the normal offline sequence: fetch candles, train, evaluate, backtest, local audit, then readiness checks; it stops at the first failed step
- `risk --paper` or `risk --live` prints the local risk policy before an
  operator loop: endpoint safety, credentials, managed cash, leverage,
  position sizing, stop-loss exposure, daily caps, drawdown stops,
  fee/slippage assumptions, external-signal quorum, and model path status
- `audit` runs no-network diagnostics for candle quality, feature stability,
  model metadata, and risk posture; `prepare` runs it before the final
  readiness check
- `prepare` exposes fetch batch size, preset, epochs, learning rate, L2 penalty, seed, walk-forward windows, threshold calibration, and backtest starting cash so one command can reproduce many training configurations
- `report` prints the current dashboard, recent artifacts, and readiness checks by default; add `--no-doctor`, `--online`, or `--account` when you need to omit readiness, check connectivity, or include authenticated account state
- evaluation and backtesting use the current saved model artifact
- backtests report fee/slippage-aware buy-and-hold BTCUSDC P&L and
  `edge_vs_buy_hold` beside strategy P&L
- the live loop supports paper mode and explicit authenticated testnet/demo execution; `--paper` forces paper mode, while `--live` forces authenticated non-mainnet execution
- `live --model PATH` loads that model before the loop; paper runs can regenerate a missing or incompatible model from current rows, but authenticated live runs fail fast instead
- `live --sleep 0` is preserved as a real zero-delay loop for scripted paper/test runs; authenticated `--live` mode clamps this to a one-second minimum
- `spot-roundtrip --mode auto --yes` performs the smallest signed spot testnet/demo exchange check from the CLI; it uses BUY then SELL when USDC is available, or SELL then BUY when only test BTC is available
- training stores early-stopping loss metadata and a model-quality report; `doctor`, `train`, and `evaluate` surface weak validation, overfit, class-balance, and probability-collapse warnings
- authenticated live runs inspect exchange account state before the loop; futures positions are resumed, while spot BTC is resumed only up to the explicit managed BTC allocation
- configured `recvWindow` is used for signed Binance requests, and startup credential validation calls an authenticated account endpoint when keys are present
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

BNB Smart Chain faucet note: the official BNB Chain faucet at
https://www.bnbchain.org/en/testnet-faucet funds BSC testnet wallet addresses
with tBNB/BEP20 test tokens after an hCaptcha-backed browser request and the
published 0.002 BNB BSC mainnet prerequisite. It does not directly fund a
Binance Spot Testnet exchange account, so BTCUSDC exchange-order testing uses
the Spot Testnet account balances exposed by the Binance API.

Binance Demo Trading note: set `"demo": true` in runtime settings to route Spot
requests to `https://demo-api.binance.com` and Futures requests to
`https://demo-fapi.binance.com`. Demo Trading uses separate demo API keys from
the regular Binance account API-key page. Binance documents it as the resettable
virtual-funds path for user-facing Spot/Futures testing; existing Spot Testnet
keys remain separate and cannot be reused on Demo Trading.

## Research reference

For verified design comparisons against high-status trading bots, exchange SDKs,
and backtesting frameworks, see `docs/SIMILAR_TRADING_REPOS_REVIEW.md`. For the
current free external signal/API inventory, see
`docs/FREE_SIGNAL_SOURCE_INVENTORY.md`. The 2026-04-28 design pass is recorded
in `docs/DESIGN_RESEARCH_NOTES_2026-04-28.md`. Agent instructions in
`AGENTS.md` require reading the comparable-repo review before broad product,
architecture, CLI, or workflow redesigns.

## Development

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m coverage run --source=src/simple_ai_bitcoin_trading_binance -m pytest -q
.venv/bin/python -m coverage report --fail-under=100
```

### Containerized run

```bash
docker build -t simple-ai-trading:dev .
docker compose run --rm simple-ai-trading                 # opens /shell
docker compose run --rm simple-ai-trading objectives      # one-shot command
```

Config and data live in named Docker volumes (`simple-ai-config`,
`simple-ai-data`).  Secrets are never baked into the image — the `configure`
command writes them to a `0600` file under the mounted config volume at
runtime.

### Push with a PAT

```bash
GITHUB_TOKEN=ghp_… python3 tools/push_with_pat.py origin feat/my-branch
```

The helper serves the token to `git push` over a short-lived UNIX socket so it
never appears in `argv`, remote URLs, `~/.git-credentials`, or shell history.

## Limitations

- The current model backend is still intentionally lightweight and conservative; it is configurable and retrainable, but it is not a large deep-learning stack.
- This is not production trading software; behavior is intentionally conservative and constrained to test-phase workflows.
- API key security depends on file-system permissions and host security; do not commit secrets to version control.
- Host selection is configurable via environment overrides, but execution scope remains BTCUSDC-only and non-mainnet-first.
