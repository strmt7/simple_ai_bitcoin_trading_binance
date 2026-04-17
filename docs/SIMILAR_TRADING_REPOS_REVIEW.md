# Similar Trading Repositories Review

Snapshot date: 2026-04-17.

Scope: verified repositories and official documentation for the main matrix. GitHub stars, forks, primary language, descriptions, and URLs were checked with `gh repo view` from this workspace. Feature notes are limited to repository descriptions, README text, or official documentation pages reviewed during this pass. Additional forum checks were used only to discover recurring operational failure modes, not as authoritative API documentation.

## Design Lessons Applied Here

1. Keep one obvious operator path. Freqtrade, LEAN CLI, and Hummingbot all expose many capabilities, but the first-run path is explicit: configure, fetch data, train or backtest, then paper or live run. This repo exposes that path through the TUI and `prepare`.
2. Keep dry-run and paper modes first-class. Freqtrade, Hummingbot, OctoBot, Alpaca, and FinRL all treat simulated or paper workflows as a normal operating mode, not an afterthought.
3. Put readiness and connectivity near execution. NautilusTrader and CCXT documentation both highlight that live execution needs reconciliation and exchange-specific state checks. This repo now has `doctor`, `report --doctor`, and a TUI readiness action.
4. Make training and risk choices discoverable. LEAN, Freqtrade, vectorbt, backtesting.py, and Qlib all emphasize repeatable research and optimization loops. This repo now has named training presets and strategy risk profiles while still allowing custom values.
5. Keep adapter boundaries explicit. CCXT, Hummingbot, python-binance, binance-connector-python, and Alpaca all separate exchange connectivity from strategy logic. This repo keeps the Binance client behind `_build_client` and tests connectivity through stubs.
6. Treat exchange filters as normal runtime failures. Binance documentation and field reports repeatedly point to quantity, precision, notional, and price-filter rejections; started live loops now persist `order_error` events for entry, close, and emergency-close order failures.
7. Batch public historical data only where it is safe. Binance kline downloads have request-size limits, so this repo pages historical fetches with `--batch-size`; live market/order loops remain sequential because signed calls and order state should not be parallelized casually.
8. Reconcile live state before acting. Live-oriented systems and exchange SDK guidance assume account/order state can outlive a process; authenticated live runs now inspect exchange BTCUSDC balances or positions before assuming the bot is flat.
9. Preserve simple local artifacts. This repo remains intentionally small: JSON data, JSON model, JSON run artifacts, no database dependency, no multi-exchange abstraction until the operational need is real. Started live loops persist halt context for post-run inspection.

## Repository Matrix

| Repository | Stars | Forks | Language | Verified URL | User Features | Technical Implementation Notes | Takeaway For This Repo |
|---|---:|---:|---|---|---|---|---|
| freqtrade/freqtrade | 48843 | 10194 | Python | https://github.com/freqtrade/freqtrade | Crypto bot with dry-run, live trade, backtesting, optimization, WebUI, Telegram control, market selection, analysis tooling. | Python strategy model, many CLI subcommands, persisted trade/history data, hyperparameter optimization surface. | Keep an explicit dry-run/live distinction, include readiness checks, and make operator actions easy to discover. |
| hummingbot/hummingbot | 18191 | 4607 | Python | https://github.com/hummingbot/hummingbot | Market-making and algorithmic bot framework with client, API, dashboard, strategy scripts/controllers, and many connectors. | Modular connector layer standardizes order and trading logic across venues; strategies call connectors rather than exchange-specific code. | Keep the Binance client behind a stable local adapter and avoid leaking exchange details into the UI. |
| Drakkar-Software/OctoBot | 5702 | 1145 | Python | https://github.com/Drakkar-Software/OctoBot | Simple interface, AI/Grid/DCA/TradingView automation, Binance/Hyperliquid plus 15+ exchanges, Telegram bot topic, paper-trading and backtest topics. | Uses a bot/product surface around exchange automation and simulated portfolio checks. | Make the TUI operationally simple and put monitoring status where the operator can always see it. |
| ccxt/ccxt | 41942 | 8623 | Python | https://github.com/ccxt/ccxt | Unified crypto exchange API for market data, trading, account data, and order history where supported. | Unified methods hide common differences but docs warn some exchange methods are unavailable and user-side tracking may be required. | Keep Binance-specific constraints explicit and tested instead of assuming all exchange data exists. |
| nautechsystems/nautilus_trader | 22042 | 2666 | Rust | https://github.com/nautechsystems/nautilus_trader | Production-grade event-driven trading engine for backtest and live markets. | Deterministic event model, live node configuration, and execution reconciliation concepts. | Add preflight checks before live-like runs and keep backtest/live assumptions visible. |
| QuantConnect/Lean | 18431 | 4700 | C# | https://github.com/QuantConnect/Lean | Local/cloud research, backtesting, optimization, live trading, reports, data download through LEAN CLI. | Engine plus CLI flow; project-oriented commands guide users through research to deployment. | Provide direct commands and guided TUI actions, not hidden workflows. |
| microsoft/qlib | 40855 | 6428 | Python | https://github.com/microsoft/qlib | AI-oriented quant research platform for supervised learning, market dynamics modeling, reinforcement learning, and production exploration. | Research pipeline separates data, model, experiment, and evaluation surfaces. | Keep model artifacts self-describing and tied to feature signatures. |
| AI4Finance-Foundation/FinRL | 14782 | 3275 | Jupyter Notebook | https://github.com/AI4Finance-Foundation/FinRL | Financial reinforcement-learning workflows, market environments, agent training, backtesting, and live-trading examples. | Layered environment/agent/evaluation approach intended for reproducible experiments. | Keep a clear offline evaluation path before testnet execution. |
| vnpy/vnpy | 39493 | 11461 | Python | https://github.com/vnpy/vnpy | Open-source quantitative trading framework used for strategy development and order routing across markets. | Event-engine architecture, strategy modules, gateway-oriented integration. | Keep runtime exchange actions and strategy decisions decoupled. |
| mementum/backtrader | 21172 | 5035 | Python | https://github.com/mementum/backtrader | Backtesting and trading framework with strategies, indicators, analyzers, and live data/trading support. | Central engine coordinates data feeds, broker, strategies, observers, and analyzers. | Maintain small composable modules: API, features, model, backtest, CLI. |
| stefan-jansen/zipline-reloaded | 1714 | 291 | Python | https://github.com/stefan-jansen/zipline-reloaded | Pythonic event-driven algorithmic trading library focused on backtesting. | Event-driven algorithm lifecycle and data bundle model. | Do not blur historical evaluation and live execution; label each action clearly. |
| polakowo/vectorbt | 7190 | 922 | Python | https://github.com/polakowo/vectorbt | Fast portfolio modeling and strategy testing across many parameters and assets. | Vectorized and callback-based portfolio simulation with strong performance orientation. | Add presets and tuning loops without sacrificing quick default runs. |
| kernc/backtesting.py | 8206 | 1429 | Python | https://github.com/kernc/backtesting.py | Backtest trading strategies in Python with statistics, plotting, and optimization support. | Compact API around strategy classes and result inspection. | Keep one-command backtest and evaluation paths easy to run. |
| pmorissette/bt | 2849 | 471 | Python | https://github.com/pmorissette/bt | Flexible Python backtesting library. | Composable strategy/backtest primitives from a small package surface. | Avoid overbuilding; add only features that improve the operator loop. |
| gbeced/pyalgotrade | 4647 | 1395 | Python | https://github.com/gbeced/pyalgotrade | Algorithmic trading library with backtesting, paper trading, live trading, event-driven flow, order types, indicators, and metrics. | Event-driven strategy framework with feed and broker abstractions. | Keep paper/live checks around order placement and expose metrics after runs. |
| ricequant/rqalpha | 6299 | 1733 | Python | https://github.com/ricequant/rqalpha | Extendable, replaceable backtest and trading framework supporting multiple securities. | Plugin-like replacement of framework components. | Keep boundaries clean so future exchange/model changes remain local. |
| jesse-ai/jesse | 7676 | 1085 | JavaScript | https://github.com/jesse-ai/jesse | Advanced crypto trading bot written in Python according to the repository description. | Repo indicates a bot-oriented product surface, with mixed primary language metadata from GitHub. | Keep user-facing commands simple even when internals grow. |
| Superalgos/Superalgos | 5412 | 6076 | JavaScript | https://github.com/Superalgos/Superalgos | Visual bot design, charting, data mining, backtesting, paper trading, and multi-server deployments. | Broad visual platform with integrated research and deployment surfaces. | This repo should stay smaller, but status and recent artifacts should always be visible. |
| pst-group/pysystemtrade | 3257 | 1013 | Python | https://github.com/pst-group/pysystemtrade | Systematic trading in Python. | Project is oriented around systematic trading workflows rather than a single exchange CLI. | Keep strategy/risk configuration explicit and saved separately from credentials. |
| sammchardy/python-binance | 7136 | 2291 | Python | https://github.com/sammchardy/python-binance | Binance API Python implementation for automated trading. | Exchange SDK abstraction around REST and streaming Binance APIs. | Keep direct Binance API calls encapsulated and fully stubbed in tests. |
| binance/binance-connector-python | 2822 | 675 | Python | https://github.com/binance/binance-connector-python | Official Binance connector packages for public and signed APIs. | Modular generated connectors, package-per-service direction, PEP 8 and Black guidance. | Keep client wrappers small and avoid hardcoding endpoint details where runtime/env config can handle them. |
| alpacahq/alpaca-py | 1252 | 335 | Python | https://github.com/alpacahq/alpaca-py | Official Alpaca SDK for market data, trading, broker API, paper/sandbox, and live account use. | Separate clients for trading, historical data, streaming, news, options, and broker operations. | Keep account/connectivity checks separate from model training and backtesting. |

## Implemented Feature Backlog From This Review

Implemented in this repo during the second refinement pass:

1. `doctor` CLI command and `Readiness check` TUI action for data, model, safety, risk, and optional connectivity preflight.
2. Real bottom-bar connection status in the TUI, refreshed on mount, every configured interval, and manual snapshot refresh.
3. Training presets: `custom`, `quick`, `balanced`, and `thorough` for easier parameter selection.
4. Guided offline pipeline includes the training preset so bulk workflow and manual training stay consistent.
5. README and agent instructions reference the simpler single-screen operation model and this comparable-repo review.

Implemented in the current operator pass:

1. `prepare` command and `Prepare system` TUI action run fetch, train, evaluate, backtest, then readiness checks.
2. `report` command and `Operator report` TUI action show dashboard state, recent artifacts, and optional readiness/account checks.
3. Strategy profiles: `custom`, `conservative`, `balanced`, and `active`.
4. Authenticated `live` runs now require a configured testnet target, credentials, and a readable compatible model; paper runs may regenerate from current rows.
5. Authenticated live mode rejects legacy model artifacts that lack a current feature signature, and live artifacts record the model signature actually used.
6. `live --sleep 0` remains a valid no-delay paper loop; authenticated live mode clamps to at least one second between iterations.
7. `fetch --batch-size` and `prepare --batch-size` page historical kline downloads instead of pretending one request is enough for larger datasets.
8. `prepare` now exposes learning rate, L2 penalty, walk-forward windows, and threshold-calibration flags in addition to preset, epoch, seed, fetch, and backtest settings.
9. Signed Binance request telemetry redacts timestamp, receive-window, and signature query values before storing request metadata.
10. Futures account reporting includes non-zero `assets` and `positions`, not only spot `balances`.
11. Entry, close, and emergency-close order rejections are caught, recorded as `order_error`, and returned as nonzero live-loop exits instead of uncaught exceptions.
12. Authenticated live starts detect existing BTCUSDC spot/futures exposure and resume it into local run state instead of assuming a previous interrupted run exited flat.
13. Futures close and emergency-close paths submit reduce-only market orders and request result responses.

## Sources Checked

- https://www.freqtrade.io/en/stable/
- https://www.freqtrade.io/en/stable/bot-usage/
- https://hummingbot.org/docs/
- https://hummingbot.org/strategies/scripts/cheatsheet/
- https://github.com/Drakkar-Software/OctoBot
- https://github.com/ccxt/ccxt/wiki/manual
- https://github.com/ccxt/binance-trade-bot
- https://nautilustrader.io/docs/latest/concepts/live/
- https://www.lean.io/cli/
- https://github.com/jesse-ai/jesse
- https://vectorbt.dev/getting-started/features/
- https://www.backtrader.com/
- https://zipline.ml4trading.io/
- https://gbeced.github.io/pyalgotrade/
- https://github.com/binance/binance-connector-python
- https://academy.binance.com/et/articles/binance-api-responses-price-filter-and-percent-price
- https://alpaca.markets/sdks/python/

## Additional Forum Checks

These were used as operational issue discovery only, then cross-checked against Binance filter behavior and local tests:

- https://www.reddit.com/r/binance/comments/1pvjvzw/how_to_automate_1000satsusdt_like_coins/
- https://www.reddit.com/r/binance/comments/tfni00/problem_with_binance_api_filter_failure_price/
- https://www.reddit.com/r/algotrading/comments/okzau7/binance_api_filter_failure_price_filter_solution/
- https://www.reddit.com/r/highfreqtrading/comments/1mo4qf3/building_an_eventdriven_execution_engine_for/
