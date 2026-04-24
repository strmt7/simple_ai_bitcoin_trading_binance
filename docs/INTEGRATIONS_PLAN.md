# Integrations plan — BTCUSDC spot testnet console

**Status:** v0 plan, drafted 2026-04-25. The repository targets BTCUSDC spot
trading on the Binance testnet only. This document captures the live-verified
free APIs that are useful for enrichment, the Binance testnet parameters worth
exposing in the UI, design notes inspired by Reflecto's Prophet Intelligence
agent, and a phased implementation plan that an engineer can execute in the
next iteration. All HTTP endpoints listed in the **Live verified** sections
were curl'd from this host on 2026-04-25 and returned the indicated payload
shape with no auth.

The file is intentionally exhaustive on the API side because the user has
asked for a "next-iteration" implementation reference: any API written below
is one we have actually tested, and the integration notes record both the
limits and the failure modes we observed so we do not re-discover them later.

---

## Part A. Free, no-auth, live-verified data APIs

For each entry: endpoint(s) tested, sample call, sample response shape,
documented rate limits and the proposed integration role inside this project.

### A1. Alternative.me — Fear & Greed Index ✅ (live tested)

* Endpoint: `GET https://api.alternative.me/fng/?limit=1&format=json`
* Auth: none. Rate limit: not formally published, but a single 24-hour value
  is updated approximately every 24 h, so polling once per hour is sufficient.
* Sample response (verified 2026-04-25):
  ```json
  {"name":"Fear and Greed Index","data":[{"value":"39","value_classification":"Fear","timestamp":"1776988800","time_until_update":"6482"}]}
  ```
* Role: regime feature. Map `value` (0..100) and a one-hot of
  `value_classification` (`Extreme Fear`, `Fear`, `Neutral`, `Greed`,
  `Extreme Greed`) into the model row. Already a known predictor for BTC
  swing direction over multi-day horizons.

### A2. CoinGecko (free public tier) ✅ (live tested)

* Price + 24h change + volume:
  `GET https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd&include_24hr_change=true&include_24hr_vol=true&include_last_updated_at=true`
* Hourly market chart (last day): `GET /api/v3/coins/bitcoin/market_chart?vs_currency=usd&days=1&interval=hourly`
* Global market state: `GET /api/v3/global` — total_market_cap, btc_dominance, active_cryptocurrencies.
* Derivatives: `GET /api/v3/derivatives/exchanges?per_page=5` — open interest and 24h volume per derivatives venue.
* Tickers depth: `GET /api/v3/coins/bitcoin/tickers?order=volume_desc&depth=true` — bid/ask and cost-to-move USD on each pair.
* Auth: none. Rate limit: ~30 req/min on the free public tier (per CoinGecko docs); we'll cap our usage at 6 req/min and cache.
* Role: cross-exchange volume confirmation, BTC dominance regime, derivatives sentiment.

### A3. Blockchain.info — base-chain stats ✅ (live tested)

* `GET https://api.blockchain.info/stats` — full snapshot incl. `market_price_usd`, `hash_rate`, `n_tx`, `n_blocks_mined`, `minutes_between_blocks`, `difficulty`, `nextretarget`, `miners_revenue_usd`.
* Quick scalars:
  * `GET https://blockchain.info/q/getdifficulty` → integer
  * `GET https://blockchain.info/q/hashrate` → integer
  * `GET https://blockchain.info/q/24hrtransactioncount` → integer
* Auth: none. Rate limit: documented as 1 req/10s per source IP for the `/q/` endpoints; the JSON `stats` endpoint accepts polling once per minute comfortably.
* Role: long-window hash-rate/difficulty regime as a "miner stress" feature, daily transaction count as a demand proxy.

### A4. mempool.space — Bitcoin network microstate ✅ (live tested)

* `GET https://mempool.space/api/v1/fees/recommended` → `{fastestFee, halfHourFee, hourFee, economyFee, minimumFee}` (sat/vB).
* `GET https://mempool.space/api/blocks/tip/height` → integer.
* `GET https://mempool.space/api/v1/difficulty-adjustment` → estimated retarget, percent change, remaining blocks.
* `GET https://mempool.space/api/mempool` → `{count, vsize, total_fee, fee_histogram}`.
* Auth: none. Rate limit: official guidance is "no key required, please be reasonable"; ~1 req/s is safe.
* Role: short-window network-stress features (mempool size, fee spike detection) often correlate with short BTC volatility.

### A5. Binance public market data ✅ (live tested)

* 24h ticker for BTCUSDC: `GET https://api.binance.com/api/v3/ticker/24hr?symbol=BTCUSDC` — open/close/high/low/last, weighted avg, volume, quoteVolume, prev close, count.
* L1 quote: `GET https://api.binance.com/api/v3/ticker/bookTicker?symbol=BTCUSDC` — bid/ask + sizes.
* USDT-M futures cross-checks (used as confirmation since USDT is the dominant futures contract):
  * Open interest: `GET https://fapi.binance.com/fapi/v1/openInterest?symbol=BTCUSDT` → `openInterest` (BTC), `time`.
  * Premium index + funding: `GET https://fapi.binance.com/fapi/v1/premiumIndex?symbol=BTCUSDT` → `markPrice`, `indexPrice`, `lastFundingRate`, `nextFundingTime`.
* Auth: none for these. Spot weight budget is 6000 weight/min per IP (most public calls cost 1–10).
* Role: same-exchange confirmation features — when our model considers BTCUSDC, we still benefit from BTCUSDT funding/OI as a leading sentiment indicator. The bookTicker L1 spread lets us model slippage realistically in the backtester.

### A6. Binance announcements feed ✅ (live tested, undocumented but stable)

* `GET https://www.binance.com/bapi/composite/v1/public/cms/article/list/query?type=1&pageNo=1&pageSize=5`
  (catalogId=48 covers "New Cryptocurrency Listing"; many trading-relevant
  catalogs exist; we'll restrict to the `New Cryptocurrency Listing`,
  `Maintenance`, and `Margin Pair Update` catalogs).
* Sample response is a JSON document with `data.catalogs[].articles[].title` and `releaseDate` (epoch ms).
* Auth: none. The endpoint is undocumented and used by the public web UI; we treat it as best-effort.
* Role: event-driven veto. If the headline contains "BTC" and falls in the past 60 minutes, flag elevated event risk so the strategy reduces position size or skips an entry.

### A7. CryptoCompare unauthenticated minimal-data ✅ (live tested)

* Multi-fiat price: `GET https://min-api.cryptocompare.com/data/price?fsym=BTC&tsyms=USD,USDC,USDT,EUR`
* Hourly OHLC fallback: `GET https://min-api.cryptocompare.com/data/v2/histohour?fsym=BTC&tsym=USD&limit=24`
* Auth: free key recommended for high volume; both above endpoints work without a key for our polling cadence.
* Note: their `news` endpoint requires an API key (verified — returns "valid auth key" error).
* Role: redundant price source for outage resilience; histohour is a backup if Binance candles are unavailable.

### A8. Other exchange tickers (free, no-auth) ✅ (live tested)

These are useful for slippage/cross-arb features and as failover price sources.

* Kraken: `GET https://api.kraken.com/0/public/Ticker?pair=XBTUSD`
* Bitstamp: `GET https://www.bitstamp.net/api/v2/ticker/btcusd/`
* Coinbase Retail: `GET https://api.coinbase.com/v2/prices/BTC-USD/spot`
* Bitfinex: `GET https://api-pub.bitfinex.com/v2/ticker/tBTCUSD`
* CoinPaprika: `GET https://api.coinpaprika.com/v1/tickers/btc-bitcoin`
* CoinLore: `GET https://api.coinlore.net/api/ticker/?id=90`

Auth: none in any case. Rate limits range from 1 req/s to 60 req/min; we'll
poll each at no more than 30s cadence.

Role: derive a **cross-venue dispersion feature** = stdev of BTC-USD prices
across Kraken, Bitstamp, Coinbase, Bitfinex, CoinPaprika, CoinLore, Binance
spot. Wider dispersion historically precedes short-window mean reversion on
the dominant venue.

### A9. CoinPaprika events calendar ✅ (live tested)

* `GET https://api.coinpaprika.com/v1/coins/btc-bitcoin/events?limit=5` — known
  upcoming + historical events with title and link.
* Auth: none. Limit ~25k req/month free tier.
* Role: a calendar feature ("event_in_next_24h: bool") that can be sourced
  daily and used to bias position sizing.

### A10. DeFiLlama — stablecoin and TVL signals ✅ (live tested)

* Stablecoins overview: `GET https://stablecoins.llama.fi/stablecoins?includePrices=true` — `circulating` per pegged asset, mechanism, chain breakdown.
* Chain TVL history: `GET https://api.llama.fi/v2/historicalChainTvl/Ethereum`
* Auth: none. Rate limit: documented as 300 req/5 min for the `api.llama.fi` host.
* Role: USDC supply growth is a useful "fiat ramp" feature; sudden contractions correlate with risk-off in BTC. ETH chain TVL growth is correlated with broader risk-on sentiment.

### A11. Public RSS / community feeds ✅ (live tested)

* CoinDesk RSS: `GET https://www.coindesk.com/arc/outboundfeeds/rss/` (returns RSS XML; redirects from `https://www.coindesk.com` need `-L`).
* Google News RSS: `GET https://news.google.com/rss/search?q=bitcoin&hl=en-US`
* Reddit JSON: `GET https://www.reddit.com/r/Bitcoin/hot.json?limit=5` with `User-Agent: simple-ai-trading/0.1`
* Auth: none. Rate limit: Reddit JSON is OK at one request per several seconds with a custom UA; CoinDesk and Google News RSS are unmetered for normal polling.
* Role: lightweight news headlines feature. In v1 we use simple keyword matching ("ETF", "approval", "rejected", "halt", "exploit", "SEC"); v2 may apply a lexicon-based sentiment scorer.

### A12. Not recommended (verified failures or auth-required)

The following looked promising but failed our zero-friction criteria:

* **CryptoCompare news / `data/v2/news`** — returns "You need a valid auth key
  or api key" without a key.
* **CryptoPanic** — every endpoint requires `auth_token`.
* **AlphaVantage `NEWS_SENTIMENT`** — the `demo` key returns a notice;
  production use requires registration and rate-limits to 25 req/day/free.
* **Coinglass public v2** — endpoints we tested returned no body; their
  free tier appears to be private-key-only now.
* **CoinCap v3** — public endpoints now require `Authorization: Bearer`.
* **Bybit `x-api`** — blocked by their CDN for non-browser clients.
* **Coinbase Exchange / Pro** — `BTC-USDC` is "Not allowed for delisted
  products" on their public v2 product feed.

---

## Part B. Reflecto / Prophet Intelligence — design takeaways

Reflecto's "Prophet Intelligence" is a closed-source LLM-augmented trading
agent built on the SafuNet platform; the only public details available are
marketing-grade. From the Reflecto and SafuNet write-ups (Sources at the
bottom of this section) we extract three reproducible patterns we can
borrow without violating their IP:

1. **Multi-modal signal fusion.** Prophet Intelligence presents signals as
   the *intersection* of statistical models, ML classifiers, and an LLM
   reasoning step. The first two are already in our codebase; we propose
   extending feature ingestion with the APIs in Part A as the primary new
   modality, and adding an optional LLM filter (offline) as a later phase.
2. **Explicit entry / SL / TP packaging.** Their UI surfaces every signal as
   an entry price, stop-loss, take-profit triplet. Our `StrategyConfig`
   already encodes risk-per-trade, stop_loss_pct, take_profit_pct; we should
   surface the *resolved* per-signal triplet in the Snapshot panel and the
   live loop's per-step log so the operator sees the same packaging.
3. **Telegram delivery.** Out of scope for the testnet console, but worth
   leaving an extension hook: a single `Notifier` interface in code with a
   noop default. A future telegram backend can plug into it without UI churn.

Other, broader inspiration drawn from the open-source landscape (see
[asavinov/intelligent-trading-bot](https://github.com/asavinov/intelligent-trading-bot)
and [HKUDS/AI-Trader](https://github.com/HKUDS/AI-Trader)):

* Walk-forward training with leakage controls — already partly implemented
  via the prepare/tune commands.
* Periodic retraining — already supported by `--retrain-interval` and
  `--retrain-window`.
* Sentiment-conditioned feature gates — *new*: only enable certain features
  during specific Fear-Greed regimes; planned for Phase 3 below.

Sources for this section:

- [Reflecto](https://reflectocoin.com/) — official site, Prophet Intelligence
  description.
- [Reflecto on X](https://x.com/reflectocoin) — recent product updates.
- [asavinov/intelligent-trading-bot](https://github.com/asavinov/intelligent-trading-bot) —
  open-source feature-engineering reference.
- [HKUDS/AI-Trader](https://github.com/HKUDS/AI-Trader) — agent-style automated
  trading reference.

---

## Part C. Binance testnet parameters audit

The repository's HTTP client lives in
`src/simple_ai_bitcoin_trading_binance/api.py`; this section enumerates
**parameters that the Binance spot testnet supports** and notes whether each
one is currently exposed by the UI / client, and where it should land if
not.

### Already exposed

| Parameter | UI surface | Code location |
|---|---|---|
| `symbol` (locked to BTCUSDC) | n/a (constant) | `api.py:438`, `cli.py:1956` |
| `interval` | Runtime form | `cli.py:_ui_edit_runtime` |
| `testnet` | Runtime form | same |
| `api_key`, `api_secret` | Runtime form (password) | same |
| `recvWindow` | Runtime form (new in this PR) | same |
| `max_rate_calls_per_minute` (client throttle) | Runtime form | same |
| Order side (`BUY`/`SELL`) | Strategy / live loop | `cli.py:place_order` |
| Order quantity (with LOT_SIZE quantize) | live loop | `api.py:normalize_quantity` |
| Spot order type `MARKET` | hard-coded | `api.py:place_order` |
| Futures `reduceOnly` / `closePosition` | spot-only build, parked | (not used) |

### Missing — Phase 1 (already wired in this PR via Settings hub)

The following were stored in `StrategyConfig` and a sub-form was added under
the new Settings -> Execution menu, but the `BinanceClient.place_order`
signature still hard-codes `MARKET`. Phase 1 of the integration work is to
plumb these values through:

| Parameter | Spot testnet enum | Default | Notes |
|---|---|---|---|
| `type` (orderType) | `MARKET`, `LIMIT`, `LIMIT_MAKER` | `MARKET` | `STOP_LOSS_LIMIT` / `TAKE_PROFIT_LIMIT` skipped — they require a `stopPrice` and triggers that we do not yet model. |
| `timeInForce` | `GTC`, `IOC`, `FOK` | `GTC` | Required for `LIMIT`; ignored for `MARKET`. |
| `price` | float | derived from book mid | Required for non-`MARKET` orders. |
| `newOrderRespType` | `ACK`, `RESULT`, `FULL` | `RESULT` | Already used in futures path; spot benefits from `FULL` for slippage measurement. |
| Post-only flag (`type=LIMIT_MAKER`) | bool | false | If true, the client must translate to `LIMIT_MAKER` instead of `LIMIT`. |
| `selfTradePreventionMode` | `EXPIRE_TAKER`, `EXPIRE_MAKER`, `EXPIRE_BOTH`, `NONE` | `NONE` | Spot testnet supports STP; default `NONE` for the testnet account because the operator is the only counterparty. |
| `icebergQty` | float | unset | Optional; only relevant for sizes > 1 BTC. Hide behind an "Advanced" toggle. |

### Missing — Phase 2 (operator workflow; not yet in the UI)

| Parameter | Endpoint | Why we want it |
|---|---|---|
| `userDataStream` listenKey | `POST /api/v3/userDataStream` (spot) | Real-time order/balance updates over WebSocket without polling. |
| `aggTrades` for slippage profile | `GET /api/v3/aggTrades?symbol=BTCUSDC` | Improves the realism of our offline backtester. |
| `depth` snapshot + diffs | `GET /api/v3/depth?symbol=BTCUSDC&limit=100` | Adds order-book imbalance as a feature. |
| `tickerPrice` for cross-pair sanity | `GET /api/v3/ticker/price?symbol=BTCUSDT` | Detects testnet/mainnet drift. |
| `myTrades` history | `GET /api/v3/myTrades?symbol=BTCUSDC` (signed) | Reconciles the local Funds allocation against actual fills. |

### Skipped — out of scope for "spot testnet only"

The following are valid Binance parameters but explicitly out of scope
because the user has narrowed the project to BTCUSDC spot testnet:

* All `/fapi/*` futures-only routes (positionSide, dualSidePosition,
  multiAssetsMargin, leverage brackets per symbol, marginType per symbol).
* All `/sapi/*` margin / staking / convert routes.
* All OCO / OTO endpoints (`/api/v3/orderList/*`) — useful but a larger UI
  story; deferred until basic LIMIT / LIMIT_MAKER works end-to-end.

Sources for this section:

- [Binance Spot REST API docs](https://developers.binance.com/docs/binance-spot-api-docs/rest-api)
- [Binance Spot enums (orderType, timeInForce, STP)](https://developers.binance.com/docs/binance-spot-api-docs/enums)
- [Binance Spot testnet setup](https://testnet.binance.vision/)

---

## Part D. Phased implementation plan

The plan is structured so each phase ends in a self-contained, testable
state. The user has indicated this document is the input to the *next*
iteration; we therefore err toward concrete tickets.

### Phase 0 — already done in this PR

* Removed `j`/`k` navigation; arrows + Tab only.
* Active panel is bordered in bright green (`heavy #1ad48f` :focus-within).
* Form labels with `[...]` no longer mangled by Rich markup.
* Form rows are explicitly sized so Inputs no longer overlap.
* Snapshot panel scrolls instead of clipping.
* Action list reorganized; `Settings` is penultimate, `Help` is last.
* New `Funds` action manages a virtual USDC/BTC allocation that caps
  trading independently of the testnet wallet.
* `RuntimeConfig` gained `recv_window_ms` and `compute_backend`.
* `StrategyConfig` gained `order_type`, `time_in_force`, `post_only`,
  `reduce_only_on_close`.
* New `compute.py` module + `Settings -> Compute backend` form let the
  operator opt into CUDA / ROCm / auto-detect; CPU fallback is silent and
  reports the reason in the activity log.
* Resize bindings: `<` `>` for nav width, `-` `+` for activity height,
  `Ctrl-L` clears activity, `r` refreshes snapshot.
* Update guards: status, connection, details, preview only re-render if
  their content actually changed (kills observable flicker).

### Phase 1 — wire execution settings into the live loop

* Extend `BinanceClient.place_order(symbol, side, quantity, *, dry_run, leverage, reduce_only, order_type="MARKET", time_in_force=None, price=None, post_only=False, stp_mode="NONE", new_order_resp_type="RESULT")`.
* For `order_type="LIMIT"` derive `price` from the book ticker (bid for SELL,
  ask for BUY) ± a configurable offset; require `time_in_force`.
* For `post_only=True` translate to `LIMIT_MAKER` and reject if it crosses
  the spread.
* Emit the resolved entry/SL/TP triple to the activity log on every signal,
  matching Reflecto Prophet's signal packaging.
* Respect the Funds allocation: every BUY clips quantity to
  `(managed_usdc / price) * (1 - taker_fee_bps/1e4)` and every SELL clips to
  `managed_btc`. Update the allocation atomically on every fill.

### Phase 2 — enrichment features

Implement a new module `enrichment.py` exposing one function per source from
Part A. Each function returns a small dict of named numeric features. The
training pipeline will join these into the existing `make_rows` output.
Targeted features (initial set):

* `fng_value` (0..100) and `fng_state` one-hot from A1.
* `btc_dom`, `total_mc_change_24h` from A2 global.
* `funding_rate_btcusdt`, `open_interest_btcusdt` from A5.
* `mempool_fast_fee`, `mempool_count` from A4.
* `xchange_dispersion_bps` from A8.
* `usdc_supply_change_7d` from A10 stablecoins.
* `news_event_in_window` (0/1 from A6+A11 keyword match).

Caching: each function persists a JSON snapshot under
`data/enrichment/<source>.json` with a TTL; the training pipeline only
re-fetches if expired. Operator can refresh on demand from a new
`Refresh enrichment` action (Phase 2 ticket).

### Phase 3 — regime-aware feature gating

* Extend `StrategyConfig` with `regime_gates: dict[str, list[str]]` mapping
  Fear-Greed buckets to allowed feature subsets. Default = identity (all
  features always allowed).
* The training pipeline trains separate model heads per regime and the live
  loop selects the head matching the current bucket. Persist as one model
  artifact with sub-models keyed by regime.

### Phase 4 — quality-of-life, observability

* Optional Telegram notifier behind a `Notifier` ABC; default noop. Config
  lives in `Settings -> Notifications`.
* Operator report exports to Markdown.
* Add `Refresh enrichment` and `Show enrichment status` actions.

---

## Part E. Operational rules for the next-iteration engineer

* All new HTTP calls go through a thin wrapper that:
  1. respects per-source TTL caches under `data/enrichment/`.
  2. tags every request with `User-Agent: simple-ai-trading/<version>`.
  3. caps total external HTTP rate at 30 req/min across all enrichment
     sources combined.
  4. fails open: if any source 5xx's or times out, the corresponding
     features are NaN-filled and the model's training is informed.
* Each new selectable parameter is added to `StrategyConfig` or
  `RuntimeConfig`, gets a default that preserves current behavior, and is
  immediately accompanied by an entry in the relevant Settings sub-form
  with `markup=False` to avoid the bracket-as-markup bug fixed in this PR.
* Tests:
  * Unit tests for every new `enrichment.*` function with a mocked
    `requests.Session`. Real HTTP must be opt-in via an env flag.
  * One end-to-end test that loads a stubbed enrichment cache and confirms
    `make_rows` emits the expected feature columns.
  * One TUI smoke test per new modal screen using the existing
    `OperatorApp.run_test` Pilot harness.
* Centralization rules (per the user's directive):
  * All visual focus indicators use the existing `:focus-within` rule on
    `#nav, #action-panel, #snapshot-panel, #activity-panel`. New panels
    must declare the same rule, not a custom border.
  * All status-text writes go through `OperatorApp.set_status` (which
    already de-dupes).
  * All log writes go through `OperatorApp.append_log` /
    `TerminalUI.append_log`, never directly to a `RichLog`.

---

*End of plan. Concrete tickets for Phase 1–3 should be filed as separate
issues; this document is an architectural brief, not a sprint backlog.*
