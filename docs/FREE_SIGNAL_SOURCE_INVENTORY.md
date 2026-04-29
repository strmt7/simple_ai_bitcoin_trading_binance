# Free Signal Source Inventory for BTCUSDC

Last checked: 2026-04-29.

This inventory is for future enrichment of the testnet-first BTCUSDC trader. It
intentionally separates exchange execution from external signal ingestion. A
source appearing here is not a trading recommendation and does not mean it
should be added to live order flow without caching, validation, and tests.

## Integration Rules

1. Keep Binance execution isolated. External sources may enrich features,
   backtests, reports, and veto/risk flags, but they should not bypass the
   BTCUSDC-only and testnet-first contracts.
2. Prefer no-key or generous official APIs first. Keyed free tiers are allowed
   for offline enrichment, but runtime must keep missing keys non-fatal.
3. Cache every external source. Recommended minimum TTLs are listed below; do
   not call social/news/macro endpoints once per live tick.
4. Store source metadata with artifacts: provider, endpoint family, timestamp,
   freshness, and whether the value came from a keyless, keyed, or unofficial
   endpoint.
5. Treat vendor limits as volatile. The table records the limits checked on
   2026-04-28; rate-limit code must still read response headers and back off.
6. Never train on rows that contain future information. Event/news/macro
   values must be joined only by the timestamp at which the signal was known.

## Highest Value First

These sources should be considered before broader integration work because
they are high-signal, cheap, and fit the current repo shape.

| Priority | Source | Auth / free tier | Candidate feature | Cadence | Why it matters |
|---|---|---|---|---|---|
| P0 | Binance Spot REST + WebSocket | Existing no-key public market data; IP weighted limits | closed candles, L1 spread, rolling volume, trade imbalance | REST per fetch; WS later | Same venue as the current BTCUSDC workflow. |
| P0 | Binance USD-M futures public data | No key for mark price, funding, OI | BTCUSDT funding, basis, open-interest delta | 1-5 min | Derivatives positioning often leads spot BTC. |
| P0 | CoinGecko | Public/demo free tier around 30 calls/min | BTC dominance, global cap, exchange dispersion, derivatives OI | 5-30 min | Fast multi-market confirmation without building exchange adapters. |
| P0 | DefiLlama | Free API, no key | stablecoin supply, chain TVL, DEX volume, fees | hourly/daily | Good liquidity/risk-on proxy, especially USDC supply changes. |
| P0 | GDELT DOC/Event/GKG | Free/open | bitcoin headline volume, tone, geopolitical stress terms | 15-60 min | Broad, timestamped event pressure signal with historical replay. |
| P0 | FRED | Free key required | rates, DXY proxies, liquidity, stress spreads | daily/hourly per series | Macro regime features for BTC risk appetite. |
| P0 | mempool.space | No key | fee pressure, mempool size, difficulty adjustment | 1-5 min | Bitcoin-native congestion and volatility proxy. |
| P0 | Alternative.me Fear and Greed | No key | sentiment regime | hourly/daily | Cheap, simple regime gate. |
| P1 | DexScreener | No key; 60 or 300 req/min endpoint limits | DEX liquidity and risk-on/off activity | 5-15 min | Useful altcoin/DEX froth proxy without execution scope creep. |
| P1 | Etherscan V2 | Free key; 3 calls/sec and 100k/day on selected chains | USDC/USDT transfer bursts, exchange wallet flow proxy | 5-30 min | On-chain stablecoin movement can predate volatility. |
| P1 | SEC EDGAR | No key; 10 req/sec with declared UA | ETF/issuer filings and enforcement headlines | 5-30 min | Regulatory event and spot ETF issuer activity. |
| P1 | Hacker News Firebase API | No key; official docs say no current rate limit | crypto/dev attention | 15-60 min | Lightweight tech/community attention proxy. |

## Source Matrix

### Exchange And Market Microstructure

| Source | Auth / limits checked | Relevant endpoints or feeds | Signal value | Suggested use |
|---|---|---|---|---|
| Binance Spot REST | No key for market data. Official docs expose `X-MBX-USED-WEIGHT-*` headers and require backoff on 429/418. | `/api/v3/klines`, `/ticker/24hr`, `/ticker/bookTicker`, `/exchangeInfo` | Native BTCUSDC candles, L1 spread, volume, symbol filters. | Already core. Add spread/volume features and persist request weight. |
| Binance Spot WebSocket | No key for market streams. 5 incoming messages/sec, 1024 streams/connection, 24h reconnect. | `btcusdc@kline_15m`, `btcusdc@trade`, `btcusdc@bookTicker` | Lower latency closed-kline and trade-flow features. | Later streaming module; still drop unclosed klines before feature rows. |
| Binance USD-M Futures | No key for many public endpoints. | `openInterest`, `premiumIndex`, funding, mark price | Futures basis, funding, OI trend. | Add as external confirmation, not execution expansion. |
| Binance announcements | Public web JSON, undocumented. | CMS article list by catalog | Listing, maintenance, BTC-related event risk. | Best-effort veto only; cache and tolerate breakage. |
| Kraken | No key public market data. | ticker, OHLC, spread, order book | Cross-venue price dispersion. | Poll every 30-60s for BTC/USD reference. |
| Bitstamp | No key public ticker/order book. | `/api/v2/ticker/btcusd/`, order book | Cross-venue price and volume. | Reference only. |
| Coinbase | No key public retail/exchange endpoints; exchange APIs have product constraints. | spot price, exchange ticker/book where available | USD spot reference and outage check. | Reference only; do not assume BTCUSDC support. |
| Bitfinex | No key public ticker/book/candles. | `tBTCUSD` ticker/candles | Cross-venue reference. | Reference only. |
| OKX | Public market data available; official limit should be checked before use. | ticker, candles, funding, open interest | Exchange dispersion and derivatives sentiment. | P2 after Binance futures. |
| Bybit | Public market data exists but CDN/API access can be stricter. | ticker, kline, funding, OI | Derivatives confirmation. | P2; require live smoke before adding. |
| KuCoin | Public market data exists with rate pools. | ticker, candles, order book | Cross-venue reference. | P2. |
| Gate.io | Public spot/futures market data. | tickers, order book, funding | Cross-venue and smaller-exchange stress. | P2. |
| Gemini | Public ticker and order book. | BTCUSD ticker/book | Regulated US venue reference. | P2 low cadence. |

### Crypto Aggregators, Indexes, And Calendars

| Source | Auth / limits checked | Relevant data | Signal value | Suggested use |
|---|---|---|---|---|
| CoinGecko | Public/demo free tier around 30 calls/min, variable by traffic. | simple price, market chart, global, exchanges, derivatives, trending, news | BTC dominance, total cap, derivatives OI, trend attention. | P0 aggregator cache. |
| CoinPaprika | Free base URL; 20,000 calls/month, 2,000 assets, limited history. | BTC ticker, events, exchanges, social/project metadata | Independent price, event calendar, market breadth. | P1 daily/hourly cache. |
| CoinCap | V3 docs route through API manager and should be treated as keyed unless verified otherwise. | assets, rates, markets, exchanges | Price/volume redundancy. | P2 only after auth model is clear. |
| CryptoCompare | Free key recommended; some endpoints work without key, news requires key. | price, histohour, indexes, social endpoints on paid tiers | Price redundancy and historical fallback. | P2; avoid news without key. |
| CoinMarketCap | Free Basic key available, attribution and redistribution rules apply. | global metrics, listings, quotes | Market breadth and dominance. | P2 offline enrichment only. |
| LiveCoinWatch | Free key tier exists. | coin prices and history | Redundant market data. | P3 if CoinGecko unavailable. |
| CoinLore | No-key public endpoints historically available. | BTC ticker, global stats | Lightweight fallback. | P3 sanity check only. |
| Alternative.me Fear and Greed | No key; low-frequency index. | sentiment value/classification | Regime gate and risk sizing input. | P0 daily/hourly cache. |
| CoinMarketCal | Free/keyed access; verify current terms. | crypto event calendar | Event risk windows. | P2 daily cache. |
| CoinPaprika events | Included in free/public docs. | coin-specific events | Event-in-next-N-hours feature. | P1. |

### On-Chain, DeFi, And Stablecoin Liquidity

| Source | Auth / limits checked | Relevant data | Signal value | Suggested use |
|---|---|---|---|---|
| DefiLlama Free API | No auth on `api.llama.fi`; separate pro API uses key. | stablecoins, TVL, DEX volume, fees/revenue, open interest | Stablecoin supply and DeFi risk appetite. | P0 hourly/daily. |
| DexScreener | No key; docs list 60 req/min for profiles/ads/boosts and 300 req/min for pairs/search/token pools. | pool liquidity, volume, pair search, boosts | DEX froth and liquidity shifts. | P1, capped well below limits. |
| Etherscan V2 | Free key; 3 calls/sec and 100k/day on selected chains. | token transfers, address balances, gas tracker, logs | USDC/USDT transfer bursts, gas stress. | P1 keyed optional cache. |
| Blockchain.com / blockchain.info | No-key public BTC stats. | difficulty, hash rate, transaction count, market price | Miner/network regime. | P1 daily/hourly. |
| mempool.space | No-key public API, no formal high-throughput guarantee. | recommended fees, mempool, blocks, difficulty adjustment | Congestion and volatility stress. | P0 1-5 min. |
| Blockstream Esplora | No-key public endpoints; rate courtesy required. | blocks, txs, mempool | BTC network fallback. | P2 fallback for mempool.space. |
| Blockchair | Free/keyed tiers vary; verify before runtime. | multi-chain stats, address/transaction metadata | Cross-chain activity. | P3 offline only. |
| Dune | Free public dashboards and API plans; API key usually required. | community SQL metrics | Curated stablecoin/exchange-flow dashboards. | P2 offline snapshots. |
| Flipside | Free community data with API/SDK workflows. | SQL metrics on chain data | Custom stablecoin/exchange features. | P2 research pipeline. |
| The Graph | Decentralized subgraphs; gateway/key details vary. | DEX subgraph data | Protocol-level flow. | P3 after DefiLlama/DexScreener. |
| Glassnode | Free tier limited. | exchange flows, active addresses | Strong but often paid. | P3, only if free tier suffices. |
| CryptoQuant | Mostly paid/community. | exchange reserves, miner flows | Strong but not reliably free. | Track as paid/optional, not default. |
| Whale Alert | Keyed free tier historically limited. | large transfers | Volatility event detector. | Optional, cache aggressively. |

### Macro, Rates, Liquidity, And Traditional Finance

| Source | Auth / limits checked | Relevant data | Signal value | Suggested use |
|---|---|---|---|---|
| FRED | Free API key required for web-service requests. | Fed funds, yields, DXY proxies, M2, SOFR, stress spreads | Macro regime and liquidity. | P0 daily/hourly by release cadence. |
| BLS Public Data API | V1 no registration: 25 queries/day; V2 registered: 500/day; both 50 requests/10s. | CPI, PPI, jobs, wages | Macro event and inflation regime. | P1 daily/monthly; never tick-time. |
| BEA API | Free key; official rate/terms should be checked before implementation. | GDP, PCE, income, trade | Macro regime. | P2 monthly/quarterly. |
| U.S. Treasury FiscalData | Public API. | debt, cash balance, auctions, receipts | Liquidity and fiscal impulse. | P1 daily. |
| Treasury yield curve data | Public files/API. | daily treasury rates | Rates regime. | P1 daily. |
| SEC EDGAR data APIs/RSS | No auth/key; 10 req/sec max and declared user-agent required. | ETF issuer filings, 8-K, S-1, enforcement releases | Regulatory and institutional event signals. | P1 event watcher. |
| CFTC COT | Public weekly reports. | futures positioning | Risk appetite/positioning. | P2 weekly. |
| Federal Reserve RSS/speeches/calendar | Public web/RSS. | policy speeches, press releases | Event risk. | P2 headline/event features. |
| ECB Data Portal / SDW | Public API. | EUR rates, liquidity, EURUSD macro | Global liquidity backdrop. | P2 daily. |
| World Bank API | Public data API under terms. | macro country indicators | Low-frequency macro. | P3 research only. |
| IMF Data | Public portal/API access varies by dataset. | global liquidity, FX reserves | Macro backdrop. | P3 research only. |
| OECD API | Public statistical data. | leading indicators | Low-frequency macro. | P3 research only. |
| Alpha Vantage | Free key; 25 requests/day. | FX, equities, commodities, technical indicators, news sentiment | Cheap keyed enrichment, but tiny free quota. | P2 offline only. |
| Twelve Data | Free Basic: 8 API credits/min and 800/day. | crypto, FX, equities, indicators | Cross-asset/indicator enrichment. | P2 with strict quota manager. |
| Nasdaq Data Link | Some free datasets; key/account for API. | macro/finance datasets | Research features. | P3. |
| Stooq | Free CSV endpoints, unofficial support. | FX, indexes, equities | Fallback cross-asset prices. | P3, not critical runtime. |

### News, Events, Social, And Developer Attention

| Source | Auth / limits checked | Relevant data | Signal value | Suggested use |
|---|---|---|---|---|
| GDELT DOC 2.0 / Event / GKG | Free/open. GDELT 2.0 updates every 15 minutes for core datasets. | global news volume, tone, themes, locations, event codes | Bitcoin news pressure and geopolitical stress. | P0, timestamped features. |
| Hacker News Firebase API | No key; official docs say no current rate limit. | top/new/best stories, item metadata | Tech/dev attention spikes. | P1 low cadence. |
| Reddit API | Official API docs exist; use OAuth and rate-limit headers for robust use. | r/Bitcoin, r/CryptoCurrency, subreddit activity | Retail/social attention. | P2, optional OAuth. |
| Reddit public JSON/RSS | Unofficial/less stable behavior. | subreddit hot/new headlines | Backup social attention. | Avoid as default; use only with strict cache. |
| CoinDesk RSS | Public RSS. | crypto news headlines | Keyword/event feature. | P1, cache 15-60 min. |
| The Block / Bitcoin Magazine / Decrypt RSS | Public feed availability varies. | crypto headlines | Redundant news source. | P2 if terms permit. |
| SEC RSS feeds | Public and official. | filings, rules, enforcement | Regulatory event features. | P1. |
| GitHub REST API | Public unauthenticated quota, higher with token. | stars, releases, issues for BTC/crypto repos | Developer attention/security events. | P2 daily. |
| GitHub Advisory Database | Public GraphQL/API with token for scale. | crypto dependency/security events | Infrastructure risk. | P3. |
| CryptoPanic | Auth token required. | curated crypto news/sentiment | Strong signal, not zero-friction. | Optional keyed P2. |
| NewsAPI | Free developer key, licensing/reuse restrictions. | headline search | General news fallback. | Research/offline only. |
| GNews | Free keyed tier. | headline search | News fallback. | Research/offline only. |
| The Guardian Open Platform | Free key for Guardian content. | finance/crypto headlines | Licensed news source. | P3. |
| NYT Developer API | Free key with limits. | business/regulatory headlines | Licensed news source. | P3. |
| Wikipedia / Wikimedia Pageviews | No-key public endpoints. | Bitcoin pageview spikes | Retail attention proxy. | P2 daily/hourly. |
| Google Trends | No official public API. | search interest | Useful but avoid unofficial runtime dependency. | Manual/research only unless official path chosen. |

## Suggested Feature Groups

| Group | Inputs | Example features | Notes |
|---|---|---|---|
| Venue quality | Binance bookTicker, trades, klines | spread_bps, quote_volume_z, taker_flow_proxy | Start here; native to current app. |
| Derivatives pressure | Binance futures, CoinGecko derivatives, DefiLlama OI | funding_rate, mark_basis, oi_change_1h | Useful for spot BTC even if execution stays spot/testnet. |
| Liquidity regime | DefiLlama stablecoins, FRED/Treasury, DXY proxy | usdc_supply_delta, fed_rate_level, tga_delta | Join at known timestamps only. |
| Network stress | mempool.space, Blockchain.com | fee_zscore, mempool_vsize, difficulty_change | BTC-native, cheap. |
| News/event pressure | GDELT, SEC, CoinPaprika events, Binance announcements | headline_count_1h, negative_tone_z, event_window_flag | Use as sizing/veto first, not directional alpha. |
| Social/dev attention | HN, Reddit, GitHub, Wikimedia | mention_count_z, pageview_z, repo_activity_z | High noise; smooth heavily. |

## Implementation Backlog

Implemented in this prototype:

- `data-sync` stores Binance closed candles, 24h ticker, L1 book ticker, USD-M
  premium index, open interest, and funding history in SQLite with the existing
  rate-limited Binance client.
- `signals` and `live --external-signals` cache and blend Alternative.me,
  CoinGecko, Binance futures positioning, and mempool.space into a bounded
  score/risk adjustment with provider quorum checks.
- `train --source auto|db --download-missing` can train from the SQLite store
  and prompt for a missing backfill when the selected interval lacks enough
  rows.

Remaining backlog:

1. Add a source registry module with provider name, endpoint family, TTL,
   auth mode, and parser contract.
2. Persist non-Binance raw provider snapshots under `data/signals/<provider>/YYYYMMDD/`
   with atomic writes and redacted config metadata.
3. Build a timestamp-safe joiner that only uses values whose `known_at_ms` is
   at or before the candle close time.
4. Add offline feature columns behind explicit feature names. Keep them disabled
   by default until backtest coverage exists.
5. Add richer `signals` subcommands for per-provider freshness, cache paths,
   and any rate-limit/backoff events.
6. Only after offline tests pass, allow `backtest` to read cached
   external features. Runtime should never block on a slow news/social API.

## Sources Checked

- Binance Spot API limits: https://developers.binance.com/docs/binance-spot-api-docs/rest-api/limits
- Binance Spot WebSocket streams: https://developers.binance.com/docs/binance-spot-api-docs/web-socket-streams
- Binance USD-M futures market data: https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Open-Interest
- CoinGecko rate limits and endpoint overview: https://docs.coingecko.com/docs/common-errors-rate-limit and https://docs.coingecko.com/reference/endpoint-overview
- CoinPaprika docs and plans: https://docs.coinpaprika.com/ and https://docs.coinpaprika.com/api-plans
- DefiLlama API docs: https://api-docs.defillama.com/
- DexScreener API reference: https://docs.dexscreener.com/api/reference
- Etherscan V2 rate limits: https://docs.etherscan.io/resources/rate-limits
- GDELT data and API documentation: https://www.gdeltproject.org/data.html and https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/
- FRED API and API-key docs: https://fred.stlouisfed.org/docs/api/fred/ and https://fred.stlouisfed.org/docs/api/api_key.html
- BLS Public Data API and FAQ: https://www.bls.gov/developers/ and https://www.bls.gov/developers/api_faqs.htm
- SEC EDGAR APIs and fair-access guidance: https://www.sec.gov/search-filings/edgar-application-programming-interfaces and https://www.sec.gov/about/webmaster-frequently-asked-questions
- Alpha Vantage support/pricing: https://www.alphavantage.co/support/ and https://www.alphavantage.co/premium/
- Twelve Data pricing/credits: https://twelvedata.com/pricing
- Hacker News Firebase API: https://github.com/HackerNews/API
- Reddit API documentation: https://www.reddit.com/dev/api/
