"""Cached free external signal aggregation for live BTCUSDC decisions."""

from __future__ import annotations

import json
import logging
import math
import random
import re
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from functools import lru_cache
from html import unescape
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Mapping
from urllib.parse import urlparse

from defusedxml import ElementTree as ET
import requests

from .compute import BackendInfo, resolve_backend
from .storage import write_json_atomic


FetchJson = Callable[[str, float], object]
FetchText = Callable[[str, float], str]
PostJson = Callable[[str, Mapping[str, object], float], object]
_LOGGER = logging.getLogger(__name__)
_JITTER_RANDOM = random.SystemRandom()

COINGECKO_SIMPLE_PRICE_URL = (
    "https://api.coingecko.com/api/v3/simple/price"
    "?ids=bitcoin&vs_currencies=usd&include_24hr_change=true&include_24hr_vol=true"
    "&include_last_updated_at=true"
)
COINPAPRIKA_BTC_TICKER_URL = "https://api.coinpaprika.com/v1/tickers/btc-bitcoin"
COINLORE_BTC_TICKER_URL = "https://api.coinlore.net/api/ticker/?id=90"
BLOCKCHAIN_STATS_URL = "https://api.blockchain.info/stats"
KRAKEN_BTCUSD_TICKER_URL = "https://api.kraken.com/0/public/Ticker?pair=XBTUSD"
COINBASE_BTCUSD_STATS_URL = "https://api.exchange.coinbase.com/products/BTC-USD/stats"
BITSTAMP_BTCUSD_TICKER_URL = "https://www.bitstamp.net/api/v2/ticker/btcusd/"
BINANCE_SPOT_BASE_URL = "https://api.binance.com"
ALTERNATIVE_FNG_URL = "https://api.alternative.me/fng/?limit=1&format=json"
MEMPOOL_FEES_URL = "https://mempool.space/api/v1/fees/recommended"
BINANCE_FUTURES_BASE_URL = "https://fapi.binance.com"
CRYPTOCOMPARE_NEWS_URL = "https://min-api.cryptocompare.com/data/v2/news/?lang=EN&categories=BTC"
GDELT_BITCOIN_NEWS_URL = (
    "https://api.gdeltproject.org/api/v2/doc/doc"
    "?query=(bitcoin%20OR%20BTC)%20(sourceCountry:US%20OR%20sourceCountry:GB)"
    "&mode=ArtList&format=json&maxrecords=12&sort=HybridRel"
)
HACKER_NEWS_BITCOIN_URL = "https://hn.algolia.com/api/v1/search?query=bitcoin&tags=story&hitsPerPage=10"
DEFAULT_OLLAMA_NEWS_MODEL = "gemma4:e4b"
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
_OLLAMA_NEWS_MAX_ITEMS = 6
_OLLAMA_NEWS_MAX_TEXT_CHARS = 140
_OLLAMA_NEWS_NUM_CTX = 512
_OLLAMA_NEWS_NUM_PREDICT = 48
_OLLAMA_NEWS_JSON_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "score": {"type": "number"},
        "horizon": {"type": "string", "enum": ["short", "medium", "long"]},
        "reaction_required": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["score", "horizon", "reaction_required", "reason"],
    "additionalProperties": False,
}
_RSS_SAME_HOST_GAP_SECONDS = 0.12
_MAX_FUTURE_TIMESTAMP_SKEW_MS = 5 * 60_000
_CACHE_COMPONENT_MAX_AGE_MS_BY_HORIZON = {
    "short": 7 * 24 * 3_600_000,
    "medium": 30 * 24 * 3_600_000,
    "long": 365 * 24 * 3_600_000,
}
_HTTP_RETRY_STATUS_CODES = {429, 500, 502, 503, 504}
_HTTP_MAX_ATTEMPTS = 3


@dataclass(frozen=True)
class NewsFeedProvider:
    provider: str
    url: str
    weight: float = 0.25
    horizon: str = "medium"
    scope: str = "crypto"


RSS_NEWS_FEEDS: tuple[NewsFeedProvider, ...] = (
    NewsFeedProvider("bitcoin_magazine", "https://bitcoinmagazine.com/.rss/full/", 0.30, "medium"),
    NewsFeedProvider("cointelegraph", "https://cointelegraph.com/rss", 0.32, "medium"),
    NewsFeedProvider("coindesk", "https://www.coindesk.com/arc/outboundfeeds/rss/", 0.32, "medium"),
    NewsFeedProvider("decrypt", "https://decrypt.co/feed", 0.28, "medium"),
    NewsFeedProvider("the_block", "https://www.theblock.co/rss.xml", 0.30, "medium"),
    NewsFeedProvider("cryptoslate", "https://cryptoslate.com/feed/", 0.24, "medium"),
    NewsFeedProvider("newsbtc", "https://www.newsbtc.com/feed/", 0.20, "medium"),
    NewsFeedProvider("bitcoinist", "https://bitcoinist.com/feed/", 0.20, "medium"),
    NewsFeedProvider("ambcrypto", "https://ambcrypto.com/feed/", 0.18, "medium"),
    NewsFeedProvider("beincrypto", "https://beincrypto.com/feed/", 0.22, "medium"),
    NewsFeedProvider("u_today", "https://u.today/rss", 0.18, "medium"),
    NewsFeedProvider("cryptonews", "https://cryptonews.com/news/feed/", 0.22, "medium"),
    NewsFeedProvider("blockworks", "https://blockworks.co/feed", 0.24, "medium"),
    NewsFeedProvider("crypto_briefing", "https://cryptobriefing.com/feed/", 0.22, "medium"),
    NewsFeedProvider("bitcoin_com_news", "https://news.bitcoin.com/feed/", 0.20, "medium"),
    NewsFeedProvider("cryptopotato", "https://cryptopotato.com/feed/", 0.20, "medium"),
    NewsFeedProvider("dailyhodl", "https://dailyhodl.com/feed/", 0.18, "medium"),
    NewsFeedProvider("coinpedia", "https://coinpedia.org/feed/", 0.18, "medium"),
    NewsFeedProvider("cryptopolitan", "https://www.cryptopolitan.com/feed/", 0.18, "medium"),
    NewsFeedProvider("livebitcoinnews", "https://www.livebitcoinnews.com/feed/", 0.18, "medium"),
    NewsFeedProvider("bitcoin_core_releases", "https://github.com/bitcoin/bitcoin/releases.atom", 0.18, "long"),
    NewsFeedProvider("cnbc_finance", "https://www.cnbc.com/id/10000664/device/rss/rss.html", 0.20, "medium", "macro"),
    NewsFeedProvider("nasdaq_crypto", "https://www.nasdaq.com/feed/rssoutbound?category=Cryptocurrency", 0.18, "medium"),
    NewsFeedProvider("investing_crypto", "https://www.investing.com/rss/news_301.rss", 0.18, "medium"),
    NewsFeedProvider("forexlive", "https://www.forexlive.com/feed/news", 0.16, "short", "macro"),
    NewsFeedProvider("marketwatch_markets", "https://feeds.content.dowjones.io/public/rss/mw_marketpulse", 0.18, "short", "macro"),
    NewsFeedProvider("bloomberg_markets", "https://feeds.bloomberg.com/markets/news.rss", 0.20, "medium", "macro"),
    NewsFeedProvider("marketwatch_top", "https://www.marketwatch.com/rss/topstories", 0.16, "medium", "macro"),
    NewsFeedProvider("bbc_business", "https://feeds.bbci.co.uk/news/business/rss.xml", 0.16, "medium", "macro"),
    NewsFeedProvider("bbc_world", "https://feeds.bbci.co.uk/news/world/rss.xml", 0.16, "medium", "geopolitical"),
    NewsFeedProvider("guardian_business", "https://www.theguardian.com/business/rss", 0.14, "medium", "macro"),
    NewsFeedProvider("guardian_world", "https://www.theguardian.com/world/rss", 0.14, "medium", "geopolitical"),
    NewsFeedProvider("aljazeera_world", "https://www.aljazeera.com/xml/rss/all.xml", 0.14, "medium", "geopolitical"),
    NewsFeedProvider("federal_reserve", "https://www.federalreserve.gov/feeds/press_all.xml", 0.28, "medium", "macro"),
    NewsFeedProvider("cftc_press", "https://www.cftc.gov/RSS/RSSGP/rssgp.xml", 0.24, "medium", "regulatory"),
    NewsFeedProvider("dowjones_markets", "https://feeds.a.dj.com/rss/RSSMarketsMain.xml", 0.18, "medium", "macro"),
    NewsFeedProvider("npr_business", "https://www.npr.org/rss/rss.php?id=1006", 0.14, "medium", "macro"),
    NewsFeedProvider("npr_world", "https://www.npr.org/rss/rss.php?id=1004", 0.14, "medium", "geopolitical"),
    NewsFeedProvider("nytimes_business", "https://rss.nytimes.com/services/xml/rss/nyt/Business.xml", 0.16, "medium", "macro"),
    NewsFeedProvider("nytimes_world", "https://rss.nytimes.com/services/xml/rss/nyt/World.xml", 0.14, "medium", "geopolitical"),
    NewsFeedProvider("nytimes_technology", "https://rss.nytimes.com/services/xml/rss/nyt/Technology.xml", 0.14, "medium", "technology"),
    NewsFeedProvider("axios_news", "https://www.axios.com/feeds/feed.rss", 0.14, "medium", "geopolitical"),
    NewsFeedProvider("fxstreet_news", "https://www.fxstreet.com/rss/news", 0.16, "short", "macro"),
    NewsFeedProvider("coinjournal", "https://coinjournal.net/news/feed/", 0.18, "medium"),
    NewsFeedProvider("kraken_blog", "https://blog.kraken.com/feed", 0.16, "long"),
    NewsFeedProvider("bitfinex_blog", "https://blog.bitfinex.com/feed/", 0.14, "long"),
    NewsFeedProvider("bitmex_blog", "https://blog.bitmex.com/feed/", 0.14, "long"),
    NewsFeedProvider("chainalysis_blog", "https://www.chainalysis.com/blog/feed/", 0.16, "long", "regulatory"),
    NewsFeedProvider("elliptic_blog", "https://www.elliptic.co/blog/rss.xml", 0.16, "long", "regulatory"),
    NewsFeedProvider("glassnode_insights", "https://insights.glassnode.com/rss/", 0.18, "medium"),
    NewsFeedProvider("bitcoin_core_blog", "https://bitcoincore.org/en/feed.xml", 0.20, "long", "technology"),
    NewsFeedProvider("bitcoin_core_commits", "https://github.com/bitcoin/bitcoin/commits/master.atom", 0.12, "long", "technology"),
    NewsFeedProvider("bitcoin_bips_commits", "https://github.com/bitcoin/bips/commits/master.atom", 0.12, "long", "technology"),
    NewsFeedProvider("lightning_lnd_releases", "https://github.com/lightningnetwork/lnd/releases.atom", 0.12, "long", "technology"),
    NewsFeedProvider("lightning_core_releases", "https://github.com/ElementsProject/lightning/releases.atom", 0.12, "long", "technology"),
    NewsFeedProvider("eclair_releases", "https://github.com/ACINQ/eclair/releases.atom", 0.12, "long", "technology"),
    NewsFeedProvider("rust_bitcoin_releases", "https://github.com/rust-bitcoin/rust-bitcoin/releases.atom", 0.12, "long", "technology"),
    NewsFeedProvider("go_ethereum_releases", "https://github.com/ethereum/go-ethereum/releases.atom", 0.10, "long", "technology"),
    NewsFeedProvider("ethereum_consensus_releases", "https://github.com/ethereum/consensus-specs/releases.atom", 0.10, "long", "technology"),
    NewsFeedProvider("prysm_releases", "https://github.com/prysmaticlabs/prysm/releases.atom", 0.10, "long", "technology"),
    NewsFeedProvider("lighthouse_releases", "https://github.com/sigp/lighthouse/releases.atom", 0.10, "long", "technology"),
    NewsFeedProvider("flashbots_mev_boost", "https://github.com/flashbots/mev-boost/releases.atom", 0.10, "long", "technology"),
    NewsFeedProvider("ethereum_blog", "https://blog.ethereum.org/feed.xml", 0.10, "long", "technology"),
    NewsFeedProvider("ethresearch", "https://ethresear.ch/latest.rss", 0.08, "long", "technology"),
    NewsFeedProvider("makerdao_forum", "https://forum.makerdao.com/latest.rss", 0.08, "medium", "macro"),
    NewsFeedProvider("aave_governance", "https://governance.aave.com/latest.rss", 0.08, "medium", "macro"),
    NewsFeedProvider("blockchain_news", "https://blockchain.news/RSS/", 0.16, "medium"),
    NewsFeedProvider("coincu", "https://coincu.com/feed/", 0.14, "medium"),
    NewsFeedProvider("cointribune_en", "https://www.cointribune.com/en/feed/", 0.14, "medium"),
    NewsFeedProvider("bitcoin_ops", "https://bitcoinops.org/feed.xml", 0.18, "long", "technology"),
    NewsFeedProvider("blockstream_blog", "https://blog.blockstream.com/rss/", 0.16, "long", "technology"),
    NewsFeedProvider("bravenewcoin", "https://bravenewcoin.com/news/rss", 0.16, "medium"),
    NewsFeedProvider("coingecko_news", "https://www.coingecko.com/buzz.rss", 0.16, "medium"),
    NewsFeedProvider("cryptodaily", "https://cryptodaily.co.uk/feed", 0.14, "medium"),
    NewsFeedProvider("finbold", "https://finbold.com/feed/", 0.16, "medium", "macro"),
    NewsFeedProvider("protos", "https://protos.com/feed/", 0.16, "medium", "regulatory"),
    NewsFeedProvider("unchained_crypto", "https://unchainedcrypto.com/feed/", 0.16, "medium"),
    NewsFeedProvider("coinshares_blog", "https://blog.coinshares.com/rss/", 0.16, "medium", "macro"),
    NewsFeedProvider("arbitrum_blog", "https://blog.arbitrum.io/rss/", 0.10, "long", "technology"),
    NewsFeedProvider("trezor_blog", "https://blog.trezor.io/feed", 0.12, "long", "security"),
    NewsFeedProvider("bank_england_news", "https://www.bankofengland.co.uk/rss/news", 0.18, "medium", "macro"),
    NewsFeedProvider("fca_news", "https://www.fca.org.uk/news/rss.xml", 0.18, "medium", "regulatory"),
    NewsFeedProvider("bleepingcomputer", "https://www.bleepingcomputer.com/feed/", 0.14, "short", "security"),
    NewsFeedProvider("krebsonsecurity", "https://krebsonsecurity.com/feed/", 0.14, "short", "security"),
    NewsFeedProvider("thehackernews", "https://thehackernews.com/feeds/posts/default", 0.14, "short", "security"),
    NewsFeedProvider("securityweek", "https://www.securityweek.com/feed/", 0.14, "short", "security"),
    NewsFeedProvider("dw_world", "https://rss.dw.com/rdf/rss-en-world", 0.12, "medium", "geopolitical"),
    NewsFeedProvider("france24_world", "https://www.france24.com/en/rss", 0.12, "medium", "geopolitical"),
    NewsFeedProvider("skynews_world", "https://feeds.skynews.com/feeds/rss/world.xml", 0.12, "medium", "geopolitical"),
    NewsFeedProvider("abc_world", "https://abcnews.go.com/abcnews/internationalheadlines", 0.12, "medium", "geopolitical"),
    NewsFeedProvider("cbs_world", "https://www.cbsnews.com/latest/rss/world", 0.12, "medium", "geopolitical"),
    NewsFeedProvider("nbc_world", "https://feeds.nbcnews.com/nbcnews/public/world", 0.12, "medium", "geopolitical"),
    NewsFeedProvider("dowjones_world", "https://feeds.a.dj.com/rss/RSSWorldNews.xml", 0.12, "medium", "geopolitical"),
)


@dataclass(frozen=True)
class ProviderFetchResult:
    component: ExternalSignalComponent
    backend: BackendInfo | None = None
    news_texts: tuple[str, ...] = ()
    news_known_at_ms: tuple[int, ...] = ()
    raw_payload: object | None = None


class _RSSFeedPayloadError(ValueError):
    def __init__(self, message: str, raw_xml: str) -> None:
        super().__init__(message)
        self.raw_xml = raw_xml


@dataclass(frozen=True)
class OllamaNewsEvaluation:
    component: ExternalSignalComponent
    status: str
    model: str
    latency_ms: int
    reason: str
    raw_payload: object | None = None


@dataclass(frozen=True)
class NewsTrigger:
    term: str
    polarity: float
    importance: float
    urgency: float
    category: str
    horizon: str = "medium"


@dataclass(frozen=True)
class NewsClassification:
    score: float
    positive_weight: float
    negative_weight: float
    importance: int
    urgency: float
    horizon: str
    category: str
    matched_terms: tuple[str, ...]

    def asdict(self) -> dict[str, object]:
        return asdict(self)


_NEWS_TRIGGERS: tuple[NewsTrigger, ...] = (
    NewsTrigger("spot bitcoin etf approval", 1.0, 10.0, 0.90, "regulatory", "short"),
    NewsTrigger("bitcoin etf approval", 1.0, 9.0, 0.88, "regulatory", "short"),
    NewsTrigger("etf approval", 1.0, 8.5, 0.84, "regulatory", "short"),
    NewsTrigger("etf inflow", 1.0, 7.0, 0.72, "liquidity", "short"),
    NewsTrigger("institutional adoption", 1.0, 7.0, 0.55, "adoption", "medium"),
    NewsTrigger("institutional buying", 1.0, 7.0, 0.62, "liquidity", "short"),
    NewsTrigger("reserve demand", 1.0, 6.0, 0.52, "adoption", "medium"),
    NewsTrigger("upgrade", 0.7, 4.5, 0.30, "technology", "long"),
    NewsTrigger("breakout", 0.7, 5.0, 0.55, "market", "short"),
    NewsTrigger("rally", 0.6, 4.5, 0.45, "market", "short"),
    NewsTrigger("fed rate cut", 0.7, 7.0, 0.70, "macro", "short"),
    NewsTrigger("sec approval", 0.9, 8.5, 0.84, "regulatory", "short"),
    NewsTrigger("blackrock inflow", 0.8, 7.0, 0.72, "liquidity", "short"),
    NewsTrigger("etf denial", -1.0, 9.0, 0.88, "regulatory", "short"),
    NewsTrigger("etf rejection", -1.0, 9.0, 0.88, "regulatory", "short"),
    NewsTrigger("approval denied", -1.0, 8.5, 0.86, "regulatory", "short"),
    NewsTrigger("rejected etf", -1.0, 8.5, 0.86, "regulatory", "short"),
    NewsTrigger("hack", -1.0, 9.0, 0.92, "security", "short"),
    NewsTrigger("exploit", -1.0, 9.0, 0.92, "security", "short"),
    NewsTrigger("exchange exploit", -1.0, 9.5, 0.95, "security", "short"),
    NewsTrigger("exchange breach", -1.0, 9.5, 0.95, "security", "short"),
    NewsTrigger("security breach", -1.0, 9.0, 0.92, "security", "short"),
    NewsTrigger("data breach", -0.9, 8.5, 0.86, "security", "short"),
    NewsTrigger("bankruptcy", -1.0, 9.0, 0.90, "counterparty", "short"),
    NewsTrigger("liquidation", -0.9, 7.5, 0.82, "market", "short"),
    NewsTrigger("selloff", -0.9, 7.0, 0.78, "market", "short"),
    NewsTrigger("outflow", -0.7, 6.5, 0.62, "liquidity", "short"),
    NewsTrigger("ban", -1.0, 9.0, 0.92, "regulatory", "short"),
    NewsTrigger("crackdown", -1.0, 8.5, 0.86, "regulatory", "short"),
    NewsTrigger("lawsuit", -0.8, 7.0, 0.74, "regulatory", "short"),
    NewsTrigger("sec lawsuit", -0.9, 8.0, 0.80, "regulatory", "short"),
    NewsTrigger("fraud", -1.0, 9.0, 0.90, "counterparty", "short"),
    NewsTrigger("outage", -0.7, 6.5, 0.76, "infrastructure", "short"),
    NewsTrigger("fed rate hike", -0.8, 8.0, 0.82, "macro", "short"),
    NewsTrigger("hawkish fed", -0.7, 7.0, 0.70, "macro", "short"),
    NewsTrigger("war", -0.6, 7.0, 0.68, "geopolitical", "medium"),
    NewsTrigger("sanctions", -0.6, 7.0, 0.68, "geopolitical", "medium"),
    NewsTrigger("bitcoin core release", 0.5, 4.5, 0.20, "technology", "long"),
)

_NEGATION_TERMS = (
    "avoid",
    "avoids",
    "avoided",
    "contained",
    "denies",
    "fake",
    "false",
    "mitigated",
    "no",
    "not",
    "resolved",
    "without",
)

_POSITIVE_NEWS_TERMS = (
    "adoption",
    "approval",
    "breakout",
    "bull",
    "buying",
    "etf inflow",
    "institutional",
    "rally",
    "reserve",
    "upgrade",
)
_NEGATIVE_NEWS_TERMS = (
    "ban",
    "bankruptcy",
    "breach",
    "crackdown",
    "exploit",
    "fraud",
    "hack",
    "lawsuit",
    "liquidation",
    "outflow",
    "selloff",
)
_SHORT_TERM_TERMS = (
    "approval",
    "ban",
    "bankruptcy",
    "breach",
    "breaking",
    "crackdown",
    "etf inflow",
    "exploit",
    "fraud",
    "hack",
    "lawsuit",
    "liquidation",
    "outage",
    "reserve",
    "sec",
)


@dataclass(frozen=True)
class ExternalSignalComponent:
    provider: str
    status: str
    score: float
    weight: float
    value: float | None
    detail: str
    known_at_ms: int
    source_symbol: str = ""
    error: str = ""
    cached: bool = False
    horizon: str = "medium"
    urgency: float = 0.0
    source_grade: int | None = None
    source_grade_model: str = ""
    source_grade_weight_multiplier: float = 1.0

    def asdict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ExternalSignalReport:
    status: str
    score_adjustment: float
    raw_score: float
    risk_multiplier: float
    provider_count: int
    fresh_count: int
    stale_count: int
    known_at_ms: int
    cache_path: str
    warnings: list[str]
    components: list[ExternalSignalComponent]
    short_term_score: float = 0.0
    medium_term_score: float = 0.0
    long_term_score: float = 0.0
    reaction_required: bool = False
    reaction_reason: str = ""
    news_backend_requested: str = "cpu"
    news_backend_kind: str = "cpu"
    news_backend_device: str = "cpu"
    news_backend_reason: str = ""
    news_ai_enabled: bool = False
    news_ai_status: str = "disabled"
    news_ai_model: str = DEFAULT_OLLAMA_NEWS_MODEL
    news_ai_latency_ms: int = 0
    news_ai_reason: str = ""

    def asdict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["components"] = [component.asdict() for component in self.components]
        return payload


def _now_ms() -> int:
    return int(time.time() * 1000)


def _clamp(value: float, low: float, high: float) -> float:
    if value < low:
        return low
    if value > high:
        return high
    return value


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(parsed):
        return default
    return parsed


def _timestamp_is_future(timestamp_ms: int, now_ms: int) -> bool:
    return int(timestamp_ms) > int(now_ms) + _MAX_FUTURE_TIMESTAMP_SKEW_MS


def _bounded_provider_timestamp_ms(timestamp_ms: int, now_ms: int) -> int:
    timestamp = int(timestamp_ms)
    if timestamp <= 0:
        return int(now_ms)
    if _timestamp_is_future(timestamp, now_ms):
        return 0
    if timestamp > int(now_ms):
        return int(now_ms)
    return timestamp


def _cache_component_max_age_ms(horizon: str) -> int:
    return _CACHE_COMPONENT_MAX_AGE_MS_BY_HORIZON.get(
        str(horizon or "medium"),
        _CACHE_COMPONENT_MAX_AGE_MS_BY_HORIZON["medium"],
    )


def _component_cache_timestamp_is_fresh(
    component: ExternalSignalComponent,
    *,
    now_ms: int,
    report_known_at_ms: int,
) -> bool:
    known_at = int(component.known_at_ms)
    if known_at <= 0 or _timestamp_is_future(known_at, now_ms):
        return False
    if known_at > int(report_known_at_ms) + _MAX_FUTURE_TIMESTAMP_SKEW_MS:
        return False
    return int(report_known_at_ms) - known_at <= _cache_component_max_age_ms(component.horizon)


@lru_cache(maxsize=256)
def _term_pattern(term: str) -> re.Pattern[str]:
    escaped = re.escape(term.lower()).replace(r"\ ", r"\s+")
    return re.compile(rf"(?<![a-z0-9]){escaped}(?![a-z0-9])")


def _match_negated(text: str, match_start: int) -> bool:
    prefix = text[max(0, match_start - 42): match_start]
    words = re.findall(r"[a-z]+", prefix.lower())
    return any(word in _NEGATION_TERMS for word in words[-5:])


def _recency_factor(age_ms: int) -> float:
    age_minutes = max(0.0, age_ms / 60_000.0)
    if age_minutes <= 15.0:
        return 1.0
    if age_minutes <= 180.0:
        return 0.82
    if age_minutes <= 1_440.0:
        return 0.58
    return 0.28


def _classify_news_text(text: str, *, age_ms: int = 0) -> NewsClassification:
    normalized = f" {text.lower()} "
    positive = 0.0
    negative = 0.0
    urgency = 0.0
    matched_terms: list[str] = []
    category_weights: dict[str, float] = {}
    horizon_weights: dict[str, float] = {}
    recency = _recency_factor(age_ms)

    for trigger in _NEWS_TRIGGERS:
        match = _term_pattern(trigger.term).search(normalized)
        if match is None:
            continue
        negated = _match_negated(normalized, match.start())
        if negated and trigger.polarity > 0.0:
            continue
        weight = float(trigger.importance)
        polarity = float(trigger.polarity)
        if negated and polarity < 0.0:
            weight *= 0.25
            polarity = 0.0
        if polarity > 0.0:
            positive += weight * polarity
        elif polarity < 0.0:
            negative += weight * abs(polarity)
        matched_terms.append(trigger.term)
        category_weights[trigger.category] = category_weights.get(trigger.category, 0.0) + weight
        horizon_weights[trigger.horizon] = horizon_weights.get(trigger.horizon, 0.0) + weight
        urgency = max(urgency, float(trigger.urgency) * recency)

    total = positive + negative
    score = 0.0 if total <= 0.0 else _clamp((positive - negative) / total, -1.0, 1.0)
    top_weight = max(category_weights.values(), default=0.0)
    density_bonus = min(1.5, max(0, len(matched_terms) - 1) * 0.35)
    importance = int(round(_clamp((top_weight + density_bonus) * recency, 0.0, 10.0)))
    urgency = _clamp(max(urgency, 0.10 if matched_terms else 0.0) + abs(score) * 0.12 + importance / 100.0, 0.0, 1.0)
    category = max(category_weights.items(), key=lambda item: item[1], default=("general", 0.0))[0]
    if urgency >= 0.78 or (importance >= 7 and horizon_weights.get("short", 0.0) > 0.0):
        horizon = "short"
    elif horizon_weights and max(horizon_weights, key=horizon_weights.get) == "long" and age_ms >= 86_400_000:
        horizon = "long"
    elif age_ms > 86_400_000 and not horizon_weights.get("short"):
        horizon = "long"
    else:
        horizon = "medium"
    return NewsClassification(
        score=score,
        positive_weight=float(positive),
        negative_weight=float(negative),
        importance=importance,
        urgency=urgency,
        horizon=horizon,
        category=category,
        matched_terms=tuple(matched_terms),
    )


def _keyword_sentiment(text: str) -> float:
    return _classify_news_text(text).score


def _keyword_counts(text: str, *, age_ms: int = 0) -> tuple[float, float]:
    classification = _classify_news_text(text, age_ms=age_ms)
    if classification.positive_weight or classification.negative_weight:
        return classification.positive_weight, classification.negative_weight
    normalized = f" {text.lower()} "
    positive = float(sum(1 for term in _POSITIVE_NEWS_TERMS if term in normalized))
    negative = float(sum(1 for term in _NEGATIVE_NEWS_TERMS if term in normalized))
    return positive, negative


def _news_urgency(text: str, *, age_ms: int) -> float:
    classification = _classify_news_text(text, age_ms=age_ms)
    if classification.matched_terms:
        return classification.urgency
    normalized = f" {text.lower()} "
    term_hit = any(term in normalized for term in _SHORT_TERM_TERMS)
    return _clamp((0.65 if term_hit else 0.15) + _recency_factor(age_ms) * 0.35, 0.0, 1.0)


def _news_horizon(text: str, *, age_ms: int) -> str:
    classification = _classify_news_text(text, age_ms=age_ms)
    if classification.matched_terms:
        return classification.horizon
    urgency = _news_urgency(text, age_ms=age_ms)
    if urgency >= 0.75:
        return "short"
    if age_ms <= 86_400_000:
        return "medium"
    return "long"


def _torch_device_for_backend(backend: BackendInfo):  # pragma: no cover - optional GPU runtime
    if backend.kind == "directml":
        import torch_directml  # type: ignore

        return torch_directml.device()
    return backend.device


def _score_news_texts(
    texts: list[str],
    *,
    compute_backend: str | None = None,
    ages_ms: list[int] | None = None,
) -> tuple[list[float], BackendInfo]:
    if ages_ms is None:
        ages = [0 for _text in texts]
    else:
        ages = [max(0, int(age)) for age in ages_ms[: len(texts)]]
        ages.extend([0] * max(0, len(texts) - len(ages)))
    counts = [_keyword_counts(text, age_ms=age) for text, age in zip(texts, ages, strict=True)]
    backend = resolve_backend(compute_backend or "cpu")
    if backend.kind != "cpu" and counts:
        try:  # pragma: no cover - covered by host GPU smoke, not CI
            import torch  # type: ignore

            device = _torch_device_for_backend(backend)
            tensor = torch.tensor(counts, dtype=torch.float32, device=device)
            numerator = tensor[:, 0] - tensor[:, 1]
            denominator = torch.clamp(tensor[:, 0] + tensor[:, 1], min=1.0)
            values = torch.clamp(numerator / denominator, min=-1.0, max=1.0)
            return [float(value) for value in values.detach().cpu().tolist()], backend
        except Exception as exc:
            backend = BackendInfo(
                requested=backend.requested,
                kind="cpu",
                device="cpu",
                vendor="Python stdlib",
                reason=f"{backend.kind} news scoring failed ({exc.__class__.__name__}); fell back to CPU",
            )
    scores = []
    for positive, negative in counts:
        total = positive + negative
        scores.append(0.0 if total == 0 else _clamp((positive - negative) / float(total), -1.0, 1.0))
    return scores, backend


def _weighted_news_classifications(weighted_texts: list[tuple[str, float, int]], *, now_ms: int) -> list[NewsClassification]:
    return [
        _classify_news_text(text, age_ms=max(0, now_ms - known_at))
        for text, _weight, known_at in weighted_texts
    ]


def _dominant_news_classification(
    classifications: list[NewsClassification],
    *,
    default_horizon: str = "medium",
) -> NewsClassification:
    if not classifications:
        return NewsClassification(0.0, 0.0, 0.0, 0, 0.0, default_horizon, "general", ())
    return max(
        classifications,
        key=lambda item: (item.urgency, item.importance, abs(item.score)),
    )


def _average_sentiment(items: list[tuple[float, float]]) -> float:
    weighted = [(sentiment, max(0.1, weight)) for sentiment, weight in items]
    total_weight = sum(weight for _sentiment, weight in weighted)
    if total_weight <= 0.0:
        return 0.0
    return _clamp(sum(sentiment * weight for sentiment, weight in weighted) / total_weight, -1.0, 1.0)


def _parse_epoch_ms(value: Any, default_ms: int) -> int:
    parsed = _safe_float(value, default_ms / 1000.0)
    if parsed > 10_000_000_000:
        return int(parsed)
    return int(parsed * 1000)


def _parse_gdelt_seen_ms(value: Any, default_ms: int) -> int:
    text = str(value or "").strip()
    if not text:
        return default_ms
    try:
        parsed = datetime.strptime(text, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return default_ms
    return int(parsed.timestamp() * 1000)


def _parse_feed_datetime_ms(value: Any, default_ms: int) -> int:
    text = str(value or "").strip()
    if not text:
        return default_ms
    try:
        parsed = parsedate_to_datetime(text)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return int(parsed.timestamp() * 1000)
    except (TypeError, ValueError, OverflowError):
        pass
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return int(parsed.timestamp() * 1000)
    except ValueError:
        return default_ms


def _xml_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()


def _child_text(element: ET.Element, *names: str) -> str:
    wanted = {name.lower() for name in names}
    for child in list(element):
        if _xml_name(child.tag) in wanted and child.text:
            return child.text.strip()
    return ""


def _strip_markup(text: str) -> str:
    collapsed = re.sub(r"<[^>]+>", " ", unescape(text or ""))
    return " ".join(collapsed.split())


def _extract_feed_items(xml_text: str, *, now_ms: int, limit: int = 4) -> list[dict[str, object]]:
    root = ET.fromstring(str(xml_text or "").lstrip("\ufeff\xef\xbb\xbf").encode("utf-8"))
    entries = [element for element in root.iter() if _xml_name(element.tag) in {"item", "entry"}]
    items: list[dict[str, object]] = []
    for entry in entries[: max(1, int(limit))]:
        title = _strip_markup(_child_text(entry, "title"))
        summary = _strip_markup(_child_text(entry, "description", "summary", "content", "encoded"))
        published = _child_text(entry, "pubDate", "published", "updated", "date")
        known_at = _bounded_provider_timestamp_ms(_parse_feed_datetime_ms(published, now_ms), now_ms)
        link = _child_text(entry, "link")
        if not link:
            for child in list(entry):
                if _xml_name(child.tag) == "link":
                    link = str(child.attrib.get("href") or "")
                    break
        if title:
            items.append(
                {
                    "title": title,
                    "summary": summary,
                    "known_at_ms": known_at,
                    "link": link,
                }
            )
    return items


def _provider_news_payload(component: ExternalSignalComponent, items: list[dict[str, object]]) -> dict[str, object]:
    return {
        "provider": component.provider,
        "status": component.status,
        "score": component.score,
        "weight": component.weight,
        "horizon": component.horizon,
        "urgency": component.urgency,
        "known_at_ms": component.known_at_ms,
        "detail": component.detail,
        "items": items,
    }


_PROVIDER_HORIZONS = {
    "alternative_fear_greed": "long",
    "coingecko_bitcoin": "medium",
    "coinpaprika_bitcoin": "medium",
    "coinlore_bitcoin": "medium",
    "blockchain_network_stats": "long",
    "kraken_btcusd_momentum": "medium",
    "coinbase_btcusd_momentum": "medium",
    "bitstamp_btcusd_momentum": "medium",
    "binance_spot_momentum": "short",
    "binance_futures_positioning": "short",
    "mempool_fee_pressure": "short",
    "cryptocompare_btc_news": "short",
    "gdelt_bitcoin_news": "short",
    "hackernews_bitcoin_attention": "short",
}


def _provider_horizon(provider: str) -> str:
    return _PROVIDER_HORIZONS.get(str(provider or ""), "medium")


def _structured_news_items_from_payload(
    provider: str,
    payload: object,
    *,
    limit: int,
    now_ms: int,
) -> list[tuple[str, int]]:
    if not isinstance(payload, Mapping):
        return []
    provider_name = str(provider)
    items: object
    if provider_name == "cryptocompare_btc_news":
        items = payload.get("Data")
    elif provider_name == "gdelt_bitcoin_news":
        items = payload.get("articles")
    elif provider_name == "hackernews_bitcoin_attention":
        items = payload.get("hits")
    else:
        return []
    if not isinstance(items, list):
        return []
    news_items: list[tuple[str, int]] = []
    for item in items[: max(0, int(limit))]:
        if not isinstance(item, Mapping):
            continue
        title = str(item.get("title") or item.get("story_title") or "").strip()
        if not title:
            continue
        body = str(item.get("body") or item.get("summary") or item.get("tags") or "").strip()
        source = str(item.get("source") or item.get("domain") or "").strip()
        known_at = now_ms
        if provider_name == "cryptocompare_btc_news":
            known_at = _bounded_provider_timestamp_ms(_parse_epoch_ms(item.get("published_on"), now_ms), now_ms)
        elif provider_name == "gdelt_bitcoin_news":
            known_at = _bounded_provider_timestamp_ms(_parse_gdelt_seen_ms(item.get("seendate"), now_ms), now_ms)
        else:
            created_at = str(item.get("created_at") or "")
            if created_at.endswith("Z"):
                try:
                    known_at = _bounded_provider_timestamp_ms(
                        int(datetime.fromisoformat(created_at.replace("Z", "+00:00")).timestamp() * 1000),
                        now_ms,
                    )
                except ValueError:
                    known_at = now_ms
        parts = [provider_name, title]
        if body:
            parts.append(body)
        if source:
            parts.append(f"source={source}")
        news_items.append((
            ": ".join(parts[:2]) + (" " + " ".join(parts[2:]) if len(parts) > 2 else ""),
            known_at if known_at > 0 else now_ms,
        ))
    return news_items


def _structured_news_texts_from_payload(provider: str, payload: object, *, limit: int) -> list[str]:
    return [
        text
        for text, _known_at in _structured_news_items_from_payload(
            provider,
            payload,
            limit=limit,
            now_ms=_now_ms(),
        )
    ]


def _provider_name_for_url(url: str) -> str:
    lower = url.lower()
    if "alternative.me" in lower:
        return "alternative_fear_greed"
    if "coingecko" in lower:
        return "coingecko_bitcoin"
    if "coinpaprika" in lower:
        return "coinpaprika_bitcoin"
    if "coinlore" in lower:
        return "coinlore_bitcoin"
    if "blockchain.info/stats" in lower:
        return "blockchain_network_stats"
    if "kraken.com" in lower:
        return "kraken_btcusd_momentum"
    if "exchange.coinbase.com" in lower:
        return "coinbase_btcusd_momentum"
    if "bitstamp" in lower:
        return "bitstamp_btcusd_momentum"
    if "api.binance.com/api/v3/ticker/24hr" in lower:
        return "binance_spot_momentum"
    if "premiumindex" in lower or "openinterest" in lower:
        return "binance_futures_positioning"
    if "mempool.space/api" in lower:
        return "mempool_fee_pressure"
    if "cryptocompare" in lower:
        return "cryptocompare_btc_news"
    if "gdeltproject" in lower:
        return "gdelt_bitcoin_news"
    if "hn.algolia" in lower:
        return "hackernews_bitcoin_attention"
    return lower.split("//", 1)[-1].split("/", 1)[0].replace(".", "_")


def _http_retry_delay_seconds(response: object, attempt: int) -> float:
    fallback = min(5.0, 0.25 * max(1, int(attempt) + 1))
    headers = getattr(response, "headers", {})
    retry_after = headers.get("Retry-After") if isinstance(headers, Mapping) else None
    if retry_after is None:
        return fallback
    try:
        return min(5.0, max(0.0, float(retry_after)))
    except (TypeError, ValueError):
        try:
            parsed = parsedate_to_datetime(str(retry_after))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return min(5.0, max(0.0, parsed.timestamp() - time.time()))
        except (TypeError, ValueError, OverflowError):
            return fallback


def _get_response(url: str, timeout: float) -> requests.Response:
    request_timeout = max(0.1, float(timeout))
    deadline = time.monotonic() + request_timeout
    last_error: requests.RequestException | None = None
    for attempt in range(_HTTP_MAX_ATTEMPTS):
        remaining_timeout = deadline - time.monotonic()
        if attempt > 0 and remaining_timeout <= 0.0:
            if last_error is not None:
                raise last_error
            raise requests.Timeout(f"HTTP request budget exhausted for {url}")
        try:
            response = requests.get(
                url,
                timeout=max(0.1, min(request_timeout, remaining_timeout)),
                headers={"User-Agent": "simple-ai-btcusdc-cli/0.1"},
            )
            status_code = int(getattr(response, "status_code", 0) or 0)
            if status_code in _HTTP_RETRY_STATUS_CODES and attempt < _HTTP_MAX_ATTEMPTS - 1:
                remaining_budget = deadline - time.monotonic()
                if remaining_budget <= 0.0:
                    response.raise_for_status()
                time.sleep(min(_http_retry_delay_seconds(response, attempt), remaining_budget))
                continue
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            last_error = exc
            remaining_budget = deadline - time.monotonic()
            if attempt >= _HTTP_MAX_ATTEMPTS - 1:
                raise
            if remaining_budget <= 0.0:
                raise
            time.sleep(min(2.0, 0.25 * (attempt + 1), remaining_budget))
    if last_error is not None:  # pragma: no cover - loop always returns or raises
        raise last_error
    raise RuntimeError("HTTP request failed without a response")  # pragma: no cover


def _get_json(url: str, timeout: float) -> object:
    response = _get_response(url, timeout)
    return response.json()


def _get_text(url: str, timeout: float) -> str:
    return _get_response(url, timeout).text


def _post_json(url: str, payload: Mapping[str, object], timeout: float) -> object:
    request_timeout = max(0.1, float(timeout))
    deadline = time.monotonic() + request_timeout
    last_error: requests.RequestException | None = None
    for attempt in range(_HTTP_MAX_ATTEMPTS):
        remaining_timeout = max(0.1, deadline - time.monotonic())
        try:
            response = requests.post(
                url,
                json=dict(payload),
                timeout=remaining_timeout,
                headers={"User-Agent": "simple-ai-btcusdc-cli/0.1"},
            )
            status_code = int(getattr(response, "status_code", 0) or 0)
            if status_code in _HTTP_RETRY_STATUS_CODES and attempt < _HTTP_MAX_ATTEMPTS - 1:
                remaining_budget = deadline - time.monotonic()
                if remaining_budget <= 0.0:
                    response.raise_for_status()
                time.sleep(min(_http_retry_delay_seconds(response, attempt), remaining_budget))
                continue
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            last_error = exc
            remaining_budget = deadline - time.monotonic()
            if attempt >= _HTTP_MAX_ATTEMPTS - 1 or remaining_budget <= 0.0:
                raise
            time.sleep(min(5.0, 0.25 * (attempt + 1), remaining_budget))
    if last_error is not None:  # pragma: no cover - loop always returns or raises
        raise last_error
    raise RuntimeError("HTTP POST failed without a response")  # pragma: no cover


def _component(
    provider: str,
    *,
    score: float,
    weight: float,
    value: float | None,
    detail: str,
    known_at_ms: int,
    source_symbol: str = "",
    horizon: str = "medium",
    urgency: float = 0.0,
) -> ExternalSignalComponent:
    return ExternalSignalComponent(
        provider=provider,
        status="ok",
        score=float(_clamp(score, -1.0, 1.0)),
        weight=float(max(0.0, weight)),
        value=value,
        detail=detail,
        known_at_ms=int(known_at_ms),
        source_symbol=source_symbol,
        horizon=horizon,
        urgency=float(_clamp(urgency, 0.0, 1.0)),
    )


def _error_component(provider: str, error: Exception, *, known_at_ms: int, horizon: str = "medium") -> ExternalSignalComponent:
    return ExternalSignalComponent(
        provider=provider,
        status="error",
        score=0.0,
        weight=0.0,
        value=None,
        detail="provider unavailable",
        known_at_ms=int(known_at_ms),
        error=str(error)[:240],
        horizon=horizon,
    )


def _fetch_alternative_fng(fetch_json: FetchJson, timeout: float, now_ms: int) -> ExternalSignalComponent:
    payload = fetch_json(ALTERNATIVE_FNG_URL, timeout)
    if not isinstance(payload, Mapping):
        raise ValueError("unexpected Alternative.me payload")
    data = payload.get("data")
    if not isinstance(data, list) or not data or not isinstance(data[0], Mapping):
        raise ValueError("missing Alternative.me data")
    latest = data[0]
    value = _safe_float(latest.get("value"), 50.0)
    classification = str(latest.get("value_classification") or "unknown")
    timestamp = _bounded_provider_timestamp_ms(
        int(_safe_float(latest.get("timestamp"), now_ms / 1000.0) * 1000),
        now_ms,
    )
    score = _clamp((50.0 - value) / 50.0, -1.0, 1.0)
    return _component(
        "alternative_fear_greed",
        score=score,
        weight=0.85,
        value=value,
        detail=f"{classification} ({value:.0f}/100)",
        known_at_ms=timestamp,
        horizon="long",
        urgency=0.10,
    )


def _fetch_coingecko_btc(fetch_json: FetchJson, timeout: float, now_ms: int) -> ExternalSignalComponent:
    payload = fetch_json(COINGECKO_SIMPLE_PRICE_URL, timeout)
    if not isinstance(payload, Mapping) or not isinstance(payload.get("bitcoin"), Mapping):
        raise ValueError("unexpected CoinGecko payload")
    bitcoin = payload["bitcoin"]
    change = _safe_float(bitcoin.get("usd_24h_change"), 0.0)
    price = _safe_float(bitcoin.get("usd"), 0.0)
    volume = _safe_float(bitcoin.get("usd_24h_vol"), 0.0)
    score = _clamp(change / 6.0, -1.0, 1.0)
    known_at = _bounded_provider_timestamp_ms(
        int(_safe_float(bitcoin.get("last_updated_at"), now_ms / 1000.0) * 1000),
        now_ms,
    )
    if known_at <= 0:
        raise ValueError("CoinGecko timestamp is in the future")
    return _component(
        "coingecko_bitcoin",
        score=score,
        weight=0.70,
        value=change,
        detail=f"24h_change={change:+.2f}% price={price:.2f} volume={volume:.0f}",
        known_at_ms=known_at,
        source_symbol="bitcoin",
        horizon="medium",
        urgency=min(1.0, abs(score) * 0.35),
    )


def _fetch_coinpaprika_btc(fetch_json: FetchJson, timeout: float, now_ms: int) -> ExternalSignalComponent:
    payload = fetch_json(COINPAPRIKA_BTC_TICKER_URL, timeout)
    if not isinstance(payload, Mapping) or not isinstance(payload.get("quotes"), Mapping):
        raise ValueError("unexpected CoinPaprika payload")
    quotes = payload["quotes"]
    usd = quotes.get("USD") if isinstance(quotes, Mapping) else None
    if not isinstance(usd, Mapping):
        raise ValueError("missing CoinPaprika USD quote")
    price = _safe_float(usd.get("price"), 0.0)
    volume = _safe_float(usd.get("volume_24h"), 0.0)
    change_1h = _safe_float(usd.get("percent_change_1h"), 0.0)
    change_24h = _safe_float(usd.get("percent_change_24h"), 0.0)
    change_7d = _safe_float(usd.get("percent_change_7d"), 0.0)
    score = _clamp((change_24h / 6.0) * 0.75 + (change_1h / 2.0) * 0.25, -1.0, 1.0)
    known_at = _bounded_provider_timestamp_ms(
        _parse_feed_datetime_ms(payload.get("last_updated"), now_ms),
        now_ms,
    )
    if known_at <= 0:
        raise ValueError("CoinPaprika timestamp is in the future")
    return _component(
        "coinpaprika_bitcoin",
        score=score,
        weight=0.45,
        value=change_24h,
        detail=(
            f"1h={change_1h:+.2f}% 24h={change_24h:+.2f}% "
            f"7d={change_7d:+.2f}% price={price:.2f} volume={volume:.0f}"
        ),
        known_at_ms=known_at,
        source_symbol="btc-bitcoin",
        horizon="medium",
        urgency=min(1.0, abs(score) * 0.35),
    )


def _fetch_coinlore_btc(fetch_json: FetchJson, timeout: float, now_ms: int) -> ExternalSignalComponent:
    payload = fetch_json(COINLORE_BTC_TICKER_URL, timeout)
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, Mapping) and isinstance(payload.get("value"), list):
        rows = payload["value"]
    else:
        raise ValueError("unexpected CoinLore payload")
    if not rows or not isinstance(rows[0], Mapping):
        raise ValueError("missing CoinLore BTC ticker")
    row = rows[0]
    price = _safe_float(row.get("price_usd"), 0.0)
    volume = _safe_float(row.get("volume24"), 0.0)
    change_1h = _safe_float(row.get("percent_change_1h"), 0.0)
    change_24h = _safe_float(row.get("percent_change_24h"), 0.0)
    change_7d = _safe_float(row.get("percent_change_7d"), 0.0)
    score = _clamp((change_24h / 6.0) * 0.70 + (change_1h / 2.0) * 0.20 + (change_7d / 14.0) * 0.10, -1.0, 1.0)
    return _component(
        "coinlore_bitcoin",
        score=score,
        weight=0.30,
        value=change_24h,
        detail=(
            f"1h={change_1h:+.2f}% 24h={change_24h:+.2f}% "
            f"7d={change_7d:+.2f}% price={price:.2f} volume={volume:.0f}"
        ),
        known_at_ms=now_ms,
        source_symbol="BTC",
        horizon="medium",
        urgency=min(1.0, abs(score) * 0.30),
    )


def _fetch_blockchain_stats(fetch_json: FetchJson, timeout: float, now_ms: int) -> ExternalSignalComponent:
    payload = fetch_json(BLOCKCHAIN_STATS_URL, timeout)
    if not isinstance(payload, Mapping):
        raise ValueError("unexpected Blockchain.com stats payload")
    transactions = _safe_float(payload.get("n_tx"), 0.0)
    hash_rate = _safe_float(payload.get("hash_rate"), 0.0)
    block_minutes = _safe_float(payload.get("minutes_between_blocks"), 10.0)
    mempool_size = _safe_float(payload.get("mempool_size"), 0.0)
    activity_score = _clamp((transactions - 350_000.0) / 600_000.0, -0.35, 0.35)
    block_delay_penalty = _clamp((block_minutes - 10.0) / 12.0, -0.20, 0.45)
    mempool_penalty = _clamp(mempool_size / 250_000_000.0, 0.0, 0.35)
    score = _clamp(activity_score - block_delay_penalty - mempool_penalty, -1.0, 1.0)
    return _component(
        "blockchain_network_stats",
        score=score,
        weight=0.25,
        value=transactions,
        detail=(
            f"tx={transactions:.0f} hash_rate={hash_rate:.0f} "
            f"block_minutes={block_minutes:.2f} mempool_size={mempool_size:.0f}"
        ),
        known_at_ms=now_ms,
        source_symbol="BTC",
        horizon="long",
        urgency=min(1.0, max(0.0, block_delay_penalty) * 0.40 + mempool_penalty * 0.40),
    )


def _fetch_kraken_btcusd(fetch_json: FetchJson, timeout: float, now_ms: int) -> ExternalSignalComponent:
    payload = fetch_json(KRAKEN_BTCUSD_TICKER_URL, timeout)
    if not isinstance(payload, Mapping) or not isinstance(payload.get("result"), Mapping):
        raise ValueError("unexpected Kraken ticker payload")
    if payload.get("error"):
        raise ValueError("Kraken ticker returned an error")
    result = payload["result"]
    pairs = [value for value in result.values() if isinstance(value, Mapping)]
    if not pairs:
        raise ValueError("missing Kraken BTCUSD ticker")
    ticker = pairs[0]
    close = ticker.get("c")
    volume_data = ticker.get("v")
    last = _safe_float(close[0] if isinstance(close, (list, tuple)) and close else close, 0.0)
    open_price = _safe_float(ticker.get("o"), 0.0)
    volume = _safe_float(
        volume_data[1] if isinstance(volume_data, (list, tuple)) and len(volume_data) > 1 else volume_data,
        0.0,
    )
    change = ((last - open_price) / open_price * 100.0) if open_price > 0 else 0.0
    score = _clamp(change / 6.0, -1.0, 1.0)
    return _component(
        "kraken_btcusd_momentum",
        score=score,
        weight=0.30,
        value=change,
        detail=f"24h_change={change:+.2f}% last={last:.2f} open={open_price:.2f} volume={volume:.3f}",
        known_at_ms=now_ms,
        source_symbol="XBTUSD",
        horizon="medium",
        urgency=min(1.0, abs(score) * 0.30),
    )


def _fetch_coinbase_btcusd(fetch_json: FetchJson, timeout: float, now_ms: int) -> ExternalSignalComponent:
    payload = fetch_json(COINBASE_BTCUSD_STATS_URL, timeout)
    if not isinstance(payload, Mapping):
        raise ValueError("unexpected Coinbase BTC-USD stats payload")
    last = _safe_float(payload.get("last"), 0.0)
    open_price = _safe_float(payload.get("open"), 0.0)
    high = _safe_float(payload.get("high"), 0.0)
    low = _safe_float(payload.get("low"), 0.0)
    volume = _safe_float(payload.get("volume"), 0.0)
    change = ((last - open_price) / open_price * 100.0) if open_price > 0 else 0.0
    score = _clamp(change / 6.0, -1.0, 1.0)
    return _component(
        "coinbase_btcusd_momentum",
        score=score,
        weight=0.30,
        value=change,
        detail=f"24h_change={change:+.2f}% last={last:.2f} high={high:.2f} low={low:.2f} volume={volume:.3f}",
        known_at_ms=now_ms,
        source_symbol="BTC-USD",
        horizon="medium",
        urgency=min(1.0, abs(score) * 0.30),
    )


def _fetch_bitstamp_btcusd(fetch_json: FetchJson, timeout: float, now_ms: int) -> ExternalSignalComponent:
    payload = fetch_json(BITSTAMP_BTCUSD_TICKER_URL, timeout)
    if not isinstance(payload, Mapping):
        raise ValueError("unexpected Bitstamp BTCUSD ticker payload")
    last = _safe_float(payload.get("last"), 0.0)
    open_price = _safe_float(payload.get("open"), 0.0)
    volume = _safe_float(payload.get("volume"), 0.0)
    change = _safe_float(payload.get("percent_change_24"), float("nan"))
    if not math.isfinite(change):
        change = ((last - open_price) / open_price * 100.0) if open_price > 0 else 0.0
    score = _clamp(change / 6.0, -1.0, 1.0)
    return _component(
        "bitstamp_btcusd_momentum",
        score=score,
        weight=0.25,
        value=change,
        detail=f"24h_change={change:+.2f}% last={last:.2f} open={open_price:.2f} volume={volume:.3f}",
        known_at_ms=now_ms,
        source_symbol="BTCUSD",
        horizon="medium",
        urgency=min(1.0, abs(score) * 0.28),
    )


def _binance_symbol_candidates(symbol: str) -> list[str]:
    symbol = (symbol or "BTCUSDC").upper()
    candidates = [symbol]
    if symbol != "BTCUSDT":
        candidates.append("BTCUSDT")
    return candidates


def _fetch_binance_spot_momentum(
    fetch_json: FetchJson,
    timeout: float,
    now_ms: int,
    symbol: str,
) -> ExternalSignalComponent:
    errors: list[str] = []
    for candidate in _binance_symbol_candidates(symbol):
        try:
            payload = fetch_json(f"{BINANCE_SPOT_BASE_URL}/api/v3/ticker/24hr?symbol={candidate}", timeout)
            if not isinstance(payload, Mapping):
                raise ValueError("unexpected Binance spot ticker payload")
            change = _safe_float(payload.get("priceChangePercent"), 0.0)
            last = _safe_float(payload.get("lastPrice"), 0.0)
            base_volume = _safe_float(payload.get("volume"), 0.0)
            quote_volume = _safe_float(payload.get("quoteVolume"), 0.0)
            score = _clamp(change / 6.0, -1.0, 1.0)
            known_at = _bounded_provider_timestamp_ms(int(_safe_float(payload.get("closeTime"), now_ms)), now_ms)
            if known_at <= 0:
                raise ValueError("Binance spot ticker timestamp is in the future")
            return _component(
                "binance_spot_momentum",
                score=score,
                weight=0.60,
                value=change,
                detail=(
                    f"24h_change={change:+.2f}% last={last:.2f} "
                    f"base_volume={base_volume:.3f} quote_volume={quote_volume:.0f}"
                ),
                known_at_ms=known_at,
                source_symbol=candidate,
                horizon="short",
                urgency=min(1.0, abs(score) * 0.45),
            )
        except Exception as exc:
            errors.append(f"{candidate}: {exc}")
    raise ValueError("; ".join(errors) or "Binance spot ticker unavailable")


def _fetch_binance_derivatives(
    fetch_json: FetchJson,
    timeout: float,
    now_ms: int,
    symbol: str,
) -> ExternalSignalComponent:
    errors: list[str] = []
    for candidate in _binance_symbol_candidates(symbol):
        try:
            premium = fetch_json(f"{BINANCE_FUTURES_BASE_URL}/fapi/v1/premiumIndex?symbol={candidate}", timeout)
            interest = fetch_json(f"{BINANCE_FUTURES_BASE_URL}/fapi/v1/openInterest?symbol={candidate}", timeout)
            if not isinstance(premium, Mapping) or not isinstance(interest, Mapping):
                raise ValueError("unexpected Binance futures payload")
            funding = _safe_float(premium.get("lastFundingRate"), 0.0)
            mark = _safe_float(premium.get("markPrice"), 0.0)
            index = _safe_float(premium.get("indexPrice"), 0.0)
            open_interest = _safe_float(interest.get("openInterest"), 0.0)
            basis = ((mark - index) / index) if index > 0 else 0.0
            score = _clamp((-funding / 0.0015) + (basis / 0.004), -1.0, 1.0)
            known_at = _bounded_provider_timestamp_ms(int(_safe_float(premium.get("time"), now_ms)), now_ms)
            return _component(
                "binance_futures_positioning",
                score=score,
                weight=1.00,
                value=funding,
                detail=(
                    f"funding={funding:+.5f} basis={basis:+.5f} "
                    f"open_interest={open_interest:.3f}"
                ),
                known_at_ms=known_at,
                source_symbol=candidate,
                horizon="short",
                urgency=min(1.0, abs(score) * 0.65),
            )
        except Exception as exc:
            errors.append(f"{candidate}: {exc}")
    raise ValueError("; ".join(errors) or "Binance derivatives unavailable")


def _fetch_mempool_fees(fetch_json: FetchJson, timeout: float, now_ms: int) -> ExternalSignalComponent:
    payload = fetch_json(MEMPOOL_FEES_URL, timeout)
    if not isinstance(payload, Mapping):
        raise ValueError("unexpected mempool.space payload")
    fastest = _safe_float(payload.get("fastestFee"), 0.0)
    half_hour = _safe_float(payload.get("halfHourFee"), 0.0)
    pressure = max(fastest, half_hour)
    score = -_clamp((pressure - 20.0) / 80.0, 0.0, 1.0)
    return _component(
        "mempool_fee_pressure",
        score=score,
        weight=0.35,
        value=pressure,
        detail=f"fastest={fastest:.1f}sat/vB half_hour={half_hour:.1f}sat/vB",
        known_at_ms=now_ms,
        source_symbol="BTC",
        horizon="short",
        urgency=min(1.0, abs(score) * 0.50),
    )


def _fetch_cryptocompare_news(
    fetch_json: FetchJson,
    timeout: float,
    now_ms: int,
    compute_backend: str | None,
) -> tuple[ExternalSignalComponent, BackendInfo]:
    payload = fetch_json(CRYPTOCOMPARE_NEWS_URL, timeout)
    if not isinstance(payload, Mapping):
        raise ValueError("unexpected CryptoCompare news payload")
    data = payload.get("Data")
    if not isinstance(data, list) or not data:
        raise ValueError("missing CryptoCompare news data")
    weighted_texts: list[tuple[str, float, int]] = []
    newest_ms = 0
    domains: list[str] = []
    for entry in data[:12]:
        if not isinstance(entry, Mapping):
            continue
        title = str(entry.get("title") or "")
        body = str(entry.get("body") or "")
        tags = str(entry.get("tags") or "")
        source = str(entry.get("source") or "")
        if title:
            weight = 1.0 + min(2.0, len(title) / 120.0)
            published_ms = _bounded_provider_timestamp_ms(_parse_epoch_ms(entry.get("published_on"), now_ms), now_ms)
            if published_ms <= 0:
                continue
            weighted_texts.append((f"{title} {body} {tags}", weight, published_ms))
            newest_ms = max(newest_ms, published_ms)
        if source and source not in domains:
            domains.append(source)
    if not weighted_texts:
        raise ValueError("CryptoCompare news contained no usable titles")
    ages = [max(0, now_ms - known_at) for _text, _weight, known_at in weighted_texts]
    scores, backend = _score_news_texts(
        [text for text, _weight, _known_at in weighted_texts],
        compute_backend=compute_backend,
        ages_ms=ages,
    )
    classifications = _weighted_news_classifications(weighted_texts, now_ms=now_ms)
    dominant = _dominant_news_classification(classifications)
    scored = [(score, weight) for score, (_text, weight, _known_at) in zip(scores, weighted_texts, strict=True)]
    score = _average_sentiment(scored)
    return _component(
        "cryptocompare_btc_news",
        score=score,
        weight=0.55,
        value=float(len(scored)),
        detail=(
            f"articles={len(scored)} sources={','.join(domains[:3]) or 'unknown'} "
            f"importance={dominant.importance}/10 category={dominant.category}"
        ),
        known_at_ms=newest_ms or now_ms,
        source_symbol="BTC",
        horizon=dominant.horizon,
        urgency=dominant.urgency,
    ), backend


def _fetch_gdelt_news(
    fetch_json: FetchJson,
    timeout: float,
    now_ms: int,
    compute_backend: str | None,
) -> tuple[ExternalSignalComponent, BackendInfo]:
    payload = fetch_json(GDELT_BITCOIN_NEWS_URL, timeout)
    if not isinstance(payload, Mapping):
        raise ValueError("unexpected GDELT payload")
    articles = payload.get("articles")
    if not isinstance(articles, list) or not articles:
        raise ValueError("missing GDELT articles")
    weighted_texts: list[tuple[str, float, int]] = []
    newest_ms = 0
    domains: list[str] = []
    for entry in articles[:12]:
        if not isinstance(entry, Mapping):
            continue
        title = str(entry.get("title") or "")
        domain = str(entry.get("domain") or "")
        if title:
            seen_ms = _bounded_provider_timestamp_ms(_parse_gdelt_seen_ms(entry.get("seendate"), now_ms), now_ms)
            if seen_ms <= 0:
                continue
            weighted_texts.append((title, 1.0, seen_ms))
            newest_ms = max(newest_ms, seen_ms)
        if domain and domain not in domains:
            domains.append(domain)
    if not weighted_texts:
        raise ValueError("GDELT articles contained no usable titles")
    ages = [max(0, now_ms - known_at) for _text, _weight, known_at in weighted_texts]
    scores, backend = _score_news_texts(
        [text for text, _weight, _known_at in weighted_texts],
        compute_backend=compute_backend,
        ages_ms=ages,
    )
    classifications = _weighted_news_classifications(weighted_texts, now_ms=now_ms)
    dominant = _dominant_news_classification(classifications)
    scored = [(score, weight) for score, (_text, weight, _known_at) in zip(scores, weighted_texts, strict=True)]
    score = _average_sentiment(scored)
    return _component(
        "gdelt_bitcoin_news",
        score=score,
        weight=0.45,
        value=float(len(scored)),
        detail=(
            f"articles={len(scored)} domains={','.join(domains[:3]) or 'unknown'} "
            f"importance={dominant.importance}/10 category={dominant.category}"
        ),
        known_at_ms=newest_ms or now_ms,
        source_symbol="BTC",
        horizon=dominant.horizon,
        urgency=dominant.urgency,
    ), backend


def _fetch_hackernews_bitcoin(
    fetch_json: FetchJson,
    timeout: float,
    now_ms: int,
    compute_backend: str | None,
) -> tuple[ExternalSignalComponent, BackendInfo]:
    payload = fetch_json(HACKER_NEWS_BITCOIN_URL, timeout)
    if not isinstance(payload, Mapping):
        raise ValueError("unexpected Hacker News payload")
    hits = payload.get("hits")
    if not isinstance(hits, list) or not hits:
        raise ValueError("missing Hacker News hits")
    weighted_texts: list[tuple[str, float, int]] = []
    newest_ms = 0
    for entry in hits[:10]:
        if not isinstance(entry, Mapping):
            continue
        title = str(entry.get("title") or entry.get("story_title") or "")
        points = _safe_float(entry.get("points"), 0.0)
        known_at = now_ms
        if title:
            weight = 1.0 + min(3.0, points / 100.0)
        created_at = str(entry.get("created_at") or "")
        if title and created_at.endswith("Z"):
            try:
                known_at = _bounded_provider_timestamp_ms(
                    int(datetime.fromisoformat(created_at.replace("Z", "+00:00")).timestamp() * 1000),
                    now_ms,
                )
                if known_at <= 0:
                    continue
                newest_ms = max(newest_ms, known_at)
            except ValueError:
                pass
        if title:
            weighted_texts.append((title, weight, known_at))
    if not weighted_texts:
        raise ValueError("Hacker News hits contained no usable titles")
    ages = [max(0, now_ms - known_at) for _text, _weight, known_at in weighted_texts]
    scores, backend = _score_news_texts(
        [text for text, _weight, _known_at in weighted_texts],
        compute_backend=compute_backend,
        ages_ms=ages,
    )
    classifications = _weighted_news_classifications(weighted_texts, now_ms=now_ms)
    dominant = _dominant_news_classification(classifications)
    scored = [(score, weight) for score, (_text, weight, _known_at) in zip(scores, weighted_texts, strict=True)]
    score = _average_sentiment(scored)
    return _component(
        "hackernews_bitcoin_attention",
        score=score,
        weight=0.25,
        value=float(len(scored)),
        detail=f"stories={len(scored)} importance={dominant.importance}/10 category={dominant.category}",
        known_at_ms=newest_ms or now_ms,
        source_symbol="BTC",
        horizon=dominant.horizon,
        urgency=dominant.urgency,
    ), backend


def _fetch_rss_news_feed(
    provider: NewsFeedProvider,
    fetch_text: FetchText,
    timeout: float,
    now_ms: int,
    compute_backend: str | None,
    *,
    items_per_provider: int = 4,
    provider_jitter_seconds: float = 0.0,
) -> ProviderFetchResult:
    if provider_jitter_seconds > 0.0:
        time.sleep(_JITTER_RANDOM.uniform(0.0, float(provider_jitter_seconds)))
    xml_text = fetch_text(provider.url, timeout)
    try:
        items = _extract_feed_items(xml_text, now_ms=now_ms, limit=items_per_provider)
    except Exception as exc:
        raise _RSSFeedPayloadError(f"feed parse failed: {exc}", str(xml_text)) from exc
    if not items:
        raise _RSSFeedPayloadError("feed contained no usable items", str(xml_text))
    weighted_texts: list[tuple[str, float, int]] = []
    newest_ms = 0
    for item in items:
        title = str(item.get("title") or "")
        summary = str(item.get("summary") or "")
        known_at = int(_safe_float(item.get("known_at_ms"), now_ms))
        weighted_texts.append((f"{title} {summary}", 1.0 + min(1.0, len(title) / 160.0), known_at))
        newest_ms = max(newest_ms, known_at)
    scores, backend = _score_news_texts(
        [text for text, _weight, _known_at in weighted_texts],
        compute_backend=compute_backend,
        ages_ms=[max(0, now_ms - known_at) for _text, _weight, known_at in weighted_texts],
    )
    classifications = _weighted_news_classifications(weighted_texts, now_ms=now_ms)
    dominant = _dominant_news_classification(classifications, default_horizon=provider.horizon)
    scored = [(score, weight) for score, (_text, weight, _known_at) in zip(scores, weighted_texts, strict=True)]
    score = _average_sentiment(scored)
    horizon = "short" if dominant.horizon == "short" else provider.horizon
    component = _component(
        provider.provider,
        score=score,
        weight=provider.weight,
        value=float(len(scored)),
        detail=f"scope={provider.scope} items={len(scored)} importance={dominant.importance}/10 category={dominant.category}",
        known_at_ms=newest_ms or now_ms,
        source_symbol="BTCUSDC" if provider.scope == "crypto" else provider.scope,
        horizon=horizon,
        urgency=dominant.urgency,
    )
    return ProviderFetchResult(
        component=component,
        backend=backend,
        news_texts=tuple(text for text, _weight, _known_at in weighted_texts),
        news_known_at_ms=tuple(known_at for _text, _weight, known_at in weighted_texts),
        raw_payload={
            **_provider_news_payload(component, items),
            "url": provider.url,
            "raw_xml": xml_text,
            "classifications": [classification.asdict() for classification in classifications],
        },
    )


def _fetch_rss_news_feeds(
    providers: tuple[NewsFeedProvider, ...],
    fetch_text: FetchText,
    timeout: float,
    now_ms: int,
    compute_backend: str | None,
    *,
    items_per_provider: int = 4,
    max_workers: int = 12,
    provider_jitter_seconds: float = 0.0,
) -> list[ProviderFetchResult]:
    if not providers:
        return []
    results: list[ProviderFetchResult] = []
    max_workers = min(max(1, int(max_workers)), max(1, len(providers)))

    host_lock = Lock()
    next_host_time: dict[str, float] = {}
    same_host_gap = max(_RSS_SAME_HOST_GAP_SECONDS, min(1.0, max(0.0, float(provider_jitter_seconds))))

    def fetch_with_host_pacing(provider: NewsFeedProvider) -> ProviderFetchResult:
        if provider_jitter_seconds > 0.0:
            time.sleep(_JITTER_RANDOM.uniform(0.0, float(provider_jitter_seconds)))
        host = urlparse(provider.url).netloc.lower()
        if host:
            with host_lock:
                now = time.perf_counter()
                ready_at = next_host_time.get(host, now)
                wait_seconds = max(0.0, ready_at - now)
                next_host_time[host] = max(now, ready_at) + same_host_gap
            if wait_seconds > 0.0:
                time.sleep(wait_seconds)
        return _fetch_rss_news_feed(
            provider,
            fetch_text,
            timeout,
            now_ms,
            compute_backend,
            items_per_provider=items_per_provider,
            provider_jitter_seconds=0.0,
        )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(fetch_with_host_pacing, provider): provider
            for provider in providers
        }
        for future in as_completed(future_map):
            provider = future_map[future]
            try:
                results.append(future.result())
            except Exception as exc:
                error_payload: dict[str, object] = {
                    "provider": provider.provider,
                    "url": provider.url,
                    "error": str(exc)[:240],
                }
                if isinstance(exc, _RSSFeedPayloadError):
                    error_payload["raw_xml"] = exc.raw_xml[:1_000_000]
                results.append(
                    ProviderFetchResult(
                        component=_error_component(provider.provider, exc, known_at_ms=now_ms, horizon=provider.horizon),
                        raw_payload=error_payload,
                    )
                )
    order = {provider.provider: index for index, provider in enumerate(providers)}
    return sorted(results, key=lambda result: order.get(result.component.provider, len(order)))


def _normalize_horizon(value: object) -> str:
    text = str(value or "").strip().lower().replace("_", "-")
    if text.startswith("short"):
        return "short"
    if text.startswith("long"):
        return "long"
    return "medium"


def _coerce_ai_reaction(value: object, score: float, horizon: str) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    if text in {"yes", "true", "1", "urgent", "react", "reaction_required"}:
        return True
    if text in {"no", "false", "0", "monitor", "none"}:
        return False
    return horizon == "short" and abs(score) >= 0.60


def _json_mapping_from_text(text: str) -> Mapping[str, object]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise
        payload = json.loads(text[start : end + 1])
    if not isinstance(payload, Mapping):
        raise ValueError("Ollama response was not a JSON object")
    return payload


def _ollama_response_text(payload: Mapping[str, object]) -> str:
    message = payload.get("message")
    if isinstance(message, Mapping):
        content = message.get("content")
        if content is not None:
            return str(content)
    return str(payload.get("response") or "")


def _ollama_priority(text: str, *, age_ms: int = 0) -> float:
    normalized = f" {text.lower()} "
    positive, negative = _keyword_counts(text, age_ms=age_ms)
    short_hits = sum(1 for term in _SHORT_TERM_TERMS if term in normalized)
    btc_focus = int(" bitcoin " in normalized or " btc" in normalized)
    recency = 0.25 + _recency_factor(age_ms) * 0.75
    return (float(short_hits * 10 + (positive + negative) * 3 + btc_focus) * recency) + min(len(text), 180) / 1000.0


def _bounded_ollama_news_items(
    news_texts: list[str],
    *,
    news_known_at_ms: list[int] | None = None,
    now_ms: int | None = None,
    limit: int = _OLLAMA_NEWS_MAX_ITEMS,
) -> list[tuple[str, int]]:
    now = _now_ms() if now_ms is None else int(now_ms)
    known_values = list(news_known_at_ms or [])
    cleaned: list[tuple[str, int]] = []
    seen: set[str] = set()
    for index, text in enumerate(news_texts):
        item = " ".join(str(text).split())[:_OLLAMA_NEWS_MAX_TEXT_CHARS]
        key = item.lower()
        if item and key not in seen:
            raw_known_at = known_values[index] if index < len(known_values) else now
            known_at = _bounded_provider_timestamp_ms(int(_safe_float(raw_known_at, now)), now)
            cleaned.append((item, known_at if known_at > 0 else now))
            seen.add(key)
    return sorted(
        cleaned,
        key=lambda item: _ollama_priority(item[0], age_ms=max(0, now - item[1])),
        reverse=True,
    )[:limit]


def _bounded_ollama_news_texts(news_texts: list[str], *, limit: int = _OLLAMA_NEWS_MAX_ITEMS) -> list[str]:
    return [text for text, _known_at in _bounded_ollama_news_items(news_texts, limit=limit)]


def _ollama_prompt(news_texts: list[str]) -> str:
    bounded = _bounded_ollama_news_texts(news_texts)
    joined = "\n".join(f"- {text}" for text in bounded)
    return (
        "Score public BTCUSDC market headlines. -1 bearish, +1 bullish. "
        "Hacks, exploits, breaches, bans, and ETF denials are bearish. "
        "ETF approvals, inflows, and adoption are bullish. "
        "Use short horizon for breaking/emergency items needing a minutes response. JSON only.\n"
        f"{joined}"
    )


def _evaluate_news_with_ollama(
    news_texts: list[str],
    *,
    news_known_at_ms: list[int] | None = None,
    model: str = DEFAULT_OLLAMA_NEWS_MODEL,
    base_url: str = DEFAULT_OLLAMA_URL,
    timeout_seconds: float = 3.0,
    post_json: PostJson = _post_json,
    now_ms: int | None = None,
) -> OllamaNewsEvaluation:
    now = _now_ms() if now_ms is None else int(now_ms)
    if not news_texts:
        raise ValueError("no news texts available for Ollama evaluation")
    selected_news_items = _bounded_ollama_news_items(
        news_texts,
        news_known_at_ms=news_known_at_ms,
        now_ms=now,
    )
    if not selected_news_items:
        raise ValueError("no usable news texts available for Ollama evaluation")
    selected_news_texts = [text for text, _known_at in selected_news_items]
    selected_known_at_ms = [known_at for _text, known_at in selected_news_items]
    selected_ages_ms = [max(0, now - known_at) for known_at in selected_known_at_ms]
    newest_selected_ms = max(selected_known_at_ms, default=now)
    classifications = [
        _classify_news_text(text, age_ms=age_ms)
        for text, age_ms in zip(selected_news_texts, selected_ages_ms, strict=True)
    ]
    dominant = _dominant_news_classification(classifications)
    heuristic_score = _average_sentiment(
        [(item.score, max(1.0, float(item.importance))) for item in classifications]
    )
    started = time.perf_counter()
    endpoint = f"{str(base_url or DEFAULT_OLLAMA_URL).rstrip('/')}/api/chat"
    prompt = _ollama_prompt(selected_news_texts)
    request = {
        "model": str(model or DEFAULT_OLLAMA_NEWS_MODEL),
        "messages": [
            {
                "role": "system",
                "content": "Public market-news sentiment classifier. Return only the requested JSON fields.",
            },
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "format": _OLLAMA_NEWS_JSON_SCHEMA,
        "think": False,
        "keep_alive": "30m",
        "options": {
            "temperature": 0,
            "num_ctx": _OLLAMA_NEWS_NUM_CTX,
            "num_predict": _OLLAMA_NEWS_NUM_PREDICT,
        },
    }
    payload = post_json(endpoint, request, timeout_seconds)
    latency_ms = int((time.perf_counter() - started) * 1000)
    if not isinstance(payload, Mapping):
        raise ValueError("unexpected Ollama payload")
    response_text = _ollama_response_text(payload)
    parse_error = ""
    try:
        parsed = _json_mapping_from_text(response_text)
    except (json.JSONDecodeError, ValueError) as exc:
        parsed = {}
        parse_error = str(exc)[:180]
    score = _clamp(_safe_float(parsed.get("score"), heuristic_score), -1.0, 1.0)
    horizon = _normalize_horizon(parsed.get("horizon") or dominant.horizon)
    reaction = _coerce_ai_reaction(parsed.get("reaction_required"), score, horizon)
    guarded = False
    if dominant.importance >= 8 and abs(dominant.score) >= 0.75 and (score == 0.0 or score * dominant.score < 0.0):
        score = dominant.score
        horizon = dominant.horizon
        reaction = dominant.urgency >= 0.75 or (horizon == "short" and abs(score) >= 0.60)
        guarded = True
    reason = " ".join(str(parsed.get("reason") or "").split())[:180]
    if parse_error:
        reason = f"schema_guarded {dominant.category}"
        guarded = True
    elif guarded:
        reason = f"guarded {dominant.category} signal"
    elif not reason:
        reason = "news impact evaluated"
    component = _component(
        "ollama_news_ai",
        score=score,
        weight=0.70,
        value=float(len(selected_news_texts)),
        detail=f"model={model} latency_ms={latency_ms} reason={reason}",
        known_at_ms=newest_selected_ms,
        source_symbol="BTCUSDC",
        horizon=horizon,
        urgency=1.0 if reaction else min(0.75, abs(score) * 0.75),
    )
    evaluation_status = "warn" if guarded else "ok"
    return OllamaNewsEvaluation(
        component=component,
        status=evaluation_status,
        model=str(model or DEFAULT_OLLAMA_NEWS_MODEL),
        latency_ms=latency_ms,
        reason=reason,
        raw_payload={
            "provider": "ollama_news_ai",
            "horizon": horizon,
            "known_at_ms": newest_selected_ms,
            "score": score,
            "urgency": component.urgency,
            "latency_ms": latency_ms,
            "status": evaluation_status,
            "request": {
                "model": request["model"],
                "format": request["format"],
                "think": request["think"],
                "options": request["options"],
                "news_count": len(news_texts),
                "prompt_news_count": len(selected_news_texts),
                "selected_news_texts": list(selected_news_texts),
                "selected_known_at_ms": list(selected_known_at_ms),
                "selected_ages_ms": list(selected_ages_ms),
                "prompt": prompt,
                "parse_error": parse_error,
                "schema_guarded": guarded,
            },
            "response": payload,
            "parsed": dict(parsed),
        },
    )


def _component_from_payload(payload: Mapping[str, object]) -> ExternalSignalComponent | None:
    try:
        provider = str(payload.get("provider") or "")
        if not provider:
            return None
        return ExternalSignalComponent(
            provider=provider,
            status=str(payload.get("status") or "ok"),
            score=_safe_float(payload.get("score"), 0.0),
            weight=_safe_float(payload.get("weight"), 0.0),
            value=None if payload.get("value") is None else _safe_float(payload.get("value"), 0.0),
            detail=str(payload.get("detail") or ""),
            known_at_ms=int(_safe_float(payload.get("known_at_ms"), 0.0)),
            source_symbol=str(payload.get("source_symbol") or ""),
            error=str(payload.get("error") or ""),
            cached=bool(payload.get("cached", False)),
            horizon=str(payload.get("horizon") or "medium"),
            urgency=_safe_float(payload.get("urgency"), 0.0),
            source_grade=(
                None
                if payload.get("source_grade") is None
                else int(_clamp(_safe_float(payload.get("source_grade"), 0.0), 0.0, 10.0))
            ),
            source_grade_model=str(payload.get("source_grade_model") or ""),
            source_grade_weight_multiplier=_safe_float(payload.get("source_grade_weight_multiplier"), 1.0),
        )
    except (TypeError, ValueError):
        return None


def _grade_weight_multiplier(grade: object) -> float:
    bounded = int(_clamp(_safe_float(grade, 5.0), 0.0, 10.0))
    if bounded <= 5:
        return _clamp(0.25 + bounded * 0.15, 0.25, 1.0)
    return _clamp(1.0 + (bounded - 5) * 0.05, 1.0, 1.25)


def _telemetry_only_grade_weight_multiplier(grade: object) -> float:
    bounded = int(_clamp(_safe_float(grade, 5.0), 0.0, 10.0))
    if bounded <= 5:
        return _clamp(0.70 + bounded * 0.06, 0.70, 1.0)
    return _clamp(1.0 + (bounded - 5) * 0.03, 1.0, 1.15)


def _source_grade_for_component(
    component: ExternalSignalComponent,
    source_grades: Mapping[tuple[str, str], object],
) -> object | None:
    horizons = [component.horizon]
    if component.horizon != "medium":
        horizons.append("medium")
    for horizon in horizons:
        grade = source_grades.get((component.provider, horizon))
        if grade is not None:
            return grade
    return None


def _source_grade_has_outcome_evidence(source_grade: object) -> bool:
    evidence = getattr(source_grade, "evidence", {})
    if not isinstance(evidence, Mapping):
        return False
    if _safe_float(evidence.get("outcome_records"), 0.0) > 0.0:
        return True
    if evidence.get("directional_accuracy") is not None:
        return True
    if evidence.get("hit_rate") is not None:
        return True
    return False


def _apply_source_grade_weights(
    components: list[ExternalSignalComponent],
    source_grades: Mapping[tuple[str, str], object],
) -> list[ExternalSignalComponent]:
    if not source_grades:
        return components
    adjusted: list[ExternalSignalComponent] = []
    for component in components:
        source_grade = _source_grade_for_component(component, source_grades)
        if source_grade is None or component.status != "ok" or component.weight <= 0.0:
            adjusted.append(component)
            continue
        grade_value = int(_clamp(_safe_float(getattr(source_grade, "grade", 5), 5.0), 0.0, 10.0))
        has_outcome_evidence = _source_grade_has_outcome_evidence(source_grade)
        multiplier = (
            _grade_weight_multiplier(grade_value)
            if has_outcome_evidence
            else _telemetry_only_grade_weight_multiplier(grade_value)
        )
        model = str(getattr(source_grade, "model", "") or "")
        grade_note = f"source_grade={grade_value}/10 multiplier={multiplier:.2f}"
        if model:
            grade_note = f"{grade_note} grade_model={model}"
        if not has_outcome_evidence:
            grade_note = f"{grade_note} telemetry_only_bounded_no_outcomes"
        payload = component.asdict()
        payload.update(
            {
                "weight": component.weight * multiplier,
                "detail": f"{component.detail} {grade_note}".strip(),
                "source_grade": grade_value,
                "source_grade_model": model,
                "source_grade_weight_multiplier": multiplier,
            }
        )
        adjusted.append(ExternalSignalComponent(**payload))
    return adjusted


def _base_component_for_regrading(component: ExternalSignalComponent) -> ExternalSignalComponent:
    previous_multiplier = _safe_float(component.source_grade_weight_multiplier, 1.0)
    payload = component.asdict()
    if previous_multiplier > 0.0:
        payload["weight"] = component.weight / previous_multiplier
    payload["source_grade"] = None
    payload["source_grade_model"] = ""
    payload["source_grade_weight_multiplier"] = 1.0
    detail = str(component.detail or "")
    marker = " source_grade="
    payload["detail"] = detail.split(marker, 1)[0] if marker in detail else detail
    return ExternalSignalComponent(**payload)


def report_from_payload(payload: Mapping[str, object]) -> ExternalSignalReport | None:
    components_raw = payload.get("components")
    if not isinstance(components_raw, list):
        return None
    components = [
        component
        for item in components_raw
        if isinstance(item, Mapping)
        for component in [_component_from_payload(item)]
        if component is not None
    ]
    warning_values = payload.get("warnings", [])
    warnings = [str(value) for value in warning_values if isinstance(value, str)] if isinstance(warning_values, list) else []
    return ExternalSignalReport(
        status=str(payload.get("status") or "warn"),
        score_adjustment=_safe_float(payload.get("score_adjustment"), 0.0),
        raw_score=_safe_float(payload.get("raw_score"), 0.0),
        risk_multiplier=_safe_float(payload.get("risk_multiplier"), 1.0),
        provider_count=int(_safe_float(payload.get("provider_count"), len(components))),
        fresh_count=int(_safe_float(payload.get("fresh_count"), 0.0)),
        stale_count=int(_safe_float(payload.get("stale_count"), 0.0)),
        known_at_ms=int(_safe_float(payload.get("known_at_ms"), 0.0)),
        cache_path=str(payload.get("cache_path") or ""),
        warnings=warnings,
        components=components,
        short_term_score=_safe_float(payload.get("short_term_score"), 0.0),
        medium_term_score=_safe_float(payload.get("medium_term_score"), 0.0),
        long_term_score=_safe_float(payload.get("long_term_score"), 0.0),
        reaction_required=bool(payload.get("reaction_required", False)),
        reaction_reason=str(payload.get("reaction_reason") or ""),
        news_backend_requested=str(payload.get("news_backend_requested") or "cpu"),
        news_backend_kind=str(payload.get("news_backend_kind") or "cpu"),
        news_backend_device=str(payload.get("news_backend_device") or "cpu"),
        news_backend_reason=str(payload.get("news_backend_reason") or ""),
        news_ai_enabled=bool(payload.get("news_ai_enabled", False)),
        news_ai_status=str(payload.get("news_ai_status") or "disabled"),
        news_ai_model=str(payload.get("news_ai_model") or DEFAULT_OLLAMA_NEWS_MODEL),
        news_ai_latency_ms=int(_safe_float(payload.get("news_ai_latency_ms"), 0.0)),
        news_ai_reason=str(payload.get("news_ai_reason") or ""),
    )


def load_external_signal_cache(
    path: Path,
    *,
    now_ms: int,
    ttl_seconds: int,
    short_reaction_refresh_seconds: int = 30,
) -> ExternalSignalReport | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, Mapping):
        return None
    report = report_from_payload(payload)
    if report is None:
        return None
    if report.known_at_ms <= 0 or report.known_at_ms > now_ms:
        return None
    age_seconds = (now_ms - report.known_at_ms) / 1000.0
    if age_seconds > max(0, int(ttl_seconds)):
        return None
    if report.reaction_required and age_seconds > max(1, int(short_reaction_refresh_seconds)):
        return None
    usable_components = [
        component
        for component in report.components
        if component.status == "ok" and component.weight > 0.0
    ]
    if any(component.known_at_ms > now_ms for component in usable_components):
        return None
    fresh_components = [
        component
        for component in usable_components
        if _component_cache_timestamp_is_fresh(
            component,
            now_ms=now_ms,
            report_known_at_ms=report.known_at_ms,
        )
    ]
    stale_count = len(usable_components) - len(fresh_components)
    if usable_components and not fresh_components:
        return None
    return ExternalSignalReport(
        **{
            **report.asdict(),
            "status": "cached",
            "fresh_count": len(fresh_components),
            "stale_count": max(0, int(report.stale_count)) + stale_count,
            "components": [
                ExternalSignalComponent(**{**component.asdict(), "cached": True})
                for component in report.components
            ],
        }
    )


def _combine_components(
    components: list[ExternalSignalComponent],
    *,
    max_adjustment: float,
    min_providers: int,
    now_ms: int,
    cache_path: Path,
    news_backend: BackendInfo,
    news_ai_enabled: bool = False,
    news_ai_status: str = "disabled",
    news_ai_model: str = DEFAULT_OLLAMA_NEWS_MODEL,
    news_ai_latency_ms: int = 0,
    news_ai_reason: str = "",
    extra_warnings: list[str] | None = None,
) -> ExternalSignalReport:
    ok_components = [component for component in components if component.status == "ok" and component.weight > 0.0]
    usable = [
        component
        for component in ok_components
        if _component_cache_timestamp_is_fresh(component, now_ms=now_ms, report_known_at_ms=now_ms)
    ]
    stale_count = len(ok_components) - len(usable)
    warnings = [
        f"{component.provider}: {component.error}"
        for component in components
        if component.status == "error" and component.error
    ]
    warnings.extend(extra_warnings or [])
    if stale_count:
        warnings.append(f"{stale_count} stale external signal component(s) ignored")
    total_weight = sum(component.weight for component in usable)
    raw_score = sum(component.score * component.weight for component in usable) / total_weight if total_weight else 0.0
    horizon_scores: dict[str, float] = {}
    for horizon in ("short", "medium", "long"):
        horizon_components = [component for component in usable if component.horizon == horizon]
        horizon_weight = sum(component.weight for component in horizon_components)
        horizon_scores[horizon] = (
            sum(component.score * component.weight for component in horizon_components) / horizon_weight
            if horizon_weight
            else 0.0
        )
    max_adjustment = _clamp(float(max_adjustment), 0.0, 0.20)
    blended_score = (
        horizon_scores["short"] * 0.50
        + horizon_scores["medium"] * 0.30
        + horizon_scores["long"] * 0.20
    )
    if not usable:
        blended_score = raw_score
    score_adjustment = _clamp(blended_score * max_adjustment, -max_adjustment, max_adjustment)
    if len(usable) < max(0, int(min_providers)):
        warnings.append("minimum external signal provider count not met; positive boost disabled")
        score_adjustment = min(0.0, score_adjustment)
    reaction_component = max(usable, key=lambda component: component.urgency, default=None)
    reaction_required = bool(
        reaction_component is not None
        and reaction_component.horizon == "short"
        and reaction_component.urgency >= 0.80
        and abs(reaction_component.score) >= 0.50
    )
    reaction_reason = (
        f"{reaction_component.provider} score={reaction_component.score:+.2f} urgency={reaction_component.urgency:.2f}"
        if reaction_required and reaction_component is not None
        else ""
    )
    risk_reference = min(raw_score, horizon_scores["short"])
    risk_multiplier = 1.0 if risk_reference >= 0.0 else _clamp(1.0 + risk_reference * 0.35, 0.50, 1.0)
    status = "ok" if usable and not warnings else ("warn" if usable else "fail")
    return ExternalSignalReport(
        status=status,
        score_adjustment=float(score_adjustment),
        raw_score=float(raw_score),
        risk_multiplier=float(risk_multiplier),
        provider_count=len(components),
        fresh_count=len(usable),
        stale_count=stale_count,
        known_at_ms=now_ms,
        cache_path=str(cache_path),
        warnings=warnings,
        components=components,
        short_term_score=float(horizon_scores["short"]),
        medium_term_score=float(horizon_scores["medium"]),
        long_term_score=float(horizon_scores["long"]),
        reaction_required=reaction_required,
        reaction_reason=reaction_reason,
        news_backend_requested=news_backend.requested,
        news_backend_kind=news_backend.kind,
        news_backend_device=news_backend.device,
        news_backend_reason=news_backend.reason,
        news_ai_enabled=bool(news_ai_enabled),
        news_ai_status=str(news_ai_status),
        news_ai_model=str(news_ai_model or DEFAULT_OLLAMA_NEWS_MODEL),
        news_ai_latency_ms=max(0, int(news_ai_latency_ms)),
        news_ai_reason=str(news_ai_reason or ""),
    )


def collect_external_signals(
    *,
    symbol: str = "BTCUSDC",
    cache_path: str | Path = "data/signals/external_signals.json",
    ttl_seconds: int = 300,
    timeout_seconds: float = 3.0,
    max_adjustment: float = 0.04,
    min_providers: int = 1,
    force_refresh: bool = False,
    compute_backend: str | None = None,
    short_reaction_refresh_seconds: int = 30,
    fetch_json: FetchJson = _get_json,
    fetch_text: FetchText | None = None,
    news_provider_limit: int = len(RSS_NEWS_FEEDS),
    news_items_per_provider: int = 4,
    news_provider_parallelism: int = 12,
    news_provider_jitter_seconds: float = 0.0,
    ollama_news_enabled: bool = False,
    ollama_model: str = DEFAULT_OLLAMA_NEWS_MODEL,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    ollama_timeout_seconds: float = 3.0,
    post_json: PostJson = _post_json,
    telemetry_path: str | Path | None = None,
    source_grade_max_age_hours: float = 168.0,
    now_ms: int | None = None,
) -> ExternalSignalReport:
    now = _now_ms() if now_ms is None else int(now_ms)
    cache = Path(cache_path)
    source_grade_warnings: list[str] = []

    def apply_source_grades(components_to_grade: list[ExternalSignalComponent]) -> list[ExternalSignalComponent]:
        if telemetry_path is None:
            return components_to_grade
        try:
            from .telemetry_store import TradingTelemetryStore

            max_age_hours = _safe_float(source_grade_max_age_hours, 168.0)
            max_age_ms = None if max_age_hours <= 0.0 else int(max_age_hours * 3_600_000)
            with TradingTelemetryStore(telemetry_path) as store:
                return _apply_source_grade_weights(
                    components_to_grade,
                    store.latest_source_grades(max_age_ms=max_age_ms, now_ms=now),
                )
        except (ImportError, OSError, RuntimeError, TypeError, ValueError, sqlite3.Error) as exc:  # pragma: no cover - telemetry must never block trading signals
            source_grade_warnings.append(f"source grade weighting unavailable: {exc.__class__.__name__}")
            return components_to_grade

    def record_report_telemetry(report_to_record: ExternalSignalReport, raw_records_to_record: list[object]) -> None:
        if telemetry_path is None:
            return
        try:
            from .telemetry_store import TradingTelemetryStore

            with TradingTelemetryStore(telemetry_path) as store:
                store.record_signal_report(report_to_record, raw_payloads=raw_records_to_record)
        except (ImportError, OSError, RuntimeError, TypeError, ValueError, sqlite3.Error) as exc:  # pragma: no cover - telemetry must never block trading signals
            _LOGGER.debug("external signal telemetry recording skipped: %s", exc)

    if not force_refresh:
        cached = load_external_signal_cache(
            cache,
            now_ms=now,
            ttl_seconds=ttl_seconds,
            short_reaction_refresh_seconds=short_reaction_refresh_seconds,
        )
        if cached is not None:
            if telemetry_path is None:
                return cached
            graded_components = apply_source_grades([_base_component_for_regrading(component) for component in cached.components])
            cached_report = _combine_components(
                graded_components,
                max_adjustment=max_adjustment,
                min_providers=min_providers,
                now_ms=now,
                cache_path=cache,
                news_backend=BackendInfo(
                    requested=cached.news_backend_requested,
                    kind=cached.news_backend_kind,
                    device=cached.news_backend_device,
                    vendor="cached",
                    reason=cached.news_backend_reason,
                ),
                news_ai_enabled=cached.news_ai_enabled,
                news_ai_status=cached.news_ai_status,
                news_ai_model=cached.news_ai_model,
                news_ai_latency_ms=cached.news_ai_latency_ms,
                news_ai_reason=cached.news_ai_reason,
                extra_warnings=[*cached.warnings, *source_grade_warnings],
            )
            cached_report = replace(cached_report, status="cached")
            record_report_telemetry(cached_report, [{"provider": "external_signal_cache", "known_at_ms": now, "payload": cached.asdict()}])
            return cached_report

    news_backend = resolve_backend(compute_backend or "cpu")
    raw_records: list[object] = []
    raw_records_lock = Lock()

    def record_fetch_json(url: str, timeout: float) -> object:
        payload = fetch_json(url, timeout)
        provider = _provider_name_for_url(url)
        with raw_records_lock:
            raw_records.append({
                "provider": provider,
                "horizon": _provider_horizon(provider),
                "known_at_ms": now,
                "url": url,
                "payload": payload,
            })
        return payload

    fetchers = [
        ("alternative_fear_greed", "long", lambda: _fetch_alternative_fng(record_fetch_json, timeout_seconds, now)),
        ("coingecko_bitcoin", "medium", lambda: _fetch_coingecko_btc(record_fetch_json, timeout_seconds, now)),
        ("coinpaprika_bitcoin", "medium", lambda: _fetch_coinpaprika_btc(record_fetch_json, timeout_seconds, now)),
        ("coinlore_bitcoin", "medium", lambda: _fetch_coinlore_btc(record_fetch_json, timeout_seconds, now)),
        ("blockchain_network_stats", "long", lambda: _fetch_blockchain_stats(record_fetch_json, timeout_seconds, now)),
        ("kraken_btcusd_momentum", "medium", lambda: _fetch_kraken_btcusd(record_fetch_json, timeout_seconds, now)),
        ("coinbase_btcusd_momentum", "medium", lambda: _fetch_coinbase_btcusd(record_fetch_json, timeout_seconds, now)),
        ("bitstamp_btcusd_momentum", "medium", lambda: _fetch_bitstamp_btcusd(record_fetch_json, timeout_seconds, now)),
        ("binance_spot_momentum", "short", lambda: _fetch_binance_spot_momentum(record_fetch_json, timeout_seconds, now, symbol)),
        ("binance_futures_positioning", "short", lambda: _fetch_binance_derivatives(record_fetch_json, timeout_seconds, now, symbol)),
        ("mempool_fee_pressure", "short", lambda: _fetch_mempool_fees(record_fetch_json, timeout_seconds, now)),
        ("cryptocompare_btc_news", "short", lambda: _fetch_cryptocompare_news(record_fetch_json, timeout_seconds, now, compute_backend)),
        ("gdelt_bitcoin_news", "short", lambda: _fetch_gdelt_news(record_fetch_json, timeout_seconds, now, compute_backend)),
        ("hackernews_bitcoin_attention", "short", lambda: _fetch_hackernews_bitcoin(record_fetch_json, timeout_seconds, now, compute_backend)),
    ]
    components: list[ExternalSignalComponent] = []
    news_texts: list[str] = []
    news_known_at_ms: list[int] = []
    structured_results: list[tuple[int, ExternalSignalComponent, BackendInfo | None]] = []
    structured_workers = min(
        len(fetchers),
        max(1, min(16, int(news_provider_parallelism))),
    )
    structured_jitter_seconds = max(0.0, float(news_provider_jitter_seconds))

    def run_structured_fetcher(fetcher: Callable[[], object]) -> object:
        if structured_jitter_seconds > 0.0:
            time.sleep(_JITTER_RANDOM.uniform(0.0, structured_jitter_seconds))
        return fetcher()

    with ThreadPoolExecutor(max_workers=structured_workers) as executor:
        future_map = {
            executor.submit(run_structured_fetcher, fetcher): (index, provider, horizon)
            for index, (provider, horizon, fetcher) in enumerate(fetchers)
        }
        for future in as_completed(future_map):
            index, provider, horizon = future_map[future]
            try:
                fetched = future.result()
                if isinstance(fetched, tuple):
                    component, backend = fetched
                    structured_results.append((index, component, backend))
                else:
                    structured_results.append((index, fetched, None))
            except Exception as exc:
                structured_results.append((index, _error_component(provider, exc, known_at_ms=now, horizon=horizon), None))
    for _index, component, backend in sorted(structured_results, key=lambda item: item[0]):
        components.append(component)
        if backend is not None:
            news_backend = backend

    for raw_record in list(raw_records):
        if not isinstance(raw_record, Mapping):  # pragma: no cover - defensive guard for future callers
            continue
        provider = str(raw_record.get("provider") or "")
        structured_news_items = _structured_news_items_from_payload(
            provider,
            raw_record.get("payload"),
            limit=max(1, min(10, int(news_items_per_provider))),
            now_ms=now,
        )
        news_texts.extend(text for text, _known_at in structured_news_items)
        news_known_at_ms.extend(known_at for _text, known_at in structured_news_items)

    effective_fetch_text = fetch_text
    if effective_fetch_text is None and fetch_json is _get_json:  # pragma: no cover - real CLI default path
        effective_fetch_text = _get_text
    rss_limit = max(0, min(len(RSS_NEWS_FEEDS), int(news_provider_limit)))
    if rss_limit and effective_fetch_text is not None:
        for result in _fetch_rss_news_feeds(
            RSS_NEWS_FEEDS[:rss_limit],
            effective_fetch_text,
            timeout_seconds,
            now,
            compute_backend,
            items_per_provider=max(1, min(10, int(news_items_per_provider))),
            max_workers=max(1, min(48, int(news_provider_parallelism))),
            provider_jitter_seconds=max(0.0, float(news_provider_jitter_seconds)),
        ):
            components.append(result.component)
            if result.backend is not None:
                news_backend = result.backend
            news_texts.extend(result.news_texts)
            result_known_at = list(result.news_known_at_ms)
            result_known_at.extend([result.component.known_at_ms] * max(0, len(result.news_texts) - len(result_known_at)))
            news_known_at_ms.extend(result_known_at[: len(result.news_texts)])
            if result.raw_payload is not None:
                raw_records.append(result.raw_payload)

    news_ai_status = "disabled"
    news_ai_latency_ms = 0
    news_ai_reason = ""
    if ollama_news_enabled:
        try:
            evaluation = _evaluate_news_with_ollama(
                news_texts,
                news_known_at_ms=news_known_at_ms,
                model=ollama_model,
                base_url=ollama_url,
                timeout_seconds=ollama_timeout_seconds,
                post_json=post_json,
                now_ms=now,
            )
            components.append(evaluation.component)
            news_ai_status = evaluation.status
            news_ai_latency_ms = evaluation.latency_ms
            news_ai_reason = evaluation.reason
            if evaluation.raw_payload is not None:
                raw_records.append(evaluation.raw_payload)
        except Exception as exc:
            components.append(_error_component("ollama_news_ai", exc, known_at_ms=now, horizon="short"))
            news_ai_status = "error"
            news_ai_reason = str(exc)[:180]
            raw_records.append(
                {
                    "provider": "ollama_news_ai",
                    "horizon": "short",
                    "known_at_ms": now,
                    "model": str(ollama_model or DEFAULT_OLLAMA_NEWS_MODEL),
                    "endpoint": f"{str(ollama_url or DEFAULT_OLLAMA_URL).rstrip('/')}/api/chat",
                    "status": "error",
                    "error": news_ai_reason,
                    "selected_news_texts": _bounded_ollama_news_texts(news_texts),
                    "news_count": len(news_texts),
                }
            )

    components = apply_source_grades(components)

    report = _combine_components(
        components,
        max_adjustment=max_adjustment,
        min_providers=min_providers,
        now_ms=now,
        cache_path=cache,
        news_backend=news_backend,
        news_ai_enabled=ollama_news_enabled,
        news_ai_status=news_ai_status,
        news_ai_model=ollama_model,
        news_ai_latency_ms=news_ai_latency_ms,
        news_ai_reason=news_ai_reason,
        extra_warnings=source_grade_warnings,
    )
    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(cache, report.asdict(), indent=2, sort_keys=True)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        warning = f"external signal cache write unavailable: {exc.__class__.__name__}"
        _LOGGER.debug("%s", warning)
        report = replace(
            report,
            status="warn" if report.status == "ok" else report.status,
            warnings=[*report.warnings, warning],
        )
    record_report_telemetry(report, raw_records)
    return report


def render_external_signal_report(report: ExternalSignalReport) -> str:
    lines = [
        "External signal report",
        (
            f"status={report.status} providers={report.fresh_count}/{report.provider_count} "
            f"score_adjustment={report.score_adjustment:+.4f} "
            f"risk_multiplier={report.risk_multiplier:.3f}"
        ),
        (
            f"horizons short={report.short_term_score:+.3f} "
            f"medium={report.medium_term_score:+.3f} long={report.long_term_score:+.3f} "
            f"reaction={'yes' if report.reaction_required else 'no'}"
        ),
        (
            f"news_backend={report.news_backend_kind} "
            f"device={report.news_backend_device}"
        ),
    ]
    if report.news_backend_reason:
        lines.append(f"news_backend_reason={report.news_backend_reason}")
    if report.news_ai_enabled:
        lines.append(
            f"news_ai={report.news_ai_status} model={report.news_ai_model} "
            f"latency_ms={report.news_ai_latency_ms}"
        )
        if report.news_ai_reason:
            lines.append(f"news_ai_reason={report.news_ai_reason}")
    if report.reaction_required and report.reaction_reason:
        lines.append(f"reaction_reason={report.reaction_reason}")
    if report.cache_path:
        lines.append(f"cache={report.cache_path}")
    for component in report.components:
        cache_note = " cached" if component.cached else ""
        symbol_note = f" symbol={component.source_symbol}" if component.source_symbol else ""
        if component.status == "ok":
            lines.append(
                f"- {component.provider}{cache_note}: score={component.score:+.3f} "
                f"weight={component.weight:.2f} horizon={component.horizon} "
                f"urgency={component.urgency:.2f}{symbol_note} {component.detail}"
            )
        else:
            lines.append(f"- {component.provider}: {component.status} {component.error or component.detail}")
    for warning in report.warnings:
        lines.append(f"warning: {warning}")
    return "\n".join(lines)
