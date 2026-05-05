from __future__ import annotations

import json

import pytest


from simple_ai_bitcoin_trading_binance import external_signals as signals


NOW_MS = 1_700_000_000_000


def _good_fetch(url: str, timeout: float):
    assert timeout > 0
    if "alternative.me" in url:
        return {"data": [{"value": "25", "value_classification": "Fear", "timestamp": str(NOW_MS // 1000)}]}
    if "coingecko" in url:
        return {
            "bitcoin": {
                "usd": "100000",
                "usd_24h_change": "3.0",
                "usd_24h_vol": "123456",
                "last_updated_at": NOW_MS // 1000,
            }
        }
    if "coinpaprika" in url:
        return {
            "quotes": {
                "USD": {
                    "price": 100000,
                    "volume_24h": 123456,
                    "percent_change_1h": 0.5,
                    "percent_change_24h": 3.0,
                    "percent_change_7d": 6.0,
                }
            },
            "last_updated": "2023-11-14T22:13:20Z",
        }
    if "coinlore" in url:
        return [
            {
                "price_usd": "100000",
                "volume24": "123456",
                "percent_change_1h": "0.5",
                "percent_change_24h": "3.0",
                "percent_change_7d": "6.0",
            }
        ]
    if "blockchain.info/stats" in url:
        return {"n_tx": 500000, "hash_rate": 875000000000, "minutes_between_blocks": 9.5, "mempool_size": 1000}
    if "kraken" in url:
        return {
            "error": [],
            "result": {
                "XXBTZUSD": {
                    "c": ["103000", "0.1"],
                    "o": "100000",
                    "v": ["10", "20"],
                }
            },
        }
    if "exchange.coinbase" in url:
        return {"last": "103000", "open": "100000", "high": "104000", "low": "99000", "volume": "20"}
    if "bitstamp" in url:
        return {"last": "103000", "open": "100000", "percent_change_24": "3.0", "volume": "20"}
    if "api.binance.com/api/v3/ticker/24hr" in url:
        return {
            "symbol": "BTCUSDC",
            "lastPrice": "103000",
            "priceChangePercent": "3.0",
            "volume": "20",
            "quoteVolume": "2000000",
            "closeTime": NOW_MS,
        }
    if "premiumIndex" in url:
        return {"lastFundingRate": "0.0001", "markPrice": "100.5", "indexPrice": "100", "time": NOW_MS}
    if "openInterest" in url:
        return {"openInterest": "987.5"}
    if "mempool" in url:
        return {"fastestFee": 10, "halfHourFee": 12}
    if "cryptocompare" in url:
        return {
            "Data": [
                {
                    "title": "Bitcoin ETF inflow sparks institutional adoption rally",
                    "body": "Analysts cite buying and reserve demand.",
                    "tags": "BTC|ETF",
                    "source": "CryptoWire",
                    "published_on": NOW_MS // 1000,
                }
            ]
        }
    if "gdeltproject" in url:
        return {
            "articles": [
                {
                    "title": "Bitcoin rally follows approval and institutional buying",
                    "domain": "example.com",
                    "seendate": "20231114T221320Z",
                }
            ]
        }
    if "hn.algolia" in url:
        return {
            "hits": [
                {
                    "title": "Bitcoin upgrade discussion avoids hack concerns",
                    "points": 125,
                    "created_at": "2023-11-14T22:13:20Z",
                }
            ]
        }
    raise AssertionError(url)


def _feed_xml(title: str = "Bitcoin ETF approval rally", summary: str = "institutional buying") -> str:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
    <rss><channel>
      <item>
        <title>{title}</title>
        <description>{summary}</description>
        <pubDate>Tue, 14 Nov 2023 22:13:20 GMT</pubDate>
        <link>https://example.test/1</link>
      </item>
      <item>
        <title>Bitcoin hack liquidation warning</title>
        <description>exchange exploit and selloff risk</description>
        <pubDate>Tue, 14 Nov 2023 22:12:20 GMT</pubDate>
      </item>
    </channel></rss>"""


def test_collect_external_signals_success_cache_and_render(tmp_path) -> None:
    cache = tmp_path / "signals.json"
    report = signals.collect_external_signals(
        cache_path=cache,
        fetch_json=_good_fetch,
        force_refresh=True,
        min_providers=2,
        now_ms=NOW_MS,
    )
    assert report.status == "ok"
    assert report.provider_count == 14
    assert report.fresh_count == 14
    assert report.score_adjustment > 0
    assert report.short_term_score != 0.0
    assert report.medium_term_score != 0.0
    assert report.long_term_score != 0.0
    assert report.reaction_required is True
    assert report.news_backend_kind == "cpu"
    assert cache.exists()
    text = signals.render_external_signal_report(report)
    assert "alternative_fear_greed" in text
    assert "score_adjustment=" in text
    assert "horizons short=" in text
    assert "reaction_reason=" in text

    cached = signals.collect_external_signals(
        cache_path=cache,
        fetch_json=lambda _url, _timeout: (_ for _ in ()).throw(AssertionError("cache not used")),
        ttl_seconds=300,
        short_reaction_refresh_seconds=30,
        now_ms=NOW_MS + 1_000,
    )
    assert cached.status == "cached"
    assert all(component.cached for component in cached.components)
    refreshed = signals.collect_external_signals(
        cache_path=cache,
        fetch_json=_good_fetch,
        ttl_seconds=300,
        short_reaction_refresh_seconds=1,
        now_ms=NOW_MS + 5_000,
    )
    assert refreshed.status == "ok"


def test_collect_external_signals_returns_report_when_cache_write_fails(tmp_path, monkeypatch) -> None:
    def fail_write(*_args, **_kwargs) -> None:
        raise OSError("read-only cache")

    monkeypatch.setattr(signals, "write_json_atomic", fail_write)
    report = signals.collect_external_signals(
        cache_path=tmp_path / "missing" / "signals.json",
        fetch_json=_good_fetch,
        force_refresh=True,
        now_ms=NOW_MS,
    )

    assert report.fresh_count == 14
    assert report.status == "warn"
    assert any("cache write unavailable" in warning for warning in report.warnings)


def test_collect_external_signals_rss_ollama_and_telemetry(tmp_path) -> None:
    def fetch_text(url: str, timeout: float) -> str:
        assert url.startswith("https://")
        assert timeout > 0
        return _feed_xml()

    def post_json(url: str, payload: dict[str, object], timeout: float):
        assert url.endswith("/api/chat")
        assert payload["model"] == "gemma4:e4b"
        assert payload["keep_alive"] == "30m"
        assert payload["think"] is False
        assert payload["options"]["num_ctx"] == 512
        assert payload["options"]["num_predict"] == 48
        messages = str(payload["messages"])
        assert messages.count("- ") <= 6
        assert "Bitcoin ETF inflow sparks institutional adoption rally" in messages
        assert "coingecko_bitcoin" not in messages
        assert timeout == 2.0
        return {
            "message": {
                "content": json.dumps(
                    {
                        "score": -0.7,
                        "horizon": "short-term",
                        "reaction_required": "true",
                        "reason": "Hack risk outweighs inflows",
                    }
                )
            }
        }

    telemetry = tmp_path / "telemetry.sqlite"
    report = signals.collect_external_signals(
        cache_path=tmp_path / "signals.json",
        fetch_json=_good_fetch,
        fetch_text=fetch_text,
        force_refresh=True,
        min_providers=30,
        now_ms=NOW_MS,
        news_provider_limit=30,
        news_provider_parallelism=4,
        news_provider_jitter_seconds=0.0,
        ollama_news_enabled=True,
        ollama_model="gemma4:e4b",
        ollama_timeout_seconds=2.0,
        post_json=post_json,
        telemetry_path=telemetry,
    )
    assert report.status == "ok"
    assert report.provider_count == 45
    assert report.fresh_count == 45
    assert report.news_ai_status == "ok"
    assert report.news_ai_model == "gemma4:e4b"
    assert report.reaction_required is True
    text = signals.render_external_signal_report(report)
    assert "news_ai=ok" in text
    from simple_ai_bitcoin_trading_binance.telemetry_store import TradingTelemetryStore

    with TradingTelemetryStore(telemetry) as store:
        observations = store.recent_observations(since_ms=NOW_MS - 10_000, limit=500)
    assert any(item.kind == "raw_provider_payload" for item in observations)
    assert any(
        item.kind == "raw_provider_payload"
        and item.source == "cryptocompare_btc_news"
        and item.horizon == "short"
        and item.observed_at_ms == NOW_MS
        for item in observations
    )
    assert any(
        item.kind == "raw_provider_payload"
        and isinstance(item.payload, dict)
        and item.payload.get("classifications")
        for item in observations
    )
    ollama_payloads = [
        item.payload
        for item in observations
        if item.kind == "raw_provider_payload"
        and isinstance(item.payload, dict)
        and item.payload.get("request", {}).get("model") == "gemma4:e4b"
    ]
    assert ollama_payloads
    assert "prompt" in ollama_payloads[0]["request"]
    assert "selected_news_texts" in ollama_payloads[0]["request"]
    assert any(isinstance(item.payload, dict) and "parsed" in item.payload for item in observations)


def test_collect_external_signals_uses_source_grades_for_live_weights(tmp_path) -> None:
    from simple_ai_bitcoin_trading_binance.telemetry_store import TradingTelemetryStore

    telemetry = tmp_path / "graded.sqlite"
    with TradingTelemetryStore(telemetry) as store:
        high = store.record_source_grade(
            source="coingecko_bitcoin",
            horizon="medium",
            window_start_ms=NOW_MS - 3_600_000,
            window_end_ms=NOW_MS - 1,
            grade=10,
            sample_count=8,
            model="gemma4:e4b",
            reason="strong source",
            evidence={"hit_rate": 0.9},
        )
        low = store.record_source_grade(
            source="cryptocompare_btc_news",
            horizon="short",
            window_start_ms=NOW_MS - 3_600_000,
            window_end_ms=NOW_MS - 1,
            grade=0,
            sample_count=8,
            model="gemma4:e4b",
            reason="bad short-term source",
            evidence={"hit_rate": 0.1},
        )
        store.connect().execute(
            "UPDATE source_grades SET created_at_ms = ? WHERE id IN (?, ?)",
            (NOW_MS - 1_000, high.id, low.id),
        )
        store.connect().commit()

    report = signals.collect_external_signals(
        cache_path=tmp_path / "graded-signals.json",
        fetch_json=_good_fetch,
        force_refresh=True,
        now_ms=NOW_MS,
        telemetry_path=telemetry,
        source_grade_max_age_hours=1.0,
    )
    coingecko = [component for component in report.components if component.provider == "coingecko_bitcoin"][0]
    cryptocompare = [component for component in report.components if component.provider == "cryptocompare_btc_news"][0]
    assert coingecko.source_grade == 10
    assert coingecko.source_grade_model == "gemma4:e4b"
    assert coingecko.source_grade_weight_multiplier == pytest.approx(1.25)
    assert coingecko.weight == pytest.approx(0.875)
    assert "source_grade=10/10" in coingecko.detail
    assert cryptocompare.source_grade == 0
    assert cryptocompare.source_grade_weight_multiplier == pytest.approx(0.25)
    assert cryptocompare.weight == pytest.approx(0.1375)
    cached = signals.collect_external_signals(
        cache_path=tmp_path / "graded-signals.json",
        fetch_json=lambda _url, _timeout: (_ for _ in ()).throw(AssertionError("cache not used")),
        ttl_seconds=300,
        short_reaction_refresh_seconds=30,
        now_ms=NOW_MS + 1_000,
        telemetry_path=telemetry,
        source_grade_max_age_hours=1.0,
    )
    cached_coingecko = [component for component in cached.components if component.provider == "coingecko_bitcoin"][0]
    assert cached.status == "cached"
    assert cached_coingecko.cached is True
    assert cached_coingecko.source_grade == 10
    assert cached_coingecko.source_grade_weight_multiplier == pytest.approx(1.25)
    with TradingTelemetryStore(telemetry) as store:
        cached_observations = store.recent_observations(since_ms=NOW_MS, limit=500)
    assert any(
        item.kind == "external_signal_component"
        and item.source == "coingecko_bitcoin"
        and isinstance(item.payload, dict)
        and item.payload.get("cached") is True
        for item in cached_observations
    )
    assert any(item.kind == "raw_provider_payload" and item.source == "external_signal_cache" for item in cached_observations)

    unbounded = signals.collect_external_signals(
        cache_path=tmp_path / "graded-signals-unbounded.json",
        fetch_json=_good_fetch,
        force_refresh=True,
        now_ms=NOW_MS,
        telemetry_path=telemetry,
        source_grade_max_age_hours=0.0,
    )
    assert [component for component in unbounded.components if component.provider == "coingecko_bitcoin"][0].source_grade == 10


def test_rss_provider_parser_jitter_and_ollama_error(tmp_path, monkeypatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(signals._JITTER_RANDOM, "uniform", lambda _low, high: high)
    monkeypatch.setattr(signals.time, "sleep", lambda seconds: sleeps.append(seconds))
    result = signals._fetch_rss_news_feed(
        signals.NewsFeedProvider("feed", "https://example.test/rss", 0.5),
        lambda _url, _timeout: _feed_xml("Bitcoin adoption upgrade", "reserve approval"),
        1.0,
        NOW_MS,
        "cpu",
        provider_jitter_seconds=0.25,
    )
    assert sleeps == [0.25]
    assert result.component.provider == "feed"
    assert result.component.status == "ok"
    assert result.raw_payload is not None and "raw_xml" in result.raw_payload
    failed_feed = signals._fetch_rss_news_feeds(
        (signals.NewsFeedProvider("bad_feed", "https://example.test/bad.xml"),),
        lambda _url, _timeout: "<rss><channel><item><title>broken",
        1.0,
        NOW_MS,
        "cpu",
        max_workers=1,
    )[0]
    assert failed_feed.component.status == "error"
    assert isinstance(failed_feed.raw_payload, dict)
    assert failed_feed.raw_payload["raw_xml"].startswith("<rss>")
    fetch_failed = signals._fetch_rss_news_feeds(
        (signals.NewsFeedProvider("offline_feed", "https://example.test/offline.xml"),),
        lambda _url, _timeout: (_ for _ in ()).throw(RuntimeError("offline")),
        1.0,
        NOW_MS,
        "cpu",
        max_workers=1,
    )[0]
    assert isinstance(fetch_failed.raw_payload, dict)
    assert "raw_xml" not in fetch_failed.raw_payload
    assert signals._extract_feed_items("\ufeff<feed><entry><title>Atom Bitcoin ban</title><updated>2023-11-14T22:13:20Z</updated><link href='https://x.test'/></entry></feed>", now_ms=NOW_MS)[0]["title"] == "Atom Bitcoin ban"

    report = signals.collect_external_signals(
        cache_path=tmp_path / "ollama-error.json",
        fetch_json=_good_fetch,
        fetch_text=lambda _url, _timeout: _feed_xml(),
        force_refresh=True,
        now_ms=NOW_MS,
        news_provider_limit=1,
        ollama_news_enabled=True,
        post_json=lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("ollama down")),
    )
    assert report.news_ai_status == "error"
    assert any(component.provider == "ollama_news_ai" and component.status == "error" for component in report.components)


def test_structured_providers_apply_jitter(tmp_path, monkeypatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(signals._JITTER_RANDOM, "uniform", lambda _low, high: high)
    monkeypatch.setattr(signals.time, "sleep", lambda seconds: sleeps.append(seconds))

    report = signals.collect_external_signals(
        cache_path=tmp_path / "structured-jitter.json",
        fetch_json=_good_fetch,
        force_refresh=True,
        now_ms=NOW_MS,
        news_provider_limit=0,
        news_provider_jitter_seconds=0.25,
    )

    assert report.provider_count == 14
    assert len(sleeps) == 14
    assert set(sleeps) == {0.25}


def test_rss_scheduler_paces_same_host_requests(monkeypatch) -> None:
    sleeps: list[float] = []
    fetch_times: dict[str, float] = {}
    clock = {"now": 0.0}

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        clock["now"] += seconds

    monkeypatch.setattr(signals._JITTER_RANDOM, "uniform", lambda _low, high: high / 2.0)
    monkeypatch.setattr(signals.time, "perf_counter", lambda: clock["now"])
    monkeypatch.setattr(signals.time, "sleep", sleep)

    def fetch_text(url: str, _timeout: float) -> str:
        fetch_times[url] = clock["now"]
        return _feed_xml()

    results = signals._fetch_rss_news_feeds(
        (
            signals.NewsFeedProvider("same_one", "https://same.test/one.xml"),
            signals.NewsFeedProvider("same_two", "https://same.test/two.xml"),
            signals.NewsFeedProvider("other", "https://other.test/rss.xml"),
        ),
        fetch_text,
        1.0,
        NOW_MS,
        "cpu",
        max_workers=1,
        provider_jitter_seconds=0.4,
    )

    assert [result.component.provider for result in results] == ["same_one", "same_two", "other"]
    assert len(sleeps) >= 3
    assert fetch_times["https://same.test/two.xml"] - fetch_times["https://same.test/one.xml"] >= 0.4


def test_rss_scheduler_accepts_hostless_provider_without_pacing(monkeypatch) -> None:
    monkeypatch.setattr(signals.time, "sleep", lambda seconds: pytest.fail(f"unexpected sleep {seconds}"))

    results = signals._fetch_rss_news_feeds(
        (signals.NewsFeedProvider("hostless", "feed.xml"),),
        lambda _url, _timeout: _feed_xml(),
        1.0,
        NOW_MS,
        "cpu",
        max_workers=1,
        provider_jitter_seconds=0.0,
    )

    assert [result.component.provider for result in results] == ["hostless"]


def test_news_feed_helpers_and_ollama_validation(tmp_path, monkeypatch) -> None:
    assert len(signals.RSS_NEWS_FEEDS) >= 90
    assert len(signals.RSS_NEWS_FEEDS) + 14 >= 100
    assert len({provider.provider for provider in signals.RSS_NEWS_FEEDS}) == len(signals.RSS_NEWS_FEEDS)
    assert all(provider.url.startswith("https://") for provider in signals.RSS_NEWS_FEEDS)
    assert signals._parse_feed_datetime_ms("", NOW_MS) == NOW_MS
    assert signals._parse_feed_datetime_ms("Tue, 14 Nov 2023 22:13:20", 0) == NOW_MS
    assert signals._parse_feed_datetime_ms("2023-11-14T22:13:20", 0) == NOW_MS
    assert signals._parse_feed_datetime_ms("not a date", NOW_MS) == NOW_MS
    assert signals._child_text(signals.ET.fromstring("<item><x /></item>"), "title") == ""
    assert signals._strip_markup("<b>Bitcoin</b>&amp;BTC") == "Bitcoin &BTC"
    assert signals._provider_name_for_url("https://example.test/path") == "example_test"
    assert signals._provider_name_for_url(signals.COINGECKO_SIMPLE_PRICE_URL) == "coingecko_bitcoin"
    assert signals._provider_name_for_url(signals.COINPAPRIKA_BTC_TICKER_URL) == "coinpaprika_bitcoin"
    assert signals._provider_name_for_url(signals.COINLORE_BTC_TICKER_URL) == "coinlore_bitcoin"
    assert signals._provider_name_for_url(signals.BLOCKCHAIN_STATS_URL) == "blockchain_network_stats"
    assert signals._provider_name_for_url(signals.KRAKEN_BTCUSD_TICKER_URL) == "kraken_btcusd_momentum"
    assert signals._provider_name_for_url(signals.COINBASE_BTCUSD_STATS_URL) == "coinbase_btcusd_momentum"
    assert signals._provider_name_for_url(signals.BITSTAMP_BTCUSD_TICKER_URL) == "bitstamp_btcusd_momentum"
    assert signals._provider_name_for_url(f"{signals.BINANCE_SPOT_BASE_URL}/api/v3/ticker/24hr?symbol=BTCUSDC") == "binance_spot_momentum"
    assert signals._provider_name_for_url(signals.MEMPOOL_FEES_URL) == "mempool_fee_pressure"
    assert signals._fetch_rss_news_feeds((), lambda _url, _timeout: "", 1.0, NOW_MS, "cpu") == []
    failures = signals._fetch_rss_news_feeds(
        (signals.NewsFeedProvider("bad_feed", "https://bad.test/rss"),),
        lambda _url, _timeout: "<rss><channel /></rss>",
        1.0,
        NOW_MS,
        "cpu",
    )
    assert failures[0].component.status == "error"
    with pytest.raises(ValueError, match="usable items"):
        signals._fetch_rss_news_feed(
            signals.NewsFeedProvider("empty_title", "https://empty.test/rss"),
            lambda _url, _timeout: "<rss><channel><item><description>x</description></item></channel></rss>",
            1.0,
            NOW_MS,
            "cpu",
        )
    assert signals._normalize_horizon("long-term") == "long"
    assert signals._normalize_horizon("later") == "medium"
    assert signals._coerce_ai_reaction(False, 0.9, "short") is False
    assert signals._coerce_ai_reaction("monitor", 0.9, "short") is False
    assert signals._coerce_ai_reaction("maybe", -0.7, "short") is True
    assert signals._json_mapping_from_text("prefix {\"score\": 1} suffix")["score"] == 1
    assert signals._ollama_response_text({"message": {"content": "{\"score\":1}"}}) == "{\"score\":1}"
    assert signals._ollama_response_text({"message": {"content": None}, "response": "{\"score\":0}"}) == "{\"score\":0}"
    assert signals._ollama_response_text({"response": "{\"score\":-1}"}) == "{\"score\":-1}"
    with pytest.raises(json.JSONDecodeError):
        signals._json_mapping_from_text("no json here")
    with pytest.raises(ValueError, match="JSON object"):
        signals._json_mapping_from_text("[1]")
    with pytest.raises(ValueError, match="no news texts"):
        signals._evaluate_news_with_ollama([], now_ms=NOW_MS)
    with pytest.raises(ValueError, match="no usable news texts"):
        signals._evaluate_news_with_ollama(["   "], now_ms=NOW_MS)
    with pytest.raises(ValueError, match="unexpected Ollama"):
        signals._evaluate_news_with_ollama(
            ["Bitcoin rally"],
            post_json=lambda *_args, **_kwargs: [],
            now_ms=NOW_MS,
        )
    bounded = signals._bounded_ollama_news_texts(
        [
            "",
            "quiet macro update",
            "Bitcoin hack crackdown emergency",
            "bitcoin hack crackdown emergency",
            "BTC adoption approval",
        ],
        limit=2,
    )
    assert bounded == ["Bitcoin hack crackdown emergency", "BTC adoption approval"]
    prompt = signals._ollama_prompt([f"Bitcoin headline {index}" for index in range(20)])
    assert prompt.count("- ") == 6
    malformed = signals._evaluate_news_with_ollama(
        ["Breaking Bitcoin exchange breach drains reserves"],
        post_json=lambda *_args, **_kwargs: {"message": {"content": "not-json"}},
        now_ms=NOW_MS,
    )
    assert malformed.component.score < -0.9
    assert malformed.component.horizon == "short"
    assert malformed.component.urgency == pytest.approx(1.0)
    assert malformed.status == "warn"
    assert malformed.reason.startswith("schema_guarded security")
    assert malformed.raw_payload["provider"] == "ollama_news_ai"
    assert malformed.raw_payload["status"] == "warn"
    assert malformed.raw_payload["request"]["parse_error"]
    assert malformed.raw_payload["request"]["schema_guarded"] is True
    contradictory = signals._evaluate_news_with_ollama(
        ["Breaking Bitcoin exchange hack confirmed"],
        post_json=lambda *_args, **_kwargs: {
            "message": {
                "content": json.dumps(
                    {"score": 0.9, "horizon": "medium", "reaction_required": False, "reason": "bullish"}
                )
            }
        },
        now_ms=NOW_MS,
    )
    assert contradictory.component.score < -0.9
    assert contradictory.component.horizon == "short"
    assert contradictory.status == "warn"
    assert contradictory.reason == "guarded security signal"
    stale = signals._evaluate_news_with_ollama(
        ["Breaking Bitcoin exchange hack confirmed"],
        news_known_at_ms=[NOW_MS - 8 * 24 * 3_600_000],
        post_json=lambda *_args, **_kwargs: {
            "message": {
                "content": json.dumps(
                    {"score": -0.9, "horizon": "short", "reaction_required": True, "reason": "urgent"}
                )
            }
        },
        now_ms=NOW_MS,
    )
    assert stale.component.known_at_ms == NOW_MS - 8 * 24 * 3_600_000
    stale_report = signals._combine_components(
        [stale.component],
        max_adjustment=0.04,
        min_providers=0,
        now_ms=NOW_MS,
        cache_path=tmp_path / "stale-ollama.json",
        news_backend=signals.resolve_backend("cpu"),
    )
    assert stale_report.fresh_count == 0
    assert stale_report.reaction_required is False
    neutral = signals._evaluate_news_with_ollama(
        ["Bitcoin market structure update"],
        post_json=lambda *_args, **_kwargs: {
            "message": {"content": json.dumps({"score": 0.1, "horizon": "medium", "reaction_required": False})}
        },
        now_ms=NOW_MS,
    )
    assert neutral.reason == "news impact evaluated"

    class _Response:
        text = "hello"

        def raise_for_status(self) -> None:
            self.raised = True

        def json(self):
            return {"ok": True}

    observed: dict[str, object] = {}

    def fake_post(url: str, *, json: dict[str, object], timeout: float, headers: dict[str, str]):
        observed["post"] = (url, json, timeout, headers)
        return _Response()

    def fake_get(url: str, *, timeout: float, headers: dict[str, str]):
        observed["get"] = (url, timeout, headers)
        return _Response()

    monkeypatch.setattr(signals.requests, "post", fake_post)
    monkeypatch.setattr(signals.requests, "get", fake_get)
    assert signals._post_json("https://example.test", {"x": 1}, 0.0) == {"ok": True}
    assert signals._get_text("https://example.test", 0.0) == "hello"
    assert observed["post"][2] == pytest.approx(0.1, abs=0.01)
    assert observed["get"][1] == 0.1


def test_collect_external_signals_optional_none_payload_branches(tmp_path, monkeypatch) -> None:
    component = signals.ExternalSignalComponent(
        provider="rss_none",
        status="ok",
        score=0.1,
        weight=0.2,
        value=1.0,
        detail="ok",
        known_at_ms=NOW_MS,
    )
    monkeypatch.setattr(
        signals,
        "_fetch_rss_news_feeds",
        lambda *_args, **_kwargs: [signals.ProviderFetchResult(component=component)],
    )
    monkeypatch.setattr(
        signals,
        "_evaluate_news_with_ollama",
        lambda *_args, **_kwargs: signals.OllamaNewsEvaluation(
            component=signals.ExternalSignalComponent(
                provider="ollama_news_ai",
                status="ok",
                score=0.0,
                weight=0.1,
                value=0.0,
                detail="no reason",
                known_at_ms=NOW_MS,
            ),
            status="ok",
            model="gemma4:e4b",
            latency_ms=1,
            reason="",
            raw_payload=None,
        ),
    )
    report = signals.collect_external_signals(
        cache_path=tmp_path / "signals.json",
        fetch_json=_good_fetch,
        fetch_text=lambda _url, _timeout: _feed_xml(),
        force_refresh=True,
        now_ms=NOW_MS,
        news_provider_limit=1,
        ollama_news_enabled=True,
    )
    assert any(item.provider == "rss_none" for item in report.components)
    assert report.news_ai_reason == ""
    rendered = signals.render_external_signal_report(report)
    assert "news_ai=ok" in rendered


def test_external_signal_failures_min_provider_gate_and_fallback(tmp_path) -> None:
    def all_fail(_url: str, _timeout: float):
        raise RuntimeError("offline")

    failed = signals.collect_external_signals(
        cache_path=tmp_path / "failed.json",
        fetch_json=all_fail,
        force_refresh=True,
        now_ms=NOW_MS,
    )
    assert failed.status == "fail"
    assert failed.score_adjustment == 0.0
    assert len(failed.warnings) == 15
    assert "offline" in signals.render_external_signal_report(failed)

    def one_positive(url: str, _timeout: float):
        if "coingecko" in url:
            return {"bitcoin": {"usd": "100", "usd_24h_change": "5", "usd_24h_vol": "1", "last_updated_at": NOW_MS // 1000}}
        raise RuntimeError("offline")

    gated = signals.collect_external_signals(
        cache_path=tmp_path / "gated.json",
        fetch_json=one_positive,
        force_refresh=True,
        min_providers=2,
        now_ms=NOW_MS,
    )
    assert gated.status == "warn"
    assert gated.score_adjustment == 0.0
    assert any("minimum external signal provider" in warning for warning in gated.warnings)

    def fallback_fetch(url: str, _timeout: float):
        if "premiumIndex?symbol=BTCUSDC" in url:
            raise RuntimeError("missing btcusdc futures")
        if "premiumIndex?symbol=BTCUSDT" in url:
            return {"lastFundingRate": "0", "markPrice": "100", "indexPrice": "100", "time": NOW_MS}
        if "openInterest?symbol=BTCUSDT" in url:
            return {"openInterest": "1"}
        return _good_fetch(url, _timeout)

    fallback = signals.collect_external_signals(
        cache_path=tmp_path / "fallback.json",
        fetch_json=fallback_fetch,
        symbol="BTCUSDC",
        force_refresh=True,
        now_ms=NOW_MS,
    )
    binance = [component for component in fallback.components if component.provider == "binance_futures_positioning"][0]
    assert binance.source_symbol == "BTCUSDT"


def test_external_signal_payload_cache_and_helpers(tmp_path, monkeypatch) -> None:
    assert isinstance(signals._now_ms(), int)
    assert signals._safe_float("bad", 7.0) == 7.0
    assert signals._safe_float(float("inf"), 8.0) == 8.0
    assert signals._keyword_sentiment("Bitcoin adoption rally") > 0
    assert signals._keyword_sentiment("Bitcoin hack crackdown") < 0
    assert signals._keyword_sentiment("Bitcoin sideways") == 0
    security = signals._classify_news_text("Breaking SEC bitcoin ETF denial after exchange hack", age_ms=30_000)
    assert security.score < -0.9
    assert security.horizon == "short"
    assert security.importance >= 8
    assert security.urgency >= 0.85
    assert security.category in {"security", "regulatory"}
    assert "hack" in security.matched_terms
    assert "etf denial" in security.matched_terms
    negated = signals._classify_news_text("Bitcoin upgrade avoids hack concerns", age_ms=1_000)
    assert negated.score > 0
    assert negated.category == "technology"
    assert negated.importance > 0
    old_release = signals._classify_news_text("Bitcoin core release improves wallet behavior", age_ms=172_800_000)
    assert old_release.horizon == "long"
    assert 0 < old_release.importance <= 3
    fake_approval = signals._classify_news_text("Fake bitcoin ETF approval report denied by issuer", age_ms=1_000)
    assert fake_approval.score == 0.0
    assert fake_approval.category == "general"
    assert signals._recency_factor(600 * 60_000) == pytest.approx(0.58)
    assert signals._news_urgency("Bitcoin hack confirmed", age_ms=1_000) >= 0.9
    assert signals._news_horizon("Breaking Bitcoin ETF inflow", age_ms=1_000) == "short"
    assert signals._news_horizon("Breaking Bitcoin headline", age_ms=1_000) == "short"
    assert signals._news_horizon("Bitcoin sideways", age_ms=3_600_000) == "medium"
    assert signals._news_horizon("Bitcoin sideways", age_ms=172_800_000) == "long"
    assert signals._average_sentiment([(1.0, 1.0), (-1.0, 1.0)]) == 0.0
    assert signals._average_sentiment([]) == 0.0
    default_classification = signals._dominant_news_classification([], default_horizon="long")
    assert default_classification.horizon == "long"
    assert default_classification.importance == 0
    scores, backend = signals._score_news_texts(["Bitcoin adoption rally", "Bitcoin hack"], compute_backend="cpu", ages_ms=[0, 1_000])
    assert scores == [1.0, -1.0]
    assert backend.kind == "cpu"
    assert signals._grade_weight_multiplier(0) == pytest.approx(0.25)
    assert signals._grade_weight_multiplier(5) == pytest.approx(1.0)
    assert signals._grade_weight_multiplier(10) == pytest.approx(1.25)
    assert signals._apply_source_grade_weights([], {}) == []
    fallback_component = signals.ExternalSignalComponent(
        provider="graded_provider",
        status="ok",
        score=0.0,
        weight=1.0,
        value=None,
        detail="",
        known_at_ms=NOW_MS,
        horizon="short",
    )
    fallback_grade = type("Grade", (), {"grade": 5, "model": ""})()
    fallback_adjusted = signals._apply_source_grade_weights(
        [fallback_component],
        {("graded_provider", "medium"): fallback_grade},
    )[0]
    assert fallback_adjusted.source_grade == 5
    assert fallback_adjusted.source_grade_model == ""
    telemetry_only_grade = type("Grade", (), {"grade": 0, "model": "scheduler", "evidence": {"outcome_records": 0}})()
    telemetry_only_adjusted = signals._apply_source_grade_weights(
        [fallback_component],
        {("graded_provider", "medium"): telemetry_only_grade},
    )[0]
    assert telemetry_only_adjusted.source_grade_weight_multiplier == pytest.approx(0.70)
    assert telemetry_only_adjusted.weight == pytest.approx(0.70)
    assert "telemetry_only_bounded_no_outcomes" in telemetry_only_adjusted.detail
    high_telemetry_grade = type("Grade", (), {"grade": 10, "model": "scheduler", "evidence": {"outcome_records": 0}})()
    high_telemetry_adjusted = signals._apply_source_grade_weights(
        [fallback_component],
        {("graded_provider", "medium"): high_telemetry_grade},
    )[0]
    assert high_telemetry_adjusted.source_grade_weight_multiplier == pytest.approx(1.15)
    zero_multiplier_component = signals.ExternalSignalComponent(
        provider="graded_provider",
        status="ok",
        score=0.0,
        weight=2.0,
        value=None,
        detail="base source_grade=9/10 source_grade_weight=0.00",
        known_at_ms=NOW_MS,
        horizon="short",
        source_grade=9,
        source_grade_model="old",
        source_grade_weight_multiplier=0.0,
    )
    reset_component = signals._base_component_for_regrading(zero_multiplier_component)
    assert reset_component.weight == 2.0
    assert reset_component.source_grade is None
    assert reset_component.detail == "base"
    assert signals._parse_epoch_ms(NOW_MS, 0) == NOW_MS
    assert signals._parse_gdelt_seen_ms("", NOW_MS) == NOW_MS
    assert signals._parse_gdelt_seen_ms("bad", NOW_MS) == NOW_MS
    assert signals._binance_symbol_candidates("btcusdt") == ["BTCUSDT"]
    assert signals._binance_symbol_candidates("btcusdc") == ["BTCUSDC", "BTCUSDT"]
    assert signals._structured_news_texts_from_payload("cryptocompare_btc_news", {"Data": "bad"}, limit=2) == []
    assert signals._structured_news_texts_from_payload("cryptocompare_btc_news", {"Data": ["bad"]}, limit=2) == []
    hn_items = signals._structured_news_items_from_payload(
        "hackernews_bitcoin_attention",
        {"hits": [{"title": "Bitcoin breach", "created_at": "not-a-dateZ"}, {"title": "Bitcoin ETF", "created_at": "not-a-date"}]},
        limit=2,
        now_ms=NOW_MS,
    )
    assert hn_items == [("hackernews_bitcoin_attention: Bitcoin breach", NOW_MS), ("hackernews_bitcoin_attention: Bitcoin ETF", NOW_MS)]
    assert signals._source_grade_has_outcome_evidence(type("Grade", (), {"evidence": "bad"})()) is False
    assert signals._source_grade_has_outcome_evidence(type("Grade", (), {"evidence": {"outcome_records": 1}})()) is True
    assert signals._source_grade_has_outcome_evidence(type("Grade", (), {"evidence": {"directional_accuracy": 0.5}})()) is True
    assert signals.report_from_payload({"components": "bad"}) is None
    assert signals.report_from_payload({"components": [{"provider": ""}, {"provider": "x", "score": "bad"}]}) is not None
    graded_payload = signals._component_from_payload(
        {
            "provider": "graded",
            "weight": 1.0,
            "source_grade": 9,
            "source_grade_model": "gemma4:e4b",
            "source_grade_weight_multiplier": 1.2,
        }
    )
    assert graded_payload is not None
    assert graded_payload.source_grade == 9
    assert graded_payload.source_grade_model == "gemma4:e4b"
    assert graded_payload.source_grade_weight_multiplier == pytest.approx(1.2)
    monkeypatch.setattr(signals, "ExternalSignalComponent", lambda **_kwargs: (_ for _ in ()).throw(TypeError("bad")))
    assert signals._component_from_payload({"provider": "x"}) is None
    monkeypatch.undo()

    cache = tmp_path / "bad.json"
    cache.write_text("not-json", encoding="utf-8")
    assert signals.load_external_signal_cache(cache, now_ms=NOW_MS, ttl_seconds=300) is None
    cache.write_text("[]", encoding="utf-8")
    assert signals.load_external_signal_cache(cache, now_ms=NOW_MS, ttl_seconds=300) is None
    cache.write_text(json.dumps({"components": "bad"}), encoding="utf-8")
    assert signals.load_external_signal_cache(cache, now_ms=NOW_MS, ttl_seconds=300) is None
    payload = {
        "status": "ok",
        "score_adjustment": 0.01,
        "raw_score": 0.5,
        "risk_multiplier": 1.0,
        "provider_count": 1,
        "fresh_count": 1,
        "stale_count": 0,
        "known_at_ms": NOW_MS - 1_000_000,
        "cache_path": str(cache),
        "warnings": ["old"],
        "components": [{"provider": "x", "status": "ok", "score": 0.1, "weight": 1.0, "known_at_ms": NOW_MS}],
    }
    cache.write_text(json.dumps(payload), encoding="utf-8")
    assert signals.load_external_signal_cache(cache, now_ms=NOW_MS, ttl_seconds=1) is None
    payload["known_at_ms"] = NOW_MS - 2_000
    payload["reaction_required"] = True
    cache.write_text(json.dumps(payload), encoding="utf-8")
    assert signals.load_external_signal_cache(
        cache,
        now_ms=NOW_MS,
        ttl_seconds=300,
        short_reaction_refresh_seconds=1,
    ) is None
    payload["reaction_required"] = False
    payload["known_at_ms"] = NOW_MS + 1
    cache.write_text(json.dumps(payload), encoding="utf-8")
    assert signals.load_external_signal_cache(cache, now_ms=NOW_MS, ttl_seconds=300) is None
    payload["known_at_ms"] = NOW_MS - 1_000
    payload["components"] = [
        {"provider": "x", "status": "ok", "score": 0.1, "weight": 1.0, "known_at_ms": NOW_MS + 1}
    ]
    cache.write_text(json.dumps(payload), encoding="utf-8")
    assert signals.load_external_signal_cache(cache, now_ms=NOW_MS, ttl_seconds=300) is None
    payload["components"] = [
        {
            "provider": "old_short",
            "status": "ok",
            "score": 0.1,
            "weight": 1.0,
            "known_at_ms": NOW_MS - 8 * 24 * 3_600_000,
            "horizon": "short",
        }
    ]
    cache.write_text(json.dumps(payload), encoding="utf-8")
    assert signals.load_external_signal_cache(cache, now_ms=NOW_MS, ttl_seconds=300) is None
    payload["components"].append(
        {
            "provider": "fresh_medium",
            "status": "ok",
            "score": 0.1,
            "weight": 1.0,
            "known_at_ms": NOW_MS - 1_000,
            "horizon": "medium",
        }
    )
    cache.write_text(json.dumps(payload), encoding="utf-8")
    cached = signals.load_external_signal_cache(cache, now_ms=NOW_MS, ttl_seconds=300)
    assert cached is not None
    assert cached.status == "cached"
    assert cached.fresh_count == 1
    assert cached.stale_count == 1


def test_combine_components_counts_and_ignores_stale_successes(tmp_path, monkeypatch) -> None:
    fresh = signals._component(
        "fresh_short",
        score=0.8,
        weight=1.0,
        value=None,
        detail="fresh",
        known_at_ms=NOW_MS - 1_000,
        horizon="short",
    )
    stale = signals._component(
        "stale_short",
        score=-0.8,
        weight=1.0,
        value=None,
        detail="stale",
        known_at_ms=NOW_MS - 8 * 24 * 3_600_000,
        horizon="short",
    )
    report = signals._combine_components(
        [fresh, stale],
        max_adjustment=0.04,
        min_providers=1,
        now_ms=NOW_MS,
        cache_path=tmp_path / "signals.json",
        news_backend=signals.resolve_backend("cpu"),
    )
    assert report.fresh_count == 1
    assert report.stale_count == 1
    assert report.raw_score == pytest.approx(0.8)
    assert any("stale external signal" in warning for warning in report.warnings)

    class _Response:
        def raise_for_status(self) -> None:
            self.raised = True

        def json(self):
            return {"ok": True}

    observed: dict[str, object] = {}

    def fake_get(url: str, *, timeout: float, headers: dict[str, str]):
        observed["url"] = url
        observed["timeout"] = timeout
        observed["headers"] = headers
        return _Response()

    monkeypatch.setattr(signals.requests, "get", fake_get)
    assert signals._get_json("https://example.test", 0.0) == {"ok": True}
    assert observed["timeout"] == 0.1


def test_external_http_get_retries_rate_limits(monkeypatch) -> None:
    sleeps: list[float] = []
    calls = {"count": 0}

    class _Response:
        def __init__(self, status_code: int) -> None:
            self.status_code = status_code
            self.headers = {"Retry-After": "0.01"} if status_code == 429 else {}
            self.text = "ok"

        def raise_for_status(self) -> None:
            if self.status_code >= 400:
                raise signals.requests.HTTPError(f"status {self.status_code}")

        def json(self):
            return {"ok": self.status_code}

    def fake_get(_url: str, *, timeout: float, headers: dict[str, str]):
        calls["count"] += 1
        return _Response(429 if calls["count"] == 1 else 200)

    monkeypatch.setattr(signals.requests, "get", fake_get)
    monkeypatch.setattr(signals.time, "sleep", lambda seconds: sleeps.append(seconds))

    assert signals._get_json("https://example.test", 1.0) == {"ok": 200}
    assert calls["count"] == 2
    assert sleeps == [0.01]

    clock = {"now": 0.0}
    observed_timeouts: list[float] = []

    def budget_exhausted_get(_url: str, *, timeout: float, headers: dict[str, str]):
        observed_timeouts.append(timeout)
        clock["now"] = 1.0
        return _Response(429)

    monkeypatch.setattr(signals.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(signals.requests, "get", budget_exhausted_get)
    sleeps.clear()
    with pytest.raises(signals.requests.HTTPError):
        signals._get_json("https://example.test", 0.5)
    assert observed_timeouts == [0.5]
    assert sleeps == []


def test_external_http_retry_delay_and_request_exception_paths(monkeypatch) -> None:
    class _Headers:
        def __init__(self, headers: dict[str, str] | None = None) -> None:
            self.headers = headers or {}

    monkeypatch.setattr(signals.time, "time", lambda: 1_700_000_000.0)
    assert signals._http_retry_delay_seconds(_Headers(), 0) == pytest.approx(0.25)
    assert signals._http_retry_delay_seconds(_Headers({"Retry-After": "bad"}), 1) == pytest.approx(0.5)
    assert signals._http_retry_delay_seconds(_Headers({"Retry-After": "Tue, 14 Nov 2023 22:13:21 GMT"}), 0) == pytest.approx(1.0)
    assert signals._http_retry_delay_seconds(_Headers({"Retry-After": "Tue, 14 Nov 2023 22:13:21"}), 0) == pytest.approx(1.0)

    sleeps: list[float] = []
    calls = {"count": 0}

    class _Response:
        status_code = 200
        headers: dict[str, str] = {}
        text = "ok"

        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {"ok": True}

    def flaky_get(_url: str, *, timeout: float, headers: dict[str, str]):
        calls["count"] += 1
        if calls["count"] == 1:
            raise signals.requests.Timeout("temporary")
        return _Response()

    monkeypatch.setattr(signals.requests, "get", flaky_get)
    monkeypatch.setattr(signals.time, "sleep", lambda seconds: sleeps.append(seconds))
    assert signals._get_json("https://example.test", 1.0) == {"ok": True}
    assert sleeps == [0.25]

    monkeypatch.setattr(signals.requests, "get", lambda *_args, **_kwargs: (_ for _ in ()).throw(signals.requests.Timeout("down")))
    with pytest.raises(signals.requests.Timeout):
        signals._get_json("https://example.test", 1.0)

    clock = {"now": 0.0}

    def sleep_past_deadline(_seconds: float) -> None:
        clock["now"] = 2.0

    monkeypatch.setattr(signals.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(signals.time, "sleep", sleep_past_deadline)
    monkeypatch.setattr(signals.requests, "get", lambda *_args, **_kwargs: (_ for _ in ()).throw(signals.requests.Timeout("first")))
    with pytest.raises(signals.requests.Timeout, match="first"):
        signals._get_json("https://example.test", 1.0)

    clock["now"] = 0.0

    class _RetryResponse(_Response):
        status_code = 429
        headers = {"Retry-After": "0.5"}

    monkeypatch.setattr(signals.requests, "get", lambda *_args, **_kwargs: _RetryResponse())
    with pytest.raises(signals.requests.Timeout, match="budget exhausted"):
        signals._get_json("https://example.test", 1.0)


def test_external_http_post_retries_status_and_exceptions(monkeypatch) -> None:
    sleeps: list[float] = []
    calls = {"count": 0}

    class _Response:
        def __init__(self, status_code: int) -> None:
            self.status_code = status_code
            self.headers = {"Retry-After": "0.01"} if status_code == 503 else {}

        def raise_for_status(self) -> None:
            if self.status_code >= 400:
                raise signals.requests.HTTPError(f"status {self.status_code}")

        def json(self):
            return {"ok": self.status_code}

    def flaky_post(_url: str, *, json: dict[str, object], timeout: float, headers: dict[str, str]):
        calls["count"] += 1
        return _Response(503 if calls["count"] == 1 else 200)

    monkeypatch.setattr(signals.requests, "post", flaky_post)
    monkeypatch.setattr(signals.time, "sleep", lambda seconds: sleeps.append(seconds))
    assert signals._post_json("https://example.test", {"x": 1}, 1.0) == {"ok": 200}
    assert sleeps == [0.01]

    calls["count"] = 0

    def exception_then_success(_url: str, *, json: dict[str, object], timeout: float, headers: dict[str, str]):
        calls["count"] += 1
        if calls["count"] == 1:
            raise signals.requests.Timeout("temporary")
        return _Response(200)

    monkeypatch.setattr(signals.requests, "post", exception_then_success)
    sleeps.clear()
    assert signals._post_json("https://example.test", {"x": 1}, 1.0) == {"ok": 200}
    assert sleeps == [0.25]

    clock = {"now": 0.0}
    observed_timeouts: list[float] = []

    def budget_exhausted_post(_url: str, *, json: dict[str, object], timeout: float, headers: dict[str, str]):
        observed_timeouts.append(timeout)
        clock["now"] = 1.0
        return _Response(503)

    monkeypatch.setattr(signals.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(signals.requests, "post", budget_exhausted_post)
    sleeps.clear()
    with pytest.raises(signals.requests.HTTPError):
        signals._post_json("https://example.test", {"x": 1}, 0.5)
    assert observed_timeouts == [0.5]
    assert sleeps == []

    monkeypatch.setattr(signals.requests, "post", lambda *_args, **_kwargs: (_ for _ in ()).throw(signals.requests.Timeout("down")))
    with pytest.raises(signals.requests.Timeout):
        signals._post_json("https://example.test", {"x": 1}, 1.0)


def test_component_cache_timestamp_freshness_rejects_invalid_and_future_report_times() -> None:
    assert signals._bounded_provider_timestamp_ms(0, NOW_MS) == NOW_MS
    assert signals._bounded_provider_timestamp_ms(NOW_MS + 10 * 60_000, NOW_MS) == 0
    assert signals._bounded_provider_timestamp_ms(NOW_MS + 60_000, NOW_MS) == NOW_MS
    zero = signals._component(
        "zero_time",
        score=0.1,
        weight=1.0,
        value=None,
        detail="zero",
        known_at_ms=0,
    )
    future_vs_report = signals._component(
        "future_vs_report",
        score=0.1,
        weight=1.0,
        value=None,
        detail="future",
        known_at_ms=NOW_MS + 10 * 60_000,
    )
    assert signals._component_cache_timestamp_is_fresh(zero, now_ms=NOW_MS, report_known_at_ms=NOW_MS) is False
    assert (
        signals._component_cache_timestamp_is_fresh(
            future_vs_report,
            now_ms=NOW_MS + 20 * 60_000,
            report_known_at_ms=NOW_MS,
        )
        is False
    )


def test_market_provider_future_timestamps_are_rejected() -> None:
    future_seconds = (NOW_MS + 10 * 60_000) // 1000
    with pytest.raises(ValueError, match="CoinGecko timestamp"):
        signals._fetch_coingecko_btc(
            lambda _url, _timeout: {"bitcoin": {"usd": 100, "usd_24h_change": 1, "usd_24h_vol": 1, "last_updated_at": future_seconds}},
            1.0,
            NOW_MS,
        )
    with pytest.raises(ValueError, match="CoinPaprika timestamp"):
        signals._fetch_coinpaprika_btc(
            lambda _url, _timeout: {
                "last_updated": "2023-11-14T22:23:20Z",
                "quotes": {"USD": {"price": 100, "volume_24h": 1, "percent_change_1h": 0, "percent_change_24h": 1, "percent_change_7d": 2}},
            },
            1.0,
            NOW_MS,
        )
    with pytest.raises(ValueError, match="Binance spot ticker timestamp"):
        signals._fetch_binance_spot_momentum(
            lambda _url, _timeout: {
                "lastPrice": "100",
                "priceChangePercent": "1",
                "volume": "1",
                "quoteVolume": "1",
                "closeTime": NOW_MS + 10 * 60_000,
            },
            1.0,
            NOW_MS,
            "BTCUSDC",
        )
    with pytest.raises(ValueError, match="GDELT articles contained no usable titles"):
        signals._fetch_gdelt_news(
            lambda _url, _timeout: {"articles": [{"title": "Bitcoin future item", "seendate": "20231114T222320Z"}]},
            1.0,
            NOW_MS,
            "cpu",
        )
    with pytest.raises(ValueError, match="Hacker News hits contained no usable titles"):
        signals._fetch_hackernews_bitcoin(
            lambda _url, _timeout: {"hits": [{"title": "Bitcoin future item", "created_at": "2023-11-14T22:23:20Z"}]},
            1.0,
            NOW_MS,
            "cpu",
        )


def test_external_signal_bad_provider_payloads_and_no_cache_path(tmp_path) -> None:
    bad_payloads = [
        lambda url, _timeout: [] if "alternative" in url else _good_fetch(url, _timeout),
        lambda url, _timeout: {"data": []} if "alternative" in url else _good_fetch(url, _timeout),
        lambda url, _timeout: [] if "coingecko" in url else _good_fetch(url, _timeout),
        lambda url, _timeout: [] if "coinpaprika" in url else _good_fetch(url, _timeout),
        lambda url, _timeout: [] if "coinlore" in url else _good_fetch(url, _timeout),
        lambda url, _timeout: [] if "blockchain.info/stats" in url else _good_fetch(url, _timeout),
        lambda url, _timeout: [] if "kraken" in url else _good_fetch(url, _timeout),
        lambda url, _timeout: [] if "exchange.coinbase" in url else _good_fetch(url, _timeout),
        lambda url, _timeout: [] if "bitstamp" in url else _good_fetch(url, _timeout),
        lambda url, _timeout: [] if "api.binance.com/api/v3/ticker/24hr" in url else _good_fetch(url, _timeout),
        lambda url, _timeout: [] if "premiumIndex" in url else _good_fetch(url, _timeout),
        lambda url, _timeout: [] if "mempool" in url else _good_fetch(url, _timeout),
        lambda url, _timeout: [] if "cryptocompare" in url else _good_fetch(url, _timeout),
        lambda url, _timeout: [] if "gdeltproject" in url else _good_fetch(url, _timeout),
        lambda url, _timeout: [] if "hn.algolia" in url else _good_fetch(url, _timeout),
    ]
    for index, fetch in enumerate(bad_payloads):
        report = signals.collect_external_signals(
            cache_path=tmp_path / f"bad-provider-{index}.json",
            fetch_json=fetch,
            force_refresh=True,
            now_ms=NOW_MS,
        )
        assert report.status in {"ok", "warn"}
        assert report.provider_count == 14

    fresh = signals.collect_external_signals(
        cache_path=tmp_path / "fresh.json",
        fetch_json=_good_fetch,
        force_refresh=False,
        now_ms=NOW_MS,
    )
    assert fresh.fresh_count == 14
    no_cache_text = signals.render_external_signal_report(
        signals.ExternalSignalReport(
            status="ok",
            score_adjustment=0.0,
            raw_score=0.0,
            risk_multiplier=1.0,
            provider_count=0,
            fresh_count=0,
            stale_count=0,
            known_at_ms=NOW_MS,
            cache_path="",
            warnings=[],
            components=[],
        )
    )
    assert "cache=" not in no_cache_text


def test_news_provider_edge_payloads_and_backend_reason_render(tmp_path) -> None:
    with pytest.raises(ValueError, match="missing CoinPaprika USD"):
        signals._fetch_coinpaprika_btc(lambda _url, _timeout: {"quotes": {}}, 1.0, NOW_MS)
    coinlore = signals._fetch_coinlore_btc(
        lambda _url, _timeout: {
            "value": [
                {
                    "price_usd": "100000",
                    "volume24": "10",
                    "percent_change_1h": "0",
                    "percent_change_24h": "1",
                    "percent_change_7d": "2",
                }
            ]
        },
        1.0,
        NOW_MS,
    )
    assert coinlore.provider == "coinlore_bitcoin"
    with pytest.raises(ValueError, match="unexpected CoinLore"):
        signals._fetch_coinlore_btc(lambda _url, _timeout: {}, 1.0, NOW_MS)
    with pytest.raises(ValueError, match="Kraken ticker returned an error"):
        signals._fetch_kraken_btcusd(lambda _url, _timeout: {"error": ["bad"], "result": {}}, 1.0, NOW_MS)
    with pytest.raises(ValueError, match="missing Kraken"):
        signals._fetch_kraken_btcusd(lambda _url, _timeout: {"error": [], "result": {"x": []}}, 1.0, NOW_MS)
    bitstamp = signals._fetch_bitstamp_btcusd(
        lambda _url, _timeout: {"last": "110", "open": "100", "volume": "1"},
        1.0,
        NOW_MS,
    )
    assert bitstamp.value == pytest.approx(10.0)

    def crypto_empty_data(url: str, _timeout: float):
        if "cryptocompare" in url:
            return {"Data": []}
        return _good_fetch(url, _timeout)

    report = signals.collect_external_signals(
        cache_path=tmp_path / "crypto-empty.json",
        fetch_json=crypto_empty_data,
        force_refresh=True,
        now_ms=NOW_MS,
    )
    assert any(component.provider == "cryptocompare_btc_news" and component.status == "error" for component in report.components)

    def crypto_no_titles(url: str, _timeout: float):
        if "cryptocompare" in url:
            return {"Data": [{"source": "x"}]}
        return _good_fetch(url, _timeout)

    report = signals.collect_external_signals(
        cache_path=tmp_path / "crypto-no-title.json",
        fetch_json=crypto_no_titles,
        force_refresh=True,
        now_ms=NOW_MS,
    )
    assert any(component.provider == "cryptocompare_btc_news" and component.status == "error" for component in report.components)

    component, _backend = signals._fetch_cryptocompare_news(
        lambda _url, _timeout: {
            "Data": [
                "skip-me",
                {"title": "Bitcoin adoption rally", "source": "dup", "published_on": NOW_MS // 1000},
                {"title": "Bitcoin hack", "source": "dup", "published_on": NOW_MS // 1000},
                {"title": "Bitcoin sideways", "source": "", "published_on": NOW_MS // 1000},
            ]
        },
        1.0,
        NOW_MS,
        "cpu",
    )
    assert component.provider == "cryptocompare_btc_news"
    stale_ms = NOW_MS - 10 * 24 * 3_600_000
    stale_component, _backend = signals._fetch_cryptocompare_news(
        lambda _url, _timeout: {
            "Data": [
                {
                    "title": "Bitcoin adoption rally",
                    "source": "stale",
                    "published_on": stale_ms // 1000,
                },
                {"source": "untitled-now"},
            ]
        },
        1.0,
        NOW_MS,
        "cpu",
    )
    assert stale_component.known_at_ms == stale_ms
    with pytest.raises(ValueError, match="no usable titles"):
        signals._fetch_cryptocompare_news(
            lambda _url, _timeout: {
                "Data": [
                    {
                        "title": "Bitcoin adoption rally",
                        "source": "future",
                        "published_on": (NOW_MS + 3_600_000) // 1000,
                    }
                ]
            },
            1.0,
            NOW_MS,
            "cpu",
        )

    def gdelt_empty_articles(url: str, _timeout: float):
        if "gdeltproject" in url:
            return {"articles": []}
        return _good_fetch(url, _timeout)

    report = signals.collect_external_signals(
        cache_path=tmp_path / "gdelt-empty.json",
        fetch_json=gdelt_empty_articles,
        force_refresh=True,
        now_ms=NOW_MS,
    )
    assert any(component.provider == "gdelt_bitcoin_news" and component.status == "error" for component in report.components)

    def gdelt_no_titles(url: str, _timeout: float):
        if "gdeltproject" in url:
            return {"articles": [{"domain": "example.com"}]}
        return _good_fetch(url, _timeout)

    report = signals.collect_external_signals(
        cache_path=tmp_path / "gdelt-no-title.json",
        fetch_json=gdelt_no_titles,
        force_refresh=True,
        now_ms=NOW_MS,
    )
    assert any(component.provider == "gdelt_bitcoin_news" and component.status == "error" for component in report.components)

    component, _backend = signals._fetch_gdelt_news(
        lambda _url, _timeout: {
            "articles": [
                "skip-me",
                {"title": "Bitcoin approval", "domain": "dup", "seendate": "20231114T221320Z"},
                {"title": "Bitcoin ban", "domain": "dup", "seendate": "20231114T221320Z"},
                {"title": "Bitcoin sideways", "domain": "", "seendate": "bad"},
            ]
        },
        1.0,
        NOW_MS,
        "cpu",
    )
    assert component.provider == "gdelt_bitcoin_news"

    def hn_empty_hits(url: str, _timeout: float):
        if "hn.algolia" in url:
            return {"hits": []}
        return _good_fetch(url, _timeout)

    report = signals.collect_external_signals(
        cache_path=tmp_path / "hn-empty.json",
        fetch_json=hn_empty_hits,
        force_refresh=True,
        now_ms=NOW_MS,
    )
    assert any(component.provider == "hackernews_bitcoin_attention" and component.status == "error" for component in report.components)

    def hn_no_titles(url: str, _timeout: float):
        if "hn.algolia" in url:
            return {"hits": [{"points": 1, "created_at": "bad-date"}]}
        return _good_fetch(url, _timeout)

    report = signals.collect_external_signals(
        cache_path=tmp_path / "hn-no-title.json",
        fetch_json=hn_no_titles,
        force_refresh=True,
        now_ms=NOW_MS,
    )
    assert any(component.provider == "hackernews_bitcoin_attention" and component.status == "error" for component in report.components)

    component, _backend = signals._fetch_hackernews_bitcoin(
        lambda _url, _timeout: {
            "hits": [
                "skip-me",
                {"title": "Bitcoin hack", "points": 1, "created_at": "bad-dateZ"},
                {"story_title": "Bitcoin adoption", "points": 1, "created_at": "2023-11-14T22:13:20Z"},
            ]
        },
        1.0,
        NOW_MS,
        "cpu",
    )
    assert component.provider == "hackernews_bitcoin_attention"

    rendered = signals.render_external_signal_report(
        signals.ExternalSignalReport(
            status="ok",
            score_adjustment=0.0,
            raw_score=0.0,
            risk_multiplier=1.0,
            provider_count=0,
            fresh_count=0,
            stale_count=0,
            known_at_ms=NOW_MS,
            cache_path="",
            warnings=[],
            components=[],
            news_backend_reason="fallback",
        )
    )
    assert "news_backend_reason=fallback" in rendered


def test_combine_components_empty_and_nonreaction_paths(tmp_path) -> None:
    backend = signals.BackendInfo("cpu", "cpu", "cpu", "Python stdlib", "")
    empty = signals._combine_components(
        [],
        max_adjustment=0.04,
        min_providers=1,
        now_ms=NOW_MS,
        cache_path=tmp_path / "empty.json",
        news_backend=backend,
    )
    assert empty.status == "fail"
    assert empty.reaction_required is False

    report = signals._combine_components(
        [
            signals.ExternalSignalComponent(
                provider="slow-news",
                status="ok",
                score=0.2,
                weight=1.0,
                value=None,
                detail="medium",
                known_at_ms=NOW_MS,
                horizon="medium",
                urgency=0.2,
            )
        ],
        max_adjustment=0.04,
        min_providers=2,
        now_ms=NOW_MS,
        cache_path=tmp_path / "one.json",
        news_backend=backend,
    )
    assert report.status == "warn"
    assert report.score_adjustment == 0.0
    assert report.reaction_required is False


def test_news_scoring_gpu_fallback_metadata(monkeypatch) -> None:
    backend = signals.BackendInfo("directml", "directml", "privateuseone:0", "DirectML", "")
    monkeypatch.setattr(signals, "resolve_backend", lambda _backend: backend)

    def fail_device(_backend):
        raise RuntimeError("no device")

    monkeypatch.setattr(signals, "_torch_device_for_backend", fail_device)
    scores, resolved = signals._score_news_texts(["Bitcoin adoption", "Bitcoin hack"], compute_backend="directml")
    assert scores == [1.0, -1.0]
    assert resolved.kind == "cpu"
    assert resolved.requested == "directml"
    assert "news scoring failed" in resolved.reason


def test_report_payload_roundtrips_horizon_and_backend_metadata() -> None:
    payload = {
        "status": "ok",
        "score_adjustment": 0.01,
        "raw_score": 0.2,
        "risk_multiplier": 0.9,
        "provider_count": 1,
        "fresh_count": 1,
        "stale_count": 0,
        "known_at_ms": NOW_MS,
        "cache_path": "cache.json",
        "warnings": [],
        "short_term_score": -0.5,
        "medium_term_score": 0.1,
        "long_term_score": 0.2,
        "reaction_required": True,
        "reaction_reason": "provider score=-1",
        "news_backend_requested": "directml",
        "news_backend_kind": "cpu",
        "news_backend_device": "cpu",
        "news_backend_reason": "fallback",
        "components": [
            {
                "provider": "news",
                "status": "ok",
                "score": -1.0,
                "weight": 1.0,
                "value": 1,
                "detail": "breaking",
                "known_at_ms": NOW_MS,
                "horizon": "short",
                "urgency": 1.0,
            }
        ],
    }
    report = signals.report_from_payload(payload)
    assert report is not None
    assert report.short_term_score == -0.5
    assert report.reaction_required is True
    assert report.news_backend_requested == "directml"
    assert report.components[0].horizon == "short"
    assert report.components[0].urgency == 1.0
