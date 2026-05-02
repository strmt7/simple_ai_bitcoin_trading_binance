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
        return {"bitcoin": {"usd": "100000", "usd_24h_change": "3.0", "usd_24h_vol": "123456"}}
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
    assert report.provider_count == 7
    assert report.fresh_count == 7
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


def test_collect_external_signals_rss_ollama_and_telemetry(tmp_path) -> None:
    def fetch_text(url: str, timeout: float) -> str:
        assert url.startswith("https://")
        assert timeout > 0
        return _feed_xml()

    def post_json(url: str, payload: dict[str, object], timeout: float):
        assert url.endswith("/api/generate")
        assert payload["model"] == "gemma4:e4b"
        assert payload["keep_alive"] == "30m"
        assert payload["options"]["num_ctx"] == 1024
        assert payload["options"]["num_predict"] == 64
        assert str(payload["prompt"]).count("- ") <= 12
        assert timeout == 2.0
        return {
            "response": json.dumps(
                {
                    "score": -0.7,
                    "horizon": "short-term",
                    "reaction_required": "true",
                    "reason": "Hack risk outweighs inflows",
                }
            )
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
    assert report.provider_count == 38
    assert report.fresh_count == 38
    assert report.news_ai_status == "ok"
    assert report.news_ai_model == "gemma4:e4b"
    assert report.reaction_required is True
    text = signals.render_external_signal_report(report)
    assert "news_ai=ok" in text
    from simple_ai_bitcoin_trading_binance.telemetry_store import TradingTelemetryStore

    with TradingTelemetryStore(telemetry) as store:
        observations = store.recent_observations(since_ms=NOW_MS - 10_000, limit=500)
    assert any(item.kind == "raw_provider_payload" for item in observations)
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
    assert signals._extract_feed_items("<feed><entry><title>Atom Bitcoin ban</title><updated>2023-11-14T22:13:20Z</updated><link href='https://x.test'/></entry></feed>", now_ms=NOW_MS)[0]["title"] == "Atom Bitcoin ban"

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


def test_news_feed_helpers_and_ollama_validation(monkeypatch) -> None:
    assert signals._parse_feed_datetime_ms("", NOW_MS) == NOW_MS
    assert signals._parse_feed_datetime_ms("Tue, 14 Nov 2023 22:13:20", 0) == NOW_MS
    assert signals._parse_feed_datetime_ms("2023-11-14T22:13:20", 0) == NOW_MS
    assert signals._parse_feed_datetime_ms("not a date", NOW_MS) == NOW_MS
    assert signals._child_text(signals.ET.fromstring("<item><x /></item>"), "title") == ""
    assert signals._strip_markup("<b>Bitcoin</b>&amp;BTC") == "Bitcoin &BTC"
    assert signals._provider_name_for_url("https://example.test/path") == "example_test"
    assert signals._provider_name_for_url(signals.COINGECKO_SIMPLE_PRICE_URL) == "coingecko_bitcoin"
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
    assert prompt.count("- ") == 12

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
    assert observed["post"][2] == 0.1
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
    assert len(failed.warnings) == 8
    assert "offline" in signals.render_external_signal_report(failed)

    def one_positive(url: str, _timeout: float):
        if "coingecko" in url:
            return {"bitcoin": {"usd": "100", "usd_24h_change": "5", "usd_24h_vol": "1"}}
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
    assert signals._news_horizon("Breaking Bitcoin ETF inflow", age_ms=1_000) == "short"
    assert signals._news_horizon("Bitcoin sideways", age_ms=3_600_000) == "medium"
    assert signals._news_horizon("Bitcoin sideways", age_ms=172_800_000) == "long"
    assert signals._average_sentiment([(1.0, 1.0), (-1.0, 1.0)]) == 0.0
    assert signals._average_sentiment([]) == 0.0
    scores, backend = signals._score_news_texts(["Bitcoin adoption rally", "Bitcoin hack"], compute_backend="cpu")
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
    assert signals._parse_epoch_ms(NOW_MS, 0) == NOW_MS
    assert signals._parse_gdelt_seen_ms("", NOW_MS) == NOW_MS
    assert signals._parse_gdelt_seen_ms("bad", NOW_MS) == NOW_MS
    assert signals._binance_symbol_candidates("btcusdt") == ["BTCUSDT"]
    assert signals._binance_symbol_candidates("btcusdc") == ["BTCUSDC", "BTCUSDT"]
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


def test_external_signal_bad_provider_payloads_and_no_cache_path(tmp_path) -> None:
    bad_payloads = [
        lambda url, _timeout: [] if "alternative" in url else _good_fetch(url, _timeout),
        lambda url, _timeout: {"data": []} if "alternative" in url else _good_fetch(url, _timeout),
        lambda url, _timeout: [] if "coingecko" in url else _good_fetch(url, _timeout),
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
        assert report.provider_count == 7

    fresh = signals.collect_external_signals(
        cache_path=tmp_path / "fresh.json",
        fetch_json=_good_fetch,
        force_refresh=False,
    )
    assert fresh.fresh_count == 7
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
