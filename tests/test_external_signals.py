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
    raise AssertionError(url)


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
    assert report.provider_count == 4
    assert report.fresh_count == 4
    assert report.score_adjustment > 0
    assert cache.exists()
    text = signals.render_external_signal_report(report)
    assert "alternative_fear_greed" in text
    assert "score_adjustment=" in text

    cached = signals.collect_external_signals(
        cache_path=cache,
        fetch_json=lambda _url, _timeout: (_ for _ in ()).throw(AssertionError("cache not used")),
        ttl_seconds=300,
        now_ms=NOW_MS + 1_000,
    )
    assert cached.status == "cached"
    assert all(component.cached for component in cached.components)


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
    assert len(failed.warnings) == 5
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
    assert signals._binance_symbol_candidates("btcusdt") == ["BTCUSDT"]
    assert signals._binance_symbol_candidates("btcusdc") == ["BTCUSDC", "BTCUSDT"]
    assert signals.report_from_payload({"components": "bad"}) is None
    assert signals.report_from_payload({"components": [{"provider": ""}, {"provider": "x", "score": "bad"}]}) is not None
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
    ]
    for index, fetch in enumerate(bad_payloads):
        report = signals.collect_external_signals(
            cache_path=tmp_path / f"bad-provider-{index}.json",
            fetch_json=fetch,
            force_refresh=True,
            now_ms=NOW_MS,
        )
        assert report.status in {"ok", "warn"}
        assert report.provider_count == 4

    fresh = signals.collect_external_signals(
        cache_path=tmp_path / "fresh.json",
        fetch_json=_good_fetch,
        force_refresh=False,
    )
    assert fresh.fresh_count == 4
    no_cache_text = signals.render_external_signal_report(
        signals.ExternalSignalReport(
            "ok",
            0.0,
            0.0,
            1.0,
            0,
            0,
            0,
            NOW_MS,
            "",
            [],
            [],
        )
    )
    assert "cache=" not in no_cache_text
