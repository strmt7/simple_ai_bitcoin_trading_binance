"""Cached free external signal aggregation for live BTCUSDC decisions."""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Mapping

import requests

from .storage import write_json_atomic


FetchJson = Callable[[str, float], object]

COINGECKO_SIMPLE_PRICE_URL = (
    "https://api.coingecko.com/api/v3/simple/price"
    "?ids=bitcoin&vs_currencies=usd&include_24hr_change=true&include_24hr_vol=true"
)
ALTERNATIVE_FNG_URL = "https://api.alternative.me/fng/?limit=1&format=json"
MEMPOOL_FEES_URL = "https://mempool.space/api/v1/fees/recommended"
BINANCE_FUTURES_BASE_URL = "https://fapi.binance.com"


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


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(parsed):
        return default
    return parsed


def _get_json(url: str, timeout: float) -> object:
    response = requests.get(
        url,
        timeout=max(0.1, float(timeout)),
        headers={"User-Agent": "simple-ai-btcusdc-cli/0.1"},
    )
    response.raise_for_status()
    return response.json()


def _component(
    provider: str,
    *,
    score: float,
    weight: float,
    value: float | None,
    detail: str,
    known_at_ms: int,
    source_symbol: str = "",
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
    )


def _error_component(provider: str, error: Exception, *, known_at_ms: int) -> ExternalSignalComponent:
    return ExternalSignalComponent(
        provider=provider,
        status="error",
        score=0.0,
        weight=0.0,
        value=None,
        detail="provider unavailable",
        known_at_ms=int(known_at_ms),
        error=str(error)[:240],
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
    timestamp = int(_safe_float(latest.get("timestamp"), now_ms / 1000.0) * 1000)
    score = _clamp((50.0 - value) / 50.0, -1.0, 1.0)
    return _component(
        "alternative_fear_greed",
        score=score,
        weight=0.85,
        value=value,
        detail=f"{classification} ({value:.0f}/100)",
        known_at_ms=timestamp,
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
    return _component(
        "coingecko_bitcoin",
        score=score,
        weight=0.70,
        value=change,
        detail=f"24h_change={change:+.2f}% price={price:.2f} volume={volume:.0f}",
        known_at_ms=now_ms,
        source_symbol="bitcoin",
    )


def _binance_symbol_candidates(symbol: str) -> list[str]:
    symbol = (symbol or "BTCUSDC").upper()
    candidates = [symbol]
    if symbol != "BTCUSDT":
        candidates.append("BTCUSDT")
    return candidates


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
            known_at = int(_safe_float(premium.get("time"), now_ms))
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
        )
    except (TypeError, ValueError):
        return None


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
        warnings=[str(value) for value in payload.get("warnings", []) if isinstance(value, str)]
        if isinstance(payload.get("warnings"), list)
        else [],
        components=components,
    )


def load_external_signal_cache(path: Path, *, now_ms: int, ttl_seconds: int) -> ExternalSignalReport | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, Mapping):
        return None
    report = report_from_payload(payload)
    if report is None:
        return None
    age_seconds = max(0.0, (now_ms - report.known_at_ms) / 1000.0)
    if age_seconds > max(0, int(ttl_seconds)):
        return None
    return ExternalSignalReport(
        **{
            **report.asdict(),
            "status": "cached",
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
) -> ExternalSignalReport:
    usable = [component for component in components if component.status == "ok" and component.weight > 0.0]
    warnings = [
        f"{component.provider}: {component.error}"
        for component in components
        if component.status == "error" and component.error
    ]
    total_weight = sum(component.weight for component in usable)
    raw_score = sum(component.score * component.weight for component in usable) / total_weight if total_weight else 0.0
    max_adjustment = _clamp(float(max_adjustment), 0.0, 0.20)
    score_adjustment = _clamp(raw_score * max_adjustment, -max_adjustment, max_adjustment)
    if len(usable) < max(0, int(min_providers)):
        warnings.append("minimum external signal provider count not met; positive boost disabled")
        score_adjustment = min(0.0, score_adjustment)
    risk_multiplier = 1.0 if raw_score >= 0.0 else _clamp(1.0 + raw_score * 0.35, 0.50, 1.0)
    status = "ok" if usable and not warnings else ("warn" if usable else "fail")
    return ExternalSignalReport(
        status=status,
        score_adjustment=float(score_adjustment),
        raw_score=float(raw_score),
        risk_multiplier=float(risk_multiplier),
        provider_count=len(components),
        fresh_count=len(usable),
        stale_count=0,
        known_at_ms=now_ms,
        cache_path=str(cache_path),
        warnings=warnings,
        components=components,
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
    fetch_json: FetchJson = _get_json,
    now_ms: int | None = None,
) -> ExternalSignalReport:
    now = _now_ms() if now_ms is None else int(now_ms)
    cache = Path(cache_path)
    if not force_refresh:
        cached = load_external_signal_cache(cache, now_ms=now, ttl_seconds=ttl_seconds)
        if cached is not None:
            return cached

    fetchers = [
        lambda: _fetch_alternative_fng(fetch_json, timeout_seconds, now),
        lambda: _fetch_coingecko_btc(fetch_json, timeout_seconds, now),
        lambda: _fetch_binance_derivatives(fetch_json, timeout_seconds, now, symbol),
        lambda: _fetch_mempool_fees(fetch_json, timeout_seconds, now),
    ]
    provider_names = [
        "alternative_fear_greed",
        "coingecko_bitcoin",
        "binance_futures_positioning",
        "mempool_fee_pressure",
    ]
    components: list[ExternalSignalComponent] = []
    for provider, fetcher in zip(provider_names, fetchers):
        try:
            components.append(fetcher())
        except Exception as exc:
            components.append(_error_component(provider, exc, known_at_ms=now))

    report = _combine_components(
        components,
        max_adjustment=max_adjustment,
        min_providers=min_providers,
        now_ms=now,
        cache_path=cache,
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(cache, report.asdict(), indent=2, sort_keys=True)
    return report


def render_external_signal_report(report: ExternalSignalReport) -> str:
    lines = [
        "External signal report",
        (
            f"status={report.status} providers={report.fresh_count}/{report.provider_count} "
            f"score_adjustment={report.score_adjustment:+.4f} "
            f"risk_multiplier={report.risk_multiplier:.3f}"
        ),
    ]
    if report.cache_path:
        lines.append(f"cache={report.cache_path}")
    for component in report.components:
        cache_note = " cached" if component.cached else ""
        symbol_note = f" symbol={component.source_symbol}" if component.source_symbol else ""
        if component.status == "ok":
            lines.append(
                f"- {component.provider}{cache_note}: score={component.score:+.3f} "
                f"weight={component.weight:.2f}{symbol_note} {component.detail}"
            )
        else:
            lines.append(f"- {component.provider}: {component.status} {component.error or component.detail}")
    for warning in report.warnings:
        lines.append(f"warning: {warning}")
    return "\n".join(lines)

