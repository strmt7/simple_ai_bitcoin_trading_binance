from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
from json import JSONDecodeError

import pytest
import requests

from simple_ai_bitcoin_trading_binance.api import (
    BinanceAPIError,
    BinanceClient,
    SymbolConstraints,
    _default_base_url,
)


class _FakeResponse:
    def __init__(self, status_code: int, payload=None, text: str = "", headers: dict[str, str] | None = None) -> None:
        self.status_code = status_code
        self._payload = payload
        self.headers = headers or {}
        if text:
            self.text = text
        elif isinstance(payload, Exception):
            self.text = str(payload)
        else:
            self.text = json.dumps(payload) if payload is not None else ""

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


def test_default_base_url_matches_market_and_environment() -> None:
    assert _default_base_url(True, "spot") == ("https://testnet.binance.vision", "api")
    assert _default_base_url(False, "spot") == ("https://api.binance.com", "api")
    assert _default_base_url(True, "futures") == ("https://testnet.binancefuture.com", "fapi")
    assert _default_base_url(False, "futures") == ("https://fapi.binance.com", "fapi")


def test_client_rejects_unknown_market_type() -> None:
    with pytest.raises(ValueError, match="market_type must be 'spot' or 'futures'"):
        BinanceClient("k", "s", market_type="swap")


def test_request_parses_json_and_raises_http_errors(monkeypatch) -> None:
    client = BinanceClient(api_key="k", api_secret="s", market_type="spot", max_retries=0)

    def request(method: str, url: str, params=None, timeout=None):
        assert method == "GET"
        assert "ping" in url
        return _FakeResponse(429, text="{\"code\": -1003}")

    monkeypatch.setattr(client.session, "request", request)
    with pytest.raises(BinanceAPIError, match="Binance returned 429"):
        client.ping()


def test_request_retries_on_rate_limits(monkeypatch) -> None:
    client = BinanceClient(api_key="k", api_secret="s", market_type="spot", max_retries=1)
    responses = [
        _FakeResponse(429, text="{\"code\": -1003}"),
        _FakeResponse(200, payload={"ok": True}),
    ]
    calls: list[tuple[str, str]] = []
    sleep_calls: list[float] = []

    def request(method: str, url: str, params=None, timeout=None):
        calls.append((method, url))
        return responses.pop(0)

    monkeypatch.setattr(client.session, "request", request)
    monkeypatch.setattr(time, "sleep", lambda seconds: sleep_calls.append(seconds))
    payload = client._request("GET", "/api/v3/ping")
    assert payload == {"ok": True}
    assert len(calls) == 2
    assert payload == {"ok": True}
    assert client.last_request_info["attempts"] == 2
    assert client.last_request_info["retries"] == 1
    assert sleep_calls


def test_request_retries_on_retry_after_header(monkeypatch) -> None:
    client = BinanceClient(api_key="k", api_secret="s", market_type="spot", max_retries=1)
    responses = [
        _FakeResponse(429, text="rate limit", headers={"Retry-After": "1"}),
        _FakeResponse(200, payload={"ok": True}),
    ]
    sleep_calls: list[float] = []

    def request(method: str, url: str, params=None, timeout=None):
        return responses.pop(0)

    monkeypatch.setattr(client.session, "request", request)
    monkeypatch.setattr(time, "sleep", lambda seconds: sleep_calls.append(seconds))
    payload = client._request("GET", "/api/v3/ping")
    assert payload == {"ok": True}
    assert any(abs(item - 1.0) < 1e-9 for item in sleep_calls)


def test_request_retries_on_api_rate_limit_code(monkeypatch) -> None:
    client = BinanceClient(api_key="k", api_secret="s", market_type="spot", max_retries=1)
    responses = [
        _FakeResponse(200, payload={"code": -1003, "msg": "Too many requests"}),
        _FakeResponse(200, payload={"ok": True}),
    ]
    calls = 0

    def request(method: str, url: str, params=None, timeout=None):
        nonlocal calls
        calls += 1
        return responses.pop(0)

    monkeypatch.setattr(client.session, "request", request)
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)
    payload = client._request("GET", "/api/v3/ping")
    assert payload == {"ok": True}
    assert calls == 2


def test_request_raises_for_malformed_json(monkeypatch) -> None:
    client = BinanceClient(api_key="k", api_secret="s")

    def request(method: str, url: str, params=None, timeout=None):
        return _FakeResponse(200, payload=JSONDecodeError("bad json", "{}", 0), text="bad json")

    monkeypatch.setattr(client.session, "request", request)
    with pytest.raises(BinanceAPIError, match="Malformed response"):
        client.ping()


def test_last_request_info_records_success(monkeypatch) -> None:
    client = BinanceClient(api_key="k", api_secret="s", market_type="spot")
    monkeypatch.setattr(client.session, "request", lambda _method, _url, params=None, timeout=None: _FakeResponse(200, {"ok": True}))
    assert client._request("GET", "/api/v3/ping") == {"ok": True}
    assert client.last_request_info["status"] == 200
    assert client.last_request_info["method"] == "GET"
    assert client.last_request_info["path"] == "/api/v3/ping"
    assert client.last_request_info["attempts"] == 1


def test_request_raises_for_transport_errors(monkeypatch) -> None:
    client = BinanceClient(api_key="k", api_secret="s", max_retries=0)

    def request(method: str, url: str, params=None, timeout=None):
        raise requests.Timeout("timeout")

    monkeypatch.setattr(client.session, "request", request)
    with pytest.raises(BinanceAPIError, match="Binance request failed"):
        client.ping()


def test_request_raises_for_binance_payload_errors(monkeypatch) -> None:
    client = BinanceClient(api_key="k", api_secret="s")

    def request(method: str, url: str, params=None, timeout=None):
        return _FakeResponse(200, payload={"code": -1, "msg": "boom"})

    monkeypatch.setattr(client.session, "request", request)
    with pytest.raises(BinanceAPIError, match="Binance API error"):
        client.ping()


def test_request_signed_payload_and_unsigned_payload_paths(monkeypatch) -> None:
    client = BinanceClient(api_key="k", api_secret="s", market_type="futures")

    def request_unsigned(method: str, url: str, params=None, timeout=None):
        assert method == "GET"
        assert url.endswith("/fapi/v1/time")
        assert params == {"symbol": "BTCUSDC"}
        return _FakeResponse(200, payload={"serverTime": 123}, text="{\"serverTime\": 123}")

    monkeypatch.setattr(client.session, "request", request_unsigned)
    assert client._request("GET", "/fapi/v1/time", {"symbol": "BTCUSDC"}) == {"serverTime": 123}

    captured: list[str] = []

    def request_signed(method: str, url: str, params=None, timeout=None):
        assert method == "POST"
        assert "/fapi/v1/order" in url
        assert params is None
        assert "symbol=BTCUSDC" in url
        assert "signature=" in url
        captured.append(url)
        return _FakeResponse(200, payload={"ok": True}, text="{\"ok\": true}")

    monkeypatch.setattr(client.session, "request", request_signed)
    client._request("POST", "/fapi/v1/order", {"symbol": "BTCUSDC"}, signed=True)
    assert any("signature=" in item for item in captured)


def test_throttle_waits_when_cadence_exceeded(monkeypatch) -> None:
    client = BinanceClient(api_key="k", api_secret="s", market_type="spot", max_calls_per_minute=2)
    client._rate_limit_at = datetime.now(timezone.utc) + timedelta(seconds=1)

    observed: list[float] = []

    def fake_sleep(seconds: float) -> None:
        observed.append(seconds)

    monkeypatch.setattr(time, "sleep", fake_sleep)
    client._throttle()
    assert observed and observed[0] > 0.99


def test_parse_and_quantize_helpers() -> None:
    assert BinanceClient._parse_float(None) == 0.0
    assert BinanceClient._parse_float("1.23") == 1.23
    assert BinanceClient._parse_float("bad") == 0.0

    filters = [
        {"filterType": "LOT_SIZE", "minQty": "0.001", "stepSize": "0.005"},
        {"filterType": "MIN_NOTIONAL", "notional": "10"},
    ]
    assert BinanceClient._parse_filter(filters, "LOT_SIZE") == filters[0]
    assert BinanceClient._parse_filter(filters, "NOTHING") == {}

    assert BinanceClient._quantize_to_step(1.37, 0.0) == 1.37
    assert BinanceClient._quantize_to_step(1.377, 0.05) == 1.35
    assert BinanceClient._quantize_to_step(-1.2, 0.1) == 0.0


def test_parse_filter_skips_non_dict_entries() -> None:
    filters = ["a", {"filterType": "NOTHING"}, {"filterType": "LOT_SIZE", "minQty": "1"}]
    assert BinanceClient._parse_filter(filters, "LOT_SIZE") == {"filterType": "LOT_SIZE", "minQty": "1"}


def test_symbol_constraints_uses_market_filters_and_notional_fallback(monkeypatch) -> None:
    client = BinanceClient(api_key="k", api_secret="s", market_type="futures")
    exchange_info = {
        "symbols": [
            {
                "symbol": "BTCUSDC",
                "status": "TRADING",
                "filters": [
                    {"filterType": "LOT_SIZE", "minQty": "0.010", "maxQty": "5", "stepSize": "0.010"},
                    {"filterType": "MARKET_LOT_SIZE", "minQty": "0.020", "maxQty": "2.5", "stepSize": "0.020"},
                    {"filterType": "MIN_NOTIONAL", "notional": "20"},
                ],
            }
        ]
    }

    def request(method: str, path: str, params=None, signed: bool = False):
        assert path == "/fapi/v1/exchangeInfo"
        return exchange_info

    monkeypatch.setattr(client, "_request", request)
    constraints = client.get_symbol_constraints("BTCUSDC")
    assert constraints == SymbolConstraints(
        symbol="BTCUSDC",
        min_qty=0.02,
        max_qty=2.5,
        step_size=0.02,
        min_notional=20.0,
        max_notional=0.0,
    )

    normalized, parsed = client.normalize_quantity("BTCUSDC", 0.015)
    assert normalized == 0.0
    assert parsed == constraints
    normalized, _ = client.normalize_quantity("BTCUSDC", 1.01)
    assert normalized == 1.0


def test_symbol_constraints_reject_unknown_symbol_or_bad_filters(monkeypatch) -> None:
    client = BinanceClient(api_key="k", api_secret="s")

    def request_unknown(_method: str, _path: str, _params=None, _signed: bool = False):
        return {"symbols": []}

    monkeypatch.setattr(client, "_request", request_unknown)
    with pytest.raises(BinanceAPIError, match="Unknown symbol"):
        client.get_symbol_constraints("BTCUSDC")

    def request_bad_filters(_method: str, _path: str, _params=None, _signed: bool = False):
        return {"symbols": [{"symbol": "BTCUSDC", "filters": {"bad": True}}]}

    monkeypatch.setattr(client, "_request", request_bad_filters)
    with pytest.raises(BinanceAPIError, match="Unexpected symbol filters"):
        client.get_symbol_constraints("BTCUSDC")


def test_ensure_btcusdc_rejects_non_trading_symbol(monkeypatch) -> None:
    client = BinanceClient(api_key="k", api_secret="s")

    def request(_method: str, _path: str, _params=None, _signed: bool = False):
        return {"symbols": [{"symbol": "BTCUSDC", "status": "HALT"}]}

    monkeypatch.setattr(client, "_request", request)
    with pytest.raises(BinanceAPIError, match="BTCUSDC is not trading"):
        client.ensure_btcusdc()


def test_leverage_fetch_and_set(monkeypatch) -> None:
    client = BinanceClient(api_key="k", api_secret="s", market_type="futures")
    calls: list[tuple[str, str]] = []

    def request(method: str, path: str, params=None, signed: bool = False):
        calls.append((method, path))
        if path == "/fapi/v1/leverageBracket":
            return [{"symbol": "BTCUSDC", "brackets": [{"initialLeverage": "2", "maxLeverage": "7"}]}]
        if path == "/fapi/v1/leverage":
            assert params["leverage"] == 7
            return {"symbol": params["symbol"], "leverage": params["leverage"]}
        raise AssertionError(f"unexpected endpoint: {path}")

    monkeypatch.setattr(client, "_request", request)
    assert client.get_max_leverage("BTCUSDC") == 7
    payload = client.set_leverage("BTCUSDC", 100)
    assert payload["leverage"] == 7
    assert calls == [
        ("GET", "/fapi/v1/leverageBracket"),
        ("GET", "/fapi/v1/leverageBracket"),
        ("POST", "/fapi/v1/leverage"),
    ]


def test_leverage_and_klines_errors(monkeypatch) -> None:
    futures_client = BinanceClient(api_key="k", api_secret="s", market_type="futures")
    assert BinanceClient(api_key="k", api_secret="s", market_type="spot").get_max_leverage("BTCUSDC") == 1

    def request_brackets(_method: str, _path: str, _params=None, **_kwargs):
        return "not a list"

    monkeypatch.setattr(futures_client, "_request", request_brackets)
    with pytest.raises(BinanceAPIError, match="Unexpected leverage bracket"):
        futures_client.get_max_leverage("BTCUSDC")

    def request_ok(method: str, path: str, params=None, signed: bool = False):
        if path == "/fapi/v1/exchangeInfo":
            return {"symbols": [{"symbol": "BTCUSDC", "status": "TRADING"}]}
        if path == "/fapi/v1/klines":
            return [[1, "100", "101", "99", "100", "1", "2"]]
        raise AssertionError(path)

    monkeypatch.setattr(futures_client, "_request", request_ok)
    assert futures_client.ensure_btcusdc()["symbol"] == "BTCUSDC"
    candles = futures_client.get_klines("BTCUSDC", "15m")
    assert len(candles) == 1
    assert candles[0].open_time == 1

    with pytest.raises(BinanceAPIError, match="This CLI supports BTCUSDC only"):
        futures_client.get_klines("ETHUSDC", "15m")

    def request_bad_klines(_method: str, _path: str, _params=None, **_kwargs):
        return [[1, "100"]]

    monkeypatch.setattr(futures_client, "_request", request_bad_klines)
    with pytest.raises(BinanceAPIError, match="Unexpected kline row"):
        futures_client.get_klines("BTCUSDC", "15m")


def test_klines_uses_start_and_end_filters(monkeypatch) -> None:
    client = BinanceClient(api_key="k", api_secret="s", market_type="spot")

    captured: list[tuple[str, dict]] = []

    def request(_method: str, _path: str, params=None, signed: bool = False):
        assert not signed
        assert params is not None
        captured.append((params["symbol"], params["interval"]))
        assert params["limit"] == 5
        assert params["startTime"] == 111
        assert params["endTime"] == 222
        return [[1, "100", "101", "99", "100", "1", "2"]]

    monkeypatch.setattr(client, "_request", request)
    candles = client.get_klines("BTCUSDC", "1m", limit=5, start_time=111, end_time=222)
    assert len(candles) == 1
    assert captured == [("BTCUSDC", "1m")]


def test_get_symbol_price_returns_numeric_tuple(monkeypatch) -> None:
    client = BinanceClient(api_key="k", api_secret="s", market_type="spot")

    monkeypatch.setattr(
        client,
        "_request",
        lambda _method, _path, _params=None, _signed=False: {"price": "123.45"},
    )
    price, _ts = client.get_symbol_price("BTCUSDC")
    assert price == 123.45


def test_client_initialization_clamps_rate_limit_bounds() -> None:
    low = BinanceClient(api_key="k", api_secret="s", max_calls_per_minute=0)
    assert low._call_delay == 60.0

    high = BinanceClient(api_key="k", api_secret="s", max_calls_per_minute=9999)
    assert high._call_delay == 0.03


def test_last_request_info_is_populated(monkeypatch) -> None:
    client = BinanceClient(api_key="k", api_secret="s", market_type="spot")

    monkeypatch.setattr(
        client.session,
        "request",
        lambda _method, _url, params=None, timeout=None: _FakeResponse(200, {"ok": True}),
    )
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)

    payload = client._request("GET", "/api/v3/ping")
    assert payload == {"ok": True}
    assert client.last_request_info["method"] == "GET"
    assert client.last_request_info["path"] == "/api/v3/ping"
    assert client.last_request_info["status"] == 200
    assert client.last_request_info["attempts"] == 1


def test_get_account_uses_market_endpoint(monkeypatch) -> None:
    spot = BinanceClient(api_key="k", api_secret="s", market_type="spot")
    futures = BinanceClient(api_key="k", api_secret="s", market_type="futures")

    def spot_request(method: str, path: str, params=None, signed: bool = False):
        assert path == "/api/v3/account"
        return {"accountType": "SPOT"}

    def futures_request(method: str, path: str, params=None, signed: bool = False):
        assert path == "/fapi/v2/account"
        return {"canTrade": True}

    monkeypatch.setattr(spot, "_request", spot_request)
    monkeypatch.setattr(futures, "_request", futures_request)
    assert spot.get_account()["accountType"] == "SPOT"
    assert futures.get_account()["canTrade"]


def test_get_max_leverage_missing_symbol_uses_default(monkeypatch) -> None:
    client = BinanceClient(api_key="k", api_secret="s", market_type="futures")

    def request(method: str, path: str, params=None, signed: bool = False):
        assert path == "/fapi/v1/leverageBracket"
        return [
            {"symbol": "ETHUSDC", "brackets": [{"maxLeverage": "50"}]},
        ]

    monkeypatch.setattr(client, "_request", request)
    assert client.get_max_leverage("BTCUSDC") == 125


def test_set_leverage_low_and_high_clamp(monkeypatch) -> None:
    client = BinanceClient(api_key="k", api_secret="s", market_type="futures")

    def request(method: str, path: str, params=None, signed: bool = False):
        if path == "/fapi/v1/leverageBracket":
            return [{"symbol": "BTCUSDC", "brackets": [{"initialLeverage": "200", "maxLeverage": "90"}]}]
        if path == "/fapi/v1/leverage":
            return {"symbol": "BTCUSDC", "leverage": params["leverage"]}
        raise AssertionError

    monkeypatch.setattr(client, "_request", request)
    assert client.set_leverage("BTCUSDC", 0)["leverage"] == 1
    assert client.set_leverage("BTCUSDC", 999)["leverage"] == 125


def test_quantize_and_symbol_constraints_handle_bad_data(monkeypatch) -> None:
    client = BinanceClient(api_key="k", api_secret="s")
    assert BinanceClient._quantize_to_step(1.23, "bad-step") == 0.0

    bad_payload = {
        "symbols": [
            {
                "symbol": "BTCUSDC",
                "filters": [
                    {"filterType": "LOT_SIZE", "minQty": "0", "maxQty": "0", "stepSize": "0"},
                    {"filterType": "NOTIONAL", "minNotional": "bad", "maxNotional": "-10"},
                    {"filterType": "MIN_NOTIONAL", "notional": "bad"},
                ],
            }
        ]
    }

    def request(_method: str, _path: str, _params=None, _signed: bool = False):
        return bad_payload

    monkeypatch.setattr(client, "_request", request)
    constraints = client.get_symbol_constraints("BTCUSDC")
    assert constraints.min_qty == 0.0
    assert constraints.max_qty == 0.0
    assert constraints.step_size == 0.0
    assert constraints.min_notional == 0.0
    assert constraints.max_notional == 0.0


def test_get_max_leverage_skips_invalid_payload_parts(monkeypatch) -> None:
    client = BinanceClient(api_key="k", api_secret="s", market_type="futures")

    def request(_method: str, _path: str, _params=None, _signed: bool = False, **_kwargs):
        assert _path == "/fapi/v1/leverageBracket"
        return [
            {"symbol": "BTCUSDC", "brackets": [None, {}, {"initialLeverage": "bad", "maxLeverage": "none"}]},
        ]

    monkeypatch.setattr(client, "_request", request)
    assert client.get_max_leverage("BTCUSDC") == 125


def test_set_leverage_spot_is_rejected() -> None:
    client = BinanceClient(api_key="k", api_secret="s", market_type="spot")
    with pytest.raises(BinanceAPIError, match="Leverage is available only in futures mode"):
        client.set_leverage("BTCUSDC", 2)


def test_place_order_uses_spot_live_endpoint(monkeypatch) -> None:
    client = BinanceClient(api_key="k", api_secret="s", market_type="spot")
    called: list[tuple[str, str, bool]] = []

    def request(method: str, path: str, params=None, signed: bool = False):
        called.append((method, path, signed))
        return {"ok": True, "method": method, "path": path, "signed": signed, "params": params}

    monkeypatch.setattr(client, "_request", request)
    response = client.place_order("BTCUSDC", "BUY", 1.0, dry_run=False)
    assert response["ok"] is True
    assert called == [("POST", "/api/v3/order", True)]


def test_ensure_btcusdc_raises_for_missing_symbol(monkeypatch) -> None:
    client = BinanceClient(api_key="k", api_secret="s")

    def request(_method: str, _path: str, _params=None, _signed: bool = False):
        return {"symbols": []}

    monkeypatch.setattr(client, "_request", request)
    with pytest.raises(BinanceAPIError, match="BTCUSDC is unavailable"):
        client.ensure_btcusdc()


def test_symbol_constraints_and_klines_cover_fallbacks_and_spot_endpoint(monkeypatch) -> None:
    client = BinanceClient(api_key="k", api_secret="s", market_type="spot")

    payload = {
        "symbols": [
            {
                "symbol": "BTCUSDC",
                "filters": [
                    {"filterType": "LOT_SIZE", "minQty": "0.1", "maxQty": "2", "stepSize": "0.1"},
                    {"filterType": "MARKET_LOT_SIZE", "minQty": "0", "maxQty": "0", "stepSize": "0"},
                    {"filterType": "NOTIONAL", "minNotional": "0", "maxNotional": "0"},
                ],
                "status": "TRADING",
            }
        ]
    }

    def request(method: str, path: str, params=None, signed: bool = False):
        if path == "/api/v3/exchangeInfo":
            return payload
        if path == "/api/v3/klines":
            return [[1, "100", "101", "99", "100", "1", "2"]]
        raise AssertionError(f"unexpected path {path}")

    monkeypatch.setattr(client, "_request", request)
    constraints = client.get_symbol_constraints("BTCUSDC")
    assert constraints.min_qty == 0.1
    assert constraints.max_qty == 2.0
    assert constraints.min_notional == 0.0
    assert constraints.max_notional == 0.0
    assert client.ensure_btcusdc()["symbol"] == "BTCUSDC"
    assert client.get_klines("BTCUSDC", "15m").__class__ is list
