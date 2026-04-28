from __future__ import annotations

import pytest

from simple_ai_bitcoin_trading_binance.api import BinanceAPIError, BinanceClient
from simple_ai_bitcoin_trading_binance.api import SymbolConstraints


def test_futures_leverage_bracket_parsing(monkeypatch) -> None:
    client = BinanceClient(api_key="k", api_secret="s", market_type="futures")
    calls: list[tuple[str, str]] = []

    def fake_request(method: str, path: str, params=None, signed: bool = False):
        calls.append((method, path))
        if path == "/fapi/v1/leverageBracket":
            return [
                {"symbol": "BTCUSDC", "brackets": [{"initialLeverage": "3", "maxLeverage": "75"}]},
            ]
        if path == "/fapi/v1/leverage":
            return {"symbol": params["symbol"], "leverage": params["leverage"]}
        raise AssertionError(f"unexpected endpoint: {path}")

    monkeypatch.setattr(client, "_request", fake_request)
    assert client.get_max_leverage("BTCUSDC") == 75
    response = client.set_leverage("BTCUSDC", 100)
    assert response["leverage"] == 75
    assert calls == [
        ("GET", "/fapi/v1/leverageBracket"),
        ("GET", "/fapi/v1/leverageBracket"),
        ("POST", "/fapi/v1/leverage"),
    ]


def test_spot_leverage_methods_rejected() -> None:
    client = BinanceClient(api_key="k", api_secret="s", market_type="spot")
    assert client.get_max_leverage("BTCUSDC") == 1
    with pytest.raises(BinanceAPIError):
        client.set_leverage("BTCUSDC", 10)


def test_symbol_constraints_and_normalize_quantity(monkeypatch) -> None:
    client = BinanceClient(api_key="k", api_secret="s", market_type="futures")

    def fake_request(method: str, path: str, params=None, signed: bool = False):
        if path == "/fapi/v1/exchangeInfo":
            return {
                "symbols": [
                    {
                        "symbol": "BTCUSDC",
                        "filters": [
                            {"filterType": "LOT_SIZE", "minQty": "0.001", "maxQty": "5", "stepSize": "0.001"},
                            {"filterType": "NOTIONAL", "minNotional": "10", "maxNotional": "3000"},
                        ],
                    }
                ]
            }
        raise AssertionError(f"unexpected endpoint: {path}")

    monkeypatch.setattr(client, "_request", fake_request)
    constraints = client.get_symbol_constraints("BTCUSDC")
    assert constraints == SymbolConstraints(
        symbol="BTCUSDC",
        min_qty=0.001,
        max_qty=5.0,
        step_size=0.001,
        min_notional=10.0,
        max_notional=3000.0,
    )

    normalized, parsed = client.normalize_quantity("BTCUSDC", 0.0004)
    assert normalized == 0.0
    assert parsed == constraints

    normalized, parsed = client.normalize_quantity("BTCUSDC", 3.2)
    assert normalized == 3.2
    assert parsed == constraints

    normalized, parsed = client.normalize_quantity("BTCUSDC", 10.0)
    assert normalized == 5.0
    assert parsed == constraints


def test_request_signed_requires_credentials() -> None:
    client = BinanceClient(api_key="", api_secret="", market_type="spot")

    with pytest.raises(BinanceAPIError, match="signed endpoint requires api_key/api_secret"):
        client._request("POST", "/api/v3/order", {"symbol": "BTCUSDC"}, signed=True)


def test_request_supports_unsigned_and_signed_paths(monkeypatch) -> None:
    client = BinanceClient(api_key="k", api_secret="s", market_type="futures", recv_window_ms=12345)
    calls: list[tuple[str, str, dict]] = []

    class Response:
        status_code = 200
        text = ""

        def json(self):
            return {}

    def fake_request(method: str, url: str, params=None, timeout=None):
        calls.append((method, url.split("?")[0], params or {}))
        if "?" in url:
            assert "signature=" in url
            assert "recvWindow=12345" in url
        return Response()

    monkeypatch.setattr(client.session, "request", fake_request)
    client._request("GET", "/fapi/v1/time", {"symbol": "BTCUSDC"})
    client._request("POST", "/fapi/v1/order", {"symbol": "BTCUSDC"}, signed=True)
    assert len(calls) == 2
    assert calls[0][0] == "GET"
    assert calls[0][1] == "https://testnet.binancefuture.com/fapi/v1/time"
    assert calls[1][1] == "https://testnet.binancefuture.com/fapi/v1/order"


def test_quantize_and_parse_filter_handle_invalid_inputs() -> None:
    assert BinanceClient._quantize_to_step(1.23, 0.0) == 1.23
    assert BinanceClient._quantize_to_step(1.23, -1.0) == 1.23
    assert BinanceClient._quantize_to_step(1.23, float("nan")) == 0.0

    assert BinanceClient._parse_filter([1, {"filterType": "X", "v": 1}, {"filterType": "Y"}], "Y") == {"filterType": "Y"}


def test_get_symbol_constraints_handles_non_list_filters_and_allocation_fallbacks(monkeypatch) -> None:
    client = BinanceClient(api_key="k", api_secret="s", market_type="futures")

    def fake_request(method: str, path: str, params=None, signed: bool = False):
        assert path == "/fapi/v1/exchangeInfo"
        return {"symbols": [{"symbol": "BTCUSDC", "filters": {"bad": 1}}]}

    monkeypatch.setattr(client, "_request", fake_request)
    with pytest.raises(BinanceAPIError, match="Unexpected symbol filters"):
        client.get_symbol_constraints("BTCUSDC")


def test_get_symbol_constraints_supports_market_lot_and_notional_fallbacks(monkeypatch) -> None:
    client = BinanceClient(api_key="k", api_secret="s", market_type="futures")

    def fake_request(method: str, path: str, params=None, signed: bool = False):
        if path == "/fapi/v1/exchangeInfo":
            return {
                "symbols": [
                    {
                        "symbol": "BTCUSDC",
                        "filters": [
                            {"filterType": "LOT_SIZE", "minQty": "0.001", "maxQty": "5", "stepSize": "0.001"},
                            {"filterType": "MARKET_LOT_SIZE", "minQty": "0.002", "maxQty": "3", "stepSize": "0.005"},
                            {"filterType": "MIN_NOTIONAL", "notional": "15"},
                        ]
                    }
                ]
            }
        raise AssertionError

    monkeypatch.setattr(client, "_request", fake_request)
    constraints = client.get_symbol_constraints("BTCUSDC")
    assert constraints.min_qty == 0.002
    assert constraints.max_qty == 3.0
    assert constraints.step_size == 0.005
    assert constraints.min_notional == 15.0
    assert constraints.max_notional == 0.0


def test_get_symbol_constraints_falls_back_to_market_lot_notional(monkeypatch) -> None:
    client = BinanceClient(api_key="k", api_secret="s", market_type="futures")

    def fake_request(method: str, path: str, params=None, signed: bool = False):
        if path == "/fapi/v1/exchangeInfo":
            return {
                "symbols": [
                    {
                        "symbol": "BTCUSDC",
                        "filters": [
                            {"filterType": "LOT_SIZE", "minQty": "0.1", "maxQty": "2", "stepSize": "0.1"},
                            {"filterType": "NOTIONAL", "minNotional": "5"},
                        ]
                    }
                ]
            }
        raise AssertionError

    monkeypatch.setattr(client, "_request", fake_request)
    constraints = client.get_symbol_constraints("BTCUSDC")
    assert constraints.min_qty == 0.1
    assert constraints.max_qty == 2.0
    assert constraints.max_notional == 0.0


def test_get_klines_rejects_bad_symbol_and_payload_shape(monkeypatch) -> None:
    client = BinanceClient(api_key="k", api_secret="s")
    with pytest.raises(BinanceAPIError, match="BTCUSDC only"):
        client.get_klines("ETHUSDC", "15m")

    def bad_payload(method, path, params=None, signed=False):
        return {"symbol": "BTCUSDC"}

    monkeypatch.setattr(client, "_request", bad_payload)
    with pytest.raises(BinanceAPIError, match="Unexpected kline payload"):
        client.get_klines("BTCUSDC", "15m", limit=1)

    def short_row(method, path, params=None, signed=False):
        return [[1, "100", "101", "99", "100", "1"]]

    monkeypatch.setattr(client, "_request", short_row)
    with pytest.raises(BinanceAPIError, match="Unexpected kline row"):
        client.get_klines("BTCUSDC", "15m", limit=1)


def test_get_klines_passes_optional_range_filters(monkeypatch) -> None:
    client = BinanceClient(api_key="k", api_secret="s")
    started = {"value": False}

    def request(method, path, params=None, signed=False):
        if params is None:
            return []
        assert "startTime" in params
        assert "endTime" in params
        started["value"] = True
        return []

    monkeypatch.setattr(client, "_request", request)
    client.get_klines("BTCUSDC", "15m", start_time=111, end_time=222, limit=5)
    assert started["value"]


def test_exchange_info_helpers_and_symbol_price(monkeypatch) -> None:
    spot = BinanceClient(api_key="k", api_secret="s", market_type="spot", testnet=False)
    futures = BinanceClient(api_key="k", api_secret="s", market_type="futures", testnet=False)

    def request(method: str, path: str, params=None, signed: bool = False):
        if path == "/api/v3/time":
            return {"serverTime": 42}
        if path == "/api/v3/ticker/price":
            return {"symbol": params["symbol"], "price": "50000.0"}
        if path == "/fapi/v1/time":
            return {"serverTime": 77}
        if path == "/fapi/v1/ticker/price":
            return {"symbol": params["symbol"], "price": "50000.0"}
        raise AssertionError(path)

    monkeypatch.setattr(spot, "_request", request)
    assert spot.get_exchange_time() == {"serverTime": 42}
    assert isinstance(spot.get_symbol_price("BTCUSDC")[0], float)

    monkeypatch.setattr(futures, "_request", request)
    assert futures.get_exchange_time() == {"serverTime": 77}


def test_place_order_branches_and_leverage_clamping(monkeypatch) -> None:
    client = BinanceClient(api_key="k", api_secret="s", market_type="futures")

    call_log: list[tuple[str, str, dict]] = []

    class _FakeResponse:
        def __init__(self) -> None:
            self.status_code = 200
            self.text = "{\"ok\": true}"

        def json(self):
            return {"ok": True}

    def request(method: str, url: str, params=None, timeout=None):
        call_log.append((method, url, params or {}))
        return _FakeResponse()

    monkeypatch.setattr(client.session, "request", request)
    monkeypatch.setattr(client, "get_max_leverage", lambda _symbol: 2)

    result_dry = client.place_order("BTCUSDC", "BUY", 0.123, dry_run=True, leverage=10.0)
    assert result_dry["dryRun"] is True
    assert result_dry["leverage"] == 10.0

    result_live = client.place_order("BTCUSDC", "SELL", 0.1, dry_run=False, leverage=10.0)
    assert result_live["ok"] is True
    assert call_log[-1][0] == "POST"
    assert "/fapi/v1/order" in call_log[-1][1]
