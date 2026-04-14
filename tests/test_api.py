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
