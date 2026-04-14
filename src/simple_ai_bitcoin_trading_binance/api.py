"""Binance HTTP client for free/public endpoints and constrained test trading calls."""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple
from urllib.parse import urlencode

import requests


BINANCE_SPOT_TESTNET = "https://testnet.binance.vision"
BINANCE_SPOT_LIVE = "https://api.binance.com"
BINANCE_FUTURES_TESTNET = "https://testnet.binancefuture.com"
BINANCE_FUTURES_LIVE = "https://fapi.binance.com"
_MAX_FUTURES_LEVERAGE = 125


def _default_base_url(testnet: bool, market_type: str) -> tuple[str, str]:
    if market_type == "futures":
        return (BINANCE_FUTURES_TESTNET if testnet else BINANCE_FUTURES_LIVE, "fapi")
    return (BINANCE_SPOT_TESTNET if testnet else BINANCE_SPOT_LIVE, "api")


class BinanceAPIError(RuntimeError):
    """Raised for non-2xx responses from Binance endpoints."""


@dataclass(frozen=True)
class Candle:
    open_time: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    close_time: int


class BinanceClient:
    """Small HTTP client wrapping only the endpoints required by this tool."""

    def __init__(self, api_key: str, api_secret: str, *, testnet: bool = True,
                 market_type: str = "spot", timeout: int = 10, max_calls_per_minute: int = 1200):
        if market_type not in {"spot", "futures"}:
            raise ValueError("market_type must be 'spot' or 'futures'")

        self.api_key = api_key
        self.api_secret = api_secret.encode("utf-8")
        self.market_type = market_type
        self.base_url, self.api_prefix = _default_base_url(testnet, market_type)
        self.session = requests.Session()
        self.session.headers.update({"X-MBX-APIKEY": api_key})
        self.session.headers.update({"User-Agent": "simple-ai-btcusdc-cli"})
        self.timeout = timeout
        if max_calls_per_minute < 1:
            max_calls_per_minute = 1
        if max_calls_per_minute > 2000:
            max_calls_per_minute = 2000
        self._call_delay = 60.0 / max_calls_per_minute
        self._rate_limit_at: datetime = datetime.utcnow()

    def _throttle(self) -> None:
        now = datetime.utcnow()
        min_interval = timedelta(seconds=self._call_delay)
        delay = (self._rate_limit_at - now).total_seconds()
        if delay > 0:
            time.sleep(delay)
        self._rate_limit_at = datetime.utcnow() + min_interval

    def _request(self, method: str, path: str, params: Dict[str, object] | None = None,
                 signed: bool = False) -> Dict[str, object] | List[Dict[str, object]]:
        self._throttle()
        if params is None:
            params = {}

        if signed:
            if not self.api_key or not self.api_secret:
                raise BinanceAPIError("signed endpoint requires api_key/api_secret")
            params = dict(params)
            params.setdefault("timestamp", int(time.time() * 1000))
            params.setdefault("recvWindow", 5000)
            query = urlencode(sorted((k, v) for k, v in params.items()))
            signature = hmac.new(self.api_secret, query.encode("utf-8"), hashlib.sha256).hexdigest()
            query += f"&signature={signature}"
            url = f"{self.base_url}{path}?{query}"
            payload = None
        else:
            payload = params or {}
            url = f"{self.base_url}{path}"

        response = self.session.request(method, url, params=payload, timeout=self.timeout)
        if response.status_code >= 400:
            raise BinanceAPIError(f"Binance returned {response.status_code}: {response.text}")
        try:
            data = response.json()
        except json.JSONDecodeError as err:
            raise BinanceAPIError("Malformed response from Binance") from err

        if isinstance(data, dict) and data.get("code") and data.get("msg"):
            raise BinanceAPIError(f"Binance API error {data['code']}: {data['msg']}")
        return data

    def ping(self) -> Dict[str, object] | None:
        endpoint = "/api/v3/ping" if self.market_type == "spot" else "/fapi/v1/ping"
        return self._request("GET", endpoint)

    def get_exchange_info(self) -> Dict[str, object]:
        endpoint = "/api/v3/exchangeInfo" if self.market_type == "spot" else "/fapi/v1/exchangeInfo"
        return self._request("GET", endpoint)

    def ensure_btcusdc(self) -> Dict[str, object]:
        info = self.get_exchange_info()
        symbols = [s for s in info.get("symbols", []) if s.get("symbol") == "BTCUSDC"]
        if not symbols:
            raise BinanceAPIError("BTCUSDC is unavailable on this endpoint. Check Binance support for the current market")
        symbol_info = symbols[0]
        if symbol_info.get("status") != "TRADING":
            raise BinanceAPIError(f"BTCUSDC is not trading. Status: {symbol_info.get('status')}")
        return symbol_info

    def get_klines(self, symbol: str, interval: str, *, limit: int = 500,
                   start_time: int | None = None, end_time: int | None = None) -> List[Candle]:
        if symbol.upper() != "BTCUSDC":
            raise BinanceAPIError("This CLI supports BTCUSDC only")
        params: Dict[str, object] = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
        }
        if start_time is not None:
            params["startTime"] = start_time
        if end_time is not None:
            params["endTime"] = end_time

        endpoint = "/api/v3/klines" if self.market_type == "spot" else "/fapi/v1/klines"
        payload = self._request("GET", endpoint, params=params)
        if not isinstance(payload, list):
            raise BinanceAPIError("Unexpected kline payload")

        candles = []
        for row in payload:
            if len(row) < 7:
                raise BinanceAPIError("Unexpected kline row")
            candles.append(
                Candle(
                    open_time=int(row[0]),
                    open=float(row[1]),
                    high=float(row[2]),
                    low=float(row[3]),
                    close=float(row[4]),
                    volume=float(row[5]),
                    close_time=int(row[6]),
                )
            )
        return candles

    def get_account(self) -> Dict[str, object]:
        endpoint = "/api/v3/account" if self.market_type == "spot" else "/fapi/v2/account"
        return self._request("GET", endpoint, {}, signed=True)

    def get_symbol_price(self, symbol: str) -> Tuple[float, int]:
        endpoint = "/api/v3/ticker/price" if self.market_type == "spot" else "/fapi/v1/ticker/price"
        data = self._request("GET", endpoint, {"symbol": symbol})
        return float(data["price"]), int(time.time() * 1000)

    def set_leverage(self, symbol: str, leverage: int) -> Dict[str, object]:
        if self.market_type != "futures":
            raise BinanceAPIError("Leverage is available only in futures mode")
        leverage = int(leverage)
        if leverage < 1:
            leverage = 1
        if leverage > _MAX_FUTURES_LEVERAGE:
            leverage = _MAX_FUTURES_LEVERAGE
        payload = {"symbol": symbol, "leverage": leverage}
        return self._request("POST", "/fapi/v1/leverage", payload, signed=True)

    def place_order(self, symbol: str, side: str, quantity: float, *, dry_run: bool,
                   leverage: float = 1.0) -> Dict[str, object]:
        payload = {
            "symbol": symbol,
            "side": side,
            "type": "MARKET",
            "quantity": f"{quantity:.8f}",
        }

        if dry_run:
            return {
                "dryRun": True,
                "symbol": symbol,
                "side": side,
                "type": "MARKET",
                "quantity": payload["quantity"],
                "leverage": leverage,
            }

        if self.market_type == "spot":
            return self._request("POST", "/api/v3/order", payload, signed=True)

        # futures: configure leverage before market order submission
        self.set_leverage(symbol, int(max(1, round(leverage))))
        return self._request("POST", "/fapi/v1/order", payload, signed=True)

    def get_exchange_time(self) -> Dict[str, object] | None:
        endpoint = "/api/v3/time" if self.market_type == "spot" else "/fapi/v1/time"
        return self._request("GET", endpoint)
