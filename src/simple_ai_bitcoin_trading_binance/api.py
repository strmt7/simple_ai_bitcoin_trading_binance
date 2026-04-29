"""Binance HTTP client for free/public endpoints and constrained test trading calls."""

from __future__ import annotations

import json
import os
import re
import time
import hashlib
import hmac
from decimal import Decimal, ROUND_DOWN, InvalidOperation
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Tuple
from urllib.parse import parse_qsl, urlencode, urlsplit

import requests


BINANCE_SPOT_TESTNET = "https://testnet.binance.vision"
BINANCE_SPOT_LIVE = "https://api.binance.com"
BINANCE_SPOT_DEMO = "https://demo-api.binance.com"
BINANCE_FUTURES_TESTNET = "https://testnet.binancefuture.com"
BINANCE_FUTURES_LIVE = "https://fapi.binance.com"
BINANCE_FUTURES_DEMO = "https://demo-fapi.binance.com"
_MAX_FUTURES_LEVERAGE = 125
_RETRY_HTTP_STATUSES = {418, 429, 500, 502, 503, 504}
_RETRY_BAPI_CODES = {-1003, -1007}
_SENSITIVE_QUERY_FIELDS = {"signature", "timestamp", "recvWindow"}


def _extract_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    if not re.fullmatch(r"\d+(\.\d+)?", value.strip()):
        return None
    try:
        return max(0.0, float(value.strip()))
    except ValueError:
        return None


def _redact_request_url(url: str) -> str:
    parts = urlsplit(url)
    if not parts.query:
        return parts.path or url
    query = []
    for key, value in parse_qsl(parts.query, keep_blank_values=True):
        query.append((key, "<redacted>" if key in _SENSITIVE_QUERY_FIELDS else value))
    return f"{parts.path}?{urlencode(query)}"


def _redact_sensitive_text(text: str, request_url: str | None = None) -> str:
    redacted = text
    if request_url:
        redacted = redacted.replace(request_url, _redact_request_url(request_url))
    for field in _SENSITIVE_QUERY_FIELDS:
        redacted = re.sub(
            rf"([?&]{re.escape(field)}=)[^&\s'\"<>)]*",
            rf"\1<redacted>",
            redacted,
        )
    return redacted


def _default_base_url(testnet: bool, market_type: str, *, demo: bool = False) -> tuple[str, str]:
    common_override = os.getenv("BINANCE_BASE_URL", "").strip()
    if market_type == "futures":
        futures_override = os.getenv("BINANCE_FUTURES_BASE_URL", "").strip()
        if futures_override:
            return futures_override, "fapi"
        if common_override:
            return common_override, "fapi"
        if demo:
            return BINANCE_FUTURES_DEMO, "fapi"
        return (BINANCE_FUTURES_TESTNET if testnet else BINANCE_FUTURES_LIVE, "fapi")
    spot_override = os.getenv("BINANCE_SPOT_BASE_URL", "").strip()
    if spot_override:
        return spot_override, "api"
    if common_override:
        return common_override, "api"
    if demo:
        return BINANCE_SPOT_DEMO, "api"
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
    quote_volume: float = 0.0
    trade_count: int = 0
    taker_buy_base_volume: float = 0.0
    taker_buy_quote_volume: float = 0.0


@dataclass(frozen=True)
class SymbolConstraints:
    symbol: str
    min_qty: float
    max_qty: float
    step_size: float
    min_notional: float
    max_notional: float


class BinanceClient:
    """Small HTTP client wrapping only the endpoints required by this tool."""

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        *,
        testnet: bool = True,
        demo: bool = False,
        market_type: str = "spot",
        timeout: int = 10,
        max_calls_per_minute: int = 1200,
        max_retries: int = 4,
        recv_window_ms: int = 5000,
    ):
        if market_type not in {"spot", "futures"}:
            raise ValueError("market_type must be 'spot' or 'futures'")

        self.api_key = api_key
        self.api_secret = api_secret.encode("utf-8")
        self.market_type = market_type
        self.demo = bool(demo)
        self.base_url, self.api_prefix = _default_base_url(testnet, market_type, demo=self.demo)
        self.session = requests.Session()
        self.session.headers.update({"X-MBX-APIKEY": api_key})
        self.session.headers.update({"User-Agent": "simple-ai-btcusdc-cli"})
        self.timeout = timeout
        try:
            recv_window_ms = int(recv_window_ms)
        except (TypeError, ValueError):
            recv_window_ms = 5000
        self.recv_window_ms = max(1, min(60000, recv_window_ms))
        if max_calls_per_minute < 1:
            max_calls_per_minute = 1
        if max_calls_per_minute > 2000:
            max_calls_per_minute = 2000
        self._call_delay = 60.0 / max_calls_per_minute
        self._rate_limit_at: datetime = datetime.now(timezone.utc)
        self.max_retries = max(0, max_retries)
        self.last_request_info: Dict[str, object] = {
            "attempts": 0,
            "status": None,
            "retries": 0,
            "method": None,
            "path": None,
        }

    def _record_request(
        self,
        attempt: int,
        response_status: int | None,
        method: str,
        path: str,
        url: str,
        last_error: str | None,
    ) -> None:
        self.last_request_info = {
            "attempts": attempt,
            "status": response_status,
            "retries": max(0, attempt - 1),
            "method": method,
            "path": path,
            "last_error": last_error,
            "url": _redact_request_url(url),
        }

    def _throttle(self) -> None:
        now = datetime.now(timezone.utc)
        min_interval = timedelta(seconds=self._call_delay)
        delay = (self._rate_limit_at - now).total_seconds()
        if delay > 0:
            time.sleep(delay)
        self._rate_limit_at = datetime.now(timezone.utc) + min_interval

    @staticmethod
    def _is_retryable_code(raw_code: object) -> bool:
        try:
            value = int(raw_code)
        except (TypeError, ValueError):
            return False
        return value in _RETRY_BAPI_CODES

    def _retry_delay(self, attempt: int, response_status: int | None, *, retry_after: float | None = None) -> float:
        if retry_after is not None:
            return min(60.0, max(0.0, retry_after))
        base = 0.5
        return min(30.0, base * (2 ** max(0, attempt)))

    def _request(self, method: str, path: str, params: Dict[str, object] | None = None,
                 signed: bool = False) -> Dict[str, object] | List[Dict[str, object]]:
        if params is None:
            params = {}
        if not isinstance(params, dict):
            params = {}

        max_attempts = self.max_retries + 1
        last_error = None
        response_status: int | None = None
        last_url: str | None = None

        for attempt in range(max_attempts):
            self._throttle()

            if signed:
                if not self.api_key or not self.api_secret:
                    raise BinanceAPIError("signed endpoint requires api_key/api_secret")
                request_params = dict(params)
                request_params["timestamp"] = int(time.time() * 1000)
                request_params.setdefault("recvWindow", self.recv_window_ms)
                query = urlencode(sorted((k, v) for k, v in request_params.items()))
                signature = hmac.new(self.api_secret, query.encode("utf-8"), hashlib.sha256).hexdigest()
                query += f"&signature={signature}"
                url = f"{self.base_url}{path}?{query}"
                payload = None
            else:
                payload = dict(params)
                url = f"{self.base_url}{path}"
            last_url = url

            try:
                response = self.session.request(method, url, params=payload, timeout=self.timeout)
            except requests.RequestException as err:
                last_error = _redact_sensitive_text(str(err), url)
                if attempt < self.max_retries:
                    delay = self._retry_delay(attempt, response_status=response_status)
                    time.sleep(delay)
                    continue
                self._record_request(attempt + 1, response_status, method, path, url, last_error)
                raise BinanceAPIError(f"Binance request failed: {last_error}") from err

            response_status = response.status_code
            if response_status >= 400:
                retry_after = _extract_retry_after(response.headers.get("Retry-After"))
                response_text = _redact_sensitive_text(response.text, url)
                last_error = f"HTTP {response_status}: {response_text}"
                if response_status in _RETRY_HTTP_STATUSES and attempt < self.max_retries:
                    delay = self._retry_delay(attempt, response_status=response_status, retry_after=retry_after)
                    time.sleep(delay)
                    continue
                self._record_request(attempt + 1, response_status, method, path, url, last_error)
                raise BinanceAPIError(f"Binance returned {response.status_code}: {response_text}")

            try:
                data = response.json()
            except json.JSONDecodeError as err:
                last_error = "Malformed response from Binance"
                if attempt < self.max_retries:
                    delay = self._retry_delay(attempt, response_status=response_status)
                    time.sleep(delay)
                    continue
                self._record_request(attempt + 1, response_status, method, path, url, last_error)
                raise BinanceAPIError("Malformed response from Binance") from err

            if isinstance(data, dict) and data.get("code") and data.get("msg"):
                if self._is_retryable_code(data.get("code")) and attempt < self.max_retries:
                    code = data.get("code")
                    last_error = _redact_sensitive_text(f"Binance API error {code}: {data['msg']}", url)
                    delay = self._retry_delay(attempt, response_status=response_status)
                    time.sleep(delay)
                    continue
                api_error = _redact_sensitive_text(
                    f"Binance API error {data['code']}: {data['msg']}",
                    url,
                )
                self._record_request(
                    attempt + 1,
                    response_status,
                    method,
                    path,
                    url,
                    api_error,
                )
                raise BinanceAPIError(api_error)

            self._record_request(attempt + 1, response_status, method, path, url, None)
            return data

        last_error = _redact_sensitive_text(last_error or "", last_url) or None
        self._record_request(max_attempts, response_status, method, path, last_url or path, last_error)
        raise BinanceAPIError(last_error or "Binance request failed")

    def ping(self) -> Dict[str, object] | None:
        endpoint = "/api/v3/ping" if self.market_type == "spot" else "/fapi/v1/ping"
        return self._request("GET", endpoint)

    def get_exchange_info(self) -> Dict[str, object]:
        endpoint = "/api/v3/exchangeInfo" if self.market_type == "spot" else "/fapi/v1/exchangeInfo"
        return self._request("GET", endpoint)

    @staticmethod
    def _parse_float(value: object) -> float:
        if value is None:
            return 0.0
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _parse_filter(filters: list[object], filter_type: str) -> Dict[str, str]:
        for item in filters:
            if not isinstance(item, dict):
                continue
            if item.get("filterType") == filter_type:
                return item
        return {}

    @staticmethod
    def _quantize_to_step(value: float, step_size: float) -> float:
        try:
            value_dec = Decimal(str(value))
            if not value_dec.is_finite():
                return 0.0
            if value_dec <= 0:
                return 0.0
        except (InvalidOperation, ValueError, TypeError):
            return 0.0

        try:
            step = Decimal(str(step_size))
        except (InvalidOperation, ValueError, TypeError):
            return 0.0
        if not step.is_finite():
            return 0.0
        if step <= 0:
            return float(value_dec)

        normalized = (value_dec // step) * step
        try:
            return float(normalized.quantize(step, rounding=ROUND_DOWN))
        except (InvalidOperation, ValueError):
            return 0.0

    def get_symbol_constraints(self, symbol: str) -> SymbolConstraints:
        symbol = symbol.upper()
        info = self.get_exchange_info()
        symbols = [item for item in info.get("symbols", []) if isinstance(item, dict) and item.get("symbol") == symbol]
        if not symbols:
            raise BinanceAPIError(f"Unknown symbol in exchangeInfo: {symbol}")

        symbol_info = symbols[0]
        filters = symbol_info.get("filters", [])
        if not isinstance(filters, list):
            raise BinanceAPIError(f"Unexpected symbol filters for {symbol}")

        lot_filter = self._parse_filter(filters, "LOT_SIZE")
        market_lot_filter = self._parse_filter(filters, "MARKET_LOT_SIZE")

        min_qty = self._parse_float(market_lot_filter.get("minQty") if market_lot_filter else lot_filter.get("minQty"))
        max_qty = self._parse_float(market_lot_filter.get("maxQty") if market_lot_filter else lot_filter.get("maxQty"))
        step_size = self._parse_float(market_lot_filter.get("stepSize") if market_lot_filter else lot_filter.get("stepSize"))

        if min_qty <= 0:
            min_qty = self._parse_float(lot_filter.get("minQty"))
        if max_qty <= 0:
            max_qty = self._parse_float(lot_filter.get("maxQty"))
        if step_size <= 0:
            step_size = self._parse_float(lot_filter.get("stepSize"))

        if min_qty <= 0:
            min_qty = 0.0
        if max_qty <= 0:
            max_qty = 0.0
        if step_size <= 0:
            step_size = 0.0

        notional_filter = self._parse_filter(filters, "NOTIONAL")
        min_notional = self._parse_float(notional_filter.get("minNotional")) if notional_filter else 0.0
        max_notional = self._parse_float(notional_filter.get("maxNotional")) if notional_filter else 0.0

        if min_notional <= 0:
            min_notional = self._parse_float(self._parse_filter(filters, "MIN_NOTIONAL").get("notional"))

        if min_notional <= 0:
            min_notional = 0.0
        if max_notional <= 0:
            max_notional = 0.0

        return SymbolConstraints(
            symbol=symbol,
            min_qty=min_qty,
            max_qty=max_qty,
            step_size=step_size,
            min_notional=min_notional,
            max_notional=max_notional,
        )

    def normalize_quantity(self, symbol: str, quantity: float) -> tuple[float, SymbolConstraints]:
        constraints = self.get_symbol_constraints(symbol)
        normalized = self._quantize_to_step(quantity, constraints.step_size)

        if normalized <= 0:
            return 0.0, constraints

        if normalized < constraints.min_qty:
            return 0.0, constraints

        if constraints.max_qty > 0:
            normalized = min(normalized, constraints.max_qty)
        return normalized, constraints

    def get_leverage_brackets(self, symbol: str) -> List[Dict[str, object]]:
        if self.market_type != "futures":
            raise BinanceAPIError("Leverage brackets are available only in futures mode")

        payload = self._request("GET", "/fapi/v1/leverageBracket", {"symbol": symbol.upper()}, signed=True)
        if not isinstance(payload, list):
            raise BinanceAPIError("Unexpected leverage bracket payload")
        return payload  # type: ignore[return-value]

    def get_max_leverage(self, symbol: str) -> int:
        if self.market_type != "futures":
            return 1
        payload = self.get_leverage_brackets(symbol)
        symbol = symbol.upper()
        for item in payload:
            if not isinstance(item, dict) or item.get("symbol") != symbol:
                continue
            brackets = item.get("brackets")
            if not isinstance(brackets, list) or not brackets:
                continue
            max_leverage = 0
            for bracket in brackets:
                if not isinstance(bracket, dict):
                    continue
                for key in ("maxLeverage", "initialLeverage"):
                    value = bracket.get(key)
                    if value is None:
                        continue
                    try:
                        parsed = int(float(value))
                    except (TypeError, ValueError):
                        continue
                    if parsed > max_leverage:
                        max_leverage = parsed
            if max_leverage > 0:
                return min(max_leverage, _MAX_FUTURES_LEVERAGE)
        return _MAX_FUTURES_LEVERAGE

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
                    quote_volume=self._parse_float(row[7]) if len(row) > 7 else 0.0,
                    trade_count=int(self._parse_float(row[8])) if len(row) > 8 else 0,
                    taker_buy_base_volume=self._parse_float(row[9]) if len(row) > 9 else 0.0,
                    taker_buy_quote_volume=self._parse_float(row[10]) if len(row) > 10 else 0.0,
                )
            )
        return candles

    def get_ticker_24h(self, symbol: str) -> Dict[str, object]:
        endpoint = "/api/v3/ticker/24hr" if self.market_type == "spot" else "/fapi/v1/ticker/24hr"
        payload = self._request("GET", endpoint, {"symbol": symbol.upper()})
        if not isinstance(payload, dict):
            raise BinanceAPIError("Unexpected 24h ticker payload")
        return payload

    def get_book_ticker(self, symbol: str) -> Dict[str, object]:
        endpoint = "/api/v3/ticker/bookTicker" if self.market_type == "spot" else "/fapi/v1/ticker/bookTicker"
        payload = self._request("GET", endpoint, {"symbol": symbol.upper()})
        if not isinstance(payload, dict):
            raise BinanceAPIError("Unexpected book ticker payload")
        return payload

    def get_futures_premium_index(self, symbol: str) -> Dict[str, object]:
        if self.market_type != "futures":
            raise BinanceAPIError("Premium index is available only in futures mode")
        payload = self._request("GET", "/fapi/v1/premiumIndex", {"symbol": symbol.upper()})
        if not isinstance(payload, dict):
            raise BinanceAPIError("Unexpected premium index payload")
        return payload

    def get_futures_open_interest(self, symbol: str) -> Dict[str, object]:
        if self.market_type != "futures":
            raise BinanceAPIError("Open interest is available only in futures mode")
        payload = self._request("GET", "/fapi/v1/openInterest", {"symbol": symbol.upper()})
        if not isinstance(payload, dict):
            raise BinanceAPIError("Unexpected open interest payload")
        return payload

    def get_futures_funding_rate(
        self,
        symbol: str,
        *,
        limit: int = 100,
        start_time: int | None = None,
        end_time: int | None = None,
    ) -> List[Dict[str, object]]:
        if self.market_type != "futures":
            raise BinanceAPIError("Funding rate history is available only in futures mode")
        params: Dict[str, object] = {"symbol": symbol.upper(), "limit": max(1, min(1000, int(limit)))}
        if start_time is not None:
            params["startTime"] = int(start_time)
        if end_time is not None:
            params["endTime"] = int(end_time)
        payload = self._request("GET", "/fapi/v1/fundingRate", params)
        if not isinstance(payload, list):
            raise BinanceAPIError("Unexpected funding rate payload")
        return payload  # type: ignore[return-value]

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
        max_leverage = self.get_max_leverage(symbol)
        if leverage > max_leverage:
            leverage = max_leverage
        payload = {"symbol": symbol, "leverage": leverage}
        return self._request("POST", "/fapi/v1/leverage", payload, signed=True)

    def place_order(self, symbol: str, side: str, quantity: float, *, dry_run: bool,
                   leverage: float = 1.0, reduce_only: bool = False) -> Dict[str, object]:
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
                "reduceOnly": bool(reduce_only),
            }

        if self.market_type == "spot":
            return self._request("POST", "/api/v3/order", payload, signed=True)

        payload["newOrderRespType"] = "RESULT"
        if reduce_only:
            payload["reduceOnly"] = "true"
        # futures: configure leverage before market order submission
        self.set_leverage(symbol, int(max(1, round(leverage))))
        return self._request("POST", "/fapi/v1/order", payload, signed=True)

    def get_exchange_time(self) -> Dict[str, object] | None:
        endpoint = "/api/v3/time" if self.market_type == "spot" else "/fapi/v1/time"
        return self._request("GET", endpoint)
