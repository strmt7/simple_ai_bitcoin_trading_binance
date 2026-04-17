from __future__ import annotations

import argparse
import asyncio
import json
import runpy
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path
from types import SimpleNamespace

import pytest

from simple_ai_bitcoin_trading_binance import api, cli, features, model
from simple_ai_bitcoin_trading_binance.api import BinanceAPIError, Candle, SymbolConstraints
from simple_ai_bitcoin_trading_binance.backtest import run_backtest
from simple_ai_bitcoin_trading_binance.config import RuntimeConfig, save_runtime, save_strategy
from simple_ai_bitcoin_trading_binance.dashboard import load_artifact_preview
from simple_ai_bitcoin_trading_binance.features import ModelRow
from simple_ai_bitcoin_trading_binance.model import (
    ModelFeatureMismatchError,
    ModelLoadError,
    TrainedModel,
    calibrate_threshold,
    load_model,
    train,
    walk_forward_report,
)
from simple_ai_bitcoin_trading_binance.types import StrategyConfig


class _ScriptedUI:
    def __init__(self, *, forms=(), confirms=(), multi_selects=()) -> None:
        self.forms = list(forms)
        self.confirms = list(confirms)
        self.multi_selects = list(multi_selects)
        self.logs: list[str] = []

    async def form(self, _title, _fields):
        return self.forms.pop(0)

    async def confirm(self, _message):
        return self.confirms.pop(0)

    async def multi_select(self, _title, _options, _selected, *, help_text=""):
        return self.multi_selects.pop(0)

    def append_log(self, text: str) -> None:
        self.logs.append(text)


def _action(title: str):
    actions = {action.title: action for action in cli._tui_actions()}
    return actions[title]


def _runtime_form(**overrides: str) -> dict[str, str]:
    payload = {
        "market_type": "spot",
        "interval": "15m",
        "testnet": "yes",
        "api_key": "",
        "api_secret": "",
        "dry_run": "yes",
        "validate_account": "no",
        "max_rate_calls_per_minute": "1100",
    }
    payload.update(overrides)
    return payload


def _strategy_form(**overrides: str) -> dict[str, str]:
    cfg = StrategyConfig()
    payload = {
        "leverage": str(cfg.leverage),
        "risk": str(cfg.risk_per_trade),
        "max_position": str(cfg.max_position_pct),
        "stop": str(cfg.stop_loss_pct),
        "take": str(cfg.take_profit_pct),
        "cooldown": str(cfg.cooldown_minutes),
        "max_open": str(cfg.max_open_positions),
        "max_trades_per_day": str(cfg.max_trades_per_day),
        "signal_threshold": str(cfg.signal_threshold),
        "max_drawdown": str(cfg.max_drawdown_limit),
        "taker_fee_bps": str(cfg.taker_fee_bps),
        "slippage_bps": str(cfg.slippage_bps),
        "label_threshold": str(cfg.label_threshold),
        "model_lookback": str(cfg.model_lookback),
        "training_epochs": str(cfg.training_epochs),
        "confidence_beta": str(cfg.confidence_beta),
        "feature_window_short": str(cfg.feature_windows[0]),
        "feature_window_long": str(cfg.feature_windows[1]),
    }
    payload.update(overrides)
    return payload


def _base_candles(n: int = 80, *, start: float = 100.0) -> list[Candle]:
    candles: list[Candle] = []
    for index in range(n):
        close = start + index
        candles.append(
            Candle(
                open_time=index * 60_000,
                open=close,
                high=close * 1.01,
                low=close * 0.99,
                close=close,
                volume=1.0 + index,
                close_time=(index + 1) * 60_000,
            )
        )
    return candles


def _flat_model(probability: float, *, feature_dim: int = 1) -> TrainedModel:
    bias = 0.0 if probability == 0.5 else 10.0 if probability > 0.5 else -10.0
    return TrainedModel(
        weights=[0.0] * feature_dim,
        bias=bias,
        feature_dim=feature_dim,
        epochs=1,
        feature_means=[0.0] * feature_dim,
        feature_stds=[1.0] * feature_dim,
    )


def _strategy_args(**overrides):
    defaults = {
        "leverage": None,
        "risk": None,
        "max_position": None,
        "stop": None,
        "take": None,
        "cooldown": None,
        "max_open": None,
        "max_trades_per_day": None,
        "signal_threshold": None,
        "max_drawdown": None,
        "taker_fee_bps": None,
        "slippage_bps": None,
        "label_threshold": None,
        "model_lookback": None,
        "training_epochs": None,
        "confidence_beta": None,
        "feature_window_short": None,
        "feature_window_long": None,
        "set_features": None,
        "enable_feature": None,
        "disable_feature": None,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _tune_args(**overrides):
    defaults = {
        "input": "candles.json",
        "save_best": False,
        "min_risk": 0.01,
        "max_risk": 0.01,
        "steps": 1,
        "min_leverage": 1.0,
        "max_leverage": 1.0,
        "min_threshold": 0.6,
        "max_threshold": 0.6,
        "min_take": 0.01,
        "max_take": 0.01,
        "min_stop": 0.01,
        "max_stop": 0.01,
        "lookback_days": None,
        "from_date": None,
        "to_date": None,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _live_args(**overrides):
    defaults = {
        "steps": 1,
        "sleep": 0,
        "paper": True,
        "live": False,
        "leverage": None,
        "retrain_interval": 0,
        "retrain_window": 1,
        "retrain_min_rows": 1,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class _SequenceModel:
    def __init__(self, scores: list[float]) -> None:
        self.scores = list(scores)
        self.feature_signature = "test-signature"

    def predict_proba(self, _features) -> float:
        if len(self.scores) == 1:
            return self.scores[0]
        return self.scores.pop(0)


class _LiveClient:
    base_url = "mock://binance"

    def __init__(self, *, candles=None, set_response=None, set_error: Exception | None = None) -> None:
        self.candles = candles if candles is not None else _base_candles(80)
        self.set_response = set_response if set_response is not None else {}
        self.set_error = set_error
        self.orders: list[dict[str, object]] = []

    def get_symbol_constraints(self, _symbol):
        return None

    def normalize_quantity(self, _symbol, quantity):
        return quantity, SymbolConstraints("BTCUSDC", 0.0, 0.0, 0.0, 0.0, 0.0)

    def get_klines(self, *_args, **_kwargs):
        return self.candles

    def place_order(self, symbol, side, size, *, dry_run, leverage):
        payload = {"symbol": symbol, "side": side, "size": size, "dry_run": dry_run, "leverage": leverage}
        self.orders.append(payload)
        return {"status": "FILLED", **payload}

    def set_leverage(self, _symbol, _leverage):
        if self.set_error is not None:
            raise self.set_error
        return self.set_response


def test_api_defensive_and_constraint_edges(monkeypatch) -> None:
    monkeypatch.setattr(api, "float", lambda _value: (_ for _ in ()).throw(ValueError("bad")), raising=False)
    assert api._extract_retry_after("1") is None
    monkeypatch.delattr(api, "float", raising=False)

    client = api.BinanceClient("k", "s", max_retries=0)
    client.max_retries = -1
    with pytest.raises(BinanceAPIError, match="Binance request failed"):
        client._request("GET", "/api/v3/ping")

    assert api.BinanceClient._quantize_to_step(float("inf"), 0.1) == 0.0
    assert api.BinanceClient._quantize_to_step("bad", 0.1) == 0.0

    class RaisingDecimal:
        def __init__(self, _raw) -> None:
            pass

        def is_finite(self):
            return True

        def __le__(self, _other):
            return False

        def __floordiv__(self, _other):
            return self

        def __mul__(self, _other):
            return self

        def quantize(self, *_args, **_kwargs):
            raise InvalidOperation

    monkeypatch.setattr(api, "Decimal", RaisingDecimal)
    assert api.BinanceClient._quantize_to_step(1.23, 0.1) == 0.0
    monkeypatch.setattr(api, "Decimal", Decimal)

    client.get_symbol_constraints = lambda _symbol: SymbolConstraints("BTCUSDC", 2.0, 10.0, 1.0, 0.0, 0.0)
    assert client.normalize_quantity("BTCUSDC", 1.0)[0] == 0.0
    client.get_symbol_constraints = lambda _symbol: SymbolConstraints("BTCUSDC", 0.0, 0.0, 1.0, 0.0, 0.0)
    assert client.normalize_quantity("BTCUSDC", 3.0)[0] == 3.0

    with pytest.raises(BinanceAPIError, match="futures mode"):
        api.BinanceClient("k", "s", market_type="spot").get_leverage_brackets("BTCUSDC")

    futures = api.BinanceClient("k", "s", market_type="futures")
    futures.get_leverage_brackets = lambda _symbol: [{"symbol": "BTCUSDC", "brackets": []}]
    assert futures.get_max_leverage("BTCUSDC") == 125


def test_feature_model_dashboard_and_backtest_edges(tmp_path, monkeypatch) -> None:
    assert features.normalize_enabled_features(["rsi", "rsi"]) == ("rsi",)

    invalid = [
        Candle(0, float("nan"), 1, 1, 1, 1, 1),
        Candle(0, 1, 1, 1, 1, 1, -1),
        Candle(0, 5, 4, 6, 5, 1, 1),
        Candle(0, 7, 6, 4, 5, 1, 1),
        Candle(0, 5, 6, 4, 7, 1, 1),
        Candle(2, 5, 6, 4, 5, 1, 1),
    ]
    assert features.make_rows(invalid + _base_candles(4), 2, 2) == []
    assert features._rsi([5, 4, 3, 2, 1], 4) < 50
    assert features.make_rows(_base_candles(8), 2, 2) == []

    non_object = tmp_path / "artifact.json"
    non_object.write_text("[1, 2, 3]", encoding="utf-8")
    assert load_artifact_preview(non_object).endswith("[non-object]")

    zero_dim_row = ModelRow(timestamp=1, close=1.0, features=(), label=1)
    with pytest.raises(ValueError, match="at least one feature"):
        train([zero_dim_row])

    row = ModelRow(timestamp=1, close=1.0, features=(1.0,), label=1)
    no_stats = TrainedModel([1.0], 0.0, 1, 1, [0.0], [])
    assert no_stats._normalize((2.0,)) == (2.0,)
    assert calibrate_threshold([row], _flat_model(0.9), start=0.8, end=0.2, steps=3) >= 0.8

    monkeypatch.setattr(model, "_class_weights", lambda _rows: (0.0, 0.0))
    trained = train([row], epochs=1)
    assert trained.class_weight_pos == 1.0
    assert trained.class_weight_neg == 1.0

    rows = [ModelRow(i, 100.0 + i, (float(i),), i % 2) for i in range(26)]
    report = walk_forward_report(rows, train_window=8, test_window=10, step=4, epochs=1, calibrate=True)
    assert report["thresholds"] != [0.5] * report["folds"]

    model_payload = {
        "weights": [0.1],
        "bias": 0.0,
        "feature_dim": 1,
        "epochs": 1,
        "feature_version": "v1",
        "feature_means": [0.0],
        "feature_stds": [1.0],
    }
    path = tmp_path / "model.json"
    path.write_text(json.dumps({**model_payload, "feature_stds": None}), encoding="utf-8")
    with pytest.raises(ModelLoadError, match="feature_stds"):
        load_model(path)
    path.write_text(json.dumps({**model_payload, "feature_stds": []}), encoding="utf-8")
    with pytest.raises(ModelLoadError, match="feature_stds length"):
        load_model(path)
    path.write_text(json.dumps(model_payload), encoding="utf-8")
    with pytest.raises(ModelFeatureMismatchError, match="Feature dimension"):
        load_model(path, expected_feature_dim=2)

    always_buy = _flat_model(0.99, feature_dim=1)
    cfg = StrategyConfig(leverage=0.5, risk_per_trade=0.5, max_position_pct=0.5)
    result = run_backtest([ModelRow(1, 100.0, (0.0,), 1)], always_buy, cfg, market_type="futures")
    assert result.trades == 1

    result = run_backtest([ModelRow(1, -1.0, (0.0,), 1)], always_buy, StrategyConfig(), market_type="spot")
    assert result.trades == 0

    expensive = StrategyConfig(risk_per_trade=0.5, max_position_pct=0.5, taker_fee_bps=20_000)
    result = run_backtest([ModelRow(1, 100.0, (0.0,), 1)], always_buy, expensive, starting_cash=100.0)
    assert result.trades == 0


def test_cli_form_parsers_and_ui_edit_helpers() -> None:
    assert cli._parse_form_bool("maybe", True) is True
    with pytest.raises(ValueError, match="Count"):
        cli._parse_form_int("0", label="Count", default=1, minimum=1)
    with pytest.raises(ValueError, match="Count"):
        cli._parse_form_int("3", label="Count", default=1, maximum=2)
    with pytest.raises(ValueError, match="Ratio"):
        cli._parse_form_float("-1", label="Ratio", default=0.0, minimum=0.0)
    with pytest.raises(ValueError, match="Ratio"):
        cli._parse_form_float("2", label="Ratio", default=0.0, maximum=1.0)

    current = RuntimeConfig(market_type="spot", interval="15m", api_key="old", api_secret="secret")
    unchanged = asyncio.run(cli._ui_edit_runtime(_ScriptedUI(forms=[None]), current))
    assert unchanged is current

    edited = asyncio.run(
        cli._ui_edit_runtime(
            _ScriptedUI(forms=[_runtime_form(market_type="bad", interval="", testnet="no", dry_run="off")]),
            current,
        )
    )
    assert edited.market_type == "spot"
    assert edited.interval == "15m"
    assert edited.api_key == "old"
    assert edited.testnet is False
    assert edited.dry_run is False

    cancelled = asyncio.run(cli._ui_edit_strategy_args(_ScriptedUI(multi_selects=[None]), StrategyConfig()))
    assert cancelled.set_features is None

    payload_cancelled = asyncio.run(
        cli._ui_edit_strategy_args(_ScriptedUI(multi_selects=[["momentum_1"]], forms=[None]), StrategyConfig())
    )
    assert payload_cancelled.set_features is None

    strategy_args = asyncio.run(
        cli._ui_edit_strategy_args(
            _ScriptedUI(multi_selects=[["momentum_1", "rsi"]], forms=[_strategy_form(feature_window_short="3", feature_window_long="5")]),
            StrategyConfig(),
        )
    )
    assert strategy_args.set_features == "momentum_1,rsi"
    assert strategy_args.feature_window_short == 3
    assert strategy_args.feature_window_long == 5


def test_cli_dashboard_account_and_artifact_helpers(tmp_path, monkeypatch, capsys) -> None:
    assert cli._recent_artifacts(base_dir=tmp_path / "missing") == []
    save_runtime(RuntimeConfig(api_key="k", api_secret="s"))

    class BadClient:
        def get_account(self):
            raise BinanceAPIError("denied")

    monkeypatch.setattr(cli, "_build_client", lambda _runtime: BadClient())
    assert cli._show_account_overview() == 2
    assert "Account overview failed" in capsys.readouterr().out

    save_runtime(RuntimeConfig(api_key="k", api_secret="s", market_type="spot"))

    class EmptyClient:
        def get_account(self):
            return {"balances": [{"asset": "ETH", "free": "0", "locked": "0"}]}

    monkeypatch.setattr(cli, "_build_client", lambda _runtime: EmptyClient())
    assert cli._account_overview_lines(RuntimeConfig(api_key="k", api_secret="s"))[-1] == "No non-zero balances found."
    assert cli._dashboard_snapshot(with_account=False).account_lines == ["Load Account or Connect to fetch balances."]

    class MixedClient:
        def get_account(self):
            return {"balances": [object(), {"asset": "BTC", "free": "0", "locked": "0"}]}

    monkeypatch.setattr(cli, "_build_client", lambda _runtime: MixedClient())
    assert cli._account_overview_lines(RuntimeConfig(api_key="k", api_secret="s"))[-1] == "BTC: free=0 locked=0"


def test_cli_remaining_helper_edges(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    raw = tmp_path / "candles.json"
    raw.write_text(
        json.dumps(
            [
                {"open_time": "bad"},
                {
                    "open_time": 1,
                    "open": "1",
                    "high": "2",
                    "low": "0.5",
                    "close": "1.5",
                    "volume": "10",
                    "close_time": 2,
                },
            ]
        ),
        encoding="utf-8",
    )
    assert [row.close for row in cli._rows_from_json(str(raw))] == [1.5]

    class ZeroLeverageClient:
        def get_max_leverage(self, _symbol):
            return 0

    monkeypatch.setattr(cli, "_build_client", lambda _runtime: ZeroLeverageClient())
    runtime = RuntimeConfig(market_type="futures", api_key="k", api_secret="s")
    assert cli._resolve_futures_leverage(runtime, StrategyConfig(leverage=12)) == 12

    class RaisingLeverageClient:
        def get_max_leverage(self, _symbol):
            raise BinanceAPIError("brackets unavailable")

    monkeypatch.setattr(cli, "_build_client", lambda _runtime: RaisingLeverageClient())
    assert cli._resolve_futures_leverage(runtime, StrategyConfig(leverage=9)) == 9

    class QuantityClient:
        def __init__(self, quantities):
            self.quantities = list(quantities)

        def normalize_quantity(self, symbol, _quantity):
            return self.quantities.pop(0), SymbolConstraints(symbol, 1.0, 0.0, 0.0, 0.0, 10.0)

    cfg = StrategyConfig(risk_per_trade=1.0, max_position_pct=1.0)
    constraints = SymbolConstraints("BTCUSDC", 1.0, 0.0, 0.0, 0.0, 0.0)
    assert cli._build_order_notional(100.0, 100.0, cfg, "spot", 1.0, QuantityClient([0.5]), constraints=constraints) == (0.0, 0.0)

    constraints = SymbolConstraints("BTCUSDC", 0.0, 0.0, 0.0, 0.0, 10.0)
    assert cli._build_order_notional(100.0, 10.0, cfg, "spot", 1.0, QuantityClient([20.0, 0.0]), constraints=constraints) == (0.0, 0.0)

    order_client = _LiveClient()
    cli._paper_or_live_order(order_client, RuntimeConfig(market_type="spot"), StrategyConfig(leverage=50), side="BUY", size=0.1, dry_run=False)
    assert order_client.orders[-1]["leverage"] == 1.0
    assert "live order: BUY" in capsys.readouterr().out

    class ConnectClient:
        base_url = "mock://connect"

        def __init__(self, *, account=None, max_leverage=21) -> None:
            self.account = account
            self.max_leverage = max_leverage

        def get_exchange_time(self):
            return 123

        def ensure_btcusdc(self):
            return None

        def get_account(self):
            return self.account

        def get_max_leverage(self, _symbol):
            if isinstance(self.max_leverage, Exception):
                raise self.max_leverage
            return self.max_leverage

    save_runtime(RuntimeConfig(market_type="spot"))
    monkeypatch.setattr(cli, "_build_client", lambda _runtime: ConnectClient())
    assert cli.command_connect(argparse.Namespace()) == 0
    assert "server_time: 123" in capsys.readouterr().out

    save_runtime(RuntimeConfig(market_type="spot", api_key="k", api_secret="s"))
    monkeypatch.setattr(cli, "_build_client", lambda _runtime: ConnectClient(account="raw-account"))
    assert cli.command_connect(argparse.Namespace()) == 0
    assert "account:" in capsys.readouterr().out

    save_runtime(RuntimeConfig(market_type="futures", api_key="k", api_secret="s"))
    monkeypatch.setattr(cli, "_build_client", lambda _runtime: ConnectClient(account={}, max_leverage=33))
    assert cli.command_connect(argparse.Namespace()) == 0
    assert "max leverage on BTCUSDC: 33x" in capsys.readouterr().out

    monkeypatch.setattr(cli, "_build_client", lambda _runtime: ConnectClient(account={}, max_leverage=BinanceAPIError("no bracket")))
    assert cli.command_connect(argparse.Namespace()) == 0
    assert "unable to fetch leverage bracket" in capsys.readouterr().err

    save_runtime(RuntimeConfig(market_type="spot"))
    save_strategy(StrategyConfig(enabled_features=("rsi",)))
    assert cli.command_strategy(_strategy_args(leverage=200, enable_feature=["rsi"])) == 0
    strategy = cli.load_strategy()
    assert strategy.leverage == 200
    assert strategy.enabled_features == ("rsi",)

    save_runtime(RuntimeConfig(market_type="futures"))
    save_strategy(StrategyConfig())
    assert cli.command_strategy(_strategy_args(leverage=200)) == 0
    assert cli.load_strategy().leverage == 125


def test_cli_tui_actions_cover_cancel_invalid_and_success_paths(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    save_runtime(RuntimeConfig())
    save_strategy(StrategyConfig())

    assert asyncio.run(_action("Overview").run(_ScriptedUI())) == 0
    help_ui = _ScriptedUI()
    assert asyncio.run(_action("Help").run(help_ui)) == 0
    assert "Operator help" in help_ui.logs[0]

    async def bad_runtime(_ui, _current):
        raise ValueError("bad runtime")

    monkeypatch.setattr(cli, "_ui_edit_runtime", bad_runtime)
    assert asyncio.run(_action("Runtime settings").run(_ScriptedUI())) == 2
    assert "Runtime settings invalid" in capsys.readouterr().err

    class BadValidationClient:
        def ping(self):
            return {}

        def ensure_btcusdc(self):
            raise BinanceAPIError("bad symbol")

    async def validating_runtime(_ui, _current):
        return RuntimeConfig(api_key="k", api_secret="s", validate_account=True)

    monkeypatch.setattr(cli, "_ui_edit_runtime", validating_runtime)
    monkeypatch.setattr(cli, "_build_client", lambda _runtime: BadValidationClient())
    assert asyncio.run(_action("Runtime settings").run(_ScriptedUI())) == 2

    async def valid_runtime(_ui, _current):
        return RuntimeConfig(validate_account=False)

    monkeypatch.setattr(cli, "_ui_edit_runtime", valid_runtime)
    assert asyncio.run(_action("Runtime settings").run(_ScriptedUI())) == 0

    async def bad_strategy(_ui, _cfg):
        raise ValueError("bad strategy")

    monkeypatch.setattr(cli, "_ui_edit_strategy_args", bad_strategy)
    assert asyncio.run(_action("Strategy settings").run(_ScriptedUI())) == 2
    async def cancelled_strategy(_ui, _cfg):
        return argparse.Namespace(leverage=None, set_features=None)

    monkeypatch.setattr(cli, "_ui_edit_strategy_args", cancelled_strategy)
    assert asyncio.run(_action("Strategy settings").run(_ScriptedUI())) == 0
    captured: dict[str, object] = {}

    def capture(name: str):
        def _capture(args):
            captured[name] = args
            return 0

        return _capture

    monkeypatch.setattr(cli, "command_strategy", capture("strategy"))
    async def valid_strategy(_ui, _cfg):
        return argparse.Namespace(leverage=2.0, set_features="momentum_1")

    monkeypatch.setattr(cli, "_ui_edit_strategy_args", valid_strategy)
    assert asyncio.run(_action("Strategy settings").run(_ScriptedUI())) == 0

    monkeypatch.setattr(cli, "command_connect", lambda _args: 0)
    assert asyncio.run(_action("Connect").run(_ScriptedUI())) == 0
    monkeypatch.setattr(cli, "command_doctor", capture("doctor"))
    assert asyncio.run(_action("Readiness check").run(_ScriptedUI(forms=[None]))) == 0
    assert asyncio.run(_action("Readiness check").run(_ScriptedUI(forms=[{"input": "", "model": "", "online": "no"}]))) == 0
    assert captured["doctor"].online is False
    monkeypatch.setattr(cli, "_show_account_overview", lambda: 0)
    assert asyncio.run(_action("Account").run(_ScriptedUI())) == 0

    assert asyncio.run(_action("Fetch candles").run(_ScriptedUI(forms=[None]))) == 0
    assert asyncio.run(_action("Fetch candles").run(_ScriptedUI(forms=[{"limit": "0", "output": ""}]))) == 2
    monkeypatch.setattr(cli, "command_fetch", capture("fetch"))
    assert asyncio.run(_action("Fetch candles").run(_ScriptedUI(forms=[{"limit": "2", "output": ""}]))) == 0
    assert captured["fetch"].output == "data/historical_btcusdc.json"

    assert asyncio.run(_action("Train model").run(_ScriptedUI(forms=[None]))) == 0
    bad_train = {
        "input": "",
        "output": "",
        "preset": "custom",
        "epochs": "0",
        "seed": "7",
        "walk_forward": "no",
        "walk_forward_train": "300",
        "walk_forward_test": "60",
        "walk_forward_step": "30",
        "calibrate_threshold": "yes",
    }
    assert asyncio.run(_action("Train model").run(_ScriptedUI(forms=[bad_train]))) == 2
    good_train = {**bad_train, "epochs": "1"}
    monkeypatch.setattr(cli, "command_train", capture("train"))
    assert asyncio.run(_action("Train model").run(_ScriptedUI(forms=[good_train]))) == 0
    assert captured["train"].preset == "custom"

    assert asyncio.run(_action("Tune strategy").run(_ScriptedUI(forms=[None]))) == 0
    tune_base = {
        "input": "",
        "window_mode": "invalid",
        "lookback_days": "30",
        "from_date": "",
        "to_date": "",
        "save_best": "no",
        "min_risk": "0.002",
        "max_risk": "0.02",
        "steps": "5",
        "min_leverage": "1.0",
        "max_leverage": "20.0",
        "min_threshold": "0.52",
        "max_threshold": "0.88",
        "min_take": "0.01",
        "max_take": "0.06",
        "min_stop": "0.008",
        "max_stop": "0.04",
    }
    assert asyncio.run(_action("Tune strategy").run(_ScriptedUI(forms=[tune_base]))) == 2
    monkeypatch.setattr(cli, "command_tune", capture("tune"))
    assert asyncio.run(_action("Tune strategy").run(_ScriptedUI(forms=[{**tune_base, "window_mode": "lookback"}]))) == 0
    assert captured["tune"].lookback_days == 30
    captured.pop("tune")
    assert asyncio.run(_action("Tune strategy").run(_ScriptedUI(forms=[{**tune_base, "window_mode": "all"}]))) == 0
    assert captured["tune"].lookback_days is None
    captured.pop("tune")
    assert asyncio.run(_action("Tune strategy").run(_ScriptedUI(forms=[{**tune_base, "window_mode": "range", "from_date": "2025-01-01", "to_date": "2025-01-02"}]))) == 0
    assert captured["tune"].from_date == "2025-01-01"

    assert asyncio.run(_action("Backtest").run(_ScriptedUI(forms=[None]))) == 0
    assert asyncio.run(_action("Backtest").run(_ScriptedUI(forms=[{"input": "", "model": "", "start_cash": "0"}]))) == 2
    monkeypatch.setattr(cli, "command_backtest", capture("backtest"))
    assert asyncio.run(_action("Backtest").run(_ScriptedUI(forms=[{"input": "", "model": "", "start_cash": "100"}]))) == 0

    assert asyncio.run(_action("Evaluate").run(_ScriptedUI(forms=[None]))) == 0
    monkeypatch.setattr(cli, "command_evaluate", capture("evaluate"))
    assert asyncio.run(_action("Evaluate").run(_ScriptedUI(forms=[{"input": "", "model": "", "threshold": "0.7", "calibrate_threshold": "yes"}]))) == 0
    assert captured["evaluate"].threshold == 0.7

    assert asyncio.run(_action("Offline pipeline").run(_ScriptedUI(forms=[None]))) == 0
    pipeline = {
        "historical": "",
        "model": "",
        "limit": "0",
        "preset": "balanced",
        "epochs": "120",
        "seed": "7",
        "walk_forward": "yes",
        "walk_forward_train": "300",
        "walk_forward_test": "60",
        "walk_forward_step": "30",
        "calibrate_threshold": "yes",
        "start_cash": "1000",
    }
    assert asyncio.run(_action("Offline pipeline").run(_ScriptedUI(forms=[pipeline]))) == 2
    pipeline["limit"] = "1"
    monkeypatch.setattr(cli, "command_fetch", lambda args: 2)
    assert asyncio.run(_action("Offline pipeline").run(_ScriptedUI(forms=[pipeline]))) == 2
    monkeypatch.setattr(cli, "command_fetch", lambda args: 0)
    monkeypatch.setattr(cli, "command_train", lambda args: 0)
    monkeypatch.setattr(cli, "command_backtest", lambda args: 0)
    monkeypatch.setattr(cli, "command_evaluate", lambda args: 0)
    assert asyncio.run(_action("Offline pipeline").run(_ScriptedUI(forms=[pipeline]))) == 0

    assert asyncio.run(_action("Paper loop").run(_ScriptedUI(forms=[None]))) == 0
    paper = {"steps": "0", "sleep": "0", "retrain_interval": "0", "retrain_window": "300", "retrain_min_rows": "240"}
    assert asyncio.run(_action("Paper loop").run(_ScriptedUI(forms=[paper]))) == 2
    monkeypatch.setattr(cli, "command_live", capture("live"))
    assert asyncio.run(_action("Paper loop").run(_ScriptedUI(forms=[{**paper, "steps": "1"}]))) == 0

    assert asyncio.run(_action("Testnet loop").run(_ScriptedUI(confirms=[False]))) == 0
    assert asyncio.run(_action("Testnet loop").run(_ScriptedUI(confirms=[True], forms=[None]))) == 0
    live_form = {"steps": "0", "sleep": "0", "retrain_interval": "0", "retrain_window": "300", "retrain_min_rows": "240"}
    assert asyncio.run(_action("Testnet loop").run(_ScriptedUI(confirms=[True], forms=[live_form]))) == 2
    assert asyncio.run(_action("Testnet loop").run(_ScriptedUI(confirms=[True], forms=[{**live_form, "steps": "1"}]))) == 0

    assert asyncio.run(_action("Spot roundtrip").run(_ScriptedUI(confirms=[False]))) == 0
    assert asyncio.run(_action("Spot roundtrip").run(_ScriptedUI(confirms=[True], forms=[None]))) == 0
    assert asyncio.run(_action("Spot roundtrip").run(_ScriptedUI(confirms=[True], forms=[{"quantity": "0"}]))) == 2
    save_runtime(RuntimeConfig(market_type="futures", api_key="k", api_secret="s"))
    assert asyncio.run(_action("Spot roundtrip").run(_ScriptedUI(confirms=[True], forms=[{"quantity": "0.1"}]))) == 2
    save_runtime(RuntimeConfig(market_type="spot", testnet=False, api_key="k", api_secret="s"))
    assert asyncio.run(_action("Spot roundtrip").run(_ScriptedUI(confirms=[True], forms=[{"quantity": "0.1"}]))) == 2
    save_runtime(RuntimeConfig(market_type="spot", testnet=True, api_key="", api_secret=""))
    assert asyncio.run(_action("Spot roundtrip").run(_ScriptedUI(confirms=[True], forms=[{"quantity": "0.1"}]))) == 2

    class OrderClient:
        def __init__(self, *, fail: bool = False) -> None:
            self.fail = fail

        def place_order(self, _symbol, _side, _quantity, *, dry_run):
            if self.fail:
                raise BinanceAPIError("order failed")
            return {"status": "FILLED", "orderId": 1, "executedQty": "0.1"}

    save_runtime(RuntimeConfig(market_type="spot", testnet=True, api_key="k", api_secret="s"))
    monkeypatch.setattr(cli, "_build_client", lambda _runtime: OrderClient(fail=True))
    assert asyncio.run(_action("Spot roundtrip").run(_ScriptedUI(confirms=[True], forms=[{"quantity": "0.1"}]))) == 2
    monkeypatch.setattr(cli, "_build_client", lambda _runtime: OrderClient())
    assert asyncio.run(_action("Spot roundtrip").run(_ScriptedUI(confirms=[True], forms=[{"quantity": "0.1"}]))) == 0

    monkeypatch.setattr(cli, "_show_recent_artifacts", lambda: 0)
    assert asyncio.run(_action("Recent artifacts").run(_ScriptedUI())) == 0


def test_cli_strategy_tune_evaluate_and_live_remaining_edges(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    save_runtime(RuntimeConfig(market_type="futures", api_key="k", api_secret="s"))
    save_strategy(StrategyConfig(enabled_features=("momentum_1",)))

    class FailingLeverageClient:
        def get_max_leverage(self, _symbol):
            raise BinanceAPIError("bracket unavailable")

    monkeypatch.setattr(cli, "_build_client", lambda _runtime: FailingLeverageClient())
    args = argparse.Namespace(
        leverage=200,
        risk=None,
        max_position=None,
        stop=None,
        take=None,
        cooldown=None,
        max_open=None,
        max_trades_per_day=None,
        signal_threshold=None,
        max_drawdown=None,
        taker_fee_bps=None,
        slippage_bps=None,
        label_threshold=None,
        model_lookback=None,
        training_epochs=None,
        confidence_beta=None,
        feature_window_short=None,
        feature_window_long=None,
        set_features=None,
        enable_feature=["rsi"],
        disable_feature=["momentum_1"],
    )
    assert cli.command_strategy(args) == 0
    assert "rsi" in capsys.readouterr().out

    args.disable_feature = ["rsi"]
    args.enable_feature = None
    assert cli.command_strategy(args) == 2
    assert "Invalid feature selection" in capsys.readouterr().err

    save_strategy(StrategyConfig())
    bad_window = argparse.Namespace(
        input=str(tmp_path / "missing.json"),
        save_best=False,
        min_risk=0.01,
        max_risk=0.01,
        steps=1,
        min_leverage=1,
        max_leverage=1,
        min_threshold=0.5,
        max_threshold=0.5,
        min_take=0.01,
        max_take=0.01,
        min_stop=0.01,
        max_stop=0.01,
        lookback_days=-1,
        from_date=None,
        to_date=None,
    )
    (tmp_path / "missing.json").write_text("[]", encoding="utf-8")
    assert cli.command_tune(bad_window) == 2

    monkeypatch.setattr(cli, "_load_rows_for_command", lambda *_args, **_kwargs: [object()])
    monkeypatch.setattr(cli, "_build_model_rows", lambda _candles, _cfg: [])
    assert cli.command_evaluate(argparse.Namespace(input="x", model="missing", threshold=None, calibrate_threshold=False)) == 2
    assert "No rows available" in capsys.readouterr().out

    row = ModelRow(1, 1.0, (0.0,), 1)
    monkeypatch.setattr(cli, "_load_rows_for_command", lambda *_args, **_kwargs: [object()])
    monkeypatch.setattr(cli, "_build_model_rows", lambda _candles, _cfg: [row])
    assert cli.command_evaluate(argparse.Namespace(input="x", model=str(tmp_path / "missing-model.json"), threshold=None, calibrate_threshold=False)) == 2
    assert "Model file not found" in capsys.readouterr().out

    model_path = tmp_path / "bad-model.json"
    model_path.write_text("{bad", encoding="utf-8")
    assert cli.command_evaluate(argparse.Namespace(input="x", model=str(model_path), threshold=0.4, calibrate_threshold=False)) == 2
    assert "Model load failed" in capsys.readouterr().err

    good_model = tmp_path / "model.json"
    good_model.write_text(
        json.dumps(
            {
                "weights": [0.0],
                "bias": 10.0,
                "feature_dim": 1,
                "epochs": 1,
                "feature_version": "v1",
                "feature_means": [0.0],
                "feature_stds": [1.0],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(cli, "_load_rows_for_command", lambda *_args, **_kwargs: [row])
    assert cli.command_evaluate(argparse.Namespace(input="x", model=str(good_model), threshold=0.4, calibrate_threshold=False)) == 0
    assert "train_accuracy:" in capsys.readouterr().out

    save_runtime(RuntimeConfig(market_type="spot", testnet=True, dry_run=True))
    save_strategy(StrategyConfig(max_open_positions=0))

    class NoRowClient:
        base_url = "mock"

        def get_symbol_constraints(self, _symbol):
            return None

        def get_klines(self, *_args, **_kwargs):
            return []

        def place_order(self, *_args, **_kwargs):
            return {}

    monkeypatch.setattr(cli, "_build_client", lambda _runtime: NoRowClient())
    monkeypatch.setattr(cli.time, "sleep", lambda _seconds: None)
    assert cli.command_live(argparse.Namespace(steps=1, sleep=0, paper=True, live=False, leverage=None, retrain_interval=-1, retrain_window=0, retrain_min_rows=0)) == 0

    class MarketErrorClient(NoRowClient):
        def get_klines(self, *_args, **_kwargs):
            raise BinanceAPIError("down")

    monkeypatch.setattr(cli, "_build_client", lambda _runtime: MarketErrorClient())
    assert cli.command_live(argparse.Namespace(steps=1, sleep=0, paper=True, live=False, leverage=None, retrain_interval=0, retrain_window=1, retrain_min_rows=1)) == 2

    class EntryClient(NoRowClient):
        def get_symbol_constraints(self, _symbol):
            return SymbolConstraints("BTCUSDC", 0.0, 0.0, 0.0, 0.0, 0.0)

        def normalize_quantity(self, _symbol, quantity):
            return quantity, self.get_symbol_constraints(_symbol)

        def get_klines(self, *_args, **_kwargs):
            return _base_candles(80)

    save_strategy(StrategyConfig(max_open_positions=0, enabled_features=("momentum_1",)))
    monkeypatch.setattr(cli, "_build_client", lambda _runtime: EntryClient())
    monkeypatch.setattr(cli, "train", lambda *_args, **_kwargs: _flat_model(0.99))
    assert cli.command_live(argparse.Namespace(steps=1, sleep=0, paper=True, live=False, leverage=None, retrain_interval=0, retrain_window=1, retrain_min_rows=1)) == 0
    assert "max open positions reached" in capsys.readouterr().out


def test_cli_tune_and_evaluate_remaining_paths(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    save_runtime(RuntimeConfig())
    save_strategy(StrategyConfig(enabled_features=("momentum_1",)))

    monkeypatch.setattr(cli, "_load_rows_for_command", lambda *_args, **_kwargs: None)
    assert cli.command_tune(_tune_args()) == 2

    rows = [ModelRow(index, 100.0 + index, (float(index),), index % 2) for index in range(45)]
    monkeypatch.setattr(cli, "_load_rows_for_command", lambda *_args, **_kwargs: [object()])
    monkeypatch.setattr(cli, "_build_model_rows", lambda _candles, _cfg: rows)
    monkeypatch.setattr(cli, "train", lambda *_args, **_kwargs: _flat_model(0.99))
    assert cli.command_tune(_tune_args(steps=0)) == 2
    assert "No valid candidates evaluated." in capsys.readouterr().out

    stopped_result = SimpleNamespace(
        realized_pnl=5.0,
        total_fees=1.0,
        max_drawdown=0.1,
        closed_trades=1,
        stopped_by_drawdown=True,
    )
    monkeypatch.setattr(cli, "run_backtest", lambda *_args, **_kwargs: stopped_result)
    assert cli.command_tune(_tune_args(steps=1)) == 0
    assert "using best fallback" in capsys.readouterr().out

    scores = iter([10.0, 1.0])
    monkeypatch.setattr(cli, "_tune_score", lambda *_args, **_kwargs: next(scores, 0.0))
    assert cli.command_tune(_tune_args(steps=2)) == 0

    model_path = tmp_path / "model.json"
    model_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(cli, "_load_runtime_model", lambda *_args, **_kwargs: _flat_model(0.99))
    assert cli.command_evaluate(
        argparse.Namespace(input="x", model=str(model_path), threshold=None, calibrate_threshold=True)
    ) == 0
    output = capsys.readouterr().out
    assert "threshold:" in output
    assert "test_accuracy:" in output

    monkeypatch.setattr(cli, "_build_model_rows", lambda _candles, _cfg: [ModelRow(1, 100.0, (0.0,), 1)])
    assert cli.command_evaluate(
        argparse.Namespace(input="x", model=str(model_path), threshold=None, calibrate_threshold=False)
    ) == 0
    assert "test_accuracy:" in capsys.readouterr().out


def test_cli_train_walk_forward_without_fold_scores(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    save_runtime(RuntimeConfig())
    save_strategy(StrategyConfig(enabled_features=("momentum_1",)))
    rows = [ModelRow(index, 100.0 + index, (float(index),), index % 2) for index in range(8)]
    monkeypatch.setattr(cli, "_load_rows_for_command", lambda *_args, **_kwargs: [object()])
    monkeypatch.setattr(cli, "_build_model_rows", lambda _candles, _cfg: rows)
    monkeypatch.setattr(
        cli,
        "walk_forward_report",
        lambda *_args, **_kwargs: {
            "folds": 1,
            "average_score": 0.5,
            "train_window": 3,
            "test_window": 2,
            "step": 1,
            "scores": [],
        },
    )
    monkeypatch.setattr(cli, "train", lambda *_args, **_kwargs: _flat_model(0.99))
    assert cli.command_train(
        argparse.Namespace(
            input="x",
            output=str(tmp_path / "model.json"),
            preset="custom",
            epochs=1,
            seed=7,
            walk_forward=True,
            walk_forward_train=3,
            walk_forward_test=2,
            walk_forward_step=1,
            calibrate_threshold=False,
        )
    ) == 0
    assert "walk-forward fold scores" not in capsys.readouterr().out


def test_cli_live_guards_leverage_clamps_and_no_rows(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(cli, "_build_client", lambda _runtime: _LiveClient())

    save_runtime(RuntimeConfig(market_type="spot", testnet=True, dry_run=True))
    save_strategy(StrategyConfig())
    assert cli.command_live(_live_args(paper=False, live=True)) == 2
    assert "Live mode needs API key" in capsys.readouterr().out

    client = _LiveClient(set_response={"leverage": "200"})
    save_runtime(RuntimeConfig(market_type="futures", testnet=True, dry_run=False, api_key="k", api_secret="s"))
    save_strategy(StrategyConfig(max_trades_per_day=0))
    monkeypatch.setattr(cli, "_build_client", lambda _runtime: client)
    monkeypatch.setattr(cli, "_resolve_futures_leverage", lambda _runtime, _cfg: 200.0)
    assert cli.command_live(_live_args(steps=0, paper=False, live=True)) == 0
    assert "effective leverage: 125.0x" in capsys.readouterr().out

    monkeypatch.setattr(cli, "_build_client", lambda _runtime: _LiveClient(set_response={}))
    monkeypatch.setattr(cli, "_resolve_futures_leverage", lambda _runtime, _cfg: 0.2)
    assert cli.command_live(_live_args(steps=0, paper=False, live=True)) == 0
    assert "effective leverage: 1.0x" in capsys.readouterr().out

    save_runtime(RuntimeConfig(market_type="spot", testnet=True, dry_run=True))
    save_strategy(StrategyConfig(max_trades_per_day=0))
    (tmp_path / "data").mkdir(exist_ok=True)
    (tmp_path / "data" / "model.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(cli, "_resolve_futures_leverage", lambda _runtime, cfg: cli._effective_leverage(cfg, _runtime.market_type))
    monkeypatch.setattr(cli, "_load_runtime_model", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("corrupt")))
    monkeypatch.setattr(cli, "_build_model_rows", lambda _candles, _cfg: [])
    assert cli.command_live(_live_args(steps=1, paper=True, live=False, retrain_interval=-1, retrain_window=0, retrain_min_rows=0)) == 0
    assert "not enough historical data" in capsys.readouterr().out


def test_cli_live_spot_close_cooldown_trade_cap_and_signal_artifact(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli.time, "sleep", lambda _seconds: None)
    save_runtime(RuntimeConfig(market_type="spot", testnet=True, dry_run=True))
    save_strategy(
        StrategyConfig(
            risk_per_trade=0.1,
            max_position_pct=0.1,
            max_open_positions=1,
            max_trades_per_day=1,
            cooldown_minutes=1,
            take_profit_pct=0.01,
            stop_loss_pct=0.99,
            taker_fee_bps=0.0,
            slippage_bps=0.0,
            enabled_features=("momentum_1",),
        )
    )
    client = _LiveClient()
    model = _SequenceModel([0.99, 0.99, 0.99, 0.99])
    row_sequences = [
        [ModelRow(1, 100.0, (0.0,), 1)],
        [ModelRow(2, 110.0, (0.0,), 1)],
        [ModelRow(3, 111.0, (0.0,), 1)],
        [ModelRow(4, 112.0, (0.0,), 1)],
    ]
    monkeypatch.setattr(cli, "_build_client", lambda _runtime: client)
    monkeypatch.setattr(cli, "_build_model_rows", lambda _candles, _cfg: row_sequences.pop(0))
    monkeypatch.setattr(cli, "_build_live_model", lambda _rows, **kwargs: kwargs.get("model") or model)

    assert cli.command_live(_live_args(steps=4)) == 0
    output = capsys.readouterr().out
    assert "result: win" in output
    assert "trade cap reached" in output
    artifact = next((tmp_path / "data").glob("live_*.json"))
    events = json.loads(artifact.read_text(encoding="utf-8"))["events"]
    assert any(event["status"] == "signal_no_entry" for event in events)


def test_cli_live_entry_rejection_and_drawdown_paths(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(cli, "_build_client", lambda _runtime: _LiveClient())
    monkeypatch.setattr(cli, "_build_live_model", lambda _rows, **kwargs: kwargs.get("model") or _SequenceModel([0.99]))

    save_runtime(RuntimeConfig(market_type="futures", testnet=True, dry_run=True))
    save_strategy(StrategyConfig(slippage_bps=20_000, enabled_features=("momentum_1",)))
    monkeypatch.setattr(cli, "_build_model_rows", lambda _candles, _cfg: [ModelRow(1, 100.0, (0.0,), 0)])
    monkeypatch.setattr(cli, "_build_live_model", lambda _rows, **kwargs: kwargs.get("model") or _SequenceModel([0.01]))
    assert cli.command_live(_live_args(steps=1)) == 0

    save_runtime(RuntimeConfig(market_type="spot", testnet=True, dry_run=True))
    save_strategy(
        StrategyConfig(
            risk_per_trade=0.5,
            max_position_pct=0.5,
            taker_fee_bps=6000,
            slippage_bps=10_000,
            enabled_features=("momentum_1",),
        )
    )
    monkeypatch.setattr(cli, "_build_model_rows", lambda _candles, _cfg: [ModelRow(2, 100.0, (0.0,), 1)])
    monkeypatch.setattr(cli, "_build_live_model", lambda _rows, **kwargs: kwargs.get("model") or _SequenceModel([0.99]))
    assert cli.command_live(_live_args(steps=1)) == 0
    assert "insufficient cash after fill adjustment" in capsys.readouterr().out

    save_runtime(RuntimeConfig(market_type="futures", testnet=True, dry_run=True))
    save_strategy(
        StrategyConfig(
            leverage=1.0,
            risk_per_trade=0.5,
            max_position_pct=0.5,
            take_profit_pct=0.99,
            stop_loss_pct=0.99,
            cooldown_minutes=3,
            max_drawdown_limit=0.9,
            taker_fee_bps=0.0,
            slippage_bps=0.0,
            enabled_features=("momentum_1",),
        )
    )
    hold_model = _SequenceModel([0.99, 0.5, 0.5])
    hold_rows = [
        [ModelRow(1, 100.0, (0.0,), 1)],
        [ModelRow(2, 120.0, (0.0,), 1)],
        [ModelRow(3, 110.0, (0.0,), 1)],
    ]
    monkeypatch.setattr(cli, "_build_model_rows", lambda _candles, _cfg: hold_rows.pop(0))
    monkeypatch.setattr(cli, "_build_live_model", lambda _rows, **kwargs: kwargs.get("model") or hold_model)
    assert cli.command_live(_live_args(steps=3)) == 0
    assert "max_drawdown observed" in capsys.readouterr().out

    save_strategy(
        StrategyConfig(
            leverage=1.0,
            risk_per_trade=0.5,
            max_position_pct=0.5,
            take_profit_pct=0.99,
            stop_loss_pct=0.99,
            max_drawdown_limit=0.1,
            taker_fee_bps=0.0,
            slippage_bps=0.0,
            enabled_features=("momentum_1",),
        )
    )
    emergency_model = _SequenceModel([0.99, 0.5])
    emergency_rows = [
        [ModelRow(1, 100.0, (0.0,), 1)],
        [ModelRow(2, 50.0, (0.0,), 1)],
    ]
    monkeypatch.setattr(cli, "_build_model_rows", lambda _candles, _cfg: emergency_rows.pop(0))
    monkeypatch.setattr(cli, "_build_live_model", lambda _rows, **kwargs: kwargs.get("model") or emergency_model)
    assert cli.command_live(_live_args(steps=2)) == 0
    assert "drawdown limit reached" in capsys.readouterr().out

    save_runtime(RuntimeConfig(market_type="spot", testnet=True, dry_run=True))
    save_strategy(
        StrategyConfig(
            risk_per_trade=0.5,
            max_position_pct=0.5,
            take_profit_pct=0.99,
            stop_loss_pct=0.01,
            max_drawdown_limit=0.01,
            taker_fee_bps=0.0,
            slippage_bps=0.0,
            enabled_features=("momentum_1",),
        )
    )
    spot_loss_model = _SequenceModel([0.99, 0.99])
    spot_loss_rows = [
        [ModelRow(1, 100.0, (0.0,), 1)],
        [ModelRow(2, 50.0, (0.0,), 1)],
    ]
    monkeypatch.setattr(cli, "_build_model_rows", lambda _candles, _cfg: spot_loss_rows.pop(0))
    monkeypatch.setattr(cli, "_build_live_model", lambda _rows, **kwargs: kwargs.get("model") or spot_loss_model)
    assert cli.command_live(_live_args(steps=2)) == 0
    output = capsys.readouterr().out
    assert "drawdown limit reached" in output
    assert "emergency close from drawdown" not in output


def test_cli_main_default_and_module_exit(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(cli, "command_menu", lambda _args: 4)
    monkeypatch.setattr(sys, "argv", ["simple-ai-trading"])
    assert cli.main(None) == 4

    save_runtime(RuntimeConfig())
    save_strategy(StrategyConfig())
    monkeypatch.setattr(sys, "argv", ["simple-ai-trading", "status"])
    with pytest.warns(RuntimeWarning, match="found in sys.modules"):
        with pytest.raises(SystemExit) as exc_info:
            runpy.run_module("simple_ai_bitcoin_trading_binance.cli", run_name="__main__")
    assert exc_info.value.code == 0
