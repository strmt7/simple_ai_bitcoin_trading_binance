from __future__ import annotations

import json
import argparse
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from simple_ai_bitcoin_trading_binance import cli
from simple_ai_bitcoin_trading_binance.api import BinanceAPIError, Candle
from simple_ai_bitcoin_trading_binance.config import RuntimeConfig, load_strategy, save_runtime, save_strategy
from simple_ai_bitcoin_trading_binance.model import TrainedModel
from simple_ai_bitcoin_trading_binance.types import StrategyConfig


class _FakeClient:
    def __init__(self) -> None:
        self.base_url = "https://api.testnet.binance.vision"
        self.orders = []

    def ping(self):
        return {"ok": True}

    def ensure_btcusdc(self):
        return {"symbol": "BTCUSDC"}

    def get_exchange_time(self):
        return {"serverTime": 123}

    def get_account(self):
        return {
            "updateTime": 123,
            "canTrade": True,
            "accountType": "MARGIN",
            "positions": [],
            "assets": [],
        }

    def get_max_leverage(self, symbol: str) -> int:
        return 10

    def get_symbol_constraints(self, symbol: str):
        return SimpleNamespace()  # placeholder unused in tests

    def get_klines(self, symbol: str, interval: str, limit: int = 500, start_time=None, end_time=None):
        return []

    def place_order(self, symbol: str, side: str, size: float, *, dry_run: bool, leverage: float = 1.0):
        self.orders.append((symbol, side, size, dry_run, leverage))
        return {"symbol": symbol, "side": side, "size": size, "dry_run": dry_run, "leverage": leverage}


def _simple_candles(n: int = 320) -> list[Candle]:
    out: list[Candle] = []
    for i in range(n):
        base = 100.0 + i
        out.append(
            Candle(
                open_time=i * 60_000,
                open=base,
                high=base * 1.001,
                low=base * 0.999,
                close=base,
                volume=1.0,
                close_time=(i + 1) * 60_000,
            )
        )
    return out


def test_parse_args_and_main_dispatch(monkeypatch) -> None:
    args = cli._parse_args(["status"])
    assert callable(args.func)

    marker = []

    def fake_status(ns: argparse.Namespace) -> int:
        marker.append("status")
        return 9

    monkeypatch.setattr(cli, "command_status", fake_status)
    assert cli.main(["status"]) == 9
    assert marker == ["status"]


def test_clamp_and_direction_helpers() -> None:
    assert cli._clamp(1.2, 0.0, 1.0) == 1.0
    assert cli._clamp(-0.2, 0.0, 1.0) == 0.0
    cfg = StrategyConfig(signal_threshold=0.55)
    assert cli._score_to_direction(0.60, cfg, "spot") == 1
    assert cli._score_to_direction(0.40, cfg, "spot") == 0
    assert cli._score_to_direction(0.56, cfg, "futures") == 1
    assert cli._score_to_direction(0.40, cfg, "futures") == -1
    assert cli._score_to_direction(0.50, cfg, "futures") == 0
    assert cli._score_to_direction(0.10, cfg, "futures") == -1


def test_resolve_futures_leverage(monkeypatch) -> None:
    runtime = RuntimeConfig(market_type="futures", api_key="k", api_secret="s", symbol="BTCUSDC")
    cfg = StrategyConfig(leverage=50.0)

    fake = _FakeClient()

    def build_client(_runtime):
        return fake

    monkeypatch.setattr(cli, "_build_client", build_client)
    assert cli._resolve_futures_leverage(runtime, cfg) == 10.0

    runtime_no_key = RuntimeConfig(market_type="futures", api_key="", api_secret="")
    assert cli._resolve_futures_leverage(runtime_no_key, cfg) == 50.0

    runtime_spot = RuntimeConfig(market_type="spot")
    assert cli._resolve_futures_leverage(runtime_spot, cfg) == 1.0


def test_build_order_notional_paths() -> None:
    cfg = StrategyConfig(risk_per_trade=0.1, max_position_pct=0.4, leverage=3.0)
    client = SimpleNamespace(
        normalize_quantity=lambda symbol, qty: (
            0.0 if qty < 0.001 else qty,
            type(
                "Constraint",
                (),
                {"symbol": "BTCUSDC", "min_qty": 0.001, "step_size": 0.001, "min_notional": 5.0, "max_notional": 300.0},
            )(),
        ),
    )
    notional, qty = cli._build_order_notional(1000.0, 10_000.0, cfg, "futures", 3.0, client)
    assert math.isclose(notional, 300.0)
    assert math.isclose(qty, 0.03)

    bad_constraints = type(
        "C",
        (),
        {"symbol": "BTCUSDC", "min_qty": 10.0, "step_size": 1.0, "min_notional": 5000.0, "max_notional": 0.0},
    )()

    client_too_small = SimpleNamespace(
        normalize_quantity=lambda symbol, qty: (0.0 if qty < bad_constraints.min_qty else qty, bad_constraints)
    )
    assert cli._build_order_notional(
        1000.0,
        10_000.0,
        cfg,
        "futures",
        3.0,
        client_too_small,
        constraints=bad_constraints,
    ) == (0.0, 0.0)

    assert cli._build_order_notional(0.0, 10000.0, cfg, "futures", 3.0, client) == (0.0, 0.0)
    assert cli._build_order_notional(1000.0, -1.0, cfg, "futures", 3.0, client) == (0.0, 0.0)


def test_paper_order_is_logged(tmp_path, monkeypatch) -> None:
    client = _FakeClient()
    cfg = StrategyConfig()

    monkeypatch.setenv("HOME", str(tmp_path))
    save_runtime(RuntimeConfig())
    args = SimpleNamespace()

    cli._paper_or_live_order(client, RuntimeConfig(), cfg, side="BUY", size=0.1, dry_run=True, leverage=3.0)
    cli._paper_or_live_order(client, RuntimeConfig(), cfg, side="SELL", size=0.2, dry_run=False, leverage=2.0)
    assert client.orders[0][0] == "BTCUSDC"


def test_command_status_prints_masked_secret(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    save_runtime(RuntimeConfig(api_secret="super-secret"))
    save_strategy(StrategyConfig())
    assert cli.command_status(argparse.Namespace()) == 0
    output = capsys.readouterr().out
    assert "***" in output


def test_command_connect_spot_and_futures(tmp_path, monkeypatch, capsys) -> None:
    fake = _FakeClient()
    monkeypatch.setenv("HOME", str(tmp_path))
    runtime = RuntimeConfig(api_key="k", api_secret="s", market_type="futures", symbol="BTCUSDC")
    save_runtime(runtime)
    monkeypatch.setattr(cli, "_build_client", lambda _runtime: fake)
    assert cli.command_connect(argparse.Namespace()) == 0
    text = capsys.readouterr().out
    assert "exchange: connected" in text
    assert "max leverage on BTCUSDC: 10x" in text


def test_command_fetch_handles_binar_errors(tmp_path, monkeypatch) -> None:
    from simple_ai_bitcoin_trading_binance.api import BinanceAPIError

    class _ErrorClient(_FakeClient):
        def ensure_btcusdc(self):
            raise BinanceAPIError("bad")

    runtime = RuntimeConfig()
    output = tmp_path / "candles.json"
    monkeypatch.setenv("HOME", str(tmp_path))
    save_runtime(runtime)
    monkeypatch.setattr(cli, "_build_client", lambda _runtime: _ErrorClient())
    assert cli.command_fetch(argparse.Namespace(symbol=None, interval=None, limit=10, output=str(output))) == 2


def test_command_train_workflow(tmp_path, monkeypatch) -> None:
    save_runtime(RuntimeConfig())
    save_strategy(StrategyConfig())

    candles = [
        {
            "open_time": i * 60_000,
            "open": 100.0 + i,
            "high": 101.0 + i,
            "low": 99.0 + i,
            "close": 100.0 + i,
            "volume": 1.0,
            "close_time": (i + 1) * 60_000,
        }
        for i in range(320)
    ]
    data_file = tmp_path / "history.json"
    model_file = tmp_path / "model.json"
    data_file.write_text(json.dumps(candles), encoding="utf-8")

    monkeypatch.setenv("HOME", str(tmp_path))
    assert cli.command_train(
        argparse.Namespace(
            input=str(data_file),
            output=str(model_file),
            epochs=12,
            walk_forward=False,
            walk_forward_train=10,
            walk_forward_test=5,
            walk_forward_step=1,
            calibrate_threshold=False,
        )
    ) == 0


def test_command_evaluate_runs_with_model_file(tmp_path, monkeypatch) -> None:
    save_runtime(RuntimeConfig())
    save_strategy(StrategyConfig())

    candles = [
        {
            "open_time": i * 60_000,
            "open": 100.0 + i,
            "high": 101.0 + i,
            "low": 99.0 + i,
            "close": 100.0 + i,
            "volume": 1.0,
            "close_time": (i + 1) * 60_000,
        }
        for i in range(240)
    ]
    data_file = tmp_path / "history.json"
    model_file = tmp_path / "model.json"
    data_file.write_text(json.dumps(candles), encoding="utf-8")
    model_file.write_text(
        json.dumps(
            {
                "weights": [0.1] + [0.0] * 12,
                "bias": 0.01,
                "feature_dim": 13,
                "epochs": 5,
                "feature_means": [1.0] * 13,
                "feature_stds": [1.0] * 13,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("HOME", str(tmp_path))
    assert cli.command_evaluate(
        argparse.Namespace(
            input=str(data_file),
            model=str(model_file),
            threshold=None,
            calibrate_threshold=False,
        )
    ) == 0


def test_command_live_paper_path_runs_a_tick(tmp_path, monkeypatch) -> None:
    class _LiveClient:
        def __init__(self):
            self.calls = 0

        def ensure_btcusdc(self):
            return {"symbol": "BTCUSDC"}

        def get_klines(self, symbol: str, interval: str, limit: int = 500, start_time=None, end_time=None):
            return _simple_candles(limit)

        def set_leverage(self, symbol: str, leverage: int):
            return {"leverage": leverage}

        def place_order(self, symbol: str, side: str, size: float, *, dry_run: bool, leverage: float = 1.0):
            self.calls += 1
            return {"side": side, "symbol": symbol, "size": size, "dry_run": dry_run}

        def get_symbol_constraints(self, symbol: str):
            return SimpleNamespace(
                symbol=symbol,
                min_qty=0.0001,
                max_qty=100.0,
                step_size=0.0001,
                min_notional=5.0,
                max_notional=0.0,
            )

        def normalize_quantity(self, symbol: str, quantity: float):
            constraints = self.get_symbol_constraints(symbol)
            normalized = max(constraints.min_qty, round(quantity, 4))
            return normalized, constraints

    monkeypatch.setenv("HOME", str(tmp_path))
    save_runtime(RuntimeConfig(testnet=True, dry_run=True, market_type="spot"))
    save_strategy(StrategyConfig(risk_per_trade=0.001, max_position_pct=0.2))

    class _AlwaysLongModel:
        def predict_proba(self, _features: tuple[float, ...]) -> float:
            return 0.95

        def predict(self, _features: tuple[float, ...], threshold: float) -> int:
            return int(0.95 >= threshold)

    monkeypatch.setattr(cli, "train", lambda *_a, **_k: _AlwaysLongModel())
    monkeypatch.setattr(cli, "_build_client", lambda _runtime: _LiveClient())
    monkeypatch.setattr(cli.time, "sleep", lambda *_args: None)
    assert cli.command_live(argparse.Namespace(steps=1, sleep=5, paper=False)) == 0


def test_command_tune_saves_candidate(monkeypatch, tmp_path) -> None:
    save_runtime(RuntimeConfig())
    save_strategy(StrategyConfig(training_epochs=80))

    candles = [
        {
            "open_time": i * 60_000,
            "open": 100.0 + i,
            "high": 101.0 + i,
            "low": 99.0 + i,
            "close": 100.0 + i,
            "volume": 1.0,
            "close_time": (i + 1) * 60_000,
        }
        for i in range(240)
    ]

    data = tmp_path / "history.json"
    data.write_text(json.dumps(candles), encoding="utf-8")

    class _Model:
        def __init__(self, score: float) -> None:
            self.score = score

        def predict_proba(self, features: tuple[float, ...]) -> float:  # pragma: no cover
            return self.score

    monkeypatch.setattr(
        cli,
        "run_backtest",
        lambda rows, model, cfg, **_kwargs: SimpleNamespace(realized_pnl=1.0, total_fees=0.0),
    )
    monkeypatch.setattr(cli, "train", lambda *_a, **_k: _Model(0.8))
    monkeypatch.setattr(cli.time, "sleep", lambda *_args: None)
    monkeypatch.setenv("HOME", str(tmp_path))

    args = argparse.Namespace(
        input=str(data),
        save_best=True,
        min_risk=0.002,
        max_risk=0.003,
        steps=2,
        min_leverage=1.0,
        max_leverage=2.0,
        min_threshold=0.55,
        max_threshold=0.65,
        min_take=0.01,
        max_take=0.02,
        min_stop=0.01,
        max_stop=0.02,
    )
    assert cli.command_tune(args) == 0


def test_loaders_handle_invalid_inputs(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="Expected candle list"):
        cli._load_json_candles(str(bad))

    mixed = tmp_path / "mixed.json"
    mixed.write_text(
        json.dumps(
            [
                {
                    "open_time": 0,
                    "open": 100.0,
                    "high": 101.0,
                    "low": 99.0,
                    "close": 100.0,
                    "volume": 1.0,
                    "close_time": 60_000,
                },
                "bad-entry",
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    rows = cli._rows_from_json(str(mixed))
    assert len(rows) == 1


def test_resolve_symbol_constraints_error_returns_none() -> None:
    runtime = RuntimeConfig(testnet=True)

    class _ErrorClient:
        def get_symbol_constraints(self, symbol: str):
            raise BinanceAPIError("boom")

    assert cli._resolve_symbol_constraints(runtime, _ErrorClient()) is None


def test_command_configure_save_and_validation_paths(tmp_path, monkeypatch, capsys) -> None:
    class _ValidClient(_FakeClient):
        def ping(self):
            return {"ok": True}

    class _FailingClient(_FakeClient):
        def ping(self):
            raise BinanceAPIError("bad")

        def ensure_btcusdc(self):
            raise BinanceAPIError("bad")

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(cli, "prompt_runtime", lambda _current: RuntimeConfig(api_key="k", api_secret="s", validate_account=True))
    monkeypatch.setattr(cli, "_build_client", lambda _runtime: _ValidClient())
    assert cli.command_configure(argparse.Namespace()) == 0
    first = capsys.readouterr()
    assert "Runtime config saved to" in first.out

    monkeypatch.setattr(cli, "_build_client", lambda _runtime: _FailingClient())
    assert cli.command_configure(argparse.Namespace()) == 2
    captured = capsys.readouterr()
    assert "Configuration saved, but validation failed" in captured.err


def test_command_connect_paths_for_errors_and_leverage_exception(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    save_runtime(RuntimeConfig(api_key="k", api_secret="s", market_type="futures", symbol="BTCUSDC"))

    class _ConnectionClient(_FakeClient):
        def get_exchange_time(self):
            return {"serverTime": 999}

        def get_account(self):
            return super().get_account()

        def get_max_leverage(self, symbol: str) -> int:
            return 10

    class _LeverageFailClient(_ConnectionClient):
        def get_max_leverage(self, symbol: str) -> int:
            raise BinanceAPIError("cant fetch")

    class _ExchangeErrorClient(_ConnectionClient):
        def get_exchange_time(self):
            raise BinanceAPIError("offline")

    monkeypatch.setattr(cli, "_build_client", lambda _runtime: _ConnectionClient())
    assert cli.command_connect(argparse.Namespace()) == 0

    monkeypatch.setattr(cli, "_build_client", lambda _runtime: _LeverageFailClient())
    assert cli.command_connect(argparse.Namespace()) == 0
    assert "unable to fetch leverage bracket" in capsys.readouterr().err

    monkeypatch.setattr(cli, "_build_client", lambda _runtime: _ExchangeErrorClient())
    assert cli.command_connect(argparse.Namespace()) == 2


def test_command_strategy_covers_full_update_path(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    save_runtime(RuntimeConfig(market_type="futures", api_key="k", api_secret="s"))
    save_strategy(StrategyConfig())

    class _LeverageClient:
        def get_max_leverage(self, symbol: str) -> int:
            return 3

    monkeypatch.setattr(cli, "_build_client", lambda _runtime: _LeverageClient())
    args = argparse.Namespace(
        leverage=12.0,
        risk=0.0,
        max_position=-1.0,
        stop=-0.5,
        take=99.0,
        max_open=0,
        max_trades_per_day=-5,
        cooldown=-10,
        signal_threshold=2.0,
        max_drawdown=-1.0,
        taker_fee_bps=-1.0,
        slippage_bps=-1.0,
        label_threshold=-1.0,
    )
    assert cli.command_strategy(args) == 0

    updated = load_strategy()
    assert updated.leverage == 3.0
    assert updated.risk_per_trade == 0.0001
    assert updated.max_position_pct == 0.0
    assert updated.stop_loss_pct == 0.0
    assert updated.take_profit_pct == 0.99
    assert updated.max_open_positions == 0
    assert updated.max_trades_per_day == 0
    assert updated.cooldown_minutes == 0
    assert updated.signal_threshold == 0.99
    assert updated.max_drawdown_limit == 0.0
    assert updated.taker_fee_bps == 0.0
    assert updated.slippage_bps == 0.0
    assert updated.label_threshold == 0.0001


def test_command_fetch_success_path(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    save_runtime(RuntimeConfig())

    class _FetchClient(_FakeClient):
        def get_klines(self, symbol: str, interval: str, limit: int = 500, start_time=None, end_time=None):
            return _simple_candles(limit)

    monkeypatch.setattr(cli, "_build_client", lambda _runtime: _FetchClient())
    out = tmp_path / "history.json"
    assert cli.command_fetch(argparse.Namespace(symbol=None, interval=None, limit=10, output=str(out))) == 0
    assert out.exists()
    assert len(json.loads(out.read_text(encoding="utf-8"))) == 10


def test_command_train_empty_rows_and_walk_forward_path(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    save_runtime(RuntimeConfig())
    save_strategy(StrategyConfig())

    empty = tmp_path / "empty.json"
    empty.write_text("[]", encoding="utf-8")
    assert cli.command_train(
        argparse.Namespace(
            input=str(empty),
            output=str(tmp_path / "model.json"),
            epochs=2,
            walk_forward=False,
            walk_forward_train=10,
            walk_forward_test=5,
            walk_forward_step=1,
            calibrate_threshold=False,
        )
    ) == 2

    candles = []
    for i in range(220):
        candles.append(
            {
                "open_time": i * 60_000,
                "open": 100.0 + i,
                "high": 101.0 + i,
                "low": 99.0 + i,
                "close": 100.0 + i,
                "volume": 1.0,
                "close_time": (i + 1) * 60_000,
            },
        )
    history = tmp_path / "history.json"
    history.write_text(json.dumps(candles), encoding="utf-8")
    model = tmp_path / "model.json"

    wf_calls = []

    def fake_wf(*_args, **_kwargs):
        wf_calls.append(1)
        return {
            "folds": [0.1, 0.2],
            "scores": [0.3, 0.5],
            "average_score": 0.4,
            "train_window": 10,
            "test_window": 5,
            "step": 1,
        }

    monkeypatch.setattr(cli, "walk_forward_report", fake_wf)
    assert (
        cli.command_train(
            argparse.Namespace(
                input=str(history),
                output=str(model),
                epochs=4,
                walk_forward=True,
                walk_forward_train=10,
                walk_forward_test=5,
                walk_forward_step=1,
                calibrate_threshold=True,
            )
        )
        == 0
    )
    assert wf_calls == [1]


def test_command_train_walk_forward_errors_do_not_abort_training(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    save_runtime(RuntimeConfig())
    save_strategy(StrategyConfig())

    candles = [
        {
            "open_time": i * 60_000,
            "open": 100.0 + i,
            "high": 101.0 + i,
            "low": 99.0 + i,
            "close": 100.0 + i,
            "volume": 1.0,
            "close_time": (i + 1) * 60_000,
        }
        for i in range(240)
    ]
    history = tmp_path / "history.json"
    history.write_text(json.dumps(candles), encoding="utf-8")

    monkeypatch.setattr(cli, "walk_forward_report", lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("unavailable")))

    assert (
        cli.command_train(
            argparse.Namespace(
                input=str(history),
                output=str(tmp_path / "model.json"),
                epochs=4,
                walk_forward=True,
                walk_forward_train=10,
                walk_forward_test=5,
                walk_forward_step=1,
                calibrate_threshold=True,
            )
        )
        == 0
    )


def test_command_tune_uses_default_leverage_when_no_api_keys(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    save_runtime(RuntimeConfig(market_type="futures", api_key="", api_secret=""))
    save_strategy(StrategyConfig(training_epochs=80))

    candles = [
        {
            "open_time": i * 60_000,
            "open": 100.0 + i,
            "high": 101.0 + i,
            "low": 99.0 + i,
            "close": 100.0 + i,
            "volume": 1.0,
            "close_time": (i + 1) * 60_000,
        }
        for i in range(220)
    ]
    history = tmp_path / "history.json"
    history.write_text(json.dumps(candles), encoding="utf-8")

    class _Model:
        def predict_proba(self, _features: tuple[float, ...]) -> float:
            return 0.6

    monkeypatch.setattr(cli, "train", lambda *_a, **_k: _Model())
    monkeypatch.setattr(cli, "run_backtest", lambda *a, **k: SimpleNamespace(realized_pnl=0.0, total_fees=0.0))
    monkeypatch.setattr(cli.time, "sleep", lambda *_args: None)

    args = argparse.Namespace(
        input=str(history),
        save_best=True,
        min_risk=0.002,
        max_risk=0.003,
        steps=2,
        min_leverage=2.0,
        max_leverage=3.0,
        min_threshold=0.52,
        max_threshold=0.53,
        min_take=0.01,
        max_take=0.02,
        min_stop=0.01,
        max_stop=0.02,
    )

    assert cli.command_tune(args) == 0
    assert "tune best score" in capsys.readouterr().out


def test_tune_score_penalizes_drawdown_stops() -> None:
    bad = SimpleNamespace(realized_pnl=200.0, total_fees=1.0, max_drawdown=0.5, stopped_by_drawdown=True, closed_trades=1)
    good = SimpleNamespace(realized_pnl=120.0, total_fees=1.0, max_drawdown=0.01, stopped_by_drawdown=False, closed_trades=2)
    assert cli._tune_score(bad, starting_cash=1000.0) < cli._tune_score(good, starting_cash=1000.0)


def test_command_tune_falls_back_to_drawdown_fallback(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    save_runtime(RuntimeConfig())
    save_strategy(StrategyConfig(training_epochs=80))

    candles = []
    for i in range(240):
        candles.append(
            {
                "open_time": i * 60_000,
                "open": 100.0 + i,
                "high": 101.0 + i,
                "low": 99.0 + i,
                "close": 100.0 + i,
                "volume": 1.0,
                "close_time": (i + 1) * 60_000,
            },
        )
    history = tmp_path / "history.json"
    history.write_text(json.dumps(candles), encoding="utf-8")

    class _Model:
        def predict_proba(self, _features: tuple[float, ...]) -> float:
            return 0.6

    calls = []

    def _mock_run_backtest(*_args, **_kwargs):
        calls.append(1)
        return SimpleNamespace(realized_pnl=float(len(calls)), total_fees=0.0, max_drawdown=0.9, stopped_by_drawdown=True, closed_trades=1)

    monkeypatch.setattr(cli, "train", lambda *_a, **_k: _Model())
    monkeypatch.setattr(cli, "run_backtest", _mock_run_backtest)
    monkeypatch.setattr(cli.time, "sleep", lambda *_args: None)

    args = argparse.Namespace(
        input=str(history),
        save_best=False,
        min_risk=0.002,
        max_risk=0.004,
        steps=2,
        min_leverage=2.0,
        max_leverage=3.0,
        min_threshold=0.52,
        max_threshold=0.53,
        min_take=0.01,
        max_take=0.02,
        min_stop=0.01,
        max_stop=0.02,
    )
    assert cli.command_tune(args) == 0
    assert len(calls) > 0
    assert "all tune candidates hit drawdown limit" in capsys.readouterr().out


def test_command_live_uses_generated_model_when_saved_model_is_invalid(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    save_runtime(RuntimeConfig(testnet=True, dry_run=True, market_type="spot"))
    save_strategy(StrategyConfig(risk_per_trade=0.002, max_position_pct=0.2))

    model_file = tmp_path / "data" / "model.json"
    model_file.parent.mkdir(parents=True, exist_ok=True)
    model_file.write_text("{invalid-json}", encoding="utf-8")

    class _FlowClient:
        def get_klines(self, symbol: str, interval: str, limit: int = 500, start_time=None, end_time=None):
            return [
                Candle(open_time=i * 60_000, open=100.0 + i, high=101.0 + i, low=99.0 + i, close=100.0 + i, volume=1.0, close_time=(i + 1) * 60_000)
                for i in range(limit)
            ]

        def get_symbol_constraints(self, symbol: str):
            return SimpleNamespace(
                symbol=symbol,
                min_qty=0.0001,
                max_qty=100.0,
                step_size=0.0001,
                min_notional=1.0,
                max_notional=0.0,
            )

        def normalize_quantity(self, symbol: str, quantity: float):
            return (max(0.0001, round(quantity, 4)), self.get_symbol_constraints(symbol))

        def place_order(self, symbol: str, side: str, quantity: float, dry_run: bool, leverage: float):
            return {
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "dry_run": dry_run,
                "leverage": leverage,
            }

    class _AlwaysLongModel:
        def predict_proba(self, _features: tuple[float, ...]) -> float:
            return 0.99

    monkeypatch.setattr(cli, "_build_client", lambda _runtime: _FlowClient())
    monkeypatch.setattr(cli, "train", lambda *_a, **_k: _AlwaysLongModel())
    monkeypatch.setattr(cli.time, "sleep", lambda *_args: None)

    assert cli.command_live(argparse.Namespace(steps=1, sleep=5, paper=False)) == 0


def test_command_live_skips_entry_when_cash_is_insufficient_after_fees(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    save_runtime(RuntimeConfig(testnet=True, dry_run=True, market_type="spot"))
    save_strategy(StrategyConfig(risk_per_trade=1.0, max_position_pct=1.0, taker_fee_bps=20000.0))

    class _FlowClient:
        def get_klines(self, symbol: str, interval: str, limit: int = 500, start_time=None, end_time=None):
            return [
                Candle(
                    open_time=0,
                    open=100.0,
                    high=101.0,
                    low=99.0,
                    close=100.0,
                    volume=1.0,
                    close_time=60_000,
                )
                for _ in range(limit)
            ]

        def get_symbol_constraints(self, symbol: str):
            return SimpleNamespace(
                symbol=symbol,
                min_qty=0.01,
                max_qty=100.0,
                step_size=0.01,
                min_notional=10.0,
                max_notional=0.0,
            )

        def normalize_quantity(self, symbol: str, quantity: float):
            return (max(0.01, round(quantity, 2)), self.get_symbol_constraints(symbol))

    class _AlwaysLongModel:
        def predict_proba(self, _features: tuple[float, ...]) -> float:
            return 0.99

    monkeypatch.setattr(cli, "_build_client", lambda _runtime: _FlowClient())
    monkeypatch.setattr(cli, "train", lambda *_a, **_k: _AlwaysLongModel())
    monkeypatch.setattr(cli.time, "sleep", lambda *_args: None)
    assert cli.command_live(argparse.Namespace(steps=1, sleep=5, paper=False)) == 0
def test_command_tune_data_insufficient(tmp_path) -> None:
    save_runtime(RuntimeConfig())
    save_strategy(StrategyConfig())

    no_data = Path(tmp_path) / "small.json"
    no_data.write_text("[]", encoding="utf-8")
    args = argparse.Namespace(
        input=str(no_data),
        save_best=False,
        min_risk=0.002,
        max_risk=0.02,
        steps=5,
        min_leverage=1.0,
        max_leverage=2.0,
        min_threshold=0.52,
        max_threshold=0.88,
        min_take=0.01,
        max_take=0.06,
        min_stop=0.008,
        max_stop=0.04,
    )
    assert cli.command_tune(args) == 2


def test_command_live_rejects_live_when_not_testnet(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    save_runtime(RuntimeConfig(testnet=False, dry_run=False, api_key="k", api_secret="s"))
    assert cli.command_live(argparse.Namespace(steps=1, sleep=5, paper=False)) == 2


def test_command_live_futures_set_leverage_failure_exits(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    save_runtime(RuntimeConfig(testnet=True, dry_run=False, market_type="futures", api_key="k", api_secret="s"))
    save_strategy(StrategyConfig(leverage=5.0))

    class _FailLeverageClient:
        def get_klines(self, symbol: str, interval: str, limit: int = 500, start_time=None, end_time=None):
            raise AssertionError("should not fetch klines if leverage fails")

        def get_symbol_constraints(self, symbol: str):
            return SimpleNamespace(
                symbol=symbol,
                min_qty=0.001,
                max_qty=100.0,
                step_size=0.001,
                min_notional=1.0,
                max_notional=0.0,
            )

        def normalize_quantity(self, symbol: str, quantity: float):
            raise AssertionError("should not normalize if leverage fails")

        def set_leverage(self, _symbol: str, _leverage: int):
            raise BinanceAPIError("leverage unavailable")

        def get_max_leverage(self, symbol: str) -> int:
            return 10

    monkeypatch.setattr(cli, "_build_client", lambda _runtime: _FailLeverageClient())
    assert cli.command_live(argparse.Namespace(steps=1, sleep=5, paper=False)) == 2


def test_command_backtest_model_missing_and_success(tmp_path, monkeypatch, capsys) -> None:
    from simple_ai_bitcoin_trading_binance.model import serialize_model

    monkeypatch.setenv("HOME", str(tmp_path))
    save_runtime(RuntimeConfig())
    save_strategy(StrategyConfig())

    input_file = tmp_path / "hist.json"
    input_file.write_text("[]", encoding="utf-8")
    missing_model = tmp_path / "missing.json"
    assert cli.command_backtest(argparse.Namespace(input=str(input_file), model=str(missing_model), start_cash=1000.0)) == 2

    candles = []
    for i in range(120):
        candles.append(
            {
                "open_time": i * 60_000,
                "open": 100.0 + i,
                "high": 101.0 + i,
                "low": 99.0 + i,
                "close": 100.0 + i,
                "volume": 1.0,
                "close_time": (i + 1) * 60_000,
            },
        )
    input_file.write_text(json.dumps(candles), encoding="utf-8")
    model_file = tmp_path / "model.json"
    serialize_model(
        TrainedModel(weights=[0.0] * 13, bias=0.0, feature_dim=13, epochs=1, feature_means=[0.0] * 13, feature_stds=[1.0] * 13),
        model_file,
    )
    assert cli.command_backtest(argparse.Namespace(input=str(input_file), model=str(model_file), start_cash=1000.0)) == 0
    assert "trades:" in capsys.readouterr().out


def test_command_live_rejects_non_testnet_and_missing_keys(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    save_strategy(StrategyConfig())

    save_runtime(RuntimeConfig(testnet=False))
    assert cli.command_live(argparse.Namespace(steps=1, sleep=5, paper=False)) == 2

    save_runtime(RuntimeConfig(testnet=True, dry_run=False, api_key="", api_secret=""))
    assert cli.command_live(argparse.Namespace(steps=1, sleep=5, paper=False)) == 2


def test_command_live_detailed_flow(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(cli.time, "sleep", lambda *_args: None)

    class _FlowClient:
        def __init__(self):
            self.orders = []

        def get_klines(self, symbol: str, interval: str, limit: int = 500, start_time=None, end_time=None):
            base = 100.0
            return [
                Candle(open_time=i * 60_000, open=base + i, high=base + i + 1, low=base + i - 1, close=base + i, volume=1.0, close_time=(i + 1) * 60_000)
                for i in range(320)
            ]

        def get_symbol_constraints(self, symbol: str):
            return SimpleNamespace(
                symbol=symbol,
                min_qty=0.0001,
                max_qty=100.0,
                step_size=0.0001,
                min_notional=1.0,
                max_notional=0.0,
            )

        def normalize_quantity(self, symbol: str, quantity: float):
            constraints = self.get_symbol_constraints(symbol)
            return max(constraints.min_qty, round(quantity, 4)), constraints

        def place_order(self, symbol: str, side: str, size: float, *, dry_run: bool, leverage: float = 1.0):
            self.orders.append((side, size))
            return {"symbol": symbol, "side": side, "size": size}

    class _ScoredModel:
        def __init__(self):
            self.calls = 0

        def predict_proba(self, _features: tuple[float, ...]) -> float:
            self.calls += 1
            if self.calls == 1:
                return 0.95
            if self.calls == 2:
                return 0.5
            return 0.95

    client = _FlowClient()
    monkeypatch.setattr(cli, "_build_client", lambda _runtime: client)
    monkeypatch.setattr(cli, "train", lambda *_a, **_k: _ScoredModel())

    save_runtime(RuntimeConfig(testnet=True, dry_run=True, market_type="spot", max_rate_calls_per_minute=1))
    save_strategy(StrategyConfig(risk_per_trade=0.001, max_position_pct=0.2, max_trades_per_day=1, training_epochs=40, cooldown_minutes=1))
    assert cli.command_live(argparse.Namespace(steps=3, sleep=5, paper=False)) == 0

    # futures path should attempt leverage and fail fast when API key missing, because live futures requires credentials
    save_runtime(RuntimeConfig(testnet=True, dry_run=False, market_type="futures", api_key="k", api_secret="s", max_rate_calls_per_minute=1))

    class _SetLeverageErrorClient(_FlowClient):
        def get_max_leverage(self, symbol: str) -> int:
            return 10

        def set_leverage(self, symbol: str, leverage: int):
            raise BinanceAPIError("fail")

    monkeypatch.setattr(cli, "_build_client", lambda _runtime: _SetLeverageErrorClient())
    assert cli.command_live(argparse.Namespace(steps=1, sleep=5, paper=False)) == 2


def test_command_live_futures_leverage_override(tmp_path, monkeypatch, capsys) -> None:
    class _LeverageClient:
        def __init__(self) -> None:
            self.set_calls: list[int] = []
            self.orders: list[tuple[str, float, bool, float]] = []

        def get_klines(self, symbol: str, interval: str, limit: int = 500, start_time=None, end_time=None):
            return [
                Candle(
                    open_time=i * 60_000,
                    open=100.0 + i,
                    high=101.0 + i,
                    low=99.0 + i,
                    close=100.0 + i,
                    volume=1.0,
                    close_time=(i + 1) * 60_000,
                )
                for i in range(limit)
            ]

        def get_symbol_constraints(self, symbol: str):
            return SimpleNamespace(
                symbol=symbol,
                min_qty=0.0001,
                max_qty=100.0,
                step_size=0.0001,
                min_notional=1.0,
                max_notional=0.0,
            )

        def normalize_quantity(self, symbol: str, quantity: float):
            constraints = self.get_symbol_constraints(symbol)
            return max(constraints.min_qty, round(quantity, 4)), constraints

        def get_max_leverage(self, symbol: str) -> int:
            return 20

        def set_leverage(self, symbol: str, leverage: int):
            self.set_calls.append(leverage)
            return {"symbol": symbol, "leverage": leverage}

        def place_order(self, symbol: str, side: str, size: float, *, dry_run: bool, leverage: float = 1.0):
            self.orders.append((side, size, dry_run, leverage))
            return {"symbol": symbol, "side": side, "size": size}

    class _AlwaysLongModel:
        def predict_proba(self, _features: tuple[float, ...]) -> float:
            return 0.99

    monkeypatch.setenv("HOME", str(tmp_path))
    save_runtime(RuntimeConfig(testnet=True, dry_run=False, market_type="futures", api_key="k", api_secret="s"))
    save_strategy(StrategyConfig(risk_per_trade=0.005, max_position_pct=0.5))

    client = _LeverageClient()
    monkeypatch.setattr(cli, "_build_client", lambda _runtime: client)
    monkeypatch.setattr(cli.time, "sleep", lambda *_args: None)
    monkeypatch.setattr(cli, "train", lambda *_a, **_k: _AlwaysLongModel())

    assert (
        cli.command_live(argparse.Namespace(steps=1, sleep=5, paper=False, leverage=12.0))
        == 0
    )
    assert client.set_calls == [12]
    assert client.orders
    assert client.orders[0][3] == 12.0
    assert "effective leverage" in capsys.readouterr().out


def test_command_live_spot_leverage_override_is_logged(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    save_runtime(RuntimeConfig(testnet=True, dry_run=True, market_type="spot"))
    save_strategy(StrategyConfig())
    monkeypatch.setattr(cli.time, "sleep", lambda *_args: None)

    class _SpotClient:
        def get_klines(self, symbol: str, interval: str, limit: int = 500, start_time=None, end_time=None):
            return [
                Candle(
                    open_time=i * 60_000,
                    open=100.0 + i,
                    high=101.0 + i,
                    low=99.0 + i,
                    close=100.0 + i,
                    volume=1.0,
                    close_time=(i + 1) * 60_000,
                )
                for i in range(limit)
            ]

        def get_symbol_constraints(self, symbol: str):
            return SimpleNamespace(
                symbol=symbol,
                min_qty=0.0001,
                max_qty=100.0,
                step_size=0.0001,
                min_notional=1.0,
                max_notional=0.0,
            )

        def normalize_quantity(self, symbol: str, quantity: float):
            constraints = self.get_symbol_constraints(symbol)
            return max(constraints.min_qty, round(quantity, 4)), constraints

        def place_order(self, symbol: str, side: str, size: float, *, dry_run: bool, leverage: float = 1.0):
            return {"symbol": symbol, "side": side, "size": size}

    class _AlwaysLongModel:
        def predict_proba(self, _features: tuple[float, ...]) -> float:
            return 0.99

    monkeypatch.setattr(cli, "_build_client", lambda _runtime: _SpotClient())
    monkeypatch.setattr(cli, "train", lambda *_a, **_k: _AlwaysLongModel())
    assert cli.command_live(argparse.Namespace(steps=1, sleep=5, paper=False, leverage=10.0)) == 0
    assert "Leverage override is spot-inactive" in capsys.readouterr().out


def test_command_live_drawdown_limit_forces_emergency_close(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    save_runtime(RuntimeConfig(testnet=True, dry_run=True, market_type="spot"))
    save_strategy(
        StrategyConfig(
            risk_per_trade=0.5,
            max_position_pct=1.0,
            take_profit_pct=0.95,
            stop_loss_pct=0.99,
            max_drawdown_limit=0.20,
            feature_windows=(4, 20),
        )
    )

    class _DrawdownClient:
        call_count = 0

        def get_klines(self, symbol: str, interval: str, limit: int = 500, start_time=None, end_time=None):
            if interval != "15m":
                return []

            self.call_count += 1
            if self.call_count == 1:
                close = 100.0
            else:
                close = 10.0

            candles = [
                Candle(
                    open_time=i * 60_000,
                    open=close,
                    high=close,
                    low=close,
                    close=close,
                    volume=1.0,
                    close_time=(i + 1) * 60_000,
                )
                for i in range(60)
            ]
            return candles

        def get_symbol_constraints(self, symbol: str):
            return SimpleNamespace(
                symbol=symbol,
                min_qty=0.0001,
                max_qty=1000.0,
                step_size=0.0001,
                min_notional=1.0,
                max_notional=0.0,
            )

        def normalize_quantity(self, symbol: str, quantity: float):
            constraints = self.get_symbol_constraints(symbol)
            return max(constraints.min_qty, round(quantity, 4)), constraints

        def place_order(self, symbol: str, side: str, size: float, *, dry_run: bool, leverage: float = 1.0):
            return {
                "symbol": symbol,
                "side": side,
                "size": size,
                "dry_run": dry_run,
                "leverage": leverage,
            }

    class _AlwaysLongModel:
        def predict_proba(self, _features: tuple[float, ...]) -> float:
            return 1.0

    monkeypatch.setattr(cli.time, "sleep", lambda *_args: None)
    monkeypatch.setattr(cli, "_build_client", lambda _runtime: _DrawdownClient())
    monkeypatch.setattr(cli, "train", lambda *_a, **_k: _AlwaysLongModel())

    assert cli.command_live(argparse.Namespace(steps=3, sleep=5, paper=False)) == 0
    output = capsys.readouterr().out
    assert "emergency close from drawdown" in output
    assert "drawdown limit reached" in output
