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
from simple_ai_bitcoin_trading_binance.model import ModelLoadError, TrainedModel
from simple_ai_bitcoin_trading_binance.features import feature_signature
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
    doctor_args = cli._parse_args(["doctor", "--input", "i.json", "--model", "m.json", "--online"])
    assert doctor_args.input == "i.json"
    assert doctor_args.model == "m.json"
    assert doctor_args.online is True
    train_args = cli._parse_args(["train", "--preset", "quick"])
    assert train_args.preset == "quick"

    marker = []

    def fake_status(ns: argparse.Namespace) -> int:
        marker.append("status")
        return 9

    monkeypatch.setattr(cli, "command_status", fake_status)
    assert cli.main(["status"]) == 9
    assert marker == ["status"]

    live = cli._parse_args(["live", "--steps", "3"])
    assert live.retrain_interval == 0
    assert live.retrain_window == 300
    assert live.retrain_min_rows == 240
    assert live.paper is False


def test_training_preset_helper_and_invalid_command(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    custom = cli._apply_training_preset(
        argparse.Namespace(
            preset="custom",
            epochs=999,
            walk_forward=True,
            walk_forward_train=1,
            walk_forward_test=1,
            walk_forward_step=1,
            calibrate_threshold=True,
        )
    )
    assert custom.epochs == 999
    quick = cli._apply_training_preset(argparse.Namespace(preset="quick", epochs=999, walk_forward=True, calibrate_threshold=True))
    assert quick.epochs == 80
    assert quick.walk_forward is False
    balanced = cli._apply_training_preset(argparse.Namespace(preset="balanced"))
    assert balanced.walk_forward is True
    assert balanced.walk_forward_test == 60
    thorough = cli._apply_training_preset(argparse.Namespace(preset="thorough"))
    assert thorough.epochs == 350
    assert thorough.walk_forward_train == 360
    with pytest.raises(ValueError):
        cli._apply_training_preset(argparse.Namespace(preset="wild"))

    assert cli.command_train(argparse.Namespace(input="x", output="y", preset="wild")) == 2
    assert "Training settings invalid" in capsys.readouterr().err


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


def test_connection_status_line_branches(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    class NoAuthClient(_FakeClient):
        pass

    save_runtime(RuntimeConfig(api_key="", api_secret="", dry_run=True, testnet=True, market_type="spot"))
    monkeypatch.setattr(cli, "_build_client", lambda _runtime: NoAuthClient())
    line = cli._connection_status_line()
    assert "online spot/testnet paper-default" in line
    assert "auth missing" in line

    save_runtime(RuntimeConfig(api_key="k", api_secret="s", dry_run=False, testnet=True, market_type="futures"))
    monkeypatch.setattr(cli, "_build_client", lambda _runtime: _FakeClient())
    line = cli._connection_status_line()
    assert "online futures/testnet testnet-live-default" in line
    assert "auth ok" in line

    class OddAuthClient(_FakeClient):
        def get_account(self):
            return "accepted"

    monkeypatch.setattr(cli, "_build_client", lambda _runtime: OddAuthClient())
    assert "auth response ok" in cli._connection_status_line()

    class OfflineClient(_FakeClient):
        def ping(self):
            raise BinanceAPIError("timeout")

    monkeypatch.setattr(cli, "_build_client", lambda _runtime: OfflineClient())
    assert "offline futures/testnet" in cli._connection_status_line()


def test_readiness_report_and_command_doctor(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    save_runtime(RuntimeConfig(testnet=False, dry_run=False, api_key="", api_secret=""))
    save_strategy(StrategyConfig())
    ok, lines = cli._readiness_report(input_path=str(tmp_path / "missing.json"), model_path=str(tmp_path / "missing-model.json"))
    assert ok is False
    assert any(line.startswith("[fix] safety target") for line in lines)
    assert any("training data" in line for line in lines)

    data_file = tmp_path / "history.json"
    model_file = tmp_path / "model.json"
    data_file.write_text("[]", encoding="utf-8")
    model_file.write_text("{}", encoding="utf-8")
    save_runtime(RuntimeConfig(testnet=True, dry_run=True, api_key="", api_secret=""))
    monkeypatch.setattr(cli, "_load_rows_for_command", lambda *_args, **_kwargs: [object()] * 80)
    monkeypatch.setattr(cli, "_load_runtime_model", lambda *_args, **_kwargs: SimpleNamespace(feature_dim=3))
    monkeypatch.setattr(cli, "_connection_status_line", lambda: "Connection 00: online spot/testnet paper-default server-time ok auth missing")
    assert cli.command_doctor(argparse.Namespace(input=str(data_file), model=str(model_file), online=True)) == 0
    output = capsys.readouterr().out
    assert "Readiness report" in output
    assert "[ok] exchange connectivity" in output

    monkeypatch.setattr(cli, "_load_runtime_model", lambda *_args, **_kwargs: (_ for _ in ()).throw(ModelLoadError("bad model")))
    ok, lines = cli._readiness_report(input_path=str(data_file), model_path=str(model_file), online=False)
    assert ok is False
    assert any("bad model" in line for line in lines)


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
    save_runtime(RuntimeConfig(api_key="visible-key", api_secret="super-secret"))
    save_strategy(StrategyConfig())
    assert cli.command_status(argparse.Namespace()) == 0
    output = capsys.readouterr().out
    assert "<redacted>" in output
    assert "super-secret" not in output
    assert "visible-key" not in output


def test_command_configure_validation_failure_returns_nonzero(tmp_path, monkeypatch, capsys) -> None:
    class _BadClient(_FakeClient):
        def ensure_btcusdc(self):
            raise BinanceAPIError("no symbol")

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(cli, "prompt_runtime", lambda _current: RuntimeConfig(api_key="k", api_secret="s", validate_account=True))
    monkeypatch.setattr(cli, "_build_client", lambda _runtime: _BadClient())
    assert cli.command_configure(argparse.Namespace()) == 2
    assert "validation failed" in capsys.readouterr().err


def test_command_configure_futures_prints_mode_line(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(cli, "prompt_runtime", lambda _current: RuntimeConfig(market_type="futures", validate_account=False))
    assert cli.command_configure(argparse.Namespace()) == 0
    assert "futures-mode enabled" in capsys.readouterr().out


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


def test_command_connect_failure_returns_nonzero(tmp_path, monkeypatch, capsys) -> None:
    class _BadClient(_FakeClient):
        def get_exchange_time(self):
            raise BinanceAPIError("offline")

    monkeypatch.setenv("HOME", str(tmp_path))
    save_runtime(RuntimeConfig())
    monkeypatch.setattr(cli, "_build_client", lambda _runtime: _BadClient())
    assert cli.command_connect(argparse.Namespace()) == 2
    assert "Connection failed" in capsys.readouterr().err


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


def test_command_fetch_rejects_non_btcusdc_symbol(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    save_runtime(RuntimeConfig())
    assert cli.command_fetch(argparse.Namespace(symbol="ETHUSDC", interval=None, limit=10, output=str(tmp_path / "candles.json"))) == 2


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


def test_command_train_rejects_bad_json_input(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    save_runtime(RuntimeConfig())
    save_strategy(StrategyConfig())
    bad_input = tmp_path / "bad.json"
    bad_input.write_text("{", encoding="utf-8")
    assert cli.command_train(
        argparse.Namespace(
            input=str(bad_input),
            output=str(tmp_path / "model.json"),
            epochs=12,
            walk_forward=False,
            walk_forward_train=10,
            walk_forward_test=5,
            walk_forward_step=1,
            calibrate_threshold=False,
        )
    ) == 2


def test_command_train_walk_forward_unavailable_still_succeeds(tmp_path, monkeypatch, capsys) -> None:
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
        for i in range(260)
    ]
    data_file = tmp_path / "history.json"
    data_file.write_text(json.dumps(candles), encoding="utf-8")
    monkeypatch.setattr(cli, "walk_forward_report", lambda *_a, **_k: (_ for _ in ()).throw(ValueError("not enough rows")))
    assert cli.command_train(
        argparse.Namespace(
            input=str(data_file),
            output=str(tmp_path / "model.json"),
            epochs=12,
            walk_forward=True,
            walk_forward_train=1000,
            walk_forward_test=1000,
            walk_forward_step=10,
            calibrate_threshold=False,
        )
    ) == 0
    assert "walk-forward unavailable" in capsys.readouterr().out


def test_command_train_artifact_includes_signature(tmp_path, monkeypatch) -> None:
    strategy = StrategyConfig(feature_windows=(4, 20), label_threshold=0.002, training_epochs=7)
    monkeypatch.setenv("HOME", str(tmp_path))
    save_runtime(RuntimeConfig(api_key="secret-key", api_secret="secret-value"))
    save_strategy(strategy)

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
        for i in range(300)
    ]
    history = tmp_path / "history.json"
    model_file = tmp_path / "model.json"
    history.write_text(json.dumps(candles), encoding="utf-8")

    captured: list[tuple[str, str, dict[str, object]]] = []

    def fake_persist(kind: str, output_dir: Path, payload: dict[str, object]) -> Path:
        captured.append((kind, str(output_dir), payload))
        return output_dir / "artifact.json"

    monkeypatch.setattr(cli, "_persist_run_artifact", fake_persist)

    assert cli.command_train(
        argparse.Namespace(
            input=str(history),
            output=str(model_file),
            epochs=11,
            walk_forward=False,
            walk_forward_train=10,
            walk_forward_test=5,
            walk_forward_step=1,
            calibrate_threshold=False,
        )
    ) == 0

    assert len(captured) == 1
    kind, _output_dir, payload = captured[0]
    assert kind == "train"
    assert payload["seed"] == 7
    expected_signature = feature_signature(
        strategy.feature_windows[0],
        strategy.feature_windows[1],
        strategy.label_threshold,
        feature_version=strategy.feature_version,
    )
    assert payload["model"]["feature_signature"] == expected_signature
    assert payload["runtime"]["api_key"] == "<redacted>"
    assert payload["runtime"]["api_secret"] == "<redacted>"


def test_command_train_written_artifact_does_not_leak_credentials(tmp_path, monkeypatch) -> None:
    strategy = StrategyConfig()
    monkeypatch.setenv("HOME", str(tmp_path))
    save_runtime(RuntimeConfig(api_key="secret-key", api_secret="secret-value"))
    save_strategy(strategy)
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

    artifact_text = next(tmp_path.glob("train_run_*.json")).read_text(encoding="utf-8")
    assert "secret-key" not in artifact_text
    assert "secret-value" not in artifact_text
    assert "<redacted>" in artifact_text


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
                "feature_version": "v1",
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


def test_command_evaluate_artifact_is_emitted(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    save_runtime(RuntimeConfig(api_key="secret-key", api_secret="secret-value"))
    save_strategy(StrategyConfig())
    captured: list[tuple[str, str, dict[str, object]]] = []

    def fake_persist(kind: str, output_dir: Path, payload: dict[str, object]) -> Path:
        captured.append((kind, str(output_dir), payload))
        return output_dir / "evaluate.json"

    monkeypatch.setattr(cli, "_persist_run_artifact", fake_persist)

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
                "feature_version": "v1",
                "feature_dim": 13,
                "epochs": 5,
                "feature_means": [1.0] * 13,
                "feature_stds": [1.0] * 13,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    assert (
        cli.command_evaluate(
            argparse.Namespace(
                input=str(data_file),
                model=str(model_file),
                threshold=None,
                calibrate_threshold=False,
            )
        )
        == 0
    )

    assert len(captured) == 1
    kind, _output_dir, payload = captured[0]
    assert kind == "evaluate"
    assert payload["command"] == "evaluate"
    assert payload["runtime"]["api_key"] == "<redacted>"
    assert payload["runtime"]["api_secret"] == "<redacted>"


def test_command_evaluate_rejects_bad_json_input(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    save_runtime(RuntimeConfig())
    save_strategy(StrategyConfig())
    bad_input = tmp_path / "history.json"
    bad_input.write_text("{", encoding="utf-8")
    assert cli.command_evaluate(
        argparse.Namespace(
            input=str(bad_input),
            model=str(tmp_path / "model.json"),
            threshold=None,
            calibrate_threshold=False,
        )
    ) == 2


def test_command_evaluate_rejects_invalid_model_payload(tmp_path, monkeypatch) -> None:
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
    data_file = tmp_path / "history.json"
    data_file.write_text(json.dumps(candles), encoding="utf-8")
    model_file = tmp_path / "model.json"
    model_file.write_text("{", encoding="utf-8")
    assert cli.command_evaluate(
        argparse.Namespace(
            input=str(data_file),
            model=str(model_file),
            threshold=None,
            calibrate_threshold=False,
        )
    ) == 2


def test_command_backtest_rejects_invalid_model_payload(tmp_path, monkeypatch) -> None:
    save_runtime(RuntimeConfig())
    save_strategy(StrategyConfig())
    monkeypatch.setenv("HOME", str(tmp_path))
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
    input_file = tmp_path / "hist.json"
    model_file = tmp_path / "model.json"
    input_file.write_text(json.dumps(candles), encoding="utf-8")
    model_file.write_text("{", encoding="utf-8")
    assert cli.command_backtest(argparse.Namespace(input=str(input_file), model=str(model_file), start_cash=1000.0)) == 2


def test_command_backtest_rejects_bad_json_input(tmp_path, monkeypatch) -> None:
    save_runtime(RuntimeConfig())
    save_strategy(StrategyConfig())
    monkeypatch.setenv("HOME", str(tmp_path))
    input_file = tmp_path / "hist.json"
    model_file = tmp_path / "model.json"
    input_file.write_text("{", encoding="utf-8")
    model_file.write_text("{}", encoding="utf-8")
    assert cli.command_backtest(argparse.Namespace(input=str(input_file), model=str(model_file), start_cash=1000.0)) == 2


def test_command_tune_needs_more_rows(tmp_path, monkeypatch, capsys) -> None:
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
        for i in range(80)
    ]
    data_file = tmp_path / "history.json"
    data_file.write_text(json.dumps(candles), encoding="utf-8")
    assert cli.command_tune(
        argparse.Namespace(
            input=str(data_file),
            save_best=False,
            min_risk=0.002,
            max_risk=0.02,
            steps=2,
            min_leverage=1.0,
            max_leverage=2.0,
            min_threshold=0.52,
            max_threshold=0.6,
            min_take=0.01,
            max_take=0.02,
            min_stop=0.008,
            max_stop=0.02,
        )
    ) == 2
    assert "Need more data rows" in capsys.readouterr().out


def test_command_tune_uses_fallback_and_can_save_best(tmp_path, monkeypatch, capsys) -> None:
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
        for i in range(320)
    ]
    data_file = tmp_path / "history.json"
    data_file.write_text(json.dumps(candles), encoding="utf-8")

    monkeypatch.setattr(
        cli,
        "run_backtest",
        lambda *_a, **_k: SimpleNamespace(
            realized_pnl=-5.0,
            total_fees=1.0,
            max_drawdown=0.1,
            closed_trades=1,
            stopped_by_drawdown=True,
        ),
    )

    assert cli.command_tune(
        argparse.Namespace(
            input=str(data_file),
            save_best=True,
            min_risk=0.002,
            max_risk=0.003,
            steps=1,
            min_leverage=1.0,
            max_leverage=1.0,
            min_threshold=0.52,
            max_threshold=0.52,
            min_take=0.01,
            max_take=0.01,
            min_stop=0.008,
            max_stop=0.008,
        )
    ) == 0
    output = capsys.readouterr().out
    assert "all tune candidates hit drawdown limit" in output
    assert "Saved tuned strategy." in output


def test_command_live_paper_flag_overrides_runtime_live_without_credentials(tmp_path, monkeypatch) -> None:
    class _LiveClient:
        def ensure_btcusdc(self):
            return {"symbol": "BTCUSDC"}

        def get_klines(self, symbol: str, interval: str, limit: int = 500, start_time=None, end_time=None):
            return _simple_candles(limit)

        def place_order(self, symbol: str, side: str, size: float, *, dry_run: bool, leverage: float = 1.0):
            return {"symbol": symbol, "side": side, "size": size, "dry_run": dry_run}

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
            return max(constraints.min_qty, round(quantity, 4)), constraints

    class _AlwaysLongModel:
        def predict_proba(self, _features: tuple[float, ...]) -> float:
            return 0.95

        def predict(self, _features: tuple[float, ...], threshold: float) -> int:
            return int(0.95 >= threshold)

    monkeypatch.setenv("HOME", str(tmp_path))
    save_runtime(RuntimeConfig(testnet=True, dry_run=False, market_type="spot", api_key="", api_secret=""))
    save_strategy(StrategyConfig(risk_per_trade=0.001, max_position_pct=0.2))
    monkeypatch.setattr(cli, "train", lambda *_a, **_k: _AlwaysLongModel())
    monkeypatch.setattr(cli, "_build_client", lambda _runtime: _LiveClient())
    monkeypatch.setattr(cli.time, "sleep", lambda *_args: None)
    assert cli.command_live(argparse.Namespace(steps=1, sleep=0, paper=True, leverage=None, retrain_interval=0, retrain_window=300, retrain_min_rows=240)) == 0


def test_command_live_spot_leverage_override_is_inactive(tmp_path, monkeypatch, capsys) -> None:
    class _LiveClient:
        def ensure_btcusdc(self):
            return {"symbol": "BTCUSDC"}

        def get_klines(self, symbol: str, interval: str, limit: int = 500, start_time=None, end_time=None):
            return _simple_candles(limit)

        def place_order(self, symbol: str, side: str, size: float, *, dry_run: bool, leverage: float = 1.0):
            return {"symbol": symbol, "side": side, "size": size, "dry_run": dry_run, "leverage": leverage}

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
            return max(constraints.min_qty, round(quantity, 4)), constraints

    class _AlwaysLongModel:
        def predict_proba(self, _features: tuple[float, ...]) -> float:
            return 0.95

        def predict(self, _features: tuple[float, ...], threshold: float) -> int:
            return int(0.95 >= threshold)

    monkeypatch.setenv("HOME", str(tmp_path))
    save_runtime(RuntimeConfig(testnet=True, dry_run=True, market_type="spot"))
    save_strategy(StrategyConfig(risk_per_trade=0.001, max_position_pct=0.2))
    monkeypatch.setattr(cli, "train", lambda *_a, **_k: _AlwaysLongModel())
    monkeypatch.setattr(cli, "_build_client", lambda _runtime: _LiveClient())
    monkeypatch.setattr(cli.time, "sleep", lambda *_args: None)
    assert cli.command_live(argparse.Namespace(steps=1, sleep=0, paper=False, leverage=20.0, retrain_interval=0, retrain_window=300, retrain_min_rows=240)) == 0
    assert "Leverage override is spot-inactive" in capsys.readouterr().out


def test_command_backtest_artifact_is_emitted(tmp_path, monkeypatch) -> None:
    from simple_ai_bitcoin_trading_binance.model import serialize_model

    monkeypatch.setenv("HOME", str(tmp_path))
    save_runtime(RuntimeConfig(api_key="secret-key", api_secret="secret-value"))
    save_strategy(StrategyConfig())
    captured: list[tuple[str, str, dict[str, object]]] = []

    def fake_persist(kind: str, output_dir: Path, payload: dict[str, object]) -> Path:
        captured.append((kind, str(output_dir), payload))
        return output_dir / "backtest.json"

    monkeypatch.setattr(cli, "_persist_run_artifact", fake_persist)

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
    input_file = tmp_path / "hist.json"
    model_file = tmp_path / "model.json"
    input_file.write_text(json.dumps(candles), encoding="utf-8")
    serialize_model(
        TrainedModel(
            weights=[0.0] * 13,
            bias=0.0,
            feature_dim=13,
            epochs=5,
            feature_means=[0.0] * 13,
            feature_stds=[1.0] * 13,
        ),
        model_file,
    )

    assert (
        cli.command_backtest(argparse.Namespace(input=str(input_file), model=str(model_file), start_cash=1000.0))
        == 0
    )

    assert len(captured) == 1
    kind, _output_dir, payload = captured[0]
    assert kind == "backtest"
    assert payload["command"] == "backtest"
    assert payload["runtime"]["api_key"] == "<redacted>"
    assert payload["runtime"]["api_secret"] == "<redacted>"


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


def test_command_live_artifact_is_emitted(tmp_path, monkeypatch) -> None:
    captured: list[tuple[str, str, dict[str, object]]] = []

    class _LiveClient:
        def __init__(self) -> None:
            self.orders: list[tuple[str, str, float]] = []

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
            return max(0.0001, round(quantity, 4)), self.get_symbol_constraints(symbol)

        def place_order(self, symbol: str, side: str, size: float, *, dry_run: bool, leverage: float = 1.0):
            self.orders.append((symbol, side, size))
            return {"symbol": symbol, "side": side, "size": size, "dry_run": dry_run}

    class _AlwaysLongModel:
        def predict_proba(self, _features: tuple[float, ...]) -> float:
            return 0.95

    def fake_persist(kind: str, output_dir: Path, payload: dict[str, object]) -> Path:
        captured.append((kind, str(output_dir), payload))
        return output_dir / "live.json"

    monkeypatch.setenv("HOME", str(tmp_path))
    save_runtime(RuntimeConfig(testnet=True, dry_run=True, market_type="spot", api_key="secret-key", api_secret="secret-value"))
    save_strategy(StrategyConfig(risk_per_trade=0.001, max_position_pct=0.2))
    monkeypatch.setattr(cli, "_build_client", lambda _runtime: _LiveClient())
    monkeypatch.setattr(cli, "train", lambda *_a, **_k: _AlwaysLongModel())
    monkeypatch.setattr(cli, "_persist_run_artifact", fake_persist)
    monkeypatch.setattr(cli.time, "sleep", lambda *_args: None)

    assert cli.command_live(argparse.Namespace(steps=1, sleep=5, paper=False)) == 0
    assert len(captured) == 1
    kind, _output_dir, payload = captured[0]
    assert kind == "live"
    assert payload["command"] == "live"
    assert payload["runtime"]["api_key"] == "<redacted>"
    assert payload["runtime"]["api_secret"] == "<redacted>"


def test_command_live_daily_trade_cap_counts_entries_not_closures(tmp_path, monkeypatch) -> None:
    class _CappedClient:
        def __init__(self) -> None:
            self.iteration = 0
            self.orders: list[tuple[str, str, float, bool]] = []

        def ensure_btcusdc(self):
            return {"symbol": "BTCUSDC"}

        def get_klines(self, symbol: str, interval: str, limit: int = 500, start_time=None, end_time=None):
            day = 24 * 60 * 60 * 1000
            closes_by_call = [
                [100.0] * (limit - 1) + [100.0],
                [100.0] * (limit - 1) + [110.0],
                [100.0] * (limit - 1) + [120.0],
            ]
            closes = closes_by_call[min(self.iteration, len(closes_by_call) - 1)]
            candles = []
            base_time = 0 if self.iteration < 2 else day
            for i, close in enumerate(closes):
                candles.append(
                    Candle(
                        open_time=base_time + i * 60_000,
                        open=close,
                        high=close * 1.001,
                        low=close * 0.999,
                        close=close,
                        volume=1.0,
                        close_time=base_time + (i + 1) * 60_000,
                    )
                )
            self.iteration += 1
            return candles

        def place_order(self, symbol: str, side: str, size: float, *, dry_run: bool, leverage: float = 1.0):
            self.orders.append((symbol, side, size, dry_run))
            return {"symbol": symbol, "side": side, "size": size, "dry_run": dry_run}

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
            return max(constraints.min_qty, round(quantity, 4)), constraints

    class _StepModel:
        def __init__(self) -> None:
            self.calls = 0

        def predict_proba(self, _features: tuple[float, ...]) -> float:
            scores = [0.95, 0.0, 0.95]
            score = scores[min(self.calls, len(scores) - 1)]
            self.calls += 1
            return score

        def predict(self, _features: tuple[float, ...], threshold: float) -> int:
            return int(self.predict_proba(_features) >= threshold)

    captured: list[dict[str, object]] = []

    def fake_persist(kind: str, output_dir: Path, payload: dict[str, object]) -> Path:
        captured.append(payload)
        return output_dir / "live.json"

    monkeypatch.setenv("HOME", str(tmp_path))
    save_runtime(RuntimeConfig(testnet=True, dry_run=True, market_type="spot"))
    save_strategy(StrategyConfig(risk_per_trade=0.01, max_position_pct=0.2, max_trades_per_day=1, cooldown_minutes=0))
    monkeypatch.setattr(cli, "train", lambda *_a, **_k: _StepModel())
    monkeypatch.setattr(cli, "_build_client", lambda _runtime: _CappedClient())
    monkeypatch.setattr(cli, "_persist_run_artifact", fake_persist)
    monkeypatch.setattr(cli.time, "sleep", lambda *_args: None)
    assert cli.command_live(argparse.Namespace(steps=3, sleep=0, paper=False, leverage=None, retrain_interval=0, retrain_window=300, retrain_min_rows=240)) == 0
    assert captured[0]["result"]["entries"] == 2


def test_command_live_rejects_non_testnet_runtime(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    save_runtime(RuntimeConfig(testnet=False, dry_run=True, market_type="spot"))
    save_strategy(StrategyConfig())
    monkeypatch.setattr(cli, "_build_client", lambda _runtime: _FakeClient())
    assert cli.command_live(argparse.Namespace(steps=1, sleep=0, paper=False, leverage=None, retrain_interval=0, retrain_window=300, retrain_min_rows=240)) == 2
    assert "Real-money execution is disabled" in capsys.readouterr().out


def test_command_live_rejects_missing_credentials_for_live_mode(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    save_runtime(RuntimeConfig(testnet=True, dry_run=False, market_type="spot", api_key="", api_secret=""))
    save_strategy(StrategyConfig())
    monkeypatch.setattr(cli, "_build_client", lambda _runtime: _FakeClient())
    assert cli.command_live(argparse.Namespace(steps=1, sleep=0, paper=False, leverage=None, retrain_interval=0, retrain_window=300, retrain_min_rows=240)) == 2
    assert "Live mode needs API key and secret" in capsys.readouterr().out


def test_command_live_futures_leverage_failure_returns_nonzero(tmp_path, monkeypatch, capsys) -> None:
    class _FailLeverageClient(_FakeClient):
        def set_leverage(self, symbol: str, leverage: int):
            raise BinanceAPIError("bad leverage")

    monkeypatch.setenv("HOME", str(tmp_path))
    save_runtime(RuntimeConfig(testnet=True, dry_run=False, market_type="futures", api_key="k", api_secret="s"))
    save_strategy(StrategyConfig(leverage=5.0))
    monkeypatch.setattr(cli, "_build_client", lambda _runtime: _FailLeverageClient())
    monkeypatch.setattr(cli, "_resolve_futures_leverage", lambda _runtime, _cfg: 5.0)
    assert cli.command_live(argparse.Namespace(steps=1, sleep=0, paper=False, leverage=None, retrain_interval=0, retrain_window=300, retrain_min_rows=240)) == 2
    assert "Failed to set leverage" in capsys.readouterr().err


def test_command_live_recovers_from_invalid_saved_model(tmp_path, monkeypatch, capsys) -> None:
    class _LiveClient:
        def __init__(self) -> None:
            self.orders: list[tuple[str, str, float, bool]] = []

        def ensure_btcusdc(self):
            return {"symbol": "BTCUSDC"}

        def get_klines(self, symbol: str, interval: str, limit: int = 500, start_time=None, end_time=None):
            return _simple_candles(limit)

        def place_order(self, symbol: str, side: str, size: float, *, dry_run: bool, leverage: float = 1.0):
            self.orders.append((symbol, side, size, dry_run))
            return {"symbol": symbol, "side": side, "size": size, "dry_run": dry_run}

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

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    save_runtime(RuntimeConfig(testnet=True, dry_run=True, market_type="spot"))
    save_strategy(StrategyConfig(risk_per_trade=0.001, max_position_pct=0.2))
    model_dir = tmp_path / "data"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "model.json").write_text("{}", encoding="utf-8")

    class _AlwaysLongModel:
        def predict_proba(self, _features: tuple[float, ...]) -> float:
            return 0.95

        def predict(self, _features: tuple[float, ...], threshold: float) -> int:
            return int(0.95 >= threshold)

    monkeypatch.setattr(cli, "train", lambda *_a, **_k: _AlwaysLongModel())
    monkeypatch.setattr(cli, "_build_client", lambda _runtime: _LiveClient())
    monkeypatch.setattr(cli, "_load_runtime_model", lambda *_a, **_k: (_ for _ in ()).throw(cli.ModelLoadError("bad model")))
    monkeypatch.setattr(cli.time, "sleep", lambda *_args: None)
    assert cli.command_live(argparse.Namespace(steps=1, sleep=0, paper=False, leverage=None, retrain_interval=0, retrain_window=300, retrain_min_rows=240)) == 0
    captured = capsys.readouterr()
    assert "Model load failed; regenerating" in captured.err


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


def test_resolve_live_retrain_rows() -> None:
    rows = list(range(5))
    assert cli._resolve_live_retrain_rows(rows, retrain_window=10, retrain_min_rows=10) == []
    assert cli._resolve_live_retrain_rows(rows, retrain_window=10, retrain_min_rows=3) == rows
    assert cli._resolve_live_retrain_rows(list(range(15)), retrain_window=10, retrain_min_rows=3) == list(range(5, 15))


def test_build_live_model_retrain_interval(monkeypatch) -> None:
    cfg = StrategyConfig(training_epochs=100)

    calls: list[tuple[int, int]] = []

    def fake_train(rows, epochs: int = 100):
        calls.append((len(rows), epochs))
        return f"model-{len(calls)}"

    monkeypatch.setattr(cli, "train", fake_train)

    base_rows = list(range(200))
    model = cli._build_live_model(
        base_rows,
        model=None,
        retrain_every=2,
        step=1,
        cfg=cfg,
        retrain_window=50,
        retrain_min_rows=40,
    )
    assert model == "model-1"
    assert len(calls) == 1
    assert calls[-1][0] == 50

    model = cli._build_live_model(
        base_rows,
        model=model,
        retrain_every=2,
        step=3,
        cfg=cfg,
        retrain_window=50,
        retrain_min_rows=40,
    )
    assert model == "model-1"
    assert len(calls) == 1

    model = cli._build_live_model(
        base_rows,
        model=model,
        retrain_every=2,
        step=4,
        cfg=cfg,
        retrain_window=50,
        retrain_min_rows=40,
    )
    assert model == "model-2"
    assert len(calls) == 2
    assert calls[-1][0] == 50


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


def test_command_live_retrain_interval_rebuilds_model_in_loop(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(cli.time, "sleep", lambda *_args: None)

    class _FlowClient:
        def get_klines(self, symbol: str, interval: str, limit: int = 500, start_time=None, end_time=None):
            return _simple_candles(limit)

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
            return {"symbol": symbol, "side": side, "size": size, "dry_run": dry_run}

    class _AlwaysLongModel:
        def __init__(self):
            self.calls = 0

        def predict_proba(self, _features: tuple[float, ...]) -> float:
            self.calls += 1
            return 0.99

    train_calls = []

    def fake_train(rows, epochs: int):
        train_calls.append(len(rows))
        return _AlwaysLongModel()

    save_runtime(RuntimeConfig(testnet=True, dry_run=True, market_type="spot"))
    save_strategy(StrategyConfig(risk_per_trade=0.001, max_position_pct=0.2))

    monkeypatch.setattr(cli, "_build_client", lambda _runtime: _FlowClient())
    monkeypatch.setattr(cli, "train", fake_train)

    assert cli.command_live(
        argparse.Namespace(
            steps=3,
            sleep=5,
            paper=False,
            retrain_interval=2,
            retrain_window=120,
            retrain_min_rows=100,
        )
    ) == 0

    # initial build at step 1 + rebuild at step 2 (interval=2), no rebuild at step 3
    assert train_calls == [120, 120]



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
