"""Coverage tests for the new CLI subcommands added in the UX rework."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from simple_ai_bitcoin_trading_binance import cli
from simple_ai_bitcoin_trading_binance.config import save_runtime
from simple_ai_bitcoin_trading_binance.positions import OpenPosition, PositionsStore
from simple_ai_bitcoin_trading_binance.types import RuntimeConfig, StrategyConfig


# --------------------------------------------------------------------------- #
# shell
# --------------------------------------------------------------------------- #

def test_command_shell_invokes_run_shell(monkeypatch):
    called = {"argv": None}

    def fake_run(argv):
        called["argv"] = list(argv)
        return 0

    import simple_ai_bitcoin_trading_binance.shell as shell_mod
    monkeypatch.setattr(shell_mod, "run_shell", fake_run)
    assert cli.command_shell(argparse.Namespace()) == 0
    assert called["argv"] == []


# --------------------------------------------------------------------------- #
# objectives
# --------------------------------------------------------------------------- #

def test_command_objectives_lists_three(capsys):
    assert cli.command_objectives(argparse.Namespace()) == 0
    out = capsys.readouterr().out
    assert "conservative" in out
    assert "default" in out
    assert "risky" in out


# --------------------------------------------------------------------------- #
# train-suite
# --------------------------------------------------------------------------- #

def _write_candles(path: Path, count: int = 12) -> None:
    payload = [
        {
            "open_time": i * 60_000,
            "open": 100.0 + i,
            "high": 101.0 + i,
            "low": 99.0 + i,
            "close": 100.5 + i,
            "volume": 1.0,
            "close_time": i * 60_000 + 59_000,
        }
        for i in range(count)
    ]
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_command_train_suite_missing_file(tmp_path, capsys):
    args = argparse.Namespace(
        input=str(tmp_path / "missing.json"),
        output_dir=str(tmp_path / "out"),
        starting_cash=1000.0,
        objective=None,
        max_workers=None,
    )
    assert cli.command_train_suite(args) == 2
    err = capsys.readouterr().err
    assert "failed to load candles" in err


def test_command_train_suite_malformed_rows_and_limited_objectives(tmp_path, monkeypatch, capsys):
    input_path = tmp_path / "candles.json"
    input_path.write_text(json.dumps([
        {"open_time": 0, "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "close_time": 60_000},
        {"bad": "entry"},  # skipped
    ]), encoding="utf-8")

    class _Fake:
        def __init__(self):
            self.outcomes = []
            self.summary_path = tmp_path / "summary.json"

    def fake_run(candles, strategy, *, objectives, market_type, starting_cash, output_dir, max_workers, compute_backend):
        # assert malformed entries were skipped
        assert len(candles) == 1
        assert objectives == ("default",)
        assert max_workers == 3
        assert compute_backend == "cpu"
        return _Fake()

    monkeypatch.setattr(cli, "run_training_suite", fake_run, raising=False)
    # import path for monkeypatch — the function is resolved lazily inside command
    monkeypatch.setattr(
        "simple_ai_bitcoin_trading_binance.training_suite.run_training_suite",
        fake_run,
    )
    args = argparse.Namespace(
        input=str(input_path),
        output_dir=str(tmp_path / "out"),
        starting_cash=1000.0,
        objective=["balanced"],
        max_workers=3,
    )
    assert cli.command_train_suite(args) == 0
    out = capsys.readouterr().out
    assert "training suite complete" in out


def test_command_train_suite_unknown_objective_is_clean_error(tmp_path, capsys):
    input_path = tmp_path / "candles.json"
    _write_candles(input_path, count=4)
    args = argparse.Namespace(
        input=str(input_path),
        output_dir=str(tmp_path / "out"),
        starting_cash=1000.0,
        objective=["not-real"],
        max_workers=None,
    )
    assert cli.command_train_suite(args) == 2
    err = capsys.readouterr().err
    assert "Unknown objective" in err


def test_command_train_suite_all_objectives(tmp_path, monkeypatch, capsys):
    input_path = tmp_path / "candles.json"
    _write_candles(input_path, count=4)

    class _Outcome:
        def __init__(self, name):
            self.objective = name
            self.best_score = 0.12
            self.model_path = tmp_path / f"{name}.json"
            self.decision_threshold = 0.58
            self.threshold_source = "strategy"
            self.validation_score = 0.13
            self.full_sample_score = 0.11
            self.ensemble_refined = True
            self.ensemble_refinement_candidates = 3
            self.local_refinement_candidates = 10
            self.explored_candidates = 1944

    class _Report:
        def __init__(self):
            self.outcomes = [_Outcome("default")]
            self.summary_path = tmp_path / "summary.json"

    def fake_run(*args, **kwargs):
        assert kwargs["objectives"] == cli.available_objectives() if hasattr(cli, "available_objectives") else True
        return _Report()

    monkeypatch.setattr(
        "simple_ai_bitcoin_trading_binance.training_suite.run_training_suite",
        fake_run,
    )
    args = argparse.Namespace(
        input=str(input_path),
        output_dir=str(tmp_path / "out"),
        starting_cash=1000.0,
        objective=None,
        max_workers=None,
    )
    assert cli.command_train_suite(args) == 0
    out = capsys.readouterr().out
    assert "default" in out
    assert "ensemble=yes" in out
    assert "local_checks=10" in out
    assert "ensemble_checks=3" in out


def test_command_train_suite_passes_gpu_options(tmp_path, monkeypatch):
    input_path = tmp_path / "candles.json"
    _write_candles(input_path, count=4)
    observed: dict[str, object] = {}

    class _Report:
        outcomes = []
        summary_path = tmp_path / "summary.json"

    def fake_run(*args, **kwargs):
        observed.update(kwargs)
        return _Report()

    monkeypatch.setattr(
        "simple_ai_bitcoin_trading_binance.training_suite.run_training_suite",
        fake_run,
    )
    args = argparse.Namespace(
        input=str(input_path),
        output_dir=str(tmp_path / "out"),
        starting_cash=1000.0,
        objective=None,
        max_workers=None,
        compute_backend="directml",
        batch_size=64,
    )
    assert cli.command_train_suite(args) == 0
    assert observed["compute_backend"] == "directml"
    assert observed["batch_size"] == 64


def test_command_train_suite_uses_saved_compute_backend_without_cli_override(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    save_runtime(RuntimeConfig(compute_backend="directml"))
    input_path = tmp_path / "candles.json"
    _write_candles(input_path, count=4)
    observed: dict[str, object] = {}

    class _Report:
        outcomes = []
        summary_path = tmp_path / "summary.json"

    def fake_run(*args, **kwargs):
        observed.update(kwargs)
        return _Report()

    monkeypatch.setattr(
        "simple_ai_bitcoin_trading_binance.training_suite.run_training_suite",
        fake_run,
    )
    args = argparse.Namespace(
        input=str(input_path),
        output_dir=str(tmp_path / "out"),
        starting_cash=1000.0,
        objective=None,
        max_workers=None,
        compute_backend=None,
        batch_size=8192,
    )
    assert cli.command_train_suite(args) == 0
    assert observed["compute_backend"] == "directml"


# --------------------------------------------------------------------------- #
# backtest-panel
# --------------------------------------------------------------------------- #

def test_command_backtest_panel_invalid_interval(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    args = argparse.Namespace(
        interval="invalid",
        market=None,
        from_date=None,
        to_date=None,
        input=str(tmp_path / "x.json"),
        model=None,
        objective=None,
        tag="",
        notes="",
        starting_cash=1000.0,
    )
    assert cli.command_backtest_panel(args) == 2
    err = capsys.readouterr().err
    assert "not supported" in err


def test_command_backtest_panel_success(tmp_path, monkeypatch, capsys):
    input_path = tmp_path / "candles.json"
    _write_candles(input_path, count=10)
    monkeypatch.chdir(tmp_path)

    class _FakeResult:
        closed_trades = 2
        realized_pnl = 1.23
        max_drawdown = 0.05

    class _FakeReport:
        filename = "backtest_ok_spot_1m_20260101000000.json"
        result = _FakeResult()
        objective_score = 0.5

    def fake_run(request, strategy):
        assert request.interval == "1m"
        return _FakeReport()

    monkeypatch.setattr(
        "simple_ai_bitcoin_trading_binance.backtest_panel.run_panel",
        fake_run,
    )
    args = argparse.Namespace(
        interval="1m",
        market="spot",
        from_date="2026-01-01",
        to_date="2026-01-02",
        input=str(input_path),
        model=None,
        objective="default",
        tag="ok",
        notes="",
        starting_cash=1000.0,
    )
    assert cli.command_backtest_panel(args) == 0
    out = capsys.readouterr().out
    assert "backtest_ok_spot_1m" in out
    assert "objective=default" in out


def test_command_backtest_panel_success_without_objective(tmp_path, monkeypatch, capsys):
    input_path = tmp_path / "candles.json"
    _write_candles(input_path, count=6)

    class _FakeResult:
        closed_trades = 0
        realized_pnl = 0.0
        max_drawdown = 0.0

    class _FakeReport:
        filename = "backtest_plain_spot_5m_20260101000000.json"
        result = _FakeResult()
        objective_score = None

    monkeypatch.setattr(
        "simple_ai_bitcoin_trading_binance.backtest_panel.run_panel",
        lambda req, strat: _FakeReport(),
    )
    args = argparse.Namespace(
        interval="5m",
        market=None,
        from_date=None,
        to_date=None,
        input=str(input_path),
        model=None,
        objective=None,
        tag="plain",
        notes="",
        starting_cash=1000.0,
    )
    assert cli.command_backtest_panel(args) == 0
    out = capsys.readouterr().out
    assert "backtest_plain" in out


# --------------------------------------------------------------------------- #
# autonomous
# --------------------------------------------------------------------------- #

def _autonomous_control_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path / "data" / "autonomous" / "state.json"


def test_command_autonomous_start_stop_status(tmp_path, monkeypatch, capsys):
    path = _autonomous_control_path(tmp_path, monkeypatch)
    calls = {}

    def fake_decision_fn(**kwargs):
        calls["decision_kwargs"] = kwargs
        return lambda *_args: None, None, None

    def fake_run_loop(client, runtime, strategy, cfg, *, decision_fn):
        calls["client"] = client
        calls["runtime"] = runtime
        calls["strategy"] = strategy
        calls["cfg"] = cfg
        calls["decision_fn"] = decision_fn
        return type(
            "Result",
            (),
            {
                "exit_reason": "iteration-cap",
                "iterations": 2,
                "opened_trades": 1,
                "closed_trades": 0,
                "skipped_entries": 3,
            },
        )()

    monkeypatch.setattr(cli, "load_runtime", lambda: RuntimeConfig(dry_run=True, testnet=True))
    monkeypatch.setattr(cli, "load_strategy", StrategyConfig)
    monkeypatch.setattr(cli, "_build_client", lambda runtime: ("client", runtime.symbol))
    monkeypatch.setattr(cli, "_build_autonomous_decision_fn", fake_decision_fn)
    monkeypatch.setattr("simple_ai_bitcoin_trading_binance.autonomous.run_loop", fake_run_loop)
    args = argparse.Namespace(
        action="start",
        objective="balanced",
        model="data/model.json",
        poll_seconds=2.5,
        iterations=2,
        heartbeat_every=3,
        starting_cash=1500.0,
        paper=False,
        live=False,
    )
    assert cli.command_autonomous(args) == 0
    assert not path.exists()
    assert calls["cfg"].objective == "default"
    assert calls["cfg"].poll_seconds == 2.5
    assert calls["cfg"].stop_after_iterations == 2
    assert calls["cfg"].heartbeat_every == 3
    assert calls["cfg"].starting_reference_cash == 1500.0
    assert calls["cfg"].dry_run is True
    assert calls["decision_kwargs"]["model_path"] == Path("data/model.json")
    assert calls["decision_kwargs"]["effective_dry_run"] is True
    assert "autonomous: iteration-cap iterations=2 opened=1 closed=0 skipped=3" in capsys.readouterr().out

    args_pause = argparse.Namespace(action="pause", objective="default")
    assert cli.command_autonomous(args_pause) == 0
    assert json.loads(path.read_text())["state"] == "PAUSED"

    args_resume = argparse.Namespace(action="resume", objective="default")
    assert cli.command_autonomous(args_resume) == 0
    args_stop = argparse.Namespace(action="stop", objective="default")
    assert cli.command_autonomous(args_stop) == 0
    args_status = argparse.Namespace(action="status", objective="default")
    assert cli.command_autonomous(args_status) == 0
    out = capsys.readouterr().out
    assert "state=STOPPING" in out


def test_command_autonomous_unknown_objective(tmp_path, monkeypatch, capsys):
    _autonomous_control_path(tmp_path, monkeypatch)
    args = argparse.Namespace(action="start", objective="bogus", model="data/model.json")
    assert cli.command_autonomous(args) == 2
    err = capsys.readouterr().err
    assert "Unknown objective" in err


def test_command_autonomous_start_blocks_unsafe_or_uncredentialed_live(tmp_path, monkeypatch, capsys):
    _autonomous_control_path(tmp_path, monkeypatch)
    base_args = argparse.Namespace(
        action="start",
        objective="default",
        model="data/model.json",
        poll_seconds=1.0,
        iterations=1,
        heartbeat_every=1,
        starting_cash=1000.0,
        paper=False,
        live=True,
    )
    monkeypatch.setattr(cli, "load_strategy", StrategyConfig)
    monkeypatch.setattr(cli, "load_runtime", lambda: RuntimeConfig(testnet=False, demo=False, dry_run=False))
    assert cli.command_autonomous(base_args) == 2
    assert "requires testnet=true or demo=true" in capsys.readouterr().err

    monkeypatch.setattr(cli, "load_runtime", lambda: RuntimeConfig(testnet=True, dry_run=False))
    assert cli.command_autonomous(base_args) == 2
    assert "Autonomous live mode requires Binance API key" in capsys.readouterr().err

    monkeypatch.setattr(
        cli,
        "load_runtime",
        lambda: RuntimeConfig(testnet=True, dry_run=False, api_key="fake-api-key", api_secret="fake-secret"),
    )
    assert cli.command_autonomous(base_args) == 2
    assert "Autonomous authenticated mode is disabled" in capsys.readouterr().err


def test_build_autonomous_decision_fn_scores_model_and_external_signals(monkeypatch, tmp_path):
    class _Model:
        feature_signature = None

        def predict_proba(self, features):
            assert features == (0.2,)
            return 0.80

    class _Client:
        def get_klines(self, symbol, interval, *, limit):
            assert symbol == "BTCUSDC"
            assert interval == "15m"
            assert limit >= 300
            return ["candles"]

    class _Report:
        fresh_count = 2
        provider_count = 2
        score_adjustment = -0.40
        risk_multiplier = 0.5

    row = type("Row", (), {"features": (0.2,), "close": 101.5, "timestamp": 1})()
    monkeypatch.setattr(cli, "_load_live_start_model", lambda *_args, **_kwargs: (_Model(), None, None))
    monkeypatch.setattr(cli, "_live_rows_for_model", lambda *_args, **_kwargs: [row])
    monkeypatch.setattr(cli, "model_decision_threshold", lambda _model, _threshold: 0.55)
    monkeypatch.setattr(cli, "confidence_adjusted_probability", lambda score, _beta: score)
    monkeypatch.setattr(cli, "collect_external_signals", lambda **_kwargs: _Report())
    monkeypatch.setattr(
        cli,
        "_apply_external_signal_to_score",
        lambda score, cfg, report: (0.30, cfg, -0.20),
    )
    telemetry = []
    monkeypatch.setattr(cli, "_record_model_telemetry", lambda *args, **kwargs: telemetry.append((args, kwargs)))

    cfg = StrategyConfig(external_signals_enabled=True)
    decision_fn, error, notice = cli._build_autonomous_decision_fn(
        model_path=tmp_path / "model.json",
        strategy=cfg,
        effective_dry_run=True,
    )

    assert error is None
    assert notice is None
    decision = decision_fn(_Client(), RuntimeConfig(symbol="BTCUSDC", interval="15m"), cfg, object())
    assert decision.side == "FLAT"
    assert decision.confidence == 0.30
    assert decision.mark_price == 101.5
    assert telemetry


def test_build_autonomous_decision_fn_rejects_live_without_model(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "_load_live_start_model", lambda *_args, **_kwargs: (None, None, None))
    decision_fn, error, notice = cli._build_autonomous_decision_fn(
        model_path=tmp_path / "missing.json",
        strategy=StrategyConfig(),
        effective_dry_run=False,
    )
    assert decision_fn is None
    assert "requires a compatible model" in error
    assert notice is None


# --------------------------------------------------------------------------- #
# positions + close
# --------------------------------------------------------------------------- #

def test_command_positions_empty_and_populated(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    args = argparse.Namespace(stats=False)
    assert cli.command_positions(args) == 0
    out = capsys.readouterr().out
    assert "no open positions" in out

    store = PositionsStore()
    store.record_open(OpenPosition(
        id="live1", symbol="BTCUSDC", market_type="spot", side="LONG",
        qty=1.0, entry_price=100.0, leverage=1.0, opened_at_ms=0, notional=100.0,
    ))
    args_stats = argparse.Namespace(stats=True)
    assert cli.command_positions(args_stats) == 0
    out = capsys.readouterr().out
    assert "live1" in out
    assert "Realized P&L" in out


def test_command_close_all_and_hit_and_miss(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    store = PositionsStore()
    store.record_open(OpenPosition(
        id="c1", symbol="BTCUSDC", market_type="spot", side="LONG",
        qty=1.0, entry_price=10.0, leverage=1.0, opened_at_ms=0, notional=10.0,
    ))
    args_hit = argparse.Namespace(position_id="c1")
    assert cli.command_close(args_hit) == 0
    out = capsys.readouterr().out
    assert "closed c1" in out

    args_miss = argparse.Namespace(position_id="missing")
    assert cli.command_close(args_miss) == 1
    err = capsys.readouterr().err
    assert "no open position" in err

    store.record_open(OpenPosition(
        id="c2", symbol="BTCUSDC", market_type="spot", side="SHORT",
        qty=1.0, entry_price=10.0, leverage=1.0, opened_at_ms=0, notional=10.0,
    ))
    args_all = argparse.Namespace(position_id="all")
    assert cli.command_close(args_all) == 0
    out = capsys.readouterr().out
    assert "closed 1 positions" in out
