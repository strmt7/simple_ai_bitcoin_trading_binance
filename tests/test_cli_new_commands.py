"""Coverage tests for the new CLI subcommands added in the UX rework."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


from simple_ai_bitcoin_trading_binance import cli
from simple_ai_bitcoin_trading_binance.positions import OpenPosition, PositionsStore


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

    def fake_run(candles, strategy, *, objectives, market_type, starting_cash, output_dir, max_workers):
        # assert malformed entries were skipped
        assert len(candles) == 1
        assert objectives == ("default",)
        assert max_workers == 3
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
            self.model_path = Path(f"/tmp/{name}.json")
            self.decision_threshold = 0.58
            self.threshold_source = "strategy"
            self.explored_candidates = 432

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
    args = argparse.Namespace(action="start", objective="balanced")
    assert cli.command_autonomous(args) == 0
    assert path.exists()
    payload = json.loads(path.read_text())
    assert payload["state"] == "RUNNING"
    assert payload["note"] == "CLI start objective=default"

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
    args = argparse.Namespace(action="start", objective="bogus")
    assert cli.command_autonomous(args) == 2
    err = capsys.readouterr().err
    assert "Unknown objective" in err


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
