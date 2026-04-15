from __future__ import annotations

import argparse

from simple_ai_bitcoin_trading_binance.cli import (
    _artifact_summary,
    _build_strategy_menu_args,
    _menu_prompt_bool,
    _menu_prompt_float,
    _menu_prompt_int,
    _recent_artifacts,
    _build_order_notional,
    _build_live_model,
    _effective_leverage,
    _resolve_live_retrain_rows,
    _run_offline_pipeline,
    _run_menu,
    _run_spot_test_order_roundtrip,
    _show_account_overview,
    _show_recent_artifacts,
    _target_notional,
    command_live,
    command_strategy,
    main,
)
from simple_ai_bitcoin_trading_binance.config import load_strategy
from simple_ai_bitcoin_trading_binance.api import SymbolConstraints
from simple_ai_bitcoin_trading_binance.types import StrategyConfig


def test_effective_leverage_clamps_by_market() -> None:
    cfg = StrategyConfig(leverage=250.0)
    assert _effective_leverage(cfg, "spot") == 1.0
    assert _effective_leverage(cfg, "futures") == 125.0


def test_target_notional_scales_with_futures_leverage() -> None:
    cfg = StrategyConfig(leverage=20.0, risk_per_trade=0.01, max_position_pct=0.2)
    spot_notional = _target_notional(1000.0, cfg, "spot")
    futures_notional = _target_notional(1000.0, cfg, "futures")
    assert spot_notional == 10.0
    assert futures_notional == 200.0


class _ConstraintClient:
    def __init__(self, constraints: SymbolConstraints) -> None:
        self.constraints = constraints

    def normalize_quantity(self, symbol: str, quantity: float):
        if symbol != self.constraints.symbol:
            return 0.0, self.constraints
        if quantity <= 0:
            return 0.0, self.constraints
        if quantity < self.constraints.min_qty:
            return 0.0, self.constraints
        step = self.constraints.step_size
        quantized = int(quantity / step) * step
        if quantized > self.constraints.max_qty > 0:
            quantized = self.constraints.max_qty
        return quantized, self.constraints


def test_build_order_notional_respects_symbol_constraints() -> None:
    cfg = StrategyConfig(leverage=2.0, risk_per_trade=0.5, max_position_pct=0.75)
    constraints = SymbolConstraints(
        symbol="BTCUSDC",
        min_qty=0.5,
        max_qty=2.0,
        step_size=0.5,
        min_notional=300.0,
        max_notional=700.0,
    )
    client = _ConstraintClient(constraints)

    notional, qty = _build_order_notional(
        cash=1000.0,
        price=500.0,
        cfg=cfg,
        market_type="futures",
        leverage=2.0,
        client=client,
        constraints=constraints,
    )
    assert notional == 500.0
    assert qty == 1.0

    constraints = SymbolConstraints(
        symbol="BTCUSDC",
        min_qty=2.0,
        max_qty=5.0,
        step_size=0.5,
        min_notional=1200.0,
        max_notional=3000.0,
    )
    client = _ConstraintClient(constraints)

    notional, qty = _build_order_notional(
        cash=1000.0,
        price=500.0,
        cfg=cfg,
        market_type="futures",
        leverage=2.0,
        client=client,
        constraints=constraints,
    )
    assert notional == 0.0
    assert qty == 0.0


def test_command_strategy_updates_risk_and_rate_limits(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    cfg = StrategyConfig()
    # ensure baseline config exists
    from simple_ai_bitcoin_trading_binance.config import save_strategy

    save_strategy(cfg)

    args = argparse.Namespace(
        leverage=None,
        risk=0.003,
        max_position=None,
        stop=None,
        take=None,
        cooldown=None,
        max_open=4,
        signal_threshold=None,
        max_drawdown=None,
        taker_fee_bps=None,
        slippage_bps=None,
        label_threshold=None,
        max_trades_per_day=7,
    )
    result = command_strategy(args)
    assert result == 0
    updated = load_strategy()
    assert updated.risk_per_trade == 0.003
    assert updated.max_open_positions == 4
    assert updated.max_trades_per_day == 7


def test_resolve_live_retrain_rows_handles_short_full_and_tail_windows() -> None:
    rows = list(range(10))
    assert _resolve_live_retrain_rows(rows[:3], retrain_window=5, retrain_min_rows=4) == []
    assert _resolve_live_retrain_rows(rows[:5], retrain_window=5, retrain_min_rows=4) == rows[:5]
    assert _resolve_live_retrain_rows(rows, retrain_window=4, retrain_min_rows=4) == rows[-4:]


def test_build_live_model_respects_existing_model_and_retrain_cadence(monkeypatch) -> None:
    cfg = StrategyConfig(training_epochs=100)
    rows = list(range(12))
    existing = object()

    assert _build_live_model(
        rows,
        model=existing,
        retrain_every=0,
        step=5,
        cfg=cfg,
        retrain_window=10,
        retrain_min_rows=4,
    ) is existing

    assert _build_live_model(
        rows,
        model=existing,
        retrain_every=3,
        step=4,
        cfg=cfg,
        retrain_window=10,
        retrain_min_rows=4,
    ) is existing

    calls: list[tuple[list[int], int]] = []

    def fake_train(train_rows, *, epochs: int):
        calls.append((list(train_rows), epochs))
        return {"trained": True}

    monkeypatch.setattr("simple_ai_bitcoin_trading_binance.cli.train", fake_train)
    rebuilt = _build_live_model(
        rows,
        model=existing,
        retrain_every=2,
        step=4,
        cfg=cfg,
        retrain_window=5,
        retrain_min_rows=4,
    )
    assert rebuilt == {"trained": True}
    assert calls == [(rows[-5:], 40)]


def test_build_live_model_returns_existing_when_rows_insufficient(monkeypatch) -> None:
    cfg = StrategyConfig(training_epochs=50)
    existing = object()
    called = {"value": False}

    def fake_train(*_args, **_kwargs):
        called["value"] = True
        return {"trained": True}

    monkeypatch.setattr("simple_ai_bitcoin_trading_binance.cli.train", fake_train)
    rebuilt = _build_live_model(
        [1, 2, 3],
        model=existing,
        retrain_every=1,
        step=1,
        cfg=cfg,
        retrain_window=10,
        retrain_min_rows=5,
    )
    assert rebuilt is existing
    assert called["value"] is False


def test_build_strategy_menu_args_retries_invalid_numeric_input() -> None:
    cfg = StrategyConfig()
    responses = iter([
        "abc",
        "2",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
    ])

    args = _build_strategy_menu_args(cfg, input_fn=lambda _prompt: next(responses))
    assert args.leverage == 2.0
    assert args.risk == cfg.risk_per_trade
    assert args.max_trades_per_day == cfg.max_trades_per_day


def test_run_menu_invalid_selection_then_exit(capsys) -> None:
    responses = iter(["99", "16"])
    result = _run_menu(input_fn=lambda _prompt: next(responses))
    output = capsys.readouterr().out
    assert result == 0
    assert "Invalid selection." in output
    assert "Exiting menu." in output


def test_run_menu_live_testnet_requires_explicit_confirmation(monkeypatch, capsys) -> None:
    called = {"live": False}

    def fake_live(_args):
        called["live"] = True
        return 0

    monkeypatch.setattr("simple_ai_bitcoin_trading_binance.cli.command_live", fake_live)
    responses = iter(["12", "nope", "16"])
    result = _run_menu(input_fn=lambda _prompt: next(responses))
    output = capsys.readouterr().out
    assert result == 0
    assert called["live"] is False
    assert "Testnet execution cancelled." in output


def test_main_without_args_routes_to_menu(monkeypatch) -> None:
    called = {"menu": False}

    def fake_menu(_args):
        called["menu"] = True
        return 0

    monkeypatch.setattr("simple_ai_bitcoin_trading_binance.cli.command_menu", fake_menu)
    assert main([]) == 0
    assert called["menu"] is True


def test_menu_prompt_int_and_float_retry_until_valid(capsys) -> None:
    int_answers = iter(["bad", "-1", "4"])
    float_answers = iter(["bad", "2.5"])
    assert _menu_prompt_int("Value", 1, minimum=0, input_fn=lambda _prompt: next(int_answers)) == 4
    assert _menu_prompt_float("Float", 1.0, minimum=0.0, maximum=3.0, input_fn=lambda _prompt: next(float_answers)) == 2.5
    output = capsys.readouterr().out
    assert "Enter a whole number." in output
    assert "Value must be >= 0." in output
    assert "Enter a numeric value." in output


def test_menu_prompt_bool_retries_invalid_answer(capsys) -> None:
    answers = iter(["maybe", "y"])
    assert _menu_prompt_bool("Proceed", False, input_fn=lambda _prompt: next(answers)) is True
    assert "Enter y or n." in capsys.readouterr().out


def test_run_menu_status_then_exit(monkeypatch) -> None:
    called = {"status": 0}

    def fake_status(_args):
        called["status"] += 1
        return 0

    monkeypatch.setattr("simple_ai_bitcoin_trading_binance.cli.command_status", fake_status)
    responses = iter(["1", "16"])
    assert _run_menu(input_fn=lambda _prompt: next(responses)) == 0
    assert called["status"] == 1


def test_run_menu_configure_and_connect_then_exit(monkeypatch) -> None:
    called = {"configure": 0, "connect": 0}

    monkeypatch.setattr(
        "simple_ai_bitcoin_trading_binance.cli.command_configure",
        lambda _args: called.__setitem__("configure", called["configure"] + 1) or 0,
    )
    monkeypatch.setattr(
        "simple_ai_bitcoin_trading_binance.cli.command_connect",
        lambda _args: called.__setitem__("connect", called["connect"] + 1) or 0,
    )
    responses = iter(["2", "4", "16"])
    assert _run_menu(input_fn=lambda _prompt: next(responses)) == 0
    assert called == {"configure": 1, "connect": 1}


def test_run_menu_fetch_builds_expected_args(monkeypatch) -> None:
    captured = {}

    monkeypatch.setattr(
        "simple_ai_bitcoin_trading_binance.cli.load_runtime",
        lambda: type(
            "R",
            (),
            {"symbol": "BTCUSDC", "interval": "15m", "market_type": "spot", "testnet": True, "dry_run": True},
        )(),
    )

    def fake_fetch(args):
        captured["args"] = args
        return 0

    monkeypatch.setattr("simple_ai_bitcoin_trading_binance.cli.command_fetch", fake_fetch)
    responses = iter(["6", "400", "tmp/fetch.json", "16"])
    assert _run_menu(input_fn=lambda _prompt: next(responses)) == 0
    assert captured["args"].limit == 400
    assert captured["args"].output == "tmp/fetch.json"
    assert captured["args"].symbol == "BTCUSDC"


def test_run_menu_train_and_tune_build_expected_args(monkeypatch) -> None:
    captured = {}

    def fake_train(args):
        captured["train"] = args
        return 0

    def fake_tune(args):
        captured["tune"] = args
        return 0

    monkeypatch.setattr("simple_ai_bitcoin_trading_binance.cli.command_train", fake_train)
    monkeypatch.setattr("simple_ai_bitcoin_trading_binance.cli.command_tune", fake_tune)
    responses = iter([
        "7",
        "data/in.json",
        "data/out.json",
        "99",
        "11",
        "y",
        "310",
        "70",
        "20",
        "n",
        "8",
        "data/tune.json",
        "y",
        "0.003",
        "0.03",
        "4",
        "1",
        "10",
        "0.5",
        "0.8",
        "0.01",
        "0.04",
        "0.01",
        "0.03",
        "16",
    ])
    assert _run_menu(input_fn=lambda _prompt: next(responses)) == 0
    assert captured["train"].input == "data/in.json"
    assert captured["train"].output == "data/out.json"
    assert captured["train"].epochs == 99
    assert captured["train"].seed == 11
    assert captured["train"].walk_forward is True
    assert captured["train"].calibrate_threshold is False
    assert captured["tune"].input == "data/tune.json"
    assert captured["tune"].save_best is True
    assert captured["tune"].steps == 4


def test_run_menu_backtest_and_evaluate_build_expected_args(monkeypatch) -> None:
    captured = {}

    monkeypatch.setattr("simple_ai_bitcoin_trading_binance.cli.command_backtest", lambda args: captured.setdefault("backtest", args) or 0)
    monkeypatch.setattr("simple_ai_bitcoin_trading_binance.cli.command_evaluate", lambda args: captured.setdefault("evaluate", args) or 0)
    responses = iter([
        "9",
        "data/back.json",
        "data/model.json",
        "1500",
        "10",
        "0.61",
        "data/eval.json",
        "data/eval-model.json",
        "y",
        "16",
    ])
    assert _run_menu(input_fn=lambda _prompt: next(responses)) == 0
    assert captured["backtest"].start_cash == 1500.0
    assert captured["evaluate"].threshold == 0.61
    assert captured["evaluate"].calibrate_threshold is True


def test_run_menu_strategy_builds_expected_args(monkeypatch) -> None:
    captured = {}

    monkeypatch.setattr("simple_ai_bitcoin_trading_binance.cli.load_strategy", lambda: StrategyConfig())
    monkeypatch.setattr("simple_ai_bitcoin_trading_binance.cli.command_strategy", lambda args: captured.setdefault("strategy", args) or 0)
    responses = iter([
        "3",
        "2",
        "0.02",
        "0.3",
        "0.01",
        "0.05",
        "1",
        "2",
        "7",
        "0.6",
        "0.2",
        "2",
        "4",
        "0.002",
        "16",
    ])
    assert _run_menu(input_fn=lambda _prompt: next(responses)) == 0
    assert captured["strategy"].leverage == 2.0
    assert captured["strategy"].max_trades_per_day == 7
    assert captured["strategy"].label_threshold == 0.002


def test_run_menu_live_variants_build_expected_args(monkeypatch) -> None:
    calls = []

    def fake_live(args):
        calls.append(args)
        return 0

    monkeypatch.setattr("simple_ai_bitcoin_trading_binance.cli.command_live", fake_live)
    responses = iter([
        "11",
        "3",
        "0",
        "1",
        "120",
        "100",
        "12",
        "TESTNET",
        "2",
        "0",
        "0",
        "240",
        "120",
        "16",
    ])
    assert _run_menu(input_fn=lambda _prompt: next(responses)) == 0
    assert len(calls) == 2
    assert calls[0].paper is True
    assert calls[0].live is False
    assert calls[0].steps == 3
    assert calls[1].paper is False
    assert calls[1].live is True
    assert calls[1].steps == 2


def test_recent_artifacts_and_summary(tmp_path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text('{"command":"train","timestamp":1,"runtime":{"symbol":"BTCUSDC","market_type":"spot"}}', encoding="utf-8")
    second.write_text('{"command":"backtest","timestamp":2,"runtime":{"symbol":"BTCUSDC","market_type":"spot"}}', encoding="utf-8")
    artifacts = _recent_artifacts(base_dir=tmp_path, limit=2)
    assert len(artifacts) == 2
    assert _artifact_summary(second).startswith("second.json command=backtest")


def test_show_recent_artifacts_handles_empty_and_populated(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setattr("simple_ai_bitcoin_trading_binance.cli._recent_artifacts", lambda: [])
    assert _show_recent_artifacts() == 0
    assert "No recent artifacts" in capsys.readouterr().out

    payload = tmp_path / "artifact.json"
    payload.write_text('{"command":"evaluate","timestamp":3,"runtime":{"symbol":"BTCUSDC","market_type":"spot"}}', encoding="utf-8")
    monkeypatch.setattr("simple_ai_bitcoin_trading_binance.cli._recent_artifacts", lambda: [payload])
    assert _show_recent_artifacts() == 0
    assert "Recent artifacts:" in capsys.readouterr().out


def test_show_account_overview_handles_missing_credentials(monkeypatch, capsys) -> None:
    monkeypatch.setattr("simple_ai_bitcoin_trading_binance.cli.load_runtime", lambda: type("R", (), {"api_key": "", "api_secret": ""})())
    assert _show_account_overview() == 2
    assert "No API credentials configured." in capsys.readouterr().out


def test_show_account_overview_prints_balances(monkeypatch, capsys) -> None:
    runtime = type("R", (), {"api_key": "k", "api_secret": "s", "market_type": "spot", "testnet": True})()

    class _Client:
        def get_account(self):
            return {
                "balances": [
                    {"asset": "BTC", "free": "1.00000000", "locked": "0.00000000"},
                    {"asset": "USDC", "free": "10000.00000000", "locked": "0.00000000"},
                ]
            }

    monkeypatch.setattr("simple_ai_bitcoin_trading_binance.cli.load_runtime", lambda: runtime)
    monkeypatch.setattr("simple_ai_bitcoin_trading_binance.cli._build_client", lambda _runtime: _Client())
    assert _show_account_overview() == 0
    output = capsys.readouterr().out
    assert "Account overview" in output
    assert "BTC" in output
    assert "USDC" in output


def test_run_offline_pipeline_invokes_commands_in_order(monkeypatch) -> None:
    calls = []

    monkeypatch.setattr(
        "simple_ai_bitcoin_trading_binance.cli.load_runtime",
        lambda: type("R", (), {"symbol": "BTCUSDC", "interval": "15m"})(),
    )
    monkeypatch.setattr("simple_ai_bitcoin_trading_binance.cli.command_fetch", lambda args: calls.append(("fetch", args.output)) or 0)
    monkeypatch.setattr("simple_ai_bitcoin_trading_binance.cli.command_train", lambda args: calls.append(("train", args.output)) or 0)
    monkeypatch.setattr("simple_ai_bitcoin_trading_binance.cli.command_backtest", lambda args: calls.append(("backtest", args.model)) or 0)
    monkeypatch.setattr("simple_ai_bitcoin_trading_binance.cli.command_evaluate", lambda args: calls.append(("evaluate", args.model)) or 0)

    responses = iter([
        "tmp/history.json",
        "tmp/model.json",
        "220",
        "50",
        "7",
        "y",
        "300",
        "60",
        "30",
        "y",
        "1000",
    ])
    assert _run_offline_pipeline(input_fn=lambda _prompt: next(responses)) == 0
    assert [name for name, _value in calls] == ["fetch", "train", "backtest", "evaluate"]


def test_run_spot_test_order_roundtrip_guards_and_executes(monkeypatch, capsys) -> None:
    runtime = type(
        "R",
        (),
        {"market_type": "spot", "testnet": True, "api_key": "k", "api_secret": "s", "symbol": "BTCUSDC"},
    )()
    calls = []

    class _Client:
        def place_order(self, symbol, side, quantity, dry_run):
            calls.append((symbol, side, quantity, dry_run))
            return {"status": "FILLED", "orderId": len(calls), "executedQty": quantity}

    monkeypatch.setattr("simple_ai_bitcoin_trading_binance.cli.load_runtime", lambda: runtime)
    monkeypatch.setattr("simple_ai_bitcoin_trading_binance.cli._build_client", lambda _runtime: _Client())
    answers = iter(["ROUNDTRIP", "0.00008"])
    assert _run_spot_test_order_roundtrip(input_fn=lambda _prompt: next(answers)) == 0
    assert calls == [
        ("BTCUSDC", "BUY", 0.00008, False),
        ("BTCUSDC", "SELL", 0.00008, False),
    ]
    assert "Spot test roundtrip complete." in capsys.readouterr().out


def test_run_menu_new_options(monkeypatch) -> None:
    calls = {"account": 0, "pipeline": 0, "artifacts": 0, "roundtrip": 0}
    monkeypatch.setattr("simple_ai_bitcoin_trading_binance.cli._show_account_overview", lambda: calls.__setitem__("account", calls["account"] + 1) or 0)
    monkeypatch.setattr("simple_ai_bitcoin_trading_binance.cli._run_offline_pipeline", lambda input_fn: calls.__setitem__("pipeline", calls["pipeline"] + 1) or 0)
    monkeypatch.setattr("simple_ai_bitcoin_trading_binance.cli._show_recent_artifacts", lambda: calls.__setitem__("artifacts", calls["artifacts"] + 1) or 0)
    monkeypatch.setattr("simple_ai_bitcoin_trading_binance.cli._run_spot_test_order_roundtrip", lambda input_fn: calls.__setitem__("roundtrip", calls["roundtrip"] + 1) or 0)
    monkeypatch.setattr("simple_ai_bitcoin_trading_binance.cli.load_runtime", lambda: type("R", (), {"symbol": "BTCUSDC", "interval": "15m", "market_type": "spot", "testnet": True, "dry_run": True})())
    monkeypatch.setattr("simple_ai_bitcoin_trading_binance.cli.load_strategy", lambda: StrategyConfig())
    responses = iter(["5", "13", "14", "15", "16"])
    assert _run_menu(input_fn=lambda _prompt: next(responses)) == 0
    assert calls == {"account": 1, "pipeline": 1, "artifacts": 1, "roundtrip": 1}


def test_command_live_rejects_conflicting_force_modes(capsys) -> None:
    args = argparse.Namespace(
        paper=True,
        live=True,
        leverage=None,
        steps=1,
        sleep=0,
        retrain_interval=0,
        retrain_window=10,
        retrain_min_rows=5,
    )
    assert command_live(args) == 2
    assert "Choose either --paper or --live" in capsys.readouterr().out
