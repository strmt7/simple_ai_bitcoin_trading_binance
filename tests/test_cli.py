from __future__ import annotations

import argparse
import asyncio

from simple_ai_bitcoin_trading_binance.cli import (
    _artifact_summary,
    _filter_candles_for_time_window,
    _recent_artifacts,
    _build_order_notional,
    _build_live_model,
    _effective_leverage,
    _resolve_live_retrain_rows,
    _tui_actions,
    _show_account_overview,
    _show_recent_artifacts,
    _target_notional,
    command_live,
    command_strategy,
    main,
)
from simple_ai_bitcoin_trading_binance.config import load_strategy
from simple_ai_bitcoin_trading_binance.api import Candle, SymbolConstraints
from simple_ai_bitcoin_trading_binance.types import StrategyConfig


class _AsyncUI:
    def __init__(
        self,
        *,
        prompts: list[str] | None = None,
        confirms: list[bool] | None = None,
        forms: list[dict[str, str] | None] | None = None,
        multiselects: list[list[str] | None] | None = None,
    ) -> None:
        self._prompts = list(prompts or [])
        self._confirms = iter(confirms or [])
        self._forms = list(forms or [])
        self._multiselects = list(multiselects or [])
        self.logs: list[str] = []

    async def prompt(self, _label: str, _default: str = "") -> str:
        return self._prompts.pop(0)

    async def secret(self, _label: str, _default: str = "") -> str:
        return self._prompts.pop(0)

    async def confirm(self, _message: str) -> bool:
        return next(self._confirms)

    async def form(self, _title: str, fields) -> dict[str, str] | None:
        if self._forms:
            return self._forms.pop(0)
        return {field.key: self._prompts.pop(0) for field in fields}

    async def multi_select(self, _title: str, _options, _selected, *, help_text: str = ""):
        del help_text
        return self._multiselects.pop(0)

    async def run_blocking(self, func, *args, **kwargs):
        return func(*args, **kwargs)

    def append_log(self, text: str) -> None:
        self.logs.append(text)


def _action(title: str):
    for action in _tui_actions():
        if action.title == title:
            return action
    raise AssertionError(f"missing action: {title}")


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
        model_lookback=320,
        training_epochs=480,
        confidence_beta=0.92,
        feature_window_short=12,
        feature_window_long=48,
        max_trades_per_day=7,
        set_features="momentum_1,rsi",
        enable_feature=None,
        disable_feature=None,
    )
    result = command_strategy(args)
    assert result == 0
    updated = load_strategy()
    assert updated.risk_per_trade == 0.003
    assert updated.max_open_positions == 4
    assert updated.max_trades_per_day == 7
    assert updated.model_lookback == 320
    assert updated.training_epochs == 480
    assert updated.confidence_beta == 0.92
    assert updated.feature_windows == (12, 48)
    assert updated.enabled_features == ("momentum_1", "rsi")


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

    def fake_train(train_rows, *, epochs: int, **_kwargs):
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


def test_main_without_args_routes_to_menu(monkeypatch) -> None:
    called = {"menu": False}

    def fake_menu(_args):
        called["menu"] = True
        return 0

    monkeypatch.setattr("simple_ai_bitcoin_trading_binance.cli.command_menu", fake_menu)
    assert main([]) == 0
    assert called["menu"] is True


def test_command_menu_routes_to_tui_when_tty(monkeypatch) -> None:
    monkeypatch.setattr("simple_ai_bitcoin_trading_binance.cli.sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("simple_ai_bitcoin_trading_binance.cli.sys.stdout.isatty", lambda: True)
    monkeypatch.setattr("simple_ai_bitcoin_trading_binance.tui.launch_tui", lambda **_kwargs: 0)
    from simple_ai_bitcoin_trading_binance.cli import command_menu

    assert command_menu(argparse.Namespace()) == 0


def test_command_menu_rejects_without_tty(monkeypatch, capsys) -> None:
    monkeypatch.setattr("simple_ai_bitcoin_trading_binance.cli.sys.stdin.isatty", lambda: False)
    monkeypatch.setattr("simple_ai_bitcoin_trading_binance.cli.sys.stdout.isatty", lambda: False)
    from simple_ai_bitcoin_trading_binance.cli import command_menu

    assert command_menu(argparse.Namespace()) == 2
    assert "Interactive console requires a real terminal" in capsys.readouterr().err


def test_tui_runtime_action_saves_runtime(monkeypatch) -> None:
    current = type(
        "R",
        (),
        {
            "symbol": "BTCUSDC",
            "interval": "15m",
            "market_type": "spot",
            "testnet": True,
            "api_key": "old-key",
            "api_secret": "old-secret",
            "dry_run": True,
            "validate_account": True,
            "max_rate_calls_per_minute": 1100,
        },
    )()
    saved = {}

    monkeypatch.setattr("simple_ai_bitcoin_trading_binance.cli.load_runtime", lambda: current)
    monkeypatch.setattr("simple_ai_bitcoin_trading_binance.cli.save_runtime", lambda cfg: saved.setdefault("cfg", cfg) or cfg)
    monkeypatch.setattr("simple_ai_bitcoin_trading_binance.cli._build_client", lambda _runtime: type("C", (), {"ping": lambda self: None, "ensure_btcusdc": lambda self: None})())

    ui = _AsyncUI(
        forms=[
            {
                "market_type": "futures",
                "interval": "1h",
                "testnet": "yes",
                "api_key": "new-key",
                "api_secret": "new-secret",
                "dry_run": "no",
                "validate_account": "yes",
                "max_rate_calls_per_minute": "1500",
            }
        ]
    )
    result = asyncio.run(_action("Runtime settings").run(ui))

    assert result == 0
    assert saved["cfg"].market_type == "futures"
    assert saved["cfg"].interval == "1h"
    assert saved["cfg"].api_key == "new-key"
    assert saved["cfg"].dry_run is False
    assert saved["cfg"].max_rate_calls_per_minute == 1500


def test_tui_strategy_action_builds_full_strategy_args(monkeypatch) -> None:
    captured = {}
    monkeypatch.setattr("simple_ai_bitcoin_trading_binance.cli.load_strategy", lambda: StrategyConfig())

    def fake_strategy(args):
        captured["args"] = args
        return 0

    monkeypatch.setattr("simple_ai_bitcoin_trading_binance.cli.command_strategy", fake_strategy)

    ui = _AsyncUI(
        multiselects=[["momentum_1", "rsi"]],
        forms=[
            {
                "profile": "custom",
                "leverage": "3",
                "risk": "0.02",
                "max_position": "0.3",
                "stop": "0.01",
                "take": "0.05",
                "cooldown": "1",
                "max_open": "2",
                "max_trades_per_day": "7",
                "signal_threshold": "0.6",
                "max_drawdown": "0.2",
                "taker_fee_bps": "2",
                "slippage_bps": "4",
                "label_threshold": "0.002",
                "model_lookback": "300",
                "training_epochs": "500",
                "confidence_beta": "0.9",
                "feature_window_short": "12",
                "feature_window_long": "48",
            }
        ],
    )
    result = asyncio.run(_action("Strategy settings").run(ui))

    assert result == 0
    assert captured["args"].leverage == 3.0
    assert captured["args"].profile == "custom"
    assert captured["args"].feature_window_short == 12
    assert captured["args"].feature_window_long == 48
    assert captured["args"].training_epochs == 500
    assert captured["args"].confidence_beta == 0.9
    assert captured["args"].set_features == "momentum_1,rsi"


def test_tui_fetch_train_tune_and_backtest_actions_build_expected_args(monkeypatch) -> None:
    captured = {}
    monkeypatch.setattr(
        "simple_ai_bitcoin_trading_binance.cli.load_runtime",
        lambda: type("R", (), {"symbol": "BTCUSDC", "interval": "15m", "market_type": "spot"})(),
    )
    monkeypatch.setattr("simple_ai_bitcoin_trading_binance.cli.command_fetch", lambda args: captured.__setitem__("fetch", args) or 0)
    monkeypatch.setattr("simple_ai_bitcoin_trading_binance.cli.command_train", lambda args: captured.__setitem__("train", args) or 0)
    monkeypatch.setattr("simple_ai_bitcoin_trading_binance.cli.command_tune", lambda args: captured.__setitem__("tune", args) or 0)
    monkeypatch.setattr("simple_ai_bitcoin_trading_binance.cli.command_backtest", lambda args: captured.__setitem__("backtest", args) or 0)

    asyncio.run(
        _action("Fetch candles").run(
            _AsyncUI(forms=[{"limit": "400", "output": "tmp/fetch.json"}])
        )
    )
    asyncio.run(
        _action("Train model").run(
            _AsyncUI(
                forms=[
                    {
                        "input": "data/in.json",
                        "output": "data/out.json",
                        "preset": "custom",
                        "epochs": "99",
                        "seed": "11",
                        "walk_forward": "yes",
                        "walk_forward_train": "310",
                        "walk_forward_test": "70",
                        "walk_forward_step": "20",
                        "calibrate_threshold": "no",
                    }
                ]
            )
        )
    )
    asyncio.run(
        _action("Tune strategy").run(
            _AsyncUI(
                forms=[
                    {
                        "input": "data/tune.json",
                        "window_mode": "range",
                        "lookback_days": "30",
                        "from_date": "2024-01-01",
                        "to_date": "2024-02-01",
                        "save_best": "yes",
                        "min_risk": "0.003",
                        "max_risk": "0.03",
                        "steps": "4",
                        "min_leverage": "1",
                        "max_leverage": "10",
                        "min_threshold": "0.5",
                        "max_threshold": "0.8",
                        "min_take": "0.01",
                        "max_take": "0.04",
                        "min_stop": "0.01",
                        "max_stop": "0.03",
                    }
                ]
            )
        )
    )
    asyncio.run(
        _action("Backtest").run(
            _AsyncUI(forms=[{"input": "data/back.json", "model": "data/model.json", "start_cash": "1500"}])
        )
    )

    assert captured["fetch"].limit == 400
    assert captured["train"].walk_forward is True
    assert captured["train"].calibrate_threshold is False
    assert captured["train"].preset == "custom"
    assert captured["tune"].from_date == "2024-01-01"
    assert captured["tune"].to_date == "2024-02-01"
    assert captured["backtest"].start_cash == 1500.0


def test_tui_local_audit_action_builds_expected_args(monkeypatch) -> None:
    captured = {}
    monkeypatch.setattr("simple_ai_bitcoin_trading_binance.cli.command_audit", lambda args: captured.__setitem__("audit", args) or 0)

    result = asyncio.run(
        _action("Local audit").run(
            _AsyncUI(forms=[{"input": "data/history.json", "model": "data/model.json"}])
        )
    )

    assert result == 0
    assert captured["audit"].input == "data/history.json"
    assert captured["audit"].model == "data/model.json"


def test_tui_local_audit_action_cancelled(capsys) -> None:
    result = asyncio.run(_action("Local audit").run(_AsyncUI(forms=[None])))
    assert result == 0
    assert "Audit cancelled." in capsys.readouterr().out


def test_tui_evaluate_and_pipeline_actions(monkeypatch) -> None:
    captured = {"order": []}
    monkeypatch.setattr(
        "simple_ai_bitcoin_trading_binance.cli.load_runtime",
        lambda: type("R", (), {"symbol": "BTCUSDC", "interval": "15m", "market_type": "spot"})(),
    )
    monkeypatch.setattr("simple_ai_bitcoin_trading_binance.cli.command_evaluate", lambda args: captured.setdefault("evaluate", args) or 0)
    monkeypatch.setattr("simple_ai_bitcoin_trading_binance.cli.command_prepare", lambda args: captured["order"].append(("prepare", args.model, args.online_doctor)) or 0)
    monkeypatch.setattr("simple_ai_bitcoin_trading_binance.cli.command_evaluate", lambda args: captured["order"].append(("evaluate", args.model)) or 0)

    ui_eval = _AsyncUI(
        forms=[
            {
                "input": "data/eval.json",
                "model": "data/eval-model.json",
                "threshold": "0.61",
                "calibrate_threshold": "yes",
            }
        ]
    )
    asyncio.run(_action("Evaluate").run(ui_eval))

    ui_pipeline = _AsyncUI(
        forms=[
            {
                "historical": "tmp/history.json",
                "model": "tmp/model.json",
                "limit": "220",
                "preset": "quick",
                "epochs": "50",
                "seed": "7",
                "start_cash": "1000",
                "online_doctor": "yes",
            }
        ]
    )
    asyncio.run(_action("Prepare system").run(ui_pipeline))

    assert captured["order"] == [
        ("evaluate", "data/eval-model.json"),
        ("prepare", "tmp/model.json", True),
    ]


def test_tui_live_and_roundtrip_actions(monkeypatch, capsys) -> None:
    calls = []
    monkeypatch.setattr("simple_ai_bitcoin_trading_binance.cli.command_live", lambda args: calls.append(args) or 0)
    monkeypatch.setattr(
        "simple_ai_bitcoin_trading_binance.cli.load_runtime",
        lambda: type("R", (), {"market_type": "spot", "testnet": True, "api_key": "k", "api_secret": "s", "symbol": "BTCUSDC"})(),
    )
    monkeypatch.setattr(
        "simple_ai_bitcoin_trading_binance.cli._build_client",
        lambda _runtime: type(
            "C",
            (),
            {"place_order": lambda self, symbol, side, quantity, dry_run=False: {"status": "FILLED", "orderId": 1 if side == "BUY" else 2, "executedQty": quantity}},
        )(),
    )

    asyncio.run(
        _action("Paper loop").run(
            _AsyncUI(
                forms=[
                    {
                        "model": "data/model.json",
                        "steps": "3",
                        "sleep": "0",
                        "retrain_interval": "1",
                        "retrain_window": "120",
                        "retrain_min_rows": "100",
                    }
                ]
            )
        )
    )
    asyncio.run(
        _action("Testnet loop").run(
            _AsyncUI(
                forms=[
                    {
                        "model": "data/model.json",
                        "steps": "2",
                        "sleep": "0",
                        "retrain_interval": "0",
                        "retrain_window": "240",
                        "retrain_min_rows": "120",
                    }
                ],
                confirms=[True],
            )
        )
    )
    asyncio.run(
        _action("Testnet loop").run(
            _AsyncUI(
                forms=[
                    {
                        "model": "data/model.json",
                        "steps": "2",
                        "sleep": "0",
                        "retrain_interval": "0",
                        "retrain_window": "240",
                        "retrain_min_rows": "120",
                    }
                ],
                confirms=[False],
            )
        )
    )
    asyncio.run(
        _action("Spot roundtrip").run(
            _AsyncUI(forms=[{"quantity": "0.00008"}], confirms=[True])
        )
    )

    assert calls[0].paper is True and calls[0].steps == 3
    assert calls[1].live is True and calls[1].steps == 2
    assert calls[0].model == "data/model.json"
    assert "Spot test roundtrip complete." in capsys.readouterr().out


def test_tui_help_action_writes_operator_help() -> None:
    ui = _AsyncUI()
    result = asyncio.run(_action("Help").run(ui))
    assert result == 0
    assert any("Operator help" in entry for entry in ui.logs)


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


def test_filter_candles_for_time_window_handles_lookback_and_ranges() -> None:
    candles = [
        Candle(open_time=1_700_000_000_000, open=1.0, high=1.0, low=1.0, close=1.0, volume=1.0, close_time=1_700_000_059_999),
        Candle(open_time=1_700_086_400_000, open=1.0, high=1.0, low=1.0, close=1.0, volume=1.0, close_time=1_700_086_459_999),
        Candle(open_time=1_700_172_800_000, open=1.0, high=1.0, low=1.0, close=1.0, volume=1.0, close_time=1_700_172_859_999),
    ]
    lookback = _filter_candles_for_time_window(candles, lookback_days=1)
    assert len(lookback) == 1

    ranged = _filter_candles_for_time_window(candles, from_date="2023-11-15", to_date="2023-11-15")
    assert len(ranged) == 1


def test_filter_candles_for_time_window_rejects_invalid_inputs() -> None:
    candles = []
    try:
        _filter_candles_for_time_window(candles, lookback_days=0)
    except ValueError as exc:
        assert "lookback_days" in str(exc)
    else:
        raise AssertionError("expected ValueError")

    try:
        _filter_candles_for_time_window(candles, lookback_days=5, from_date="2024-01-01")
    except ValueError as exc:
        assert "cannot be combined" in str(exc)
    else:
        raise AssertionError("expected ValueError")

    try:
        _filter_candles_for_time_window(candles, from_date="2024-02-01", to_date="2024-01-01")
    except ValueError as exc:
        assert "from_date must be <=" in str(exc)
    else:
        raise AssertionError("expected ValueError")
