"""Tests for the Funds allocation and Settings hub flows."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from simple_ai_bitcoin_trading_binance.cli import (
    _apply_funds_change,
    _funds_summary,
    _tui_actions,
    _ui_funds_menu,
    _ui_settings_menu,
)
from simple_ai_bitcoin_trading_binance.config import (
    load_runtime,
    save_runtime,
    save_strategy,
)
from simple_ai_bitcoin_trading_binance.types import RuntimeConfig, StrategyConfig


class _ScriptedUI:
    """Records every UI call and replays scripted answers."""

    def __init__(
        self,
        *,
        menu_choices: list[str | None] | None = None,
        forms: list[dict[str, str] | None] | None = None,
        confirms: list[bool] | None = None,
    ) -> None:
        self.menu_choices = list(menu_choices or [])
        self.forms = list(forms or [])
        self.confirms = list(confirms or [])
        self.menu_calls: list[tuple[str, list[tuple[str, str]]]] = []
        self.form_calls: list[tuple[str, list[Any]]] = []
        self.confirm_calls: list[str] = []
        self.logs: list[str] = []

    async def menu(self, title, options, *, help_text=""):
        self.menu_calls.append((title, list(options)))
        if not self.menu_choices:
            return None
        return self.menu_choices.pop(0)

    async def form(self, title, fields):
        self.form_calls.append((title, list(fields)))
        if not self.forms:
            return None
        return self.forms.pop(0)

    async def confirm(self, message):
        self.confirm_calls.append(message)
        if not self.confirms:
            return False
        return self.confirms.pop(0)

    def append_log(self, text: str) -> None:
        self.logs.append(text)

    async def run_blocking(self, func, *args, **kwargs):
        return func(*args, **kwargs)


def _action(title: str):
    for action in _tui_actions():
        if action.title == title:
            return action
    raise AssertionError(f"missing action: {title}")


@pytest.fixture()
def isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    save_runtime(RuntimeConfig())
    save_strategy(StrategyConfig())
    return tmp_path


def test_funds_summary_describes_current_allocation() -> None:
    cfg = RuntimeConfig(managed_usdc=1234.5, managed_btc=0.001)
    summary = _funds_summary(cfg)
    assert "USDC available: 1234.5000" in summary
    assert "BTC available: 0.00100000" in summary


def test_apply_funds_change_deposits_and_withdraws(isolated_home) -> None:
    save_runtime(RuntimeConfig(managed_usdc=100.0, managed_btc=0.0))
    after, msg = _apply_funds_change("deposit_usdc", 250.0)
    assert after.managed_usdc == 350.0
    assert "Deposited" in msg

    after, msg = _apply_funds_change("withdraw_usdc", 1000.0)
    assert after.managed_usdc == 0.0  # capped to available
    assert "capped" in msg

    after, msg = _apply_funds_change("deposit_btc", 0.01)
    assert after.managed_btc == 0.01

    after, msg = _apply_funds_change("withdraw_btc", 0.005)
    assert pytest.approx(after.managed_btc, abs=1e-12) == 0.005


def test_apply_funds_change_reset_restores_defaults(isolated_home) -> None:
    save_runtime(RuntimeConfig(managed_usdc=42.0, managed_btc=0.5))
    after, msg = _apply_funds_change("reset", 0.0)
    assert after.managed_usdc == 1000.0
    assert after.managed_btc == 0.0
    assert "Reset" in msg


def test_apply_funds_change_unknown_action_no_op(isolated_home) -> None:
    save_runtime(RuntimeConfig(managed_usdc=10.0))
    after, msg = _apply_funds_change("teleport", 1.0)
    assert after.managed_usdc == 10.0
    assert "Unknown" in msg


def test_funds_menu_close_returns_zero(isolated_home) -> None:
    ui = _ScriptedUI(menu_choices=["close"])
    result = asyncio.run(_ui_funds_menu(ui))
    assert result == 0
    assert ui.menu_calls[0][0].startswith("Funds")


def test_funds_menu_show_logs_summary_then_closes(isolated_home) -> None:
    ui = _ScriptedUI(menu_choices=["show", "close"])
    result = asyncio.run(_ui_funds_menu(ui))
    assert result == 0
    assert any("USDC available" in line for line in ui.logs)


def test_funds_menu_deposit_usdc_persists_change(isolated_home) -> None:
    ui = _ScriptedUI(
        menu_choices=["deposit_usdc", "close"],
        forms=[{"amount": "250"}],
    )
    asyncio.run(_ui_funds_menu(ui))
    assert load_runtime().managed_usdc == pytest.approx(1250.0)


def test_funds_menu_zero_amount_rejected(isolated_home) -> None:
    ui = _ScriptedUI(
        menu_choices=["deposit_usdc", "close"],
        forms=[{"amount": "0"}],
    )
    asyncio.run(_ui_funds_menu(ui))
    assert any("Amount must be" in line for line in ui.logs)


def test_funds_menu_invalid_amount_rejected(isolated_home) -> None:
    ui = _ScriptedUI(
        menu_choices=["deposit_btc", "close"],
        forms=[{"amount": "not-a-number"}],
    )
    asyncio.run(_ui_funds_menu(ui))
    assert any("rejected" in line.lower() for line in ui.logs)


def test_funds_menu_reset_requires_confirmation(isolated_home) -> None:
    save_runtime(RuntimeConfig(managed_usdc=42.0, managed_btc=0.5))
    ui_decline = _ScriptedUI(menu_choices=["reset", "close"], confirms=[False])
    asyncio.run(_ui_funds_menu(ui_decline))
    assert load_runtime().managed_usdc == 42.0  # unchanged on decline

    ui_accept = _ScriptedUI(menu_choices=["reset", "close"], confirms=[True])
    asyncio.run(_ui_funds_menu(ui_accept))
    assert load_runtime().managed_usdc == 1000.0


def test_funds_menu_deposit_form_cancel_keeps_balance(isolated_home) -> None:
    ui = _ScriptedUI(
        menu_choices=["deposit_usdc", "close"],
        forms=[None],
    )
    asyncio.run(_ui_funds_menu(ui))
    assert load_runtime().managed_usdc == 1000.0


def test_settings_menu_close_returns_zero(isolated_home) -> None:
    ui = _ScriptedUI(menu_choices=["close"])
    result = asyncio.run(_ui_settings_menu(ui))
    assert result == 0
    assert ui.menu_calls[0][0] == "Settings"


def test_settings_menu_compute_persists_backend(isolated_home) -> None:
    ui = _ScriptedUI(
        menu_choices=["compute", "close"],
        forms=[{"backend": "auto"}],
    )
    asyncio.run(_ui_settings_menu(ui))
    runtime = load_runtime()
    assert runtime.compute_backend == "auto"


def test_settings_menu_compute_unknown_backend_does_not_persist(isolated_home) -> None:
    save_runtime(RuntimeConfig(compute_backend="cpu"))
    ui = _ScriptedUI(
        menu_choices=["compute", "close"],
        forms=[{"backend": "ferrari"}],
    )
    asyncio.run(_ui_settings_menu(ui))
    assert load_runtime().compute_backend == "cpu"


def test_settings_menu_execution_form_persists_choices(isolated_home) -> None:
    ui = _ScriptedUI(
        menu_choices=["execution", "close"],
        forms=[
            {
                "order_type": "LIMIT",
                "time_in_force": "IOC",
                "post_only": "yes",
                "reduce_only_on_close": "no",
            }
        ],
    )
    asyncio.run(_ui_settings_menu(ui))
    from simple_ai_bitcoin_trading_binance.config import load_strategy

    cfg = load_strategy()
    assert cfg.order_type == "LIMIT"
    assert cfg.time_in_force == "IOC"
    assert cfg.post_only is True
    assert cfg.reduce_only_on_close is False


def test_settings_menu_execution_unsupported_order_type_falls_back(isolated_home) -> None:
    save_strategy(StrategyConfig(order_type="MARKET", time_in_force="GTC"))
    ui = _ScriptedUI(
        menu_choices=["execution", "close"],
        forms=[
            {
                "order_type": "BOGUS",
                "time_in_force": "WHATEVER",
                "post_only": "no",
                "reduce_only_on_close": "yes",
            }
        ],
    )
    asyncio.run(_ui_settings_menu(ui))
    from simple_ai_bitcoin_trading_binance.config import load_strategy

    cfg = load_strategy()
    assert cfg.order_type == "MARKET"
    assert cfg.time_in_force == "GTC"


def test_funds_action_is_registered() -> None:
    assert _action("Funds")
    assert _action("Settings")


def test_help_action_is_last_in_action_list() -> None:
    actions = _tui_actions()
    assert actions[-1].title == "Help"
    assert actions[-2].title == "Settings"


def test_funds_action_invokes_funds_menu(isolated_home) -> None:
    ui = _ScriptedUI(menu_choices=["close"])
    result = asyncio.run(_action("Funds").run(ui))
    assert result == 0
    assert ui.menu_calls and ui.menu_calls[0][0].startswith("Funds")


def test_settings_action_invokes_settings_menu(isolated_home) -> None:
    ui = _ScriptedUI(menu_choices=["close"])
    result = asyncio.run(_action("Settings").run(ui))
    assert result == 0
    assert ui.menu_calls and ui.menu_calls[0][0] == "Settings"


def test_settings_menu_execution_form_cancellation(isolated_home) -> None:
    ui = _ScriptedUI(menu_choices=["execution", "close"], forms=[None])
    asyncio.run(_ui_settings_menu(ui))
    assert any("Execution settings cancelled" in line for line in ui.logs)


def test_settings_menu_compute_form_cancellation(isolated_home) -> None:
    ui = _ScriptedUI(menu_choices=["compute", "close"], forms=[None])
    asyncio.run(_ui_settings_menu(ui))
    assert any("Compute backend selection cancelled" in line for line in ui.logs)


def test_settings_menu_runtime_saves_when_form_valid(isolated_home) -> None:
    ui = _ScriptedUI(
        menu_choices=["runtime", "close"],
        forms=[
            {
                "market_type": "spot",
                "interval": "1h",
                "testnet": "yes",
                "api_key": "",
                "api_secret": "",
                "dry_run": "yes",
                "validate_account": "no",
                "max_rate_calls_per_minute": "200",
                "recv_window_ms": "8000",
            }
        ],
    )
    asyncio.run(_ui_settings_menu(ui))
    runtime = load_runtime()
    assert runtime.interval == "1h"
    assert runtime.recv_window_ms == 8000
    assert any("Runtime settings saved" in line for line in ui.logs)


def test_settings_menu_runtime_invalid_value_logs_error(isolated_home, monkeypatch) -> None:
    """If _ui_edit_runtime raises ValueError the hub must log and continue."""
    from simple_ai_bitcoin_trading_binance import cli as cli_mod

    async def _exploder(_ui, _current):
        raise ValueError("intentionally bad")

    monkeypatch.setattr(cli_mod, "_ui_edit_runtime", _exploder)
    ui = _ScriptedUI(menu_choices=["runtime", "close"])
    asyncio.run(_ui_settings_menu(ui))
    assert any("Runtime settings invalid" in line for line in ui.logs)


def test_settings_menu_strategy_invalid_value_logs_error(isolated_home, monkeypatch) -> None:
    from simple_ai_bitcoin_trading_binance import cli as cli_mod

    async def _exploder(_ui, _current):
        raise ValueError("strategy boom")

    monkeypatch.setattr(cli_mod, "_ui_edit_strategy_args", _exploder)
    ui = _ScriptedUI(menu_choices=["strategy", "close"])
    asyncio.run(_ui_settings_menu(ui))
    assert any("Strategy settings invalid" in line for line in ui.logs)


def test_settings_menu_strategy_custom_no_change_is_cancelled(isolated_home, monkeypatch) -> None:
    """When the strategy form returns the 'no change' sentinel, the hub logs cancellation."""
    import argparse
    from simple_ai_bitcoin_trading_binance import cli as cli_mod

    async def _no_change(_ui, _current):
        return argparse.Namespace(
            profile="custom",
            leverage=None,
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
            enable_feature=None,
            disable_feature=None,
        )

    monkeypatch.setattr(cli_mod, "_ui_edit_strategy_args", _no_change)
    ui = _ScriptedUI(menu_choices=["strategy", "close"])
    asyncio.run(_ui_settings_menu(ui))
    assert any("Strategy update cancelled" in line for line in ui.logs)


def test_settings_menu_unknown_choice_loops_to_top(isolated_home) -> None:
    """A choice that doesn't match any known sub-menu must fall through and re-prompt."""
    ui = _ScriptedUI(menu_choices=["mystery", "close"])
    asyncio.run(_ui_settings_menu(ui))
    # The hub must have re-prompted — i.e. asked the menu twice.
    assert len(ui.menu_calls) == 2


def test_settings_menu_strategy_with_change_runs_command(isolated_home, monkeypatch) -> None:
    """When the strategy form returns concrete values, command_strategy runs."""
    import argparse
    from simple_ai_bitcoin_trading_binance import cli as cli_mod

    async def _changed(_ui, _current):
        return argparse.Namespace(
            profile="custom",
            leverage=None,
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
            set_features="momentum_1,rsi",
            enable_feature=None,
            disable_feature=None,
        )

    captured: list[argparse.Namespace] = []

    def _fake_command_strategy(args):
        captured.append(args)
        return 0

    monkeypatch.setattr(cli_mod, "_ui_edit_strategy_args", _changed)
    monkeypatch.setattr(cli_mod, "command_strategy", _fake_command_strategy)
    ui = _ScriptedUI(menu_choices=["strategy", "close"])
    asyncio.run(_ui_settings_menu(ui))
    assert captured and captured[0].set_features == "momentum_1,rsi"
