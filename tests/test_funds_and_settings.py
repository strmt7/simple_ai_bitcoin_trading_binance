"""Tests for the Funds allocation and Settings hub flows."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from simple_ai_bitcoin_trading_binance.api import BinanceAPIError
from simple_ai_bitcoin_trading_binance.cli import (
    _account_free_balances,
    _apply_funds_change,
    _credential_fingerprint,
    _funds_summary,
    _load_exchange_funds,
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
        if action.title == title or title in action.aliases:
            return action
    raise AssertionError(f"missing action: {title}")


def _runtime_form(**overrides: str) -> dict[str, str]:
    payload = {
        "market_type": "spot",
        "interval": "15m",
        "testnet": "yes",
        "demo": "no",
        "api_key": "",
        "api_secret": "",
        "dry_run": "yes",
        "validate_account": "yes",
        "max_rate_calls_per_minute": "1200",
        "recv_window_ms": "5000",
    }
    payload.update(overrides)
    return payload


class _FundsClient:
    def __init__(self, account: dict[str, object] | Exception) -> None:
        self.account = account

    def get_account(self):
        if isinstance(self.account, Exception):
            raise self.account
        return self.account


@pytest.fixture()
def isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    save_runtime(RuntimeConfig())
    save_strategy(StrategyConfig())
    return tmp_path


def test_account_free_balances_reads_spot_and_futures_payloads() -> None:
    balances = _account_free_balances(
        {
            "balances": [
                object(),
                {"asset": "USDC", "free": "12.5", "locked": "1"},
                {"asset": "BTC", "free": "0.01", "locked": "0"},
                {"asset": "ETH", "free": "99", "locked": "0"},
            ],
            "assets": [
                object(),
                {"asset": "USDC", "availableBalance": "20.0", "walletBalance": "25.0"},
                {"asset": "BTC", "walletBalance": "0.005"},
                {"asset": "ETH", "availableBalance": "99"},
            ],
        }
    )
    assert balances == {"USDC": 20.0, "BTC": 0.01}
    assert _account_free_balances("bad") == {"USDC": 0.0, "BTC": 0.0}
    assert _account_free_balances({"balances": "bad", "assets": "bad"}) == {"USDC": 0.0, "BTC": 0.0}


def test_funds_summary_describes_exchange_backed_caps() -> None:
    cfg = RuntimeConfig(managed_usdc=1234.5, managed_btc=0.001)
    summary = _funds_summary(cfg)
    assert "API credentials missing" in summary
    assert "Trading caps: USDC=1234.5000 BTC=0.00100000" in summary

    cfg = RuntimeConfig(api_key="fake-api-key", api_secret="fake-secret", managed_usdc=5.0, managed_btc=0.2)
    assert "Exchange balance not loaded" in _funds_summary(cfg)
    summary = _funds_summary(cfg, {"USDC": 50.0, "BTC": 0.25})
    assert "Exchange free: USDC=50.0000 BTC=0.25000000" in summary
    assert "Trading caps: USDC=5.0000 BTC=0.20000000" in summary


def test_load_exchange_funds_requires_credentials(isolated_home) -> None:
    with pytest.raises(BinanceAPIError, match="Funds requires Binance API key"):
        _load_exchange_funds(load_runtime())


def test_load_exchange_funds_reads_authenticated_balances(isolated_home, monkeypatch) -> None:
    from simple_ai_bitcoin_trading_binance import cli as cli_mod

    save_runtime(RuntimeConfig(api_key="fake-api-key", api_secret="fake-secret"))
    monkeypatch.setattr(
        cli_mod,
        "_build_client",
        lambda _runtime: _FundsClient({"balances": [{"asset": "USDC", "free": "15"}, {"asset": "BTC", "free": "0.02"}]}),
    )
    assert _load_exchange_funds(load_runtime()) == {"USDC": 15.0, "BTC": 0.02}


def test_apply_funds_change_uses_exchange_balances(isolated_home) -> None:
    save_runtime(RuntimeConfig(api_key="fake-api-key", api_secret="fake-secret", managed_usdc=10.0, managed_btc=0.1))

    unchanged, msg = _apply_funds_change("sync", 0.0)
    assert unchanged.managed_usdc == 10.0
    assert "Funds requires Binance API key" in msg

    after, msg = _apply_funds_change("sync", 0.0, balances={"USDC": 250.0, "BTC": 0.5})
    assert after.managed_usdc == 250.0
    assert after.managed_btc == 0.5
    assert "Synced" in msg

    after, msg = _apply_funds_change("set_usdc", 500.0, balances={"USDC": 300.0, "BTC": 0.5})
    assert after.managed_usdc == 300.0
    assert "capped" in msg

    after, msg = _apply_funds_change("set_btc", 0.25, balances={"USDC": 300.0, "BTC": 0.5})
    assert after.managed_btc == 0.25
    assert "Set BTC" in msg

    after, msg = _apply_funds_change("clear", 0.0)
    assert after.managed_usdc == 0.0
    assert after.managed_btc == 0.0
    assert "No exchange balances were changed" in msg

    after, msg = _apply_funds_change("deposit_usdc", 100.0, balances={"USDC": 999.0})
    assert after.managed_usdc == 0.0
    assert "no longer supports" in msg


def test_apply_funds_change_unknown_action_no_op(isolated_home) -> None:
    save_runtime(RuntimeConfig(managed_usdc=10.0))
    after, msg = _apply_funds_change("teleport", 1.0)
    assert after.managed_usdc == 10.0
    assert "Unknown" in msg


def test_funds_menu_close_returns_zero(isolated_home) -> None:
    ui = _ScriptedUI(menu_choices=["close"])
    result = asyncio.run(_ui_funds_menu(ui))
    assert result == 0
    assert ui.menu_calls[0][0].startswith("Trading caps")
    assert [key for key, _label in ui.menu_calls[0][1]] == ["show", "close"]


def test_funds_menu_show_without_credentials_logs_requirement(isolated_home) -> None:
    ui = _ScriptedUI(menu_choices=["show", "close"])
    result = asyncio.run(_ui_funds_menu(ui))
    assert result == 0
    assert any("Funds requires Binance API key" in line for line in ui.logs)


def test_funds_menu_sync_persists_exchange_balances(isolated_home, monkeypatch) -> None:
    from simple_ai_bitcoin_trading_binance import cli as cli_mod

    save_runtime(RuntimeConfig(api_key="fake-api-key", api_secret="fake-secret"))
    monkeypatch.setattr(
        cli_mod,
        "_build_client",
        lambda _runtime: _FundsClient({"balances": [{"asset": "USDC", "free": "250"}, {"asset": "BTC", "free": "0.125"}]}),
    )
    ui = _ScriptedUI(menu_choices=["sync", "close"])
    asyncio.run(_ui_funds_menu(ui))
    runtime = load_runtime()
    assert runtime.managed_usdc == 250.0
    assert runtime.managed_btc == 0.125


def test_funds_menu_set_usdc_cap_is_capped_to_exchange_free(isolated_home, monkeypatch) -> None:
    from simple_ai_bitcoin_trading_binance import cli as cli_mod

    save_runtime(RuntimeConfig(api_key="fake-api-key", api_secret="fake-secret"))
    monkeypatch.setattr(
        cli_mod,
        "_build_client",
        lambda _runtime: _FundsClient({"balances": [{"asset": "USDC", "free": "100"}, {"asset": "BTC", "free": "0.5"}]}),
    )
    ui = _ScriptedUI(
        menu_choices=["set_usdc", "close"],
        forms=[{"amount": "999"}],
    )
    asyncio.run(_ui_funds_menu(ui))
    assert load_runtime().managed_usdc == 100.0
    assert any("capped" in line for line in ui.logs)


def test_funds_menu_invalid_amount_rejected(isolated_home, monkeypatch) -> None:
    from simple_ai_bitcoin_trading_binance import cli as cli_mod

    save_runtime(RuntimeConfig(api_key="fake-api-key", api_secret="fake-secret"))
    monkeypatch.setattr(
        cli_mod,
        "_build_client",
        lambda _runtime: _FundsClient({"balances": [{"asset": "USDC", "free": "100"}, {"asset": "BTC", "free": "0.5"}]}),
    )
    ui = _ScriptedUI(
        menu_choices=["set_btc", "close"],
        forms=[{"amount": "not-a-number"}],
    )
    asyncio.run(_ui_funds_menu(ui))
    assert any("rejected" in line.lower() for line in ui.logs)


def test_funds_menu_clear_caps(isolated_home) -> None:
    save_runtime(RuntimeConfig(api_key="fake-api-key", api_secret="fake-secret", managed_usdc=42.0, managed_btc=0.5))
    ui = _ScriptedUI(menu_choices=["clear", "close"])
    asyncio.run(_ui_funds_menu(ui))
    assert load_runtime().managed_usdc == 0.0


def test_funds_menu_form_cancel_keeps_cap(isolated_home, monkeypatch) -> None:
    from simple_ai_bitcoin_trading_binance import cli as cli_mod

    save_runtime(RuntimeConfig(api_key="fake-api-key", api_secret="fake-secret", managed_usdc=42.0))
    monkeypatch.setattr(
        cli_mod,
        "_build_client",
        lambda _runtime: _FundsClient({"balances": [{"asset": "USDC", "free": "100"}]}),
    )
    ui = _ScriptedUI(
        menu_choices=["set_usdc", "close"],
        forms=[None],
    )
    asyncio.run(_ui_funds_menu(ui))
    assert load_runtime().managed_usdc == 42.0


def test_funds_menu_exchange_failure_logs_short_credential_message(isolated_home, monkeypatch) -> None:
    from simple_ai_bitcoin_trading_binance import cli as cli_mod

    save_runtime(RuntimeConfig(api_key="fake-api-key", api_secret="fake-secret"))
    monkeypatch.setattr(cli_mod, "_build_client", lambda _runtime: _FundsClient(BinanceAPIError("bad key")))
    ui = _ScriptedUI(menu_choices=["show", "close"])
    asyncio.run(_ui_funds_menu(ui))
    assert ui.logs == ["Funds requires valid Binance API credentials: bad key"]


def test_settings_menu_close_returns_zero(isolated_home) -> None:
    ui = _ScriptedUI(menu_choices=["close"])
    result = asyncio.run(_ui_settings_menu(ui))
    assert result == 0
    assert ui.menu_calls[0][0] == "All settings"


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
    assert cfg.order_type == "MARKET"
    assert cfg.time_in_force == "IOC"
    assert cfg.post_only is False
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
    assert actions[-2].title == "All settings"


def test_tui_credential_gated_actions_reflect_credential_state(isolated_home) -> None:
    save_runtime(RuntimeConfig())
    missing = _tui_actions({"fingerprint": "missing", "status": "missing"})
    assert missing[0].title == "Connect"
    assert missing[1].title == "Dashboard"
    assert missing[0].is_enabled() is False
    assert missing[1].is_enabled() is True
    assert _action("Trading caps").is_enabled() is False

    runtime = RuntimeConfig(api_key="fake-api-key", api_secret="fake-secret")
    save_runtime(runtime)
    fingerprint = _credential_fingerprint(runtime)
    unchecked = _tui_actions({"fingerprint": fingerprint, "status": "unchecked"})
    assert unchecked[0].is_enabled() is True
    assert next(action for action in unchecked if action.title == "Trading caps").is_enabled() is False

    invalid = _tui_actions({"fingerprint": fingerprint, "status": "invalid"})
    assert invalid[0].is_enabled() is True
    assert "failed validation" in invalid[0].lock_reason()

    valid = _tui_actions({"fingerprint": fingerprint, "status": "valid"})
    assert next(action for action in valid if action.title == "Trading caps").is_enabled() is True


def test_tui_settings_action_marks_credentials_after_settings_close(isolated_home) -> None:
    runtime = RuntimeConfig(api_key="fake-api-key", api_secret="fake-secret")
    save_runtime(runtime)
    state = {"fingerprint": _credential_fingerprint(runtime), "status": "valid"}
    settings = next(action for action in _tui_actions(state) if action.title == "All settings")

    assert asyncio.run(settings.run(_ScriptedUI(menu_choices=["close"]))) == 0
    assert state["fingerprint"] == _credential_fingerprint(runtime)
    assert state["status"] == "valid"


def test_connection_settings_escape_does_not_save_or_validate(isolated_home, monkeypatch) -> None:
    from simple_ai_bitcoin_trading_binance import cli as cli_mod

    runtime = RuntimeConfig(api_key="fake-api-key", api_secret="fake-secret", validate_account=True)
    save_runtime(runtime)
    state = {"fingerprint": _credential_fingerprint(runtime), "status": "valid"}
    monkeypatch.setattr(
        cli_mod,
        "_build_client",
        lambda _runtime: pytest.fail("cancelled connection settings must not validate"),
    )

    settings = next(action for action in _tui_actions(state) if action.title == "Connection settings")
    assert asyncio.run(settings.run(_ScriptedUI(forms=[None]))) == 0
    assert load_runtime() == runtime
    assert state == {"fingerprint": _credential_fingerprint(runtime), "status": "valid"}

    save_runtime(RuntimeConfig())
    assert asyncio.run(settings.run(_ScriptedUI(menu_choices=["close"]))) == 0
    assert state["fingerprint"] == _credential_fingerprint(runtime)
    assert state["status"] == "valid"


def test_funds_action_requires_credentials_before_menu(isolated_home) -> None:
    ui = _ScriptedUI(menu_choices=["close"])
    result = asyncio.run(_action("Funds").run(ui))
    assert result == 2
    assert ui.logs and "Trading caps requires Binance API key" in ui.logs[0]
    assert ui.menu_calls == []

    save_runtime(RuntimeConfig(api_key="fake-api-key", api_secret="fake-secret"))
    ui = _ScriptedUI(menu_choices=["close"])
    result = asyncio.run(_action("Funds").run(ui))
    assert result == 0
    assert ui.menu_calls and ui.menu_calls[0][0].startswith("Trading caps")


def test_settings_action_invokes_settings_menu(isolated_home) -> None:
    ui = _ScriptedUI(menu_choices=["close"])
    result = asyncio.run(_action("Settings").run(ui))
    assert result == 0
    assert ui.menu_calls and ui.menu_calls[0][0] == "All settings"


def test_signed_tui_actions_stop_before_forms_without_credentials(isolated_home) -> None:
    for title in ("Connect", "Account", "Testnet loop", "Spot roundtrip"):
        ui = _ScriptedUI()
        assert asyncio.run(_action(title).run(ui)) == 2
        assert ui.logs and "requires Binance API key" in ui.logs[0]

    ui = _ScriptedUI(
        forms=[
            {
                "input": "",
                "model": "",
                "readiness": "no",
                "online": "no",
                "account": "yes",
            }
        ]
    )
    assert asyncio.run(_action("Operator report").run(ui)) == 2
    assert "Full report account section requires Binance API key" in ui.logs[0]


def test_direct_connection_settings_updates_credential_state(isolated_home, monkeypatch) -> None:
    from simple_ai_bitcoin_trading_binance import cli as cli_mod

    state = {"fingerprint": "", "status": "missing"}
    save_runtime(RuntimeConfig(validate_account=True))
    monkeypatch.setattr(cli_mod, "_build_client", lambda _runtime: object())
    monkeypatch.setattr(cli_mod, "_validate_runtime_connection", lambda _runtime, _client: None)

    action = next(item for item in _tui_actions(state) if item.title == "Connection settings")
    ui = _ScriptedUI(forms=[_runtime_form(api_key="fake-api-key", api_secret="fake-secret", validate_account="yes")])

    assert asyncio.run(action.run(ui)) == 0
    assert state["fingerprint"] == _credential_fingerprint(load_runtime())
    assert state["status"] == "valid"


def test_signed_actions_lock_out_non_testnet_runtime_even_with_valid_credentials(isolated_home) -> None:
    runtime = RuntimeConfig(testnet=False, demo=False, api_key="fake-api-key", api_secret="fake-secret")
    save_runtime(runtime)
    state = {"fingerprint": _credential_fingerprint(runtime), "status": "valid"}
    action = next(item for item in _tui_actions(state) if item.title == "Trading caps")
    ui = _ScriptedUI(menu_choices=["close"])

    assert asyncio.run(action.run(ui)) == 2
    assert ui.menu_calls == []
    assert ui.logs == ["Trading caps is locked. Signed actions require testnet=true or demo=true."]


def test_settings_menu_execution_form_cancellation(isolated_home) -> None:
    ui = _ScriptedUI(menu_choices=["execution", "close"], forms=[None])
    asyncio.run(_ui_settings_menu(ui))
    assert any("Order settings cancelled" in line for line in ui.logs)


def test_settings_menu_compute_form_cancellation(isolated_home) -> None:
    ui = _ScriptedUI(menu_choices=["compute", "close"], forms=[None])
    asyncio.run(_ui_settings_menu(ui))
    assert any("Compute backend selection cancelled" in line for line in ui.logs)


def test_settings_menu_runtime_saves_when_form_valid(isolated_home) -> None:
    save_runtime(RuntimeConfig(managed_usdc=42.0, managed_btc=0.5, compute_backend="auto"))
    ui = _ScriptedUI(
        menu_choices=["runtime", "close"],
        forms=[_runtime_form(interval="1h", validate_account="no", max_rate_calls_per_minute="200", recv_window_ms="8000")],
    )
    asyncio.run(_ui_settings_menu(ui))
    runtime = load_runtime()
    assert runtime.interval == "1h"
    assert runtime.recv_window_ms == 8000
    assert runtime.compute_backend == "auto"
    assert runtime.managed_usdc == 42.0
    assert runtime.managed_btc == 0.5
    assert any("Connection settings saved" in line for line in ui.logs)


def test_settings_menu_runtime_marks_missing_and_valid_credentials(isolated_home, monkeypatch) -> None:
    from simple_ai_bitcoin_trading_binance import cli as cli_mod

    marks: list[str] = []
    save_runtime(RuntimeConfig(interval="15m", validate_account=True))
    ui = _ScriptedUI(
        menu_choices=["runtime", "runtime", "close"],
        forms=[
            _runtime_form(interval="1h", validate_account="yes"),
            _runtime_form(interval="4h", api_key="fake-api-key", api_secret="fake-secret", validate_account="yes"),
        ],
    )
    monkeypatch.setattr(cli_mod, "_build_client", lambda _runtime: object())
    monkeypatch.setattr(cli_mod, "_validate_runtime_connection", lambda _runtime, _client: None)

    asyncio.run(_ui_settings_menu(ui, mark_credentials=marks.append))

    assert marks == ["missing", "unchecked", "valid"]
    assert load_runtime().api_key == "fake-api-key"
    assert any("Connection credentials validated" in line for line in ui.logs)


def test_settings_menu_runtime_marks_invalid_credentials(isolated_home, monkeypatch) -> None:
    from simple_ai_bitcoin_trading_binance import cli as cli_mod

    marks: list[str] = []
    save_runtime(RuntimeConfig(validate_account=True))
    ui = _ScriptedUI(
        menu_choices=["runtime", "close"],
        forms=[_runtime_form(api_key="fake-api-key", api_secret="fake-secret", validate_account="yes")],
    )
    monkeypatch.setattr(cli_mod, "_build_client", lambda _runtime: object())
    monkeypatch.setattr(
        cli_mod,
        "_validate_runtime_connection",
        lambda _runtime, _client: (_ for _ in ()).throw(BinanceAPIError("bad key")),
    )

    asyncio.run(_ui_settings_menu(ui, mark_credentials=marks.append))

    assert marks == ["unchecked", "invalid"]
    assert any("Connection validation failed: bad key" in line for line in ui.logs)


def test_settings_menu_runtime_validation_without_credential_callback(isolated_home, monkeypatch) -> None:
    from simple_ai_bitcoin_trading_binance import cli as cli_mod

    save_runtime(RuntimeConfig(validate_account=True))
    ui = _ScriptedUI(
        menu_choices=["runtime", "runtime", "close"],
        forms=[
            _runtime_form(interval="1h", api_key="fake-api-key", api_secret="fake-secret", validate_account="yes"),
            _runtime_form(interval="4h", api_key="fake-api-key", api_secret="fake-secret", validate_account="yes"),
        ],
    )
    calls = {"count": 0}

    def validate(_runtime, _client):
        calls["count"] += 1
        if calls["count"] == 2:
            raise BinanceAPIError("bad key")

    monkeypatch.setattr(cli_mod, "_build_client", lambda _runtime: object())
    monkeypatch.setattr(cli_mod, "_validate_runtime_connection", validate)

    asyncio.run(_ui_settings_menu(ui))

    assert any("Connection credentials validated" in line for line in ui.logs)
    assert any("Connection validation failed: bad key" in line for line in ui.logs)


def test_settings_menu_runtime_invalid_value_logs_error(isolated_home, monkeypatch) -> None:
    """If _ui_edit_runtime raises ValueError the hub must log and continue."""
    from simple_ai_bitcoin_trading_binance import cli as cli_mod

    async def _exploder(_ui, _current):
        raise ValueError("intentionally bad")

    monkeypatch.setattr(cli_mod, "_ui_edit_runtime", _exploder)
    ui = _ScriptedUI(menu_choices=["runtime", "close"])
    asyncio.run(_ui_settings_menu(ui))
    assert any("Connection settings invalid" in line for line in ui.logs)


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
