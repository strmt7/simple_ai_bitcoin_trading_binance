"""Comprehensive unit tests for the autonomous loop module."""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from pathlib import Path

import pytest

from simple_ai_bitcoin_trading_binance import autonomous
from simple_ai_bitcoin_trading_binance.api import BinanceAPIError
from simple_ai_bitcoin_trading_binance.autonomous import (
    AutonomousConfig,
    AutonomousControl,
    Decision,
    Heartbeat,
    LoopResult,
    STATE_PAUSED,
    STATE_RUNNING,
    STATE_STOPPED,
    STATE_STOPPING,
    _close_to_trade,
    _default_decision,
    _directional_confidence,
    _entry_gate,
    _evaluate_auto_close,
    _open_position_from_decision,
    ensure_credentials,
    ensure_testnet,
    run_loop,
)
from simple_ai_bitcoin_trading_binance.logging_ext import reset as reset_logger
from simple_ai_bitcoin_trading_binance.objective import get_objective
from simple_ai_bitcoin_trading_binance.positions import (
    ClosedTrade,
    OpenPosition,
    PositionsStore,
    new_position_id,
)
from simple_ai_bitcoin_trading_binance.types import RuntimeConfig, StrategyConfig


@pytest.fixture(autouse=True)
def _reset_logger():
    reset_logger()
    yield
    reset_logger()


class FakeClient:
    base_url = "https://testnet.binance.vision"

    def __init__(self, price: float = 100.0):
        self._price = price

    def get_symbol_price(self, symbol: str):
        return (float(self._price), 0)


def _make_config(tmp_path: Path, **overrides) -> AutonomousConfig:
    defaults = dict(
        objective="default",
        poll_seconds=0.0,  # intentionally below the floor
        stop_after_iterations=1,
        heartbeat_every=1,
        dry_run=True,
        control_path=tmp_path / "state.json",
        heartbeat_path=tmp_path / "heartbeat.json",
        positions_root=tmp_path / "autonomous",
        log_path=tmp_path / "autonomous.log",
    )
    defaults.update(overrides)
    return AutonomousConfig(**defaults)


def _runtime(testnet: bool = True, *, api_key: str = "k", api_secret: str = "s") -> RuntimeConfig:
    return RuntimeConfig(
        symbol="BTCUSDC",
        interval="15m",
        market_type="spot",
        testnet=testnet,
        api_key=api_key,
        api_secret=api_secret,
        dry_run=True,
    )


def _strategy() -> StrategyConfig:
    return StrategyConfig()


# ----- AutonomousControl ----------------------------------------------------


def test_control_write_rejects_invalid_state(tmp_path: Path) -> None:
    ctl = AutonomousControl(path=tmp_path / "state.json")
    with pytest.raises(ValueError):
        ctl.write("NOT_A_STATE")


def test_control_write_then_read_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    ctl = AutonomousControl(path=path)
    ctl.write(STATE_RUNNING, note="hello")
    payload = ctl.read()
    assert payload["state"] == STATE_RUNNING
    assert payload["note"] == "hello"
    assert ctl.state() == STATE_RUNNING


def test_control_read_when_missing_returns_stopped(tmp_path: Path) -> None:
    ctl = AutonomousControl(path=tmp_path / "missing.json")
    payload = ctl.read()
    assert payload == {"state": STATE_STOPPED, "note": "", "ts_ms": 0}
    # and state() falls through to the default branch
    assert ctl.state() == STATE_STOPPED


def test_control_read_malformed_json(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    path.write_text("not-valid-json", encoding="utf-8")
    ctl = AutonomousControl(path=path)
    assert ctl.read() == {"state": STATE_STOPPED, "note": "read-error", "ts_ms": 0}


def test_control_read_wrong_payload_type(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    path.write_text(json.dumps(["not", "a", "dict"]), encoding="utf-8")
    ctl = AutonomousControl(path=path)
    assert ctl.read() == {"state": STATE_STOPPED, "note": "malformed", "ts_ms": 0}


def test_control_read_unknown_state_is_malformed(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    path.write_text(json.dumps({"state": "WEIRD"}), encoding="utf-8")
    ctl = AutonomousControl(path=path)
    assert ctl.read()["note"] == "malformed"


def test_control_state_defaults_when_state_is_empty(tmp_path: Path) -> None:
    """Covers the ``state() -> str(... or STATE_STOPPED)`` branch."""
    path = tmp_path / "state.json"
    # produce a malformed payload so read() returns a dict whose state key is
    # the default STOPPED; state() returns it directly.
    ctl = AutonomousControl(path=path)
    assert ctl.state() == STATE_STOPPED


# ----- Heartbeat ------------------------------------------------------------


def test_heartbeat_write_persists_under_tmp_path(tmp_path: Path) -> None:
    hb = Heartbeat(
        iteration=1,
        state=STATE_RUNNING,
        last_signal=0.42,
        last_side="LONG",
        last_price=100.0,
        open_positions=0,
        realized_pnl=0.0,
        unrealized_pnl=0.0,
        objective="default",
        updated_at_ms=1,
    )
    target = tmp_path / "nested" / "heartbeat.json"
    hb.write(target)
    assert target.exists()
    data = json.loads(target.read_text(encoding="utf-8"))
    assert data["iteration"] == 1
    assert data["last_side"] == "LONG"
    assert data["objective"] == "default"


# ----- ensure_testnet / ensure_credentials ----------------------------------


def test_ensure_testnet_raises_when_not_testnet() -> None:
    with pytest.raises(RuntimeError):
        ensure_testnet(_runtime(testnet=False))


def test_ensure_testnet_passes_when_testnet() -> None:
    ensure_testnet(_runtime(testnet=True))  # no raise


def test_ensure_credentials_live_requires_key() -> None:
    cfg = AutonomousConfig(dry_run=False)
    with pytest.raises(RuntimeError):
        ensure_credentials(_runtime(api_key="", api_secret=""), cfg)


def test_ensure_credentials_live_with_keys_passes() -> None:
    cfg = AutonomousConfig(dry_run=False)
    ensure_credentials(_runtime(api_key="k", api_secret="s"), cfg)


def test_ensure_credentials_dry_run_short_circuits() -> None:
    cfg = AutonomousConfig(dry_run=True)
    ensure_credentials(_runtime(api_key="", api_secret=""), cfg)


# ----- _evaluate_auto_close -------------------------------------------------


def _make_position(side: str = "LONG", entry: float = 100.0) -> OpenPosition:
    return OpenPosition(
        id=new_position_id(),
        symbol="BTCUSDC",
        market_type="spot",
        side=side,
        qty=0.1,
        entry_price=entry,
        leverage=1.0,
        opened_at_ms=0,
        notional=entry * 0.1,
    )


def test_evaluate_auto_close_take_profit_hits(tmp_path: Path) -> None:
    strat = StrategyConfig(take_profit_pct=0.01, stop_loss_pct=0.01)
    pos = _make_position("LONG", entry=100.0)
    cfg = AutonomousConfig()
    should_close, reason = _evaluate_auto_close(pos, 200.0, cfg, strat)
    assert should_close is True
    assert "take-profit" in reason


def test_evaluate_auto_close_stop_loss_hits_for_short(tmp_path: Path) -> None:
    strat = StrategyConfig(take_profit_pct=0.1, stop_loss_pct=0.01)
    pos = _make_position("SHORT", entry=100.0)
    cfg = AutonomousConfig()
    should_close, reason = _evaluate_auto_close(pos, 200.0, cfg, strat)
    # SHORT at 100, mark 200 => pnl_pct = -1.0 <= -0.01 => stop-loss
    assert should_close is True
    assert "stop-loss" in reason


def test_evaluate_auto_close_cfg_max_threshold(tmp_path: Path) -> None:
    strat = StrategyConfig(take_profit_pct=10.0, stop_loss_pct=10.0)
    pos = _make_position("LONG", entry=100.0)
    cfg = AutonomousConfig(max_unrealized_close_pct=0.05)
    should_close, reason = _evaluate_auto_close(pos, 110.0, cfg, strat)
    assert should_close is True
    assert "auto-take-profit" in reason


def test_evaluate_auto_close_cfg_min_threshold(tmp_path: Path) -> None:
    strat = StrategyConfig(take_profit_pct=10.0, stop_loss_pct=10.0)
    pos = _make_position("LONG", entry=100.0)
    cfg = AutonomousConfig(min_unrealized_close_pct=-0.01)
    should_close, reason = _evaluate_auto_close(pos, 90.0, cfg, strat)
    assert should_close is True
    assert "auto-stop-loss" in reason


def test_evaluate_auto_close_no_close_path(tmp_path: Path) -> None:
    strat = StrategyConfig(take_profit_pct=0.5, stop_loss_pct=0.5)
    pos = _make_position("LONG", entry=100.0)
    cfg = AutonomousConfig()
    should_close, reason = _evaluate_auto_close(pos, 101.0, cfg, strat)
    assert should_close is False
    assert reason == ""


# ----- _open_position_from_decision ----------------------------------------


def test_open_position_from_decision_clamps_price(tmp_path: Path) -> None:
    decision = Decision(side="LONG", confidence=0.7, mark_price=-5.0)
    runtime = _runtime()
    strat = StrategyConfig()
    cfg = _make_config(tmp_path)
    position = _open_position_from_decision(
        decision, runtime, strat, get_objective("default"), cfg, clock=lambda: 1.0,
    )
    assert position.entry_price == 0.01
    assert position.side == "LONG"
    assert position.dry_run is True


def test_open_position_from_decision_live_sets_dry_run_false(tmp_path: Path) -> None:
    decision = Decision(side="LONG", confidence=0.7, mark_price=200.0)
    runtime = _runtime(api_key="k", api_secret="s")
    strat = StrategyConfig()
    cfg = _make_config(tmp_path, dry_run=False)
    position = _open_position_from_decision(
        decision, runtime, strat, get_objective("default"), cfg, clock=lambda: 2.0,
    )
    assert position.entry_price == 200.0
    assert position.dry_run is False
    assert position.opened_at_ms == 2000


# ----- _close_to_trade ------------------------------------------------------


def test_close_to_trade_long_profit(tmp_path: Path) -> None:
    pos = _make_position("LONG", entry=100.0)
    trade = _close_to_trade(pos, 200.0, "take-profit", clock=lambda: 3.0, fees=0.5)
    assert trade.realized_pnl > 0
    assert trade.realized_pnl_pct == pytest.approx(1.0)
    assert trade.fees == 0.5
    assert trade.closed_at_ms == 3000


def test_close_to_trade_short_loss(tmp_path: Path) -> None:
    pos = _make_position("SHORT", entry=100.0)
    trade = _close_to_trade(pos, 200.0, "stop-loss", clock=lambda: 0.0)
    assert trade.realized_pnl < 0
    assert trade.realized_pnl_pct == pytest.approx(-1.0)


# ----- _default_decision ----------------------------------------------------


def test_default_decision_returns_flat(tmp_path: Path) -> None:
    client = FakeClient(price=1234.0)
    decision = _default_decision(
        client, _runtime(), _strategy(), get_objective("default")
    )
    assert decision.side == "FLAT"
    assert decision.mark_price == 1234.0
    assert decision.confidence == 0.0


def test_directional_confidence_handles_short_and_invalid_values() -> None:
    assert _directional_confidence(Decision(side="LONG", confidence=0.8, mark_price=1.0)) == 0.8
    assert _directional_confidence(Decision(side="SHORT", confidence=0.2, mark_price=1.0)) == 0.8
    assert _directional_confidence(Decision(side="LONG", confidence="bad", mark_price=1.0)) == 0.0


# ----- run_loop: integration-ish branches ----------------------------------


def _tick_clock():
    counter = {"v": 0}

    def clock():
        counter["v"] += 1
        return float(counter["v"])

    return clock


def test_run_loop_normal_flow_three_iterations(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path, stop_after_iterations=3)
    client = FakeClient()
    runtime = _runtime()
    strat = _strategy()

    def decision_fn(_c, _r, _s, _o):
        return Decision(side="FLAT", confidence=0.5, mark_price=100.0)

    sleeps: list[float] = []
    result = run_loop(
        client,
        runtime,
        strat,
        cfg,
        decision_fn=decision_fn,
        sleep=lambda d: sleeps.append(d),
        clock=_tick_clock(),
    )
    assert result.iterations == 3
    assert result.heartbeats_written == 3
    assert result.exit_reason == "iteration-cap"
    # final state file persisted as STOPPED
    state_payload = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert state_payload["state"] == STATE_STOPPED
    # heartbeat file exists
    assert (tmp_path / "heartbeat.json").exists()
    # sleep was clamped to at least 1.0 (min interval)
    assert all(d >= 1.0 for d in sleeps)


def test_run_loop_paused_and_resumed_and_stopped(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path, stop_after_iterations=None)
    control_path = tmp_path / "state.json"
    client = FakeClient()

    # A sleep fake that flips the control state over time.
    sleep_calls = {"count": 0}

    def sleep_fake(_d):
        sleep_calls["count"] += 1
        if sleep_calls["count"] == 1:
            # After first sleep, flip to RUNNING so iteration 2 proceeds.
            AutonomousControl(path=control_path).write(STATE_RUNNING)
        elif sleep_calls["count"] == 2:
            # After the normal run iteration's sleep, request a stop.
            AutonomousControl(path=control_path).write(STATE_STOPPING)

    # Before the loop starts we'll let run_loop write RUNNING itself; but we want
    # the first iter to observe PAUSED. Our trick: wrap the control file write
    # by installing a fake that overrides after the initial write.
    def decision_fn(_c, _r, _s, _o):
        return Decision(side="FLAT", confidence=0.1, mark_price=100.0)

    # To force a paused iteration first, override the state after the loop's
    # own initial write. We achieve that by monkey-patching sleep to only fire
    # the transitions, and by pre-writing PAUSED AFTER run_loop's initial
    # write. That is done by a decision_fn that mutates state — but the state
    # is checked before decision_fn. So instead: monkeypatch AutonomousControl
    # inside run_loop via a wrapper sleep? Simpler: we flip to PAUSED in the
    # first iteration by overwriting before the first iteration's check.
    # The cleanest approach: write PAUSED *before* starting run_loop, and then
    # rely on run_loop's own first write to override. It WILL override, so we
    # have to intercept. Use a sentinel decision_fn to switch state.
    # Easiest solution: wrap sleep to flip; but the first state check happens
    # before the first sleep. Use a probe in the initial state: monkeypatch
    # AutonomousControl.write to skip the initial write.
    # --> Use a subclass via monkeypatching the control_path fixture.

    # Simpler scheme: set PAUSED directly in state.json and make loop skip the
    # initial write by patching control.write to a no-op before first iter.

    # We'll take the practical route: override decision_fn to swap states across
    # iterations, since it runs AFTER the state check on each iter.
    iters = {"n": 0}

    def dec(_c, _r, _s, _o):
        iters["n"] += 1
        ctl = AutonomousControl(path=control_path)
        if iters["n"] == 1:
            # flip to PAUSED so next iter hits paused branch
            ctl.write(STATE_PAUSED)
        elif iters["n"] == 2:
            # this won't actually execute because state=PAUSED short-circuits
            ctl.write(STATE_RUNNING)
        return Decision(side="FLAT", confidence=0.0, mark_price=100.0)

    # A simple sleep that, when called during paused iteration, flips to RUNNING
    # and later to STOPPING.
    phase = {"n": 0}

    def sleep_and_flip(_d):
        phase["n"] += 1
        ctl = AutonomousControl(path=control_path)
        if phase["n"] == 1:
            # first sleep was after iter 1 (normal flow) — state already PAUSED
            # so next iteration hits the paused branch. Next time sleep fires
            # it's the paused branch's sleep; transition to RUNNING.
            pass
        elif phase["n"] == 2:
            # after paused iter's sleep: flip to RUNNING so next iter is normal
            ctl.write(STATE_RUNNING)
        elif phase["n"] == 3:
            # after normal iter 3's sleep: request stop
            ctl.write(STATE_STOPPING)

    result = run_loop(
        client, runtime=_runtime(), strategy=_strategy(), cfg=cfg,
        decision_fn=dec, sleep=sleep_and_flip, clock=_tick_clock(),
    )
    assert result.exit_reason == "operator-stop"
    assert result.iterations >= 3


def test_run_loop_decision_binance_error_continues(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path, stop_after_iterations=2)

    attempts = {"n": 0}

    def dec(_c, _r, _s, _o):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise BinanceAPIError("rate-limited")
        return Decision(side="FLAT", confidence=0.0, mark_price=100.0)

    result = run_loop(
        FakeClient(), _runtime(), _strategy(), cfg,
        decision_fn=dec, sleep=lambda _d: None, clock=_tick_clock(),
    )
    # iteration 1 raised => continue; iteration 2 ran normally => iteration-cap
    assert result.exit_reason == "iteration-cap"
    assert result.iterations == 2


def test_run_loop_decision_generic_exception_breaks(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path, stop_after_iterations=5)

    def dec(_c, _r, _s, _o):
        raise RuntimeError("boom")

    result = run_loop(
        FakeClient(), _runtime(), _strategy(), cfg,
        decision_fn=dec, sleep=lambda _d: None, clock=_tick_clock(),
    )
    assert result.exit_reason == "decision-exception"
    assert result.iterations == 1


def test_run_loop_closes_position_via_auto_close(tmp_path: Path) -> None:
    cfg = _make_config(
        tmp_path,
        stop_after_iterations=1,
        max_unrealized_close_pct=0.01,
    )
    # Pre-stage an open position with a tiny profit so the auto-close trips.
    store = PositionsStore(root=cfg.positions_root)
    store.record_open(_make_position("LONG", entry=100.0))

    def dec(_c, _r, _s, _o):
        return Decision(side="FLAT", confidence=0.0, mark_price=200.0)

    result = run_loop(
        FakeClient(), _runtime(), _strategy(), cfg,
        decision_fn=dec, sleep=lambda _d: None, clock=_tick_clock(),
    )
    assert result.closed_trades == 1
    assert result.iterations == 1


def test_run_loop_opens_position_when_flat_and_long_signal(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path, stop_after_iterations=1)

    def dec(_c, _r, _s, _o):
        return Decision(side="LONG", confidence=0.8, mark_price=100.0)

    result = run_loop(
        FakeClient(), _runtime(), _strategy(), cfg,
        decision_fn=dec, sleep=lambda _d: None, clock=_tick_clock(),
    )
    assert result.opened_trades == 1
    store = PositionsStore(root=cfg.positions_root)
    opens = store.load_open()
    assert len(opens) == 1
    assert opens[0].side == "LONG"


def test_run_loop_respects_max_open_positions(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path, stop_after_iterations=2)
    # Pre-stage a position so the open branch is skipped.
    store = PositionsStore(root=cfg.positions_root)
    store.record_open(_make_position("LONG", entry=50.0))
    strat = replace(StrategyConfig(),
                    take_profit_pct=10.0, stop_loss_pct=10.0,
                    max_open_positions=1)

    def dec(_c, _r, _s, _o):
        # Signal LONG on every iter; because max_open_positions=1 and one is
        # already open, no new position should open.
        return Decision(side="LONG", confidence=0.9, mark_price=50.0)

    result = run_loop(
        FakeClient(), _runtime(), strat, cfg,
        decision_fn=dec, sleep=lambda _d: None, clock=_tick_clock(),
    )
    # No new opens despite LONG signal
    assert result.opened_trades == 0
    assert len(PositionsStore(root=cfg.positions_root).load_open()) == 1


def test_run_loop_respects_zero_max_open_positions(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path, stop_after_iterations=1)
    strat = replace(StrategyConfig(), max_open_positions=0)

    result = run_loop(
        FakeClient(), _runtime(), strat, cfg,
        decision_fn=lambda *_: Decision(side="LONG", confidence=0.9, mark_price=100.0),
        sleep=lambda _d: None,
        clock=_tick_clock(),
    )

    assert result.opened_trades == 0
    assert result.skipped_entries == 1
    assert PositionsStore(root=cfg.positions_root).load_open() == []


def test_entry_gate_blocks_daily_cap_cooldown_drawdown_and_low_confidence(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path, starting_reference_cash=1000.0)
    store = PositionsStore(root=cfg.positions_root)
    strategy = replace(
        StrategyConfig(),
        max_trades_per_day=1,
        cooldown_minutes=5,
        max_drawdown_limit=0.10,
    )
    trade = ClosedTrade(
        id="t1",
        symbol="BTCUSDC",
        market_type="spot",
        side="LONG",
        qty=1.0,
        entry_price=100.0,
        exit_price=90.0,
        leverage=1.0,
        opened_at_ms=1_000,
        closed_at_ms=2_000,
        realized_pnl=-150.0,
        realized_pnl_pct=-0.15,
    )
    store.record_close(trade)

    gate = _entry_gate(
        store,
        Decision(side="LONG", confidence=0.9, mark_price=100.0),
        strategy,
        cfg,
        get_objective("default"),
        now_ms_value=3_000,
    )
    assert gate.allowed is False
    assert gate.reason.startswith("daily-cap-reached")

    no_cap = replace(strategy, max_trades_per_day=0)
    gate = _entry_gate(
        store,
        Decision(side="LONG", confidence=0.9, mark_price=100.0),
        no_cap,
        cfg,
        get_objective("default"),
        now_ms_value=3_000,
    )
    assert gate.reason.startswith("cooldown-active")

    no_cooldown = replace(no_cap, cooldown_minutes=0)
    gate = _entry_gate(
        store,
        Decision(side="LONG", confidence=0.9, mark_price=100.0),
        no_cooldown,
        cfg,
        get_objective("default"),
        now_ms_value=3_000,
    )
    assert gate.reason.startswith("drawdown-lockout")

    no_drawdown = replace(no_cooldown, max_drawdown_limit=0.0)
    gate = _entry_gate(
        store,
        Decision(side="LONG", confidence=0.51, mark_price=100.0),
        no_drawdown,
        cfg,
        get_objective("default"),
        now_ms_value=3_000,
    )
    assert gate.reason.startswith("low-confidence")

    zero_reference = _entry_gate(
        PositionsStore(root=tmp_path / "zero-reference"),
        Decision(side="LONG", confidence=0.9, mark_price=100.0),
        replace(no_drawdown, max_open_positions=1),
        _make_config(tmp_path / "zero-cfg", starting_reference_cash=0.0),
        get_objective("default"),
        now_ms_value=3_000,
    )
    assert zero_reference.allowed is True
    assert zero_reference.drawdown == 0.0


def test_run_loop_custom_logger_is_honored(tmp_path: Path) -> None:
    """Ensure the ``logger=`` override branch runs."""
    cfg = _make_config(tmp_path, stop_after_iterations=1)
    logger = logging.getLogger("autonomous-test-custom")
    result = run_loop(
        FakeClient(), _runtime(), _strategy(), cfg,
        decision_fn=lambda *_: Decision(side="FLAT", confidence=0.0, mark_price=100.0),
        sleep=lambda _d: None,
        clock=_tick_clock(),
        logger=logger,
    )
    assert isinstance(result, LoopResult)
    assert result.iterations == 1


def test_run_loop_skips_heartbeat_when_not_on_cadence(tmp_path: Path) -> None:
    """With heartbeat_every=2 and only 1 iteration, the heartbeat branch is skipped."""
    cfg = _make_config(tmp_path, stop_after_iterations=1, heartbeat_every=2)
    logger = logging.getLogger("autonomous-test-skip-hb")
    result = run_loop(
        FakeClient(), _runtime(), _strategy(), cfg,
        decision_fn=lambda *_: Decision(side="FLAT", confidence=0.0, mark_price=100.0),
        sleep=lambda _d: None,
        clock=_tick_clock(),
        logger=logger,
    )
    assert result.heartbeats_written == 0
    assert result.iterations == 1
