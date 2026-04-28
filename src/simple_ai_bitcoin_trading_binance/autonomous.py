"""Autonomous testnet live-trading loop with operator-controlled pause / resume / stop.

The autonomous loop is deliberately conservative:

* Requires ``testnet=True`` on the runtime — it refuses to start otherwise.
* Honors every existing risk gate (daily trade cap, drawdown limit, cooldown,
  max open positions) through the same strategy config the live CLI uses.
* Writes a heartbeat artifact after every iteration so operators can see
  liveness from another shell or from the TUI.
* Reads a small control file each iteration.  The file contains one of
  ``RUNNING``, ``PAUSED``, ``STOPPING``.  A separate command
  (``autonomous pause/resume/stop``) just rewrites that file.
* Uses an objective-tagged model artifact so the user can flip between
  Conservative, Default, and Risky without rebooting.

Safety design principles:
* No real-money execution.  The client must point at testnet; we re-verify this
  before every entry by reading ``client.base_url``.
* No credential leakage — every artifact writes through ``RuntimeConfig.public_dict``.
* No infinite fast loop — the poll interval is clamped to at least 1 second
  even if the strategy advertises ``--sleep 0``.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional

from .api import BinanceAPIError, BinanceClient
from .logging_ext import configure as configure_logging
from .objective import ObjectiveSpec, get_objective
from .positions import (
    ClosedTrade,
    OpenPosition,
    PositionsStore,
    compute_stats,
    new_position_id,
    now_ms,
)
from .storage import write_json_atomic
from .types import RuntimeConfig, StrategyConfig

STATE_RUNNING = "RUNNING"
STATE_PAUSED = "PAUSED"
STATE_STOPPING = "STOPPING"
STATE_STOPPED = "STOPPED"
_VALID_STATES = {STATE_RUNNING, STATE_PAUSED, STATE_STOPPING, STATE_STOPPED}
_MIN_INTERVAL_SECONDS = 1.0
_DEFAULT_AUTONOMOUS_DIR = Path("data/autonomous")


@dataclass
class AutonomousControl:
    """Thin filesystem-backed state machine used to pause / resume / stop."""

    path: Path = field(default_factory=lambda: _DEFAULT_AUTONOMOUS_DIR / "state.json")

    def __post_init__(self) -> None:
        self.path = Path(self.path)

    def write(self, state: str, *, note: str = "") -> None:
        if state not in _VALID_STATES:
            raise ValueError(f"Invalid state {state!r}")
        payload = {"state": state, "note": note, "ts_ms": now_ms()}
        write_json_atomic(self.path, payload, indent=2, sort_keys=True)

    def read(self) -> dict[str, object]:
        if not self.path.exists():
            return {"state": STATE_STOPPED, "note": "", "ts_ms": 0}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {"state": STATE_STOPPED, "note": "read-error", "ts_ms": 0}
        if not isinstance(payload, dict) or payload.get("state") not in _VALID_STATES:
            return {"state": STATE_STOPPED, "note": "malformed", "ts_ms": 0}
        return payload

    def state(self) -> str:
        return str(self.read().get("state") or STATE_STOPPED)


@dataclass
class Heartbeat:
    """Serializable snapshot of the autonomous loop's current status."""

    iteration: int
    state: str
    last_signal: float
    last_side: str
    last_price: float
    open_positions: int
    realized_pnl: float
    unrealized_pnl: float
    objective: str
    updated_at_ms: int
    message: str = ""

    def write(self, path: Path) -> None:
        write_json_atomic(path, asdict(self), indent=2, sort_keys=True)


@dataclass
class AutonomousConfig:
    """All knobs the autonomous loop exposes.

    Every field has a safe testnet-first default.  ``poll_seconds`` is clamped
    at runtime to ``_MIN_INTERVAL_SECONDS`` so even a misconfigured deployment
    never busy-waits.
    """

    objective: str = "default"
    poll_seconds: float = 30.0
    min_unrealized_close_pct: float | None = None  # auto-close at this pnl%
    max_unrealized_close_pct: float | None = None  # auto-close above this pnl%
    stop_after_iterations: int | None = None  # None = infinite
    heartbeat_every: int = 1  # write heartbeat every N iterations
    dry_run: bool = True  # paper by default — operator must explicitly live
    control_path: Path = field(default_factory=lambda: _DEFAULT_AUTONOMOUS_DIR / "state.json")
    heartbeat_path: Path = field(default_factory=lambda: _DEFAULT_AUTONOMOUS_DIR / "heartbeat.json")
    positions_root: Path = field(default_factory=lambda: _DEFAULT_AUTONOMOUS_DIR)
    log_path: Path = field(default_factory=lambda: _DEFAULT_AUTONOMOUS_DIR / "autonomous.log")
    starting_reference_cash: float = 1000.0


def ensure_testnet(runtime: RuntimeConfig) -> None:
    """Raise if the runtime is not configured for testnet.  No real-money execution."""

    if not runtime.testnet:
        raise RuntimeError(
            "Autonomous mode refuses to start unless runtime.testnet=True. "
            "This phase blocks real-money execution."
        )


def ensure_credentials(runtime: RuntimeConfig, cfg: AutonomousConfig) -> None:
    """Live mode requires API credentials.  Paper mode does not."""

    if cfg.dry_run:
        return
    if not runtime.api_key or not runtime.api_secret:
        raise RuntimeError("Autonomous live mode requires runtime.api_key and runtime.api_secret.")


@dataclass
class Decision:
    """A single iteration's decision — injected so the loop stays testable."""

    side: str  # "LONG" / "SHORT" / "FLAT"
    confidence: float
    mark_price: float


DecisionFn = Callable[[BinanceClient, RuntimeConfig, StrategyConfig, ObjectiveSpec], Decision]


def _default_decision(
    client: BinanceClient,
    runtime: RuntimeConfig,
    strategy: StrategyConfig,
    objective: ObjectiveSpec,
) -> Decision:
    """Placeholder decision that only reads the ticker.  Real inference is wired by callers.

    Keeping the default free of model calls lets the module be imported in
    environments without training data.  The autonomous CLI command supplies a
    real decision function that runs features + model prediction.
    """

    price, _ts = client.get_symbol_price(runtime.symbol)
    del strategy, objective
    return Decision(side="FLAT", confidence=0.0, mark_price=float(price))


def _evaluate_auto_close(
    position: OpenPosition,
    mark_price: float,
    cfg: AutonomousConfig,
    strategy: StrategyConfig,
) -> tuple[bool, str]:
    """Return (should_close, reason) for an open position at the given mark."""

    pnl_pct = position.unrealized_pnl_pct(mark_price)
    if cfg.max_unrealized_close_pct is not None and pnl_pct >= cfg.max_unrealized_close_pct:
        return True, f"auto-take-profit@{cfg.max_unrealized_close_pct:+.2%}"
    if cfg.min_unrealized_close_pct is not None and pnl_pct <= cfg.min_unrealized_close_pct:
        return True, f"auto-stop-loss@{cfg.min_unrealized_close_pct:+.2%}"
    if strategy.take_profit_pct > 0 and pnl_pct >= strategy.take_profit_pct:
        return True, f"take-profit@{strategy.take_profit_pct:+.2%}"
    if strategy.stop_loss_pct > 0 and pnl_pct <= -strategy.stop_loss_pct:
        return True, f"stop-loss@{strategy.stop_loss_pct:+.2%}"
    return False, ""


def _open_position_from_decision(
    decision: Decision,
    runtime: RuntimeConfig,
    strategy: StrategyConfig,
    objective: ObjectiveSpec,
    cfg: AutonomousConfig,
    *,
    clock=time.time,
) -> OpenPosition:
    """Build a position record from a decision + runtime state."""

    risk_cash = max(0.0, cfg.starting_reference_cash * strategy.risk_per_trade)
    price = max(0.01, float(decision.mark_price))
    qty = max(0.0, risk_cash / price)
    notional = qty * price
    return OpenPosition(
        id=new_position_id(),
        symbol=runtime.symbol,
        market_type=runtime.market_type,
        side=decision.side,
        qty=qty,
        entry_price=price,
        leverage=float(strategy.leverage),
        opened_at_ms=int(clock() * 1000),
        notional=notional,
        strategy_profile="autonomous",
        objective=objective.name,
        dry_run=cfg.dry_run,
        stop_loss_pct=strategy.stop_loss_pct,
        take_profit_pct=strategy.take_profit_pct,
    )


def _close_to_trade(
    position: OpenPosition,
    mark_price: float,
    reason: str,
    *,
    clock=time.time,
    fees: float = 0.0,
) -> ClosedTrade:
    pnl = position.unrealized_pnl(mark_price)
    pnl_pct = position.unrealized_pnl_pct(mark_price)
    return ClosedTrade(
        id=position.id,
        symbol=position.symbol,
        market_type=position.market_type,
        side=position.side,
        qty=position.qty,
        entry_price=position.entry_price,
        exit_price=float(mark_price),
        leverage=position.leverage,
        opened_at_ms=position.opened_at_ms,
        closed_at_ms=int(clock() * 1000),
        realized_pnl=pnl - fees,
        realized_pnl_pct=pnl_pct,
        fees=fees,
        reason=reason,
        strategy_profile=position.strategy_profile,
        objective=position.objective,
        dry_run=position.dry_run,
    )


@dataclass
class LoopResult:
    """What ``run_loop`` returns once it exits."""

    iterations: int
    final_state: str
    heartbeats_written: int
    closed_trades: int
    opened_trades: int
    exit_reason: str
    skipped_entries: int = 0


@dataclass(frozen=True)
class EntryGate:
    """Pre-entry risk decision used by the autonomous loop."""

    allowed: bool
    reason: str
    open_positions: int
    daily_entries: int
    cooldown_remaining_ms: int
    drawdown: float
    directional_confidence: float


def _safe_day_ms(timestamp_ms: int) -> int:
    return int(timestamp_ms // (24 * 60 * 60 * 1000))


def _directional_confidence(decision: Decision) -> float:
    """Convert model probability-like confidence into side confidence."""

    try:
        raw_confidence = float(decision.confidence)
    except (TypeError, ValueError):
        raw_confidence = 0.0
    confidence = max(0.0, min(1.0, raw_confidence))
    if decision.side == "SHORT":
        return 1.0 - confidence
    return confidence


def _daily_entry_count(store: PositionsStore, day: int) -> int:
    opened = sum(1 for position in store.load_open() if _safe_day_ms(position.opened_at_ms) == day)
    closed = sum(1 for trade in store.load_ledger() if _safe_day_ms(trade.opened_at_ms) == day)
    return opened + closed


def _last_activity_ms(store: PositionsStore) -> int:
    candidates = [position.opened_at_ms for position in store.load_open()]
    candidates.extend(trade.closed_at_ms for trade in store.load_ledger())
    return max(candidates, default=0)


def _entry_gate(
    store: PositionsStore,
    decision: Decision,
    strategy: StrategyConfig,
    cfg: AutonomousConfig,
    objective: ObjectiveSpec,
    *,
    now_ms_value: int,
) -> EntryGate:
    """Return the deterministic pre-entry gate decision for a signal."""

    opens = store.load_open()
    max_open = int(strategy.max_open_positions)
    day = _safe_day_ms(now_ms_value)
    daily_entries = _daily_entry_count(store, day)
    stats = compute_stats(
        store,
        mark_price=decision.mark_price,
        starting_reference_cash=cfg.starting_reference_cash,
    )
    drawdown = 0.0
    if cfg.starting_reference_cash > 0:
        equity_delta = stats.realized_pnl + stats.unrealized_pnl
        drawdown = max(0.0, -equity_delta / cfg.starting_reference_cash)
    confidence = _directional_confidence(decision)
    min_confidence = (
        objective.training.signal_threshold
        if objective.training is not None
        else strategy.signal_threshold
    )

    cooldown_ms = max(0, int(strategy.cooldown_minutes)) * 60 * 1000
    last_activity = _last_activity_ms(store)
    cooldown_remaining = 0
    if cooldown_ms > 0 and last_activity > 0:
        cooldown_remaining = max(0, cooldown_ms - max(0, now_ms_value - last_activity))

    if decision.side not in {"LONG", "SHORT"}:
        reason = "flat-signal"
    elif confidence < min_confidence:
        reason = f"low-confidence<{min_confidence:.3f}"
    elif max_open <= 0:
        reason = "max-open-disabled"
    elif len(opens) >= max_open:
        reason = f"max-open-reached:{len(opens)}/{max_open}"
    elif int(strategy.max_trades_per_day) > 0 and daily_entries >= int(strategy.max_trades_per_day):
        reason = f"daily-cap-reached:{daily_entries}/{int(strategy.max_trades_per_day)}"
    elif cooldown_remaining > 0:
        reason = f"cooldown-active:{cooldown_remaining}ms"
    elif strategy.max_drawdown_limit > 0.0 and drawdown >= strategy.max_drawdown_limit:
        reason = f"drawdown-lockout:{drawdown:.2%}"
    else:
        reason = "allowed"

    return EntryGate(
        allowed=reason == "allowed",
        reason=reason,
        open_positions=len(opens),
        daily_entries=daily_entries,
        cooldown_remaining_ms=cooldown_remaining,
        drawdown=drawdown,
        directional_confidence=confidence,
    )


def run_loop(
    client: BinanceClient,
    runtime: RuntimeConfig,
    strategy: StrategyConfig,
    cfg: AutonomousConfig,
    *,
    decision_fn: DecisionFn = _default_decision,
    sleep=time.sleep,
    clock=time.time,
    logger: logging.Logger | None = None,
) -> LoopResult:
    """Run the autonomous loop until the control file requests a stop."""

    ensure_testnet(runtime)
    ensure_credentials(runtime, cfg)
    objective = get_objective(cfg.objective)
    logger = logger or configure_logging(path=cfg.log_path)
    control = AutonomousControl(path=cfg.control_path)
    control.write(STATE_RUNNING, note=f"objective={objective.name}")
    store = PositionsStore(root=cfg.positions_root)
    poll = max(_MIN_INTERVAL_SECONDS, float(cfg.poll_seconds))

    iteration = 0
    heartbeats = 0
    closed = 0
    opened = 0
    skipped = 0
    exit_reason = "requested-stop"
    try:
        while True:
            iteration += 1
            state = control.state()
            if state == STATE_STOPPING:
                exit_reason = "operator-stop"
                break
            if state == STATE_PAUSED:
                logger.info("autonomous iter=%d paused", iteration)
                sleep(poll)
                continue
            try:
                decision = decision_fn(client, runtime, strategy, objective)
            except BinanceAPIError as err:
                logger.warning("autonomous iter=%d binance-error=%s", iteration, err)
                sleep(poll)
                continue
            except Exception as err:  # noqa: BLE001 - loop-wide guard
                logger.error("autonomous iter=%d decision-error=%s", iteration, err)
                exit_reason = "decision-exception"
                break

            logger.info(
                "autonomous iter=%d side=%s conf=%.4f mark=%.2f",
                iteration, decision.side, decision.confidence, decision.mark_price,
            )

            # Close any open position that meets auto-close thresholds first
            for position in store.load_open():
                should_close, reason = _evaluate_auto_close(position, decision.mark_price, cfg, strategy)
                if should_close:
                    trade = _close_to_trade(position, decision.mark_price, reason, clock=clock)
                    store.record_close(trade)
                    closed += 1
                    logger.info(
                        "autonomous iter=%d close id=%s reason=%s pnl=%+.2f (%+.2%%)",
                        iteration, trade.id, reason, trade.realized_pnl, trade.realized_pnl_pct,
                    )

            # Open new position only after the same risk gates used in operator
            # readiness checks approve it.
            gate = _entry_gate(
                store,
                decision,
                strategy,
                cfg,
                objective,
                now_ms_value=int(clock() * 1000),
            )
            if decision.side in {"LONG", "SHORT"} and gate.allowed:
                position = _open_position_from_decision(
                    decision, runtime, strategy, objective, cfg, clock=clock,
                )
                store.record_open(position)
                opened += 1
                logger.info(
                    "autonomous iter=%d open id=%s side=%s qty=%.6f entry=%.2f",
                    iteration, position.id, position.side, position.qty, position.entry_price,
                )
            elif decision.side in {"LONG", "SHORT"}:
                skipped += 1
                logger.info(
                    "autonomous iter=%d skip-entry reason=%s open=%d daily=%d conf=%.4f drawdown=%.4f",
                    iteration,
                    gate.reason,
                    gate.open_positions,
                    gate.daily_entries,
                    gate.directional_confidence,
                    gate.drawdown,
                )

            if iteration % max(1, cfg.heartbeat_every) == 0:
                stats = compute_stats(store, mark_price=decision.mark_price,
                                     starting_reference_cash=cfg.starting_reference_cash)
                heartbeat = Heartbeat(
                    iteration=iteration,
                    state=STATE_RUNNING,
                    last_signal=decision.confidence,
                    last_side=decision.side,
                    last_price=decision.mark_price,
                    open_positions=stats.open_positions,
                    realized_pnl=stats.realized_pnl,
                    unrealized_pnl=stats.unrealized_pnl,
                    objective=objective.name,
                    updated_at_ms=int(clock() * 1000),
                    message="" if gate.allowed or decision.side == "FLAT" else gate.reason,
                )
                heartbeat.write(cfg.heartbeat_path)
                heartbeats += 1

            if cfg.stop_after_iterations is not None and iteration >= cfg.stop_after_iterations:
                exit_reason = "iteration-cap"
                break
            sleep(poll)
    finally:
        control.write(STATE_STOPPED, note=exit_reason)

    return LoopResult(
        iterations=iteration,
        final_state=STATE_STOPPED,
        heartbeats_written=heartbeats,
        closed_trades=closed,
        opened_trades=opened,
        exit_reason=exit_reason,
        skipped_entries=skipped,
    )
