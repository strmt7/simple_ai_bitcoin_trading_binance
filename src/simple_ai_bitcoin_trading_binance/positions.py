"""Positions ledger + realized/unrealized P&L accounting for the autonomous loop.

The existing backtest and live-loop code keep the single-position contract
``max_open_positions=1`` in memory only.  Autonomous mode needs durable state
so a restart (or an operator running ``autonomous status`` from another shell)
can see exactly what is open and what has been realized.

Two JSON files live under ``data/autonomous/``:

* ``open_positions.json`` — a list of currently open positions.
* ``ledger.json`` — a list of every closed trade in chronological order.

Entries are small and human-readable.  No credentials, no raw order IDs beyond
what the exchange already returned, and all numeric fields are plain floats so
the file loads fine with ``python -m json.tool``.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .storage import write_json_atomic


@dataclass
class OpenPosition:
    """An open position on the exchange as tracked by this process."""

    id: str
    symbol: str
    market_type: str
    side: str  # "LONG" or "SHORT"
    qty: float
    entry_price: float
    leverage: float
    opened_at_ms: int
    notional: float
    strategy_profile: str = ""
    objective: str = ""
    dry_run: bool = True
    stop_loss_pct: float = 0.0
    take_profit_pct: float = 0.0

    def unrealized_pnl(self, mark_price: float) -> float:
        if self.side == "LONG":
            return (mark_price - self.entry_price) * self.qty
        return (self.entry_price - mark_price) * self.qty

    def unrealized_pnl_pct(self, mark_price: float) -> float:
        if self.entry_price <= 0:
            return 0.0
        if self.side == "LONG":
            return (mark_price - self.entry_price) / self.entry_price
        return (self.entry_price - mark_price) / self.entry_price


@dataclass
class ClosedTrade:
    """A completed round-trip recorded in the ledger."""

    id: str
    symbol: str
    market_type: str
    side: str
    qty: float
    entry_price: float
    exit_price: float
    leverage: float
    opened_at_ms: int
    closed_at_ms: int
    realized_pnl: float
    realized_pnl_pct: float
    fees: float = 0.0
    reason: str = ""
    strategy_profile: str = ""
    objective: str = ""
    dry_run: bool = True


@dataclass
class LedgerStats:
    """Aggregate statistics across closed trades + open unrealized exposure."""

    closed_trades: int
    wins: int
    losses: int
    realized_pnl: float
    realized_pnl_pct: float
    win_rate: float
    total_fees: float
    largest_win: float
    largest_loss: float
    open_positions: int
    unrealized_pnl: float
    unrealized_pnl_pct: float
    starting_reference_cash: float

    def asdict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PositionsStore:
    """Durable storage for open positions + closed trades ledger."""

    root: Path = field(default_factory=lambda: Path("data/autonomous"))

    def __post_init__(self) -> None:
        self.root = Path(self.root)

    @property
    def open_path(self) -> Path:
        return self.root / "open_positions.json"

    @property
    def ledger_path(self) -> Path:
        return self.root / "ledger.json"

    # ---- low-level I/O ------------------------------------------------------

    def _load(self, path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []
        if not isinstance(payload, list):
            return []
        return [entry for entry in payload if isinstance(entry, dict)]

    def _write(self, path: Path, payload: list[dict[str, Any]]) -> None:
        write_json_atomic(path, payload, indent=2, sort_keys=True)

    # ---- public API ---------------------------------------------------------

    def load_open(self) -> list[OpenPosition]:
        return [OpenPosition(**entry) for entry in self._load(self.open_path)
                if self._valid_open_entry(entry)]

    def load_ledger(self) -> list[ClosedTrade]:
        return [ClosedTrade(**entry) for entry in self._load(self.ledger_path)
                if self._valid_closed_entry(entry)]

    def record_open(self, position: OpenPosition) -> OpenPosition:
        existing = self.load_open()
        existing = [p for p in existing if p.id != position.id]
        existing.append(position)
        self._write(self.open_path, [asdict(p) for p in existing])
        return position

    def record_close(self, trade: ClosedTrade) -> ClosedTrade:
        existing = self.load_ledger()
        existing.append(trade)
        self._write(self.ledger_path, [asdict(t) for t in existing])
        # also drop the matching open entry if present
        opens = [p for p in self.load_open() if p.id != trade.id]
        self._write(self.open_path, [asdict(p) for p in opens])
        return trade

    def remove_open(self, position_id: str) -> bool:
        opens = self.load_open()
        filtered = [p for p in opens if p.id != position_id]
        if len(filtered) == len(opens):
            return False
        self._write(self.open_path, [asdict(p) for p in filtered])
        return True

    def find_open(self, position_id: str) -> OpenPosition | None:
        for position in self.load_open():
            if position.id == position_id:
                return position
        return None

    # ---- validation helpers -------------------------------------------------

    @staticmethod
    def _valid_open_entry(entry: dict[str, Any]) -> bool:
        required = {"id", "symbol", "market_type", "side", "qty", "entry_price",
                    "leverage", "opened_at_ms", "notional"}
        return required.issubset(entry.keys())

    @staticmethod
    def _valid_closed_entry(entry: dict[str, Any]) -> bool:
        required = {"id", "symbol", "market_type", "side", "qty", "entry_price",
                    "exit_price", "leverage", "opened_at_ms", "closed_at_ms",
                    "realized_pnl", "realized_pnl_pct"}
        return required.issubset(entry.keys())


def new_position_id() -> str:
    """Generate a short, collision-resistant identifier for a position."""

    return uuid.uuid4().hex[:12]


def now_ms(clock=time.time) -> int:
    """Current wall clock in milliseconds (injected for deterministic tests)."""

    return int(clock() * 1000)


def unrealized_pnl_usd(position: OpenPosition, mark_price: float) -> float:
    """Return the USD-denominated unrealized P&L for ``position`` at ``mark_price``."""

    return position.unrealized_pnl(mark_price)


def unrealized_pnl_pct(position: OpenPosition, mark_price: float) -> float:
    """Return the unrealized P&L as a fraction of entry notional."""

    return position.unrealized_pnl_pct(mark_price)


def compute_stats(
    store: PositionsStore,
    *,
    mark_price: float | None,
    starting_reference_cash: float = 1000.0,
) -> LedgerStats:
    """Compute aggregate statistics from the ledger + open positions.

    ``mark_price`` is the current BTCUSDC mark used to value open positions;
    pass ``None`` if the mark is unavailable — unrealized fields will be zero
    rather than raising.
    """

    closed = store.load_ledger()
    opens = store.load_open()
    wins = sum(1 for t in closed if t.realized_pnl > 0)
    losses = sum(1 for t in closed if t.realized_pnl < 0)
    realized = sum(t.realized_pnl for t in closed)
    total_fees = sum(t.fees for t in closed)
    largest_win = max((t.realized_pnl for t in closed if t.realized_pnl > 0), default=0.0)
    largest_loss = min((t.realized_pnl for t in closed if t.realized_pnl < 0), default=0.0)
    unrealized = 0.0
    unrealized_pct = 0.0
    if mark_price is not None and opens:
        unrealized = sum(p.unrealized_pnl(float(mark_price)) for p in opens)
        entry_notional = sum(abs(p.entry_price * p.qty) for p in opens)
        if entry_notional > 0:
            unrealized_pct = unrealized / entry_notional
    realized_pct = (realized / starting_reference_cash) if starting_reference_cash > 0 else 0.0
    return LedgerStats(
        closed_trades=len(closed),
        wins=wins,
        losses=losses,
        realized_pnl=realized,
        realized_pnl_pct=realized_pct,
        win_rate=(wins / len(closed)) if closed else 0.0,
        total_fees=total_fees,
        largest_win=largest_win,
        largest_loss=largest_loss,
        open_positions=len(opens),
        unrealized_pnl=unrealized,
        unrealized_pnl_pct=unrealized_pct,
        starting_reference_cash=starting_reference_cash,
    )


def render_positions_table(
    positions: list[OpenPosition],
    *,
    mark_price: float | None,
) -> list[str]:
    """Return a 2D-aligned textual table of open positions.

    Returns an empty list if there is nothing to show so callers can branch on
    truthiness without peeking at the length.
    """

    if not positions:
        return []
    header = f"{'#':>2} {'id':<12} {'side':<5} {'qty':>10} {'entry':>12} {'mark':>12} {'pnl$':>12} {'pnl%':>7}"
    rows = [header]
    for idx, pos in enumerate(positions, start=1):
        mark_value = mark_price if mark_price is not None else pos.entry_price
        pnl_usd = pos.unrealized_pnl(mark_value) if mark_price is not None else 0.0
        pnl_pct = pos.unrealized_pnl_pct(mark_value) if mark_price is not None else 0.0
        rows.append(
            f"{idx:>2} {pos.id:<12} {pos.side:<5} "
            f"{pos.qty:>10.6f} {pos.entry_price:>12.2f} "
            f"{mark_value:>12.2f} {pnl_usd:>+12.2f} {pnl_pct:>+7.2%}"
        )
    return rows


def render_stats_lines(stats: LedgerStats) -> list[str]:
    """Human-friendly rendering of ``LedgerStats`` suitable for the shell."""

    lines = [
        f"Closed trades  : {stats.closed_trades}  (wins {stats.wins}, losses {stats.losses})",
        f"Realized P&L   : {stats.realized_pnl:+.2f} USDC  ({stats.realized_pnl_pct:+.2%})",
        f"Win rate       : {stats.win_rate:.0%}",
        f"Total fees     : {stats.total_fees:.2f} USDC",
        f"Largest win    : {stats.largest_win:+.2f} USDC",
        f"Largest loss   : {stats.largest_loss:+.2f} USDC",
        f"Open positions : {stats.open_positions}",
        f"Unrealized P&L : {stats.unrealized_pnl:+.2f} USDC  ({stats.unrealized_pnl_pct:+.2%})",
    ]
    return lines
