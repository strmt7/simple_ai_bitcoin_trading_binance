"""Claude-Code-inspired interactive shell for the trading CLI.

Single-line prompt, slash commands, tab-completion (when readline is available),
and a fall-through to the existing argparse CLI so any subcommand can be run
from inside the shell without leaving it.

Design goals:

* The shell is pure stdlib — no curses, no prompt_toolkit.  That keeps the
  dependency footprint at ``requests`` + ``textual`` like the rest of the repo.
* Every seam that would normally touch the outside world (stdout, readline,
  the CLI dispatcher, the autonomous control file, the positions store) is
  parameterizable so tests can drive the whole thing without I/O.
* Built-in commands live in ``_default_commands`` so callers can add or
  override individual commands without touching the shell's runtime.
"""

from __future__ import annotations

import shlex
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .autonomous import (
    STATE_PAUSED,
    STATE_RUNNING,
    STATE_STOPPING,
    AutonomousControl,
)
from .backtest_panel import describe_supported_intervals, list_reports
from .objective import available_objectives, describe_objectives
from .positions import (
    PositionsStore,
    compute_stats,
    render_positions_table,
    render_stats_lines,
)
from .style import (
    Palette,
    bad,
    bold,
    color,
    frame,
    muted,
    ok,
    supports_color,
    warn,
)

CliRunner = Callable[[list[str]], int]


@dataclass(frozen=True)
class SlashCommand:
    """A single slash-prefixed operator command registered with the shell."""

    name: str
    summary: str
    run: Callable[["Shell", list[str]], int]


@dataclass
class ShellState:
    """Mutable per-session state for the shell."""

    cwd: Path = field(default_factory=lambda: Path("."))
    history: list[str] = field(default_factory=list)
    last_exit: int = 0
    color_enabled: bool = False


def _default_cli_runner(argv: list[str]) -> int:
    """Lazy import so shell.py can be imported without pulling the whole CLI graph."""

    from .cli import main as cli_main

    return int(cli_main(argv))


class Shell:
    """Interactive shell session."""

    def __init__(
        self,
        *,
        commands: dict[str, SlashCommand] | None = None,
        reader: Callable[[str], str] = input,
        writer: Callable[[str], None] = print,
        clock: Callable[[], float] = time.time,
        palette: Palette | None = None,
        color_enabled: bool | None = None,
        cli_runner: CliRunner = _default_cli_runner,
        control_factory: Callable[[], AutonomousControl] = AutonomousControl,
        positions_factory: Callable[[], PositionsStore] = PositionsStore,
    ) -> None:
        self.reader = reader
        self.writer = writer
        self.clock = clock
        self.palette = palette or Palette()
        self.state = ShellState()
        self.state.color_enabled = supports_color() if color_enabled is None else bool(color_enabled)
        self.cli_runner = cli_runner
        self.control_factory = control_factory
        self.positions_factory = positions_factory
        self._completion_cache: list[str] = []
        if commands is None:
            self.commands: dict[str, SlashCommand] = {}
            for cmd in _default_commands():
                self.commands[cmd.name] = cmd
        else:
            self.commands = dict(commands)

    # ---- output helpers ----------------------------------------------------

    def print(self, text: str) -> None:
        self.writer(str(text))

    def println(self, text: str) -> None:
        self.writer(str(text))

    def banner(self) -> str:
        enabled = self.state.color_enabled
        line1 = bold(
            color("  simple-ai ", self.palette.heading, enabled=enabled)
            + color("▸ binance testnet", self.palette.accent, enabled=enabled),
            enabled=enabled,
        )
        line2 = muted("  type /help for commands, /quit to exit",
                      enabled=enabled, palette=self.palette)
        return f"{line1}\n{line2}"

    def prompt_text(self) -> str:
        return bold(
            color("simple-ai ❯ ", self.palette.primary, enabled=self.state.color_enabled),
            enabled=self.state.color_enabled,
        )

    # ---- registration ------------------------------------------------------

    def register(self, command: SlashCommand) -> None:
        self.commands[command.name] = command

    # ---- completion --------------------------------------------------------

    def complete(self, text: str, state: int) -> str | None:
        if state == 0:
            stripped = text.lstrip("/")
            slash_prefix = "/" if text.startswith("/") else ""
            matches = [
                f"{slash_prefix}{name}"
                for name in sorted(self.commands)
                if name.startswith(stripped)
            ]
            self._completion_cache = matches
        if state < 0 or state >= len(self._completion_cache):
            return None
        return self._completion_cache[state]

    # ---- dispatch ----------------------------------------------------------

    def dispatch(self, raw: str) -> int:
        line = raw.strip()
        if not line:
            return 0
        try:
            tokens = shlex.split(line)
        except ValueError as err:
            self.println(bad(f"parse error: {err}",
                             enabled=self.state.color_enabled, palette=self.palette))
            return 2
        # ``line`` is non-empty after the earlier guard, so shlex.split yields ≥1 token.
        head = tokens[0]
        rest = tokens[1:]
        if head in {"?", "help", "/help"}:
            head = "/help"
        slash = head.startswith("/")
        name = head.lstrip("/")
        if slash and name in self.commands:
            try:
                return int(self.commands[name].run(self, rest))
            except SystemExit:
                raise
            except Exception as err:  # noqa: BLE001 - surfaced for the operator
                self.println(bad(f"error: {err}",
                                 enabled=self.state.color_enabled, palette=self.palette))
                return 1
        if slash:
            self.println(warn(f"Unknown command {head!r} — try /help",
                              enabled=self.state.color_enabled, palette=self.palette))
            return 2
        # fall-through: pass the full tokens to the CLI dispatcher
        try:
            return int(self.cli_runner(tokens))
        except SystemExit as exit_err:  # argparse uses SystemExit for errors
            return int(exit_err.code or 0)
        except Exception as err:  # noqa: BLE001
            self.println(bad(f"error: {err}",
                             enabled=self.state.color_enabled, palette=self.palette))
            return 1

    # ---- main loop ---------------------------------------------------------

    def _prompt_line(self) -> str | None:
        """Read one line from the user.  Returns None on EOF, empty string on ^C."""

        try:
            return self.reader(self.prompt_text())
        except EOFError:
            self.println("")
            return None
        except KeyboardInterrupt:
            self.println(muted("^C", enabled=self.state.color_enabled, palette=self.palette))
            return ""

    def run(self) -> int:
        self.println(self.banner())
        while True:
            line = self._prompt_line()
            if line is None:
                return int(self.state.last_exit)
            if line.strip():
                self.state.history.append(line)
            try:
                self.state.last_exit = self.dispatch(line)
            except SystemExit as exit_err:
                return int(exit_err.code or 0)


# ==========================================================================
# Built-in commands
# ==========================================================================


def _cmd_help(shell: Shell, _args: list[str]) -> int:
    enabled = shell.state.color_enabled
    lines = [bold(f"{'command':<14} summary", enabled=enabled)]
    for name in sorted(shell.commands):
        cmd = shell.commands[name]
        lines.append(f"/{name:<13} {cmd.summary}")
    shell.println("\n".join(lines))
    return 0


def _cmd_quit(shell: Shell, _args: list[str]) -> int:
    raise SystemExit(shell.state.last_exit)


def _cmd_clear(shell: Shell, _args: list[str]) -> int:
    if shell.state.color_enabled:
        shell.println("\x1b[2J\x1b[H")
    else:
        shell.println("\n" * 2)
    return 0


def _cmd_status(shell: Shell, _args: list[str]) -> int:
    enabled = shell.state.color_enabled
    lines = frame(
        "shell",
        [
            f"cwd         : {shell.state.cwd}",
            f"color       : {'on' if enabled else 'off'}",
            f"last_exit   : {shell.state.last_exit}",
            f"history     : {len(shell.state.history)} entries",
            f"commands    : {len(shell.commands)} registered",
        ],
        width=60,
        enabled=enabled,
        palette=shell.palette,
    )
    shell.println("\n".join(lines))
    return 0


def _cmd_history(shell: Shell, _args: list[str]) -> int:
    if not shell.state.history:
        shell.println(muted("(history is empty)",
                            enabled=shell.state.color_enabled, palette=shell.palette))
        return 0
    for idx, line in enumerate(shell.state.history, start=1):
        shell.println(f"{idx:>4}  {line}")
    return 0


def _cmd_palette(shell: Shell, _args: list[str]) -> int:
    enabled = shell.state.color_enabled
    swatches = [
        color("primary", shell.palette.primary, enabled=enabled),
        color("accent", shell.palette.accent, enabled=enabled),
        ok("ok", enabled=enabled, palette=shell.palette),
        warn("warn", enabled=enabled, palette=shell.palette),
        bad("bad", enabled=enabled, palette=shell.palette),
        muted("muted", enabled=enabled, palette=shell.palette),
    ]
    shell.println(" ".join(swatches))
    return 0


def _cmd_intervals(shell: Shell, args: list[str]) -> int:
    market = args[0].lower() if args else "spot"
    if market not in {"spot", "futures"}:
        shell.println(bad(f"unknown market {market!r}; use spot or futures",
                          enabled=shell.state.color_enabled, palette=shell.palette))
        return 2
    shell.println(f"{market}: {describe_supported_intervals(market)}")
    return 0


def _cmd_objectives(shell: Shell, _args: list[str]) -> int:
    enabled = shell.state.color_enabled
    lines = [bold(f"{'name':<14} {'label':<14} summary", enabled=enabled)]
    for entry in describe_objectives():
        lines.append(f"{entry['name']:<14} {entry['label']:<14} {entry['summary']}")
    shell.println("\n".join(lines))
    return 0


def _cmd_positions(shell: Shell, _args: list[str]) -> int:
    store = shell.positions_factory()
    opens = store.load_open()
    if not opens:
        shell.println(muted("(no open positions)",
                            enabled=shell.state.color_enabled, palette=shell.palette))
        return 0
    rows = render_positions_table(opens, mark_price=None)
    shell.println("\n".join(rows))
    return 0


def _cmd_stats(shell: Shell, _args: list[str]) -> int:
    store = shell.positions_factory()
    stats = compute_stats(store, mark_price=None)
    shell.println("\n".join(render_stats_lines(stats)))
    return 0


def _cmd_close(shell: Shell, args: list[str]) -> int:
    if not args:
        shell.println(bad("usage: /close <id|all>",
                          enabled=shell.state.color_enabled, palette=shell.palette))
        return 2
    store = shell.positions_factory()
    target = args[0]
    if target.lower() == "all":
        opens = store.load_open()
        for position in opens:
            store.remove_open(position.id)
        shell.println(ok(f"closed {len(opens)} positions (local ledger only)",
                         enabled=shell.state.color_enabled, palette=shell.palette))
        return 0
    if store.remove_open(target):
        shell.println(ok(f"closed {target} (local ledger only)",
                         enabled=shell.state.color_enabled, palette=shell.palette))
        return 0
    shell.println(warn(f"no open position with id {target!r}",
                       enabled=shell.state.color_enabled, palette=shell.palette))
    return 1


def _cmd_auto(shell: Shell, args: list[str]) -> int:
    if not args:
        shell.println(bad("usage: /auto {start|pause|resume|stop|status} [args]",
                          enabled=shell.state.color_enabled, palette=shell.palette))
        return 2
    action = args[0].lower()
    rest = args[1:]
    control = shell.control_factory()
    if action == "start":
        objective = "default"
        if "--objective" in rest:
            idx = rest.index("--objective")
            if idx + 1 >= len(rest):
                shell.println(bad("--objective needs a value",
                                  enabled=shell.state.color_enabled, palette=shell.palette))
                return 2
            objective = rest[idx + 1]
        if objective not in available_objectives():
            shell.println(bad(f"unknown objective {objective!r}",
                              enabled=shell.state.color_enabled, palette=shell.palette))
            return 2
        control.write(STATE_RUNNING, note=f"started via shell objective={objective}")
        shell.println(ok(f"autonomous: RUNNING (objective={objective})",
                         enabled=shell.state.color_enabled, palette=shell.palette))
        return 0
    if action == "pause":
        control.write(STATE_PAUSED, note="paused via shell")
        shell.println(ok("autonomous: PAUSED",
                         enabled=shell.state.color_enabled, palette=shell.palette))
        return 0
    if action == "resume":
        control.write(STATE_RUNNING, note="resumed via shell")
        shell.println(ok("autonomous: RUNNING",
                         enabled=shell.state.color_enabled, palette=shell.palette))
        return 0
    if action == "stop":
        control.write(STATE_STOPPING, note="stop requested via shell")
        shell.println(ok("autonomous: STOPPING",
                         enabled=shell.state.color_enabled, palette=shell.palette))
        return 0
    if action == "status":
        payload = control.read()
        shell.println(
            f"state={payload.get('state')} note={payload.get('note') or ''} "
            f"ts_ms={payload.get('ts_ms')}"
        )
        return 0
    shell.println(bad(f"unknown /auto action {action!r}",
                      enabled=shell.state.color_enabled, palette=shell.palette))
    return 2


def _cmd_backtests(shell: Shell, _args: list[str]) -> int:
    listings = list_reports()
    if not listings:
        shell.println(muted("(no reports under data/backtests)",
                            enabled=shell.state.color_enabled, palette=shell.palette))
        return 0
    enabled = shell.state.color_enabled
    lines = [bold(f"{'created_at':<28} {'market':<8} {'interval':<8} {'tag':<20} path",
                  enabled=enabled)]
    for item in listings:
        lines.append(
            f"{item.created_at:<28} {item.market:<8} {item.interval:<8} "
            f"{item.tag:<20} {item.path}"
        )
    shell.println("\n".join(lines))
    return 0


def _default_commands() -> list[SlashCommand]:
    return [
        SlashCommand("help", "list registered commands", _cmd_help),
        SlashCommand("quit", "exit the shell", _cmd_quit),
        SlashCommand("exit", "exit the shell", _cmd_quit),
        SlashCommand("clear", "clear the screen", _cmd_clear),
        SlashCommand("status", "show shell state", _cmd_status),
        SlashCommand("history", "show input history", _cmd_history),
        SlashCommand("palette", "preview color palette", _cmd_palette),
        SlashCommand("intervals", "list Binance-supported intervals", _cmd_intervals),
        SlashCommand("objectives", "list registered training objectives", _cmd_objectives),
        SlashCommand("positions", "list open autonomous positions", _cmd_positions),
        SlashCommand("stats", "summarize realized + unrealized P&L", _cmd_stats),
        SlashCommand("close", "close an autonomous position locally", _cmd_close),
        SlashCommand("auto", "control the autonomous loop", _cmd_auto),
        SlashCommand("backtests", "list saved backtest reports", _cmd_backtests),
    ]


def run_shell(argv: list[str] | None = None) -> int:
    """Entry point used by the CLI subcommand."""

    del argv  # reserved for future flags like --no-color; currently ignored.
    shell = Shell()
    return int(shell.run())


def main() -> int:
    return run_shell(sys.argv[1:])


__all__ = [
    "Shell",
    "ShellState",
    "SlashCommand",
    "main",
    "run_shell",
]
