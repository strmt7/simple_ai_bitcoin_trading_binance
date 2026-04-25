"""Branch-coverage tests for the interactive shell module."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from simple_ai_bitcoin_trading_binance import shell as shell_mod
from simple_ai_bitcoin_trading_binance.autonomous import (
    STATE_PAUSED,
    STATE_RUNNING,
    STATE_STOPPING,
    AutonomousControl,
)
from simple_ai_bitcoin_trading_binance.positions import (
    OpenPosition,
    PositionsStore,
)
from simple_ai_bitcoin_trading_binance.style import Palette


class _Recorder:
    """Captures every line the shell writes for later assertions."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def __call__(self, text: str) -> None:  # acts as ``writer``
        for piece in str(text).splitlines() or [""]:
            self.lines.append(piece)

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


def _make_shell(
    tmp_path: Path,
    *,
    reader=None,
    cli_runner=None,
    color_enabled: bool = False,
) -> tuple[shell_mod.Shell, _Recorder, Path, Path]:
    control_path = tmp_path / "control.json"
    positions_root = tmp_path / "positions"
    positions_root.mkdir(exist_ok=True)
    recorder = _Recorder()

    def _reader(prompt: str) -> str:
        raise EOFError  # never exercised unless the test overrides

    shell = shell_mod.Shell(
        reader=reader or _reader,
        writer=recorder,
        palette=Palette(),
        color_enabled=color_enabled,
        cli_runner=cli_runner or (lambda argv: 0),
        control_factory=lambda: AutonomousControl(path=control_path),
        positions_factory=lambda: PositionsStore(root=positions_root),
    )
    return shell, recorder, control_path, positions_root


def test_banner_and_prompt_unstyled(tmp_path):
    shell, _recorder, _ctrl, _pos = _make_shell(tmp_path)
    banner = shell.banner()
    assert "simple-ai" in banner
    assert "\x1b[" not in banner  # color disabled
    assert "❯" in shell.prompt_text()


def test_banner_styled(tmp_path):
    shell, _recorder, _ctrl, _pos = _make_shell(tmp_path, color_enabled=True)
    banner = shell.banner()
    assert "\x1b[" in banner  # some ANSI escape
    prompt = shell.prompt_text()
    assert "\x1b[" in prompt


def test_empty_dispatch_no_output(tmp_path):
    shell, recorder, _ctrl, _pos = _make_shell(tmp_path)
    assert shell.dispatch("") == 0
    assert shell.dispatch("   ") == 0
    assert recorder.lines == []


def test_help_via_slash_and_bare_and_qmark(tmp_path):
    shell, recorder, _ctrl, _pos = _make_shell(tmp_path)
    assert shell.dispatch("/help") == 0
    assert shell.dispatch("help") == 0
    assert shell.dispatch("?") == 0
    assert any("command" in line for line in recorder.lines)
    assert any("/help" in line for line in recorder.lines)


def test_unknown_slash_command(tmp_path):
    shell, recorder, _ctrl, _pos = _make_shell(tmp_path)
    assert shell.dispatch("/nope") == 2
    assert any("Unknown command" in line for line in recorder.lines)


def test_command_raises_exception(tmp_path):
    shell, recorder, _ctrl, _pos = _make_shell(tmp_path)

    def boom(_shell, _args):
        raise RuntimeError("kaboom")

    shell.register(shell_mod.SlashCommand("boom", "explodes", boom))
    assert shell.dispatch("/boom") == 1
    assert any("error: kaboom" in line for line in recorder.lines)


def test_fallthrough_cli_runner(tmp_path):
    calls: list[list[str]] = []

    def runner(argv):
        calls.append(argv)
        return 7

    shell, _recorder, _ctrl, _pos = _make_shell(tmp_path, cli_runner=runner)
    assert shell.dispatch("status --foo bar") == 7
    assert calls == [["status", "--foo", "bar"]]


def test_fallthrough_cli_systemexit(tmp_path):
    def runner(argv):
        raise SystemExit(5)

    shell, _recorder, _ctrl, _pos = _make_shell(tmp_path, cli_runner=runner)
    assert shell.dispatch("status") == 5


def test_fallthrough_cli_systemexit_none(tmp_path):
    def runner(argv):
        raise SystemExit(None)

    shell, _recorder, _ctrl, _pos = _make_shell(tmp_path, cli_runner=runner)
    assert shell.dispatch("status") == 0


def test_fallthrough_cli_raises(tmp_path):
    def runner(argv):
        raise RuntimeError("cli boom")

    shell, recorder, _ctrl, _pos = _make_shell(tmp_path, cli_runner=runner)
    assert shell.dispatch("status") == 1
    assert any("cli boom" in line for line in recorder.lines)


def test_parse_error_returns_two(tmp_path):
    shell, recorder, _ctrl, _pos = _make_shell(tmp_path)
    # an unterminated quote forces shlex to raise ValueError
    assert shell.dispatch('/help "unterminated') == 2
    assert any("parse error" in line for line in recorder.lines)


def test_quit_raises_systemexit(tmp_path):
    shell, _recorder, _ctrl, _pos = _make_shell(tmp_path)
    shell.state.last_exit = 42
    with pytest.raises(SystemExit) as exc:
        shell.dispatch("/quit")
    assert exc.value.code == 42
    with pytest.raises(SystemExit):
        shell.dispatch("/exit")


def test_clear_on_tty_and_off(tmp_path):
    shell, recorder, _ctrl, _pos = _make_shell(tmp_path, color_enabled=True)
    assert shell.dispatch("/clear") == 0
    assert any("\x1b[2J" in line for line in recorder.lines)

    shell_off, recorder_off, _c, _p = _make_shell(tmp_path, color_enabled=False)
    assert shell_off.dispatch("/clear") == 0
    assert recorder_off.lines  # just newlines


def test_status_frame_output(tmp_path):
    shell, recorder, _ctrl, _pos = _make_shell(tmp_path)
    assert shell.dispatch("/status") == 0
    assert any("shell" in line for line in recorder.lines)
    assert any("cwd" in line for line in recorder.lines)


def test_history_empty_and_populated(tmp_path):
    shell, recorder, _ctrl, _pos = _make_shell(tmp_path)
    assert shell.dispatch("/history") == 0
    assert any("empty" in line for line in recorder.lines)

    recorder.lines.clear()
    shell.state.history.extend(["first", "second"])
    assert shell.dispatch("/history") == 0
    assert any("first" in line for line in recorder.lines)
    assert any("second" in line for line in recorder.lines)


def test_palette_output(tmp_path):
    shell, recorder, _ctrl, _pos = _make_shell(tmp_path)
    assert shell.dispatch("/palette") == 0
    assert any("primary" in line for line in recorder.lines)


def test_intervals_markets_and_unknown(tmp_path):
    shell, recorder, _ctrl, _pos = _make_shell(tmp_path)
    assert shell.dispatch("/intervals") == 0
    assert any("1m" in line for line in recorder.lines)
    recorder.lines.clear()
    assert shell.dispatch("/intervals futures") == 0
    assert any("1M" in line for line in recorder.lines)
    recorder.lines.clear()
    assert shell.dispatch("/intervals nope") == 2
    assert any("unknown market" in line for line in recorder.lines)


def test_objectives_listing(tmp_path):
    shell, recorder, _ctrl, _pos = _make_shell(tmp_path)
    assert shell.dispatch("/objectives") == 0
    assert any("conservative" in line for line in recorder.lines)
    assert any("default" in line for line in recorder.lines)
    assert any("risky" in line for line in recorder.lines)


def test_positions_empty_and_populated(tmp_path):
    shell, recorder, _ctrl, positions_root = _make_shell(tmp_path)
    assert shell.dispatch("/positions") == 0
    assert any("no open positions" in line for line in recorder.lines)

    store = PositionsStore(root=positions_root)
    store.record_open(OpenPosition(
        id="abc123", symbol="BTCUSDC", market_type="spot", side="LONG",
        qty=0.5, entry_price=100.0, leverage=1.0, opened_at_ms=1, notional=50.0,
    ))
    recorder.lines.clear()
    assert shell.dispatch("/positions") == 0
    assert any("abc123" in line for line in recorder.lines)


def test_stats_renders_even_when_empty(tmp_path):
    shell, recorder, _ctrl, _pos = _make_shell(tmp_path)
    assert shell.dispatch("/stats") == 0
    assert any("Realized P&L" in line for line in recorder.lines)


def test_close_usage_and_hit_and_miss_and_all(tmp_path):
    shell, recorder, _ctrl, positions_root = _make_shell(tmp_path)
    assert shell.dispatch("/close") == 2
    assert any("usage" in line for line in recorder.lines)

    store = PositionsStore(root=positions_root)
    store.record_open(OpenPosition(
        id="pos1", symbol="BTCUSDC", market_type="spot", side="LONG",
        qty=1.0, entry_price=10.0, leverage=1.0, opened_at_ms=1, notional=10.0,
    ))
    recorder.lines.clear()
    assert shell.dispatch("/close pos1") == 0
    assert any("closed pos1" in line for line in recorder.lines)

    recorder.lines.clear()
    assert shell.dispatch("/close ghost") == 1
    assert any("no open position" in line for line in recorder.lines)

    store.record_open(OpenPosition(
        id="pos2", symbol="BTCUSDC", market_type="spot", side="SHORT",
        qty=1.0, entry_price=20.0, leverage=1.0, opened_at_ms=2, notional=20.0,
    ))
    recorder.lines.clear()
    assert shell.dispatch("/close all") == 0
    assert any("closed 1 positions" in line for line in recorder.lines)


def test_auto_all_actions(tmp_path):
    shell, recorder, control_path, _pos = _make_shell(tmp_path)
    assert shell.dispatch("/auto") == 2

    recorder.lines.clear()
    assert shell.dispatch("/auto start") == 0
    payload = json.loads(control_path.read_text())
    assert payload["state"] == STATE_RUNNING

    recorder.lines.clear()
    assert shell.dispatch("/auto start --objective risky") == 0
    assert any("risky" in line for line in recorder.lines)

    recorder.lines.clear()
    assert shell.dispatch("/auto start --objective") == 2
    assert any("needs a value" in line for line in recorder.lines)

    recorder.lines.clear()
    assert shell.dispatch("/auto start --objective nonesuch") == 2
    assert any("unknown objective" in line for line in recorder.lines)

    assert shell.dispatch("/auto pause") == 0
    assert json.loads(control_path.read_text())["state"] == STATE_PAUSED
    assert shell.dispatch("/auto resume") == 0
    assert json.loads(control_path.read_text())["state"] == STATE_RUNNING
    assert shell.dispatch("/auto stop") == 0
    assert json.loads(control_path.read_text())["state"] == STATE_STOPPING

    recorder.lines.clear()
    assert shell.dispatch("/auto status") == 0
    assert any("state=" in line for line in recorder.lines)

    recorder.lines.clear()
    assert shell.dispatch("/auto whatever") == 2
    assert any("unknown /auto action" in line for line in recorder.lines)


def test_backtests_empty_and_populated(tmp_path, monkeypatch):
    shell, recorder, _ctrl, _pos = _make_shell(tmp_path)
    monkeypatch.setattr(shell_mod, "list_reports", lambda: [])
    assert shell.dispatch("/backtests") == 0
    assert any("no reports" in line for line in recorder.lines)

    # stub list_reports to return a fake listing
    class _Listing:
        def __init__(self):
            self.path = "x.json"
            self.tag = "t"
            self.interval = "1m"
            self.market = "spot"
            self.created_at = "2026-04-01T00:00:00+00:00"

    monkeypatch.setattr(shell_mod, "list_reports", lambda: [_Listing()])
    recorder.lines.clear()
    assert shell.dispatch("/backtests") == 0
    assert any("x.json" in line for line in recorder.lines)


def test_complete_matches_and_past_end(tmp_path):
    shell, _recorder, _ctrl, _pos = _make_shell(tmp_path)
    first = shell.complete("/h", 0)
    assert first is not None
    assert first.startswith("/h")
    assert shell.complete("/h", 999) is None
    # bare prefix without slash also completes
    assert shell.complete("p", 0) is not None
    # no match
    shell._completion_cache = []  # reset from prior call
    assert shell.complete("nopematch", 0) is None
    # negative state
    assert shell.complete("/h", -1) is None


def test_run_eof_returns_last_exit(tmp_path):
    shell, _recorder, _ctrl, _pos = _make_shell(tmp_path)
    shell.state.last_exit = 3
    assert shell.run() == 3


def test_run_dispatch_and_quit(tmp_path):
    inputs = iter(["/help", "/quit"])

    def reader(_prompt):
        return next(inputs)

    shell, _recorder, _ctrl, _pos = _make_shell(tmp_path, reader=reader)
    assert shell.run() == 0


def test_run_keyboard_interrupt_continues(tmp_path):
    state = {"count": 0}

    def reader(_prompt):
        state["count"] += 1
        if state["count"] == 1:
            raise KeyboardInterrupt
        raise EOFError

    shell, recorder, _ctrl, _pos = _make_shell(tmp_path, reader=reader)
    shell.run()
    assert any("^C" in line for line in recorder.lines)


def test_custom_commands_override(tmp_path):
    cmd = shell_mod.SlashCommand("ping", "returns pong", lambda s, a: (s.println("pong") or 0))
    shell = shell_mod.Shell(
        commands={"ping": cmd},
        reader=lambda _p: (_ for _ in ()).throw(EOFError),
        writer=_Recorder(),
    )
    assert list(shell.commands) == ["ping"]


def test_println_uses_writer(tmp_path):
    shell, recorder, _ctrl, _pos = _make_shell(tmp_path)
    shell.println("hi")
    shell.print("bye")
    assert "hi" in recorder.lines
    assert "bye" in recorder.lines


def test_run_shell_entrypoint_constructs_shell(monkeypatch):
    invoked: dict[str, object] = {}

    class _FakeShell:
        def __init__(self, *_args, **_kwargs):
            invoked["built"] = True

        def run(self):
            invoked["ran"] = True
            return 0

    monkeypatch.setattr(shell_mod, "Shell", _FakeShell)
    assert shell_mod.run_shell([]) == 0
    assert invoked == {"built": True, "ran": True}


def test_main_delegates_to_run_shell(monkeypatch):
    monkeypatch.setattr(shell_mod, "run_shell", lambda argv: 11)
    monkeypatch.setattr(shell_mod.sys, "argv", ["prog", "--foo"])
    assert shell_mod.main() == 11


def test_default_cli_runner_calls_cli_main(monkeypatch):
    captured: dict[str, object] = {}

    def fake_main(argv):
        captured["argv"] = argv
        return 42

    monkeypatch.setattr("simple_ai_bitcoin_trading_binance.cli.main", fake_main)
    assert shell_mod._default_cli_runner(["status"]) == 42
    assert captured["argv"] == ["status"]
