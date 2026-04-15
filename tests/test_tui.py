from __future__ import annotations

import asyncio

from simple_ai_bitcoin_trading_binance.tui import (
    ConfirmScreen,
    OperatorApp,
    PromptScreen,
    TUIAction,
    TerminalUI,
    launch_tui,
)


class _FakeInput:
    def __init__(self, value: str = "") -> None:
        self.value = value
        self.focused = False

    def focus(self) -> None:
        self.focused = True


class _FakeStatic:
    def __init__(self) -> None:
        self.value = ""

    def update(self, value: str) -> None:
        self.value = value


class _FakeOptionList:
    def __init__(self) -> None:
        self.highlighted = 0

    def action_cursor_down(self) -> None:
        self.highlighted = 1

    def action_cursor_up(self) -> None:
        self.highlighted = 0


class _FakeRichLog:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def write(self, text: str) -> None:
        self.lines.append(text)


def test_prompt_screen_behaviors(monkeypatch) -> None:
    screen = PromptScreen("Prompt", "seed", password=True)

    fake_input = _FakeInput("typed-value")
    dismissed: list[object] = []
    monkeypatch.setattr(screen, "query_one", lambda _selector, _cls=None: fake_input)
    monkeypatch.setattr(screen, "dismiss", lambda value: dismissed.append(value))

    screen.on_mount()
    assert fake_input.focused is True

    screen.on_input_submitted(type("Evt", (), {"value": "submitted"})())
    screen.on_button_pressed(type("Evt", (), {"button": type("Btn", (), {"id": "submit"})()})())
    screen.on_button_pressed(type("Evt", (), {"button": type("Btn", (), {"id": "cancel"})()})())

    assert dismissed == ["submitted", "typed-value", None]


def test_confirm_screen_behaviors(monkeypatch) -> None:
    screen = ConfirmScreen("Confirm?")

    dismissed: list[bool] = []
    monkeypatch.setattr(screen, "dismiss", lambda value: dismissed.append(value))

    screen.on_button_pressed(type("Evt", (), {"button": type("Btn", (), {"id": "confirm"})()})())
    screen.on_button_pressed(type("Evt", (), {"button": type("Btn", (), {"id": "cancel"})()})())

    assert dismissed == [True, False]


def test_terminal_ui_methods() -> None:
    seen = {"screens": [], "logs": []}

    class _FakeApp:
        def __init__(self) -> None:
            self.results = iter(["  typed  ", "", True])

        async def push_screen_wait(self, screen):
            seen["screens"].append(type(screen).__name__)
            return next(self.results)

        def append_log(self, text: str) -> None:
            seen["logs"].append(text)

    ui = TerminalUI(_FakeApp())

    assert asyncio.run(ui.prompt("Label", "fallback")) == "typed"
    assert asyncio.run(ui.secret("Secret", "fallback")) == ""
    assert asyncio.run(ui.confirm("Confirm")) is True
    ui.append_log("line")
    assert seen["screens"] == ["PromptScreen", "PromptScreen", "ConfirmScreen"]
    assert seen["logs"] == ["line"]


def test_operator_app_methods(monkeypatch) -> None:
    widgets = {
        "#actions": _FakeOptionList(),
        "#status": _FakeStatic(),
        "#preview": _FakeStatic(),
        "#log": _FakeRichLog(),
    }

    async def async_action(_ui):
        print("async output")
        return 3

    def sync_action(_ui):
        print("sync output")
        return 1

    app = OperatorApp(
        title_text="console",
        actions=[
            TUIAction("1", "Sync", "sync description", sync_action),
            TUIAction("2", "Async", "async description", async_action),
        ],
        snapshot_provider=lambda: "snapshot",
    )
    monkeypatch.setattr(app, "query_one", lambda selector, _cls=None: widgets[selector])

    assert list(app.compose())
    app.on_mount()
    assert widgets["#preview"].value == "snapshot"
    assert widgets["#status"].value == "sync description"

    asyncio.run(app._execute_action(app.actions_data[0]))
    assert "sync output" in widgets["#log"].lines
    assert widgets["#status"].value == "Sync complete (1)"

    widgets["#actions"].highlighted = 1
    asyncio.run(app.action_run_selected())
    assert "async output" in widgets["#log"].lines

    app.action_refresh_preview()
    assert widgets["#status"].value == "Refreshed"

    app.action_cursor_down()
    assert widgets["#status"].value == "async description"
    app.action_cursor_up()
    assert widgets["#status"].value == "sync description"

    asyncio.run(app.on_option_list_option_selected(object()))
    assert widgets["#status"].value == "Sync complete (1)"


def test_launch_tui_constructs_operator_app(monkeypatch) -> None:
    captured = {}

    class _FakeOperatorApp:
        def __init__(self, *, title_text, actions, snapshot_provider) -> None:
            captured["title"] = title_text
            captured["actions"] = actions
            captured["snapshot"] = snapshot_provider()

        def run(self):
            return 7

    monkeypatch.setattr("simple_ai_bitcoin_trading_binance.tui.OperatorApp", _FakeOperatorApp)
    result = launch_tui(title="title", actions=[TUIAction("1", "One", "desc", lambda _ui: 0)], snapshot_provider=lambda: "snap")

    assert result == 7
    assert captured["title"] == "title"
    assert captured["snapshot"] == "snap"
