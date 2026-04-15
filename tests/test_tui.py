from __future__ import annotations

import asyncio

from simple_ai_bitcoin_trading_binance.tui import (
    ConfirmScreen,
    FormField,
    FormScreen,
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
        self.size = type("Size", (), {"width": 70})()

    def update(self, value: str) -> None:
        self.value = value


class _FakeFormInput(_FakeInput):
    def __init__(self, value: str = "", identifier: str = "") -> None:
        super().__init__(value)
        self.id = identifier


class _FakeOptionList:
    def __init__(self) -> None:
        self.highlighted = 0

    def action_cursor_down(self) -> None:
        self.highlighted = 1

    def action_cursor_up(self) -> None:
        self.highlighted = 0


class _FakeWorkspace:
    def __init__(self) -> None:
        self.active = ""


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


def test_form_screen_behaviors(monkeypatch) -> None:
    screen = FormScreen(
        "Runtime",
        [
            FormField("api_key", "API key", "seed", password=True),
            FormField("interval", "Interval", "15m"),
        ],
    )
    inputs = {
        "#field-api_key": _FakeFormInput("typed-key", "field-api_key"),
        "#field-interval": _FakeFormInput("1h", "field-interval"),
    }
    dismissed: list[object] = []

    def fake_query_one(selector: str, _cls=None):
        return inputs[selector]

    def fake_query(_cls):
        return list(inputs.values())

    monkeypatch.setattr(screen, "query_one", fake_query_one)
    monkeypatch.setattr(screen, "query", fake_query)
    monkeypatch.setattr(screen, "dismiss", lambda value: dismissed.append(value))

    screen.on_mount()
    assert inputs["#field-api_key"].focused is True

    first_event = type("Evt", (), {"input": inputs["#field-api_key"]})()
    second_event = type("Evt", (), {"input": inputs["#field-interval"]})()
    screen.on_input_submitted(first_event)
    assert inputs["#field-interval"].focused is True
    screen.on_input_submitted(second_event)
    screen.on_button_pressed(type("Evt", (), {"button": type("Btn", (), {"id": "save"})()})())
    screen.on_button_pressed(type("Evt", (), {"button": type("Btn", (), {"id": "cancel"})()})())

    assert dismissed == [
        {"api_key": "typed-key", "interval": "1h"},
        {"api_key": "typed-key", "interval": "1h"},
        None,
    ]


def test_form_screen_handles_empty_and_unknown_submission(monkeypatch) -> None:
    screen = FormScreen("Empty", [])
    dismissed: list[object] = []
    monkeypatch.setattr(screen, "dismiss", lambda value: dismissed.append(value))

    screen.on_mount()
    screen.on_input_submitted(type("Evt", (), {"input": type("InputEvt", (), {"id": "field-missing"})()})())

    assert dismissed == [{}]


def test_terminal_ui_methods() -> None:
    seen = {"screens": [], "logs": []}

    class _FakeApp:
        def __init__(self) -> None:
            self.results = iter(["  typed  ", "", True, {"api_key": "typed-key"}])

        async def push_screen_wait(self, screen):
            seen["screens"].append(type(screen).__name__)
            return next(self.results)

        def append_log(self, text: str) -> None:
            seen["logs"].append(text)

    ui = TerminalUI(_FakeApp())

    assert asyncio.run(ui.prompt("Label", "fallback")) == "typed"
    assert asyncio.run(ui.secret("Secret", "fallback")) == ""
    assert asyncio.run(ui.confirm("Confirm")) is True
    assert asyncio.run(ui.form("Runtime", [FormField("api_key", "API key")])) == {"api_key": "typed-key"}
    ui.append_log("line")
    assert seen["screens"] == ["PromptScreen", "PromptScreen", "ConfirmScreen", "FormScreen"]
    assert seen["logs"] == ["line"]


def test_operator_app_methods(monkeypatch) -> None:
    widgets = {
        "#actions": _FakeOptionList(),
        "#workspace": _FakeWorkspace(),
        "#status": _FakeStatic(),
        "#details": _FakeStatic(),
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
        snapshot_provider=lambda _width=70: "snapshot",
    )
    monkeypatch.setattr(app, "query_one", lambda selector, _cls=None: widgets[selector])

    app.on_mount()
    assert widgets["#workspace"].active == "panel-overview"
    assert widgets["#preview"].value == "snapshot"
    assert widgets["#status"].value == "Ready. Enter runs the selected action."
    assert widgets["#details"].value.startswith("Sync")

    asyncio.run(app._execute_action(app.actions_data[0]))
    assert widgets["#workspace"].active == "panel-activity"
    assert "sync output" in widgets["#log"].lines
    assert widgets["#status"].value == "Sync complete (1)"

    widgets["#actions"].highlighted = 1
    asyncio.run(app.action_run_selected())
    assert "async output" in widgets["#log"].lines

    app.action_refresh_preview()
    assert widgets["#workspace"].active == "panel-overview"
    assert widgets["#status"].value == "Refreshed"

    app.action_cursor_down()
    assert widgets["#workspace"].active == "panel-action"
    assert widgets["#status"].value == "Async"
    app.action_cursor_up()
    assert widgets["#workspace"].active == "panel-action"
    assert widgets["#status"].value == "Sync"

    asyncio.run(app.on_option_list_option_selected(object()))
    assert widgets["#status"].value == "Sync complete (1)"


def test_operator_app_refresh_preview_supports_zero_arg_provider(monkeypatch) -> None:
    widgets = {
        "#preview": _FakeStatic(),
    }
    app = OperatorApp(
        title_text="console",
        actions=[TUIAction("1", "Sync", "sync description", lambda _ui: 0)],
        snapshot_provider=lambda: "snapshot-without-width",
    )
    monkeypatch.setattr(app, "query_one", lambda selector, _cls=None: widgets[selector])

    app.refresh_preview()

    assert widgets["#preview"].value == "snapshot-without-width"


def test_operator_app_execute_action_handles_silent_result(monkeypatch) -> None:
    widgets = {
        "#workspace": _FakeWorkspace(),
        "#status": _FakeStatic(),
        "#preview": _FakeStatic(),
        "#log": _FakeRichLog(),
    }

    app = OperatorApp(
        title_text="console",
        actions=[TUIAction("1", "Silent", "silent description", lambda _ui: None)],
        snapshot_provider=lambda _width=70: "snapshot",
    )
    monkeypatch.setattr(app, "query_one", lambda selector, _cls=None: widgets[selector])

    asyncio.run(app._execute_action(app.actions_data[0]))

    assert widgets["#workspace"].active == "panel-activity"
    assert widgets["#log"].lines == []
    assert widgets["#status"].value == "Silent complete (None)"


def test_operator_app_activate_workspace_is_defensive(monkeypatch) -> None:
    app = OperatorApp(
        title_text="console",
        actions=[TUIAction("1", "Sync", "sync description", lambda _ui: 0)],
        snapshot_provider=lambda _width=70: "snapshot",
    )
    monkeypatch.setattr(app, "query_one", lambda *_args, **_kwargs: (_ for _ in ()).throw(KeyError("missing")))

    app._activate_workspace("panel-action")


def test_operator_app_runs_in_textual_runtime() -> None:
    calls: list[str] = []

    async def runner() -> None:
        def sync_action(_ui):
            calls.append("run")
            return 0

        app = OperatorApp(
            title_text="console",
            actions=[TUIAction("1", "Sync", "sync description", sync_action)],
            snapshot_provider=lambda _width=70: "snapshot",
        )
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()

        assert calls == ["run"]

    asyncio.run(runner())


def test_modal_screens_compose_in_textual_runtime() -> None:
    class _PromptApp(OperatorApp):
        def on_mount(self) -> None:
            super().on_mount()
            self.push_screen(PromptScreen("Prompt", "seed"))

    class _ConfirmApp(OperatorApp):
        def on_mount(self) -> None:
            super().on_mount()
            self.push_screen(ConfirmScreen("Confirm?"))

    class _FormApp(OperatorApp):
        def on_mount(self) -> None:
            super().on_mount()
            self.push_screen(FormScreen("Runtime", [FormField("api_key", "API key", "seed", password=True)]))

    async def runner() -> None:
        prompt_app = _PromptApp(
            title_text="console",
            actions=[TUIAction("1", "Sync", "sync description", lambda _ui: 0)],
            snapshot_provider=lambda _width=70: "snapshot",
        )
        async with prompt_app.run_test() as pilot:
            await pilot.pause()
            assert isinstance(prompt_app.screen_stack[-1], PromptScreen)

        confirm_app = _ConfirmApp(
            title_text="console",
            actions=[TUIAction("1", "Sync", "sync description", lambda _ui: 0)],
            snapshot_provider=lambda _width=70: "snapshot",
        )
        async with confirm_app.run_test() as pilot:
            await pilot.pause()
            assert isinstance(confirm_app.screen_stack[-1], ConfirmScreen)

        form_app = _FormApp(
            title_text="console",
            actions=[TUIAction("1", "Sync", "sync description", lambda _ui: 0)],
            snapshot_provider=lambda _width=70: "snapshot",
        )
        async with form_app.run_test() as pilot:
            await pilot.pause()
            assert isinstance(form_app.screen_stack[-1], FormScreen)

    asyncio.run(runner())


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
