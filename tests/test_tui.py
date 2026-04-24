from __future__ import annotations

import asyncio

from textual.widgets import Button

from simple_ai_bitcoin_trading_binance.tui import (
    ConfirmScreen,
    FormField,
    FormScreen,
    MultiSelectScreen,
    OperatorApp,
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
    def __init__(self, identifier: str = "actions") -> None:
        self.id = identifier
        self.highlighted = 0
        self.focused = False

    def action_cursor_down(self) -> None:
        self.highlighted = 1

    def action_cursor_up(self) -> None:
        self.highlighted = 0

    def focus(self) -> None:
        self.focused = True


class _FakeOptionEvent:
    def __init__(self, option_list: _FakeOptionList, option_index: int) -> None:
        self.option_list = option_list
        self.option_index = option_index


class _FakeRichLog:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def write(self, text: str) -> None:
        self.lines.append(text)


def _disable_app_timers(app: OperatorApp, monkeypatch) -> list[tuple[str, str]]:
    timers: list[tuple[str, str]] = []
    monkeypatch.setattr(app, "set_timer", lambda _delay, _callback, *, name=None, **_kwargs: timers.append(("timer", name or "")))
    monkeypatch.setattr(app, "set_interval", lambda _interval, _callback, *, name=None, **_kwargs: timers.append(("interval", name or "")))
    return timers


def test_confirm_screen_behaviors(monkeypatch) -> None:
    screen = ConfirmScreen("Confirm?")

    dismissed: list[bool] = []
    monkeypatch.setattr(screen, "dismiss", lambda value: dismissed.append(value))

    screen.on_button_pressed(type("Evt", (), {"button": type("Btn", (), {"id": "confirm"})()})())
    screen.on_button_pressed(type("Evt", (), {"button": type("Btn", (), {"id": "cancel"})()})())
    screen.action_dismiss_false()

    assert dismissed == [True, False, False]


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
    screen.action_dismiss_none()

    assert dismissed == [
        {"api_key": "typed-key", "interval": "1h"},
        {"api_key": "typed-key", "interval": "1h"},
        None,
        None,
    ]


def test_form_screen_handles_empty_and_unknown_submission(monkeypatch) -> None:
    screen = FormScreen("Empty", [])
    dismissed: list[object] = []
    monkeypatch.setattr(screen, "dismiss", lambda value: dismissed.append(value))

    screen.on_mount()
    screen.on_input_submitted(type("Evt", (), {"input": type("InputEvt", (), {"id": "field-missing"})()})())

    assert dismissed == [{}]


def test_multi_select_screen_behaviors(monkeypatch) -> None:
    class _FakeSelectionList:
        def __init__(self) -> None:
            self.selected = ["momentum_1"]
            self.focused = False
            self.selected_all = False
            self.cleared = False

        def focus(self) -> None:
            self.focused = True

        def select_all(self) -> None:
            self.selected_all = True
            self.selected = ["momentum_1", "rsi"]

        def deselect_all(self) -> None:
            self.cleared = True
            self.selected = []

    screen = MultiSelectScreen("Features", ["momentum_1", "rsi"], ["momentum_1"], help_text="help")
    selection_list = _FakeSelectionList()
    dismissed: list[object] = []

    monkeypatch.setattr(screen, "query_one", lambda _selector, _cls=None: selection_list)
    monkeypatch.setattr(screen, "dismiss", lambda value: dismissed.append(value))

    screen.on_mount()
    assert selection_list.focused is True

    screen.on_button_pressed(type("Evt", (), {"button": type("Btn", (), {"id": "all"})()})())
    screen.on_button_pressed(type("Evt", (), {"button": type("Btn", (), {"id": "none"})()})())
    screen.on_button_pressed(type("Evt", (), {"button": type("Btn", (), {"id": "save"})()})())
    screen.on_button_pressed(type("Evt", (), {"button": type("Btn", (), {"id": "cancel"})()})())
    screen.action_dismiss_none()

    assert selection_list.selected_all is True
    assert selection_list.cleared is True
    assert dismissed == [[], None, None]


def test_terminal_ui_methods() -> None:
    seen = {"screens": [], "logs": []}

    class _FakeApp:
        def __init__(self) -> None:
            self.results = iter([True, {"api_key": "typed-key"}, ["momentum_1", "rsi"]])

        async def push_screen_wait(self, screen):
            seen["screens"].append(type(screen).__name__)
            return next(self.results)

        def push_screen(self, screen, callback=None):
            seen["screens"].append(type(screen).__name__)
            if callback is not None:
                callback(next(self.results))

        def append_log(self, text: str) -> None:
            seen["logs"].append(text)

    ui = TerminalUI(_FakeApp())

    assert asyncio.run(ui.confirm("Confirm")) is True
    assert asyncio.run(ui.form("Runtime", [FormField("api_key", "API key")])) == {"api_key": "typed-key"}
    assert asyncio.run(ui.multi_select("Features", ["momentum_1"], ["momentum_1"])) == ["momentum_1", "rsi"]
    ui.append_log("line")
    assert asyncio.run(ui.run_blocking(lambda left, right: left + right, 2, 5)) == 7
    assert seen["screens"] == ["ConfirmScreen", "FormScreen", "MultiSelectScreen"]
    assert seen["logs"] == ["line"]


def test_operator_app_methods(monkeypatch) -> None:
    widgets = {
        "#actions": _FakeOptionList(),
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
    timers = _disable_app_timers(app, monkeypatch)

    app.on_mount()
    assert timers == [("timer", "connection-status-initial"), ("interval", "connection-status")]
    assert widgets["#actions"].focused is True
    assert widgets["#preview"].value == "snapshot"
    assert widgets["#status"].value == "Ready. Enter runs the selected action."
    assert widgets["#details"].value.startswith("Sync")

    app.on_option_list_option_highlighted(_FakeOptionEvent(widgets["#actions"], 0))
    assert widgets["#status"].value == "Ready. Enter runs the selected action."
    assert app._ignored_initial_highlight is True

    asyncio.run(app._execute_action(app.actions_data[0]))
    assert "sync output" in widgets["#log"].lines
    assert widgets["#status"].value == "Sync complete (1)"

    widgets["#actions"].highlighted = 1
    asyncio.run(app.action_run_selected())
    assert "async output" in widgets["#log"].lines

    app.action_refresh_preview()
    assert widgets["#status"].value == "Snapshot refreshed"
    assert timers[-1] == ("timer", "connection-status-manual")

    app.action_cursor_down()
    assert widgets["#status"].value == "Async"
    app.action_cursor_up()
    assert widgets["#status"].value == "Sync"

    app.on_option_list_option_highlighted(_FakeOptionEvent(widgets["#actions"], 0))
    assert widgets["#status"].value == "Sync"
    app.on_option_list_option_highlighted(_FakeOptionEvent(widgets["#actions"], 1))
    assert widgets["#actions"].highlighted == 1
    assert widgets["#details"].value.startswith("Async")
    assert widgets["#status"].value == "Async"

    asyncio.run(app.on_option_list_option_selected(_FakeOptionEvent(widgets["#actions"], 0)))
    assert widgets["#actions"].highlighted == 0
    assert widgets["#status"].value == "Sync complete (1)"
    asyncio.run(app.on_option_list_option_selected(_FakeOptionEvent(_FakeOptionList("other"), 1)))
    assert widgets["#status"].value == "Sync complete (1)"


def test_operator_app_global_actions_are_blocked_while_modal_is_open(monkeypatch) -> None:
    widgets = {
        "#actions": _FakeOptionList(),
        "#status": _FakeStatic(),
        "#details": _FakeStatic(),
        "#preview": _FakeStatic(),
        "#log": _FakeRichLog(),
    }
    called: list[str] = []

    app = OperatorApp(
        title_text="console",
        actions=[TUIAction("1", "Sync", "sync description", lambda _ui: called.append("run"))],
        snapshot_provider=lambda _width=70: "snapshot",
    )
    monkeypatch.setattr(app, "query_one", lambda selector, _cls=None: widgets[selector])
    monkeypatch.setattr(app, "_modal_open", lambda: True)

    asyncio.run(app.action_run_selected())
    app.action_refresh_preview()
    app.action_cursor_down()
    app.action_cursor_up()
    app.on_option_list_option_highlighted(_FakeOptionEvent(widgets["#actions"], 0))
    asyncio.run(app.on_option_list_option_selected(_FakeOptionEvent(widgets["#actions"], 0)))

    assert called == []
    assert widgets["#actions"].highlighted == 0
    assert widgets["#status"].value == ""


def test_operator_app_first_nonzero_highlight_updates_action_details(monkeypatch) -> None:
    widgets = {
        "#actions": _FakeOptionList(),
        "#status": _FakeStatic(),
        "#details": _FakeStatic(),
        "#preview": _FakeStatic(),
        "#log": _FakeRichLog(),
    }
    app = OperatorApp(
        title_text="console",
        actions=[
            TUIAction("1", "Sync", "sync description", lambda _ui: 0),
            TUIAction("2", "Async", "async description", lambda _ui: 0),
        ],
        snapshot_provider=lambda _width=70: "snapshot",
    )
    monkeypatch.setattr(app, "query_one", lambda selector, _cls=None: widgets[selector])
    _disable_app_timers(app, monkeypatch)

    app.on_mount()
    app.on_option_list_option_highlighted(_FakeOptionEvent(widgets["#actions"], 1))

    assert app._ignored_initial_highlight is True
    assert widgets["#actions"].highlighted == 1
    assert widgets["#details"].value.startswith("Async")


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

    assert widgets["#log"].lines == []
    assert widgets["#status"].value == "Silent complete (None)"


def test_operator_app_select_action_clamps_index(monkeypatch) -> None:
    widgets = {
        "#actions": _FakeOptionList(),
    }
    app = OperatorApp(
        title_text="console",
        actions=[
            TUIAction("1", "Sync", "sync description", lambda _ui: 0),
            TUIAction("2", "Async", "async description", lambda _ui: 0),
        ],
        snapshot_provider=lambda _width=70: "snapshot",
    )
    monkeypatch.setattr(app, "query_one", lambda selector, _cls=None: widgets[selector])

    assert app._select_action(-10).title == "Sync"
    assert widgets["#actions"].highlighted == 0
    assert app._select_action(100).title == "Async"
    assert widgets["#actions"].highlighted == 1


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
            assert app.query_one("#actions").has_focus
            assert str(app.query_one("#actions-title").content) == "Commands"
            assert str(app.query_one("#details-title").content) == "Selected"
            assert str(app.query_one("#preview-title").content) == "Snapshot"
            assert str(app.query_one("#log-title").content) == "Activity"
            assert str(app.query_one("#preview").content) == "snapshot"
            assert "Sync" in str(app.query_one("#details").content)
            assert app.query_one("#log") is not None
            await pilot.press("enter")
            await pilot.pause()

        assert calls == ["run"]

    asyncio.run(runner())


def test_modal_screens_compose_in_textual_runtime() -> None:
    class _ConfirmApp(OperatorApp):
        def on_mount(self) -> None:
            super().on_mount()
            self.push_screen(ConfirmScreen("Confirm?"))

    class _FormApp(OperatorApp):
        def on_mount(self) -> None:
            super().on_mount()
            self.push_screen(
                FormScreen(
                    "Runtime",
                    [
                        FormField("api_key", "API key", "seed", password=True),
                        FormField("interval", "Interval", "15m"),
                    ],
                )
            )

    class _MultiApp(OperatorApp):
        def on_mount(self) -> None:
            super().on_mount()
            self.push_screen(MultiSelectScreen("Features", ["momentum_1"], ["momentum_1"]))

    async def runner() -> None:
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
            assert form_app.focused.id == "field-api_key"
            await pilot.press("tab")
            await pilot.pause()
            assert form_app.focused.id == "field-interval"
            await pilot.press("shift+tab")
            await pilot.pause()
            assert form_app.focused.id == "field-api_key"

        multi_app = _MultiApp(
            title_text="console",
            actions=[TUIAction("1", "Sync", "sync description", lambda _ui: 0)],
            snapshot_provider=lambda _width=70: "snapshot",
        )
        async with multi_app.run_test() as pilot:
            await pilot.pause()
            assert isinstance(multi_app.screen_stack[-1], MultiSelectScreen)
            await pilot.press("tab")
            await pilot.pause()
            assert multi_app.focused.id == "all"

    asyncio.run(runner())


def test_operator_app_flat_button_css_is_homogeneous() -> None:
    css = OperatorApp.CSS

    assert "border: round" not in css
    assert "ContentTabs" not in css
    assert "#workspace" not in css
    assert "#nav" in css
    assert "#action-panel" in css
    assert "#snapshot-panel" in css
    assert "#activity-panel" in css
    assert "Button.-primary" not in css
    assert "Button.-error" not in css
    assert 'variant="primary"' not in css
    assert 'variant="error"' not in css
    assert "border: none;" in css


def test_modal_buttons_use_one_default_variant_in_textual_runtime() -> None:
    class _ConfirmApp(OperatorApp):
        def on_mount(self) -> None:
            super().on_mount()
            self.push_screen(ConfirmScreen("Confirm?"))

    class _FormApp(OperatorApp):
        def on_mount(self) -> None:
            super().on_mount()
            self.push_screen(FormScreen("Runtime", [FormField("interval", "Interval", "15m")]))

    class _MultiApp(OperatorApp):
        def on_mount(self) -> None:
            super().on_mount()
            self.push_screen(MultiSelectScreen("Features", ["momentum_1"], ["momentum_1"]))

    async def runner() -> None:
        for app_cls, button_ids in (
            (_ConfirmApp, ("confirm", "cancel")),
            (_FormApp, ("save", "cancel")),
            (_MultiApp, ("all", "none", "save", "cancel")),
        ):
            app = app_cls(
                title_text="console",
                actions=[TUIAction("1", "Sync", "sync description", lambda _ui: 0)],
                snapshot_provider=lambda _width=70: "snapshot",
            )
            async with app.run_test() as pilot:
                await pilot.pause()
                variants = [app.screen.query_one(f"#{button_id}", Button).variant for button_id in button_ids]
                assert variants == ["default"] * len(button_ids)

    asyncio.run(runner())


def test_operator_app_live_keyboard_navigation_keeps_context_visible() -> None:
    calls: list[str] = []

    async def runner(size: tuple[int, int]) -> None:
        def action(name: str):
            def _run(_ui):
                calls.append(name)
                print(f"{name} output")
                return 1

            return _run

        app = OperatorApp(
            title_text="console",
            actions=[
                TUIAction("1", "One", "first description", action("one")),
                TUIAction("2", "Two", "second description", action("two")),
                TUIAction("3", "Three", "third description", action("three")),
            ],
            snapshot_provider=lambda _width=70: "snapshot",
        )
        async with app.run_test(size=size) as pilot:
            await pilot.pause()
            assert app.query_one("#actions").has_focus
            assert str(app.query_one("#preview").content) == "snapshot"
            assert app.query_one("#log") is not None

            await pilot.press("down")
            await pilot.pause()
            assert app.query_one("#actions").highlighted == 1
            assert "Two" in str(app.query_one("#details").content)

            await pilot.press("enter")
            await pilot.pause()
            assert str(app.query_one("#status").content) == "Two complete (1)"
            assert calls[-1:] == ["two"]

            await pilot.press("up")
            await pilot.pause()
            assert app.query_one("#actions").highlighted == 0
            assert "One" in str(app.query_one("#details").content)

            await pilot.press("r")
            await pilot.pause()
            assert str(app.query_one("#status").content) == "Snapshot refreshed"
            assert str(app.query_one("#preview").content) == "snapshot"

        assert calls[-1:] == ["two"]

    asyncio.run(runner((100, 32)))
    asyncio.run(runner((52, 18)))


def test_modal_keyboard_navigation_does_not_trigger_background_action() -> None:
    calls: list[str] = []

    async def runner() -> None:
        def tracked(_ui):
            calls.append("run")
            return 0

        app = OperatorApp(
            title_text="console",
            actions=[TUIAction("1", "Sync", "sync description", tracked)],
            snapshot_provider=lambda _width=70: "snapshot",
        )
        async with app.run_test() as pilot:
            await pilot.pause()
            app.push_screen(FormScreen("Runtime", [FormField("interval", "Interval", "15m")]))
            await pilot.pause()
            assert isinstance(app.screen_stack[-1], FormScreen)
            assert app.focused.id == "field-interval"

            await pilot.press("enter")
            await pilot.pause()
            assert calls == []
            assert len(app.screen_stack) == 1

    asyncio.run(runner())


def test_launch_tui_constructs_operator_app(monkeypatch) -> None:
    captured = {}

    class _FakeOperatorApp:
        def __init__(self, *, title_text, actions, snapshot_provider, connection_provider=None) -> None:
            captured["title"] = title_text
            captured["actions"] = actions
            captured["snapshot"] = snapshot_provider()
            captured["connection"] = connection_provider

        def run(self):
            return 7

    monkeypatch.setattr("simple_ai_bitcoin_trading_binance.tui.OperatorApp", _FakeOperatorApp)
    result = launch_tui(title="title", actions=[TUIAction("1", "One", "desc", lambda _ui: 0)], snapshot_provider=lambda: "snap")

    assert result == 7
    assert captured["title"] == "title"
    assert captured["snapshot"] == "snap"
    assert captured["connection"] is None


def test_operator_app_connection_status_paths(monkeypatch) -> None:
    widgets = {"#connectionbar": _FakeStatic()}
    app = OperatorApp(
        title_text="title",
        actions=[TUIAction("1", "One", "desc", lambda _ui: 0)],
        snapshot_provider=lambda _width=70: "snapshot",
        connection_provider=lambda: "Connection: online",
        connection_interval=1.0,
    )
    assert app.connection_interval == 5.0
    monkeypatch.setattr(app, "query_one", lambda selector, _cls=None: widgets[selector])

    asyncio.run(app.refresh_connection_status())
    assert widgets["#connectionbar"].value == "Connection: online"

    app.connection_provider = None
    asyncio.run(app.refresh_connection_status())
    assert widgets["#connectionbar"].value == "Connection: no checker configured"

    def failing_provider() -> str:
        raise RuntimeError("network down")

    app.connection_provider = failing_provider
    asyncio.run(app.refresh_connection_status())
    assert widgets["#connectionbar"].value == "Connection: check failed (network down)"

    monkeypatch.setattr(app, "query_one", lambda *_args, **_kwargs: (_ for _ in ()).throw(KeyError("missing")))
    app.set_connection_status("ignored")
