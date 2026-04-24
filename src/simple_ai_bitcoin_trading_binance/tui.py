"""Textual-based terminal UI for operator workflows."""

from __future__ import annotations

import asyncio
import inspect
import io
import textwrap
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from typing import Awaitable, Callable

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, OptionList, RichLog, SelectionList, Static


@dataclass(frozen=True)
class TUIAction:
    key: str
    title: str
    description: str
    run: Callable[["TerminalUI"], Awaitable[int | None] | int | None]


@dataclass(frozen=True)
class FormField:
    key: str
    label: str
    value: str = ""
    password: bool = False


class ConfirmScreen(ModalScreen[bool]):
    BINDINGS = [Binding("escape", "dismiss_false", "Cancel", show=False)]

    def __init__(self, message: str) -> None:
        super().__init__()
        self.message = message

    def compose(self) -> ComposeResult:
        yield Vertical(
            Label(self.message, id="confirm-label", markup=False),
            Horizontal(
                Button("Confirm", id="confirm"),
                Button("Cancel", id="cancel"),
                id="confirm-buttons",
            ),
            id="confirm-dialog",
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "confirm")

    def action_dismiss_false(self) -> None:
        self.dismiss(False)


class FormScreen(ModalScreen[dict[str, str] | None]):
    BINDINGS = [Binding("escape", "dismiss_none", "Cancel", show=False)]

    def __init__(self, title: str, fields: list[FormField]) -> None:
        super().__init__()
        self.title_text = title
        self.fields = fields

    def compose(self) -> ComposeResult:
        rows = []
        for field in self.fields:
            rows.append(
                Vertical(
                    Label(field.label, classes="form-label", markup=False),
                    Input(value=field.value, password=field.password, id=f"field-{field.key}"),
                    classes="form-row",
                )
            )
        yield Vertical(
            Label(self.title_text, id="form-title", markup=False),
            VerticalScroll(*rows, id="form-fields"),
            Horizontal(
                Button("Save", id="save"),
                Button("Cancel", id="cancel"),
                id="form-buttons",
            ),
            id="form-dialog",
        )

    def on_mount(self) -> None:
        if self.fields:
            self.query_one(f"#field-{self.fields[0].key}", Input).focus()

    def _values(self) -> dict[str, str]:
        payload = {}
        for field in self.fields:
            payload[field.key] = self.query_one(f"#field-{field.key}", Input).value.strip()
        return payload

    def on_input_submitted(self, event: Input.Submitted) -> None:
        current_id = event.input.id or ""
        ids = [f"field-{field.key}" for field in self.fields]
        if current_id not in ids:
            self.dismiss(self._values())
            return
        index = ids.index(current_id)
        if index >= len(ids) - 1:
            self.dismiss(self._values())
            return
        self.query_one(f"#{ids[index + 1]}", Input).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "save":
            self.dismiss(self._values())
        else:
            self.dismiss(None)

    def action_dismiss_none(self) -> None:
        self.dismiss(None)


class MenuScreen(ModalScreen[str | None]):
    """Modal list-picker used for settings hubs and similar routers."""

    BINDINGS = [Binding("escape", "dismiss_none", "Cancel", show=False)]

    def __init__(
        self,
        title: str,
        options: list[tuple[str, str]],
        *,
        help_text: str = "",
    ) -> None:
        super().__init__()
        self.title_text = title
        self.options = options
        self.help_text = help_text

    def compose(self) -> ComposeResult:
        yield Vertical(
            Label(self.title_text, id="menu-title", markup=False),
            Static(
                self.help_text or "Select an item with Up/Down and press Enter.",
                id="menu-help",
                markup=False,
            ),
            OptionList(*[label for _, label in self.options], id="menu-list"),
            Horizontal(
                Button("Close", id="close"),
                id="menu-buttons",
            ),
            id="menu-dialog",
        )

    def on_mount(self) -> None:
        option_list = self.query_one("#menu-list", OptionList)
        option_list.highlighted = 0
        option_list.focus()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option_list.id != "menu-list":
            return
        key, _label = self.options[event.option_index]
        self.dismiss(key)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(None)

    def action_dismiss_none(self) -> None:
        self.dismiss(None)


class MultiSelectScreen(ModalScreen[list[str] | None]):
    BINDINGS = [Binding("escape", "dismiss_none", "Cancel", show=False)]

    def __init__(
        self,
        title: str,
        options: list[str],
        selected: list[str] | tuple[str, ...],
        *,
        help_text: str = "",
    ) -> None:
        super().__init__()
        self.title_text = title
        self.options = options
        self.selected = set(selected)
        self.help_text = help_text

    def compose(self) -> ComposeResult:
        yield Vertical(
            Label(self.title_text, id="feature-title", markup=False),
            Static(
                self.help_text or "Use space to toggle an item. Save applies the current selection.",
                id="feature-help",
                markup=False,
            ),
            SelectionList(
                *[(option, option, option in self.selected) for option in self.options],
                id="feature-list",
            ),
            Horizontal(
                Button("All", id="all"),
                Button("None", id="none"),
                Button("Save", id="save"),
                Button("Cancel", id="cancel"),
                id="feature-buttons",
            ),
            id="feature-dialog",
        )

    def on_mount(self) -> None:
        self.query_one("#feature-list", SelectionList).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        selection_list = self.query_one("#feature-list", SelectionList)
        if event.button.id == "all":
            selection_list.select_all()
            return
        if event.button.id == "none":
            selection_list.deselect_all()
            return
        if event.button.id == "save":
            self.dismiss([str(value) for value in selection_list.selected])
            return
        self.dismiss(None)

    def action_dismiss_none(self) -> None:
        self.dismiss(None)


class TerminalUI:
    def __init__(self, app: "OperatorApp") -> None:
        self.app = app

    async def _await_screen(self, screen):
        loop = asyncio.get_event_loop()
        future: asyncio.Future = loop.create_future()

        def _settle(result):
            if not future.done():
                future.set_result(result)

        self.app.push_screen(screen, callback=_settle)
        return await future

    async def confirm(self, message: str) -> bool:
        return bool(await self._await_screen(ConfirmScreen(message)))

    async def form(self, title: str, fields: list[FormField]) -> dict[str, str] | None:
        return await self._await_screen(FormScreen(title, fields))

    async def multi_select(
        self,
        title: str,
        options: list[str],
        selected: list[str] | tuple[str, ...],
        *,
        help_text: str = "",
    ) -> list[str] | None:
        return await self._await_screen(
            MultiSelectScreen(title, options, selected, help_text=help_text)
        )

    async def menu(
        self,
        title: str,
        options: list[tuple[str, str]],
        *,
        help_text: str = "",
    ) -> str | None:
        return await self._await_screen(
            MenuScreen(title, options, help_text=help_text)
        )

    def append_log(self, text: str) -> None:
        self.app.append_log(text)

    async def run_blocking(self, func, *args, **kwargs):
        return await asyncio.to_thread(func, *args, **kwargs)


class OperatorApp(App[int]):
    CSS = """
    Screen {
        layout: vertical;
        background: #081018;
        color: #dbe8f2;
    }
    ConfirmScreen, FormScreen, MultiSelectScreen, MenuScreen {
        align: center middle;
    }
    #topbar {
        dock: top;
        height: 2;
    }
    #titlebar {
        height: 1;
        padding: 0 1;
        background: #102a3e;
        color: #f4fbff;
        text-style: bold;
    }
    #status {
        height: 1;
        border: none;
        background: #0b1a26;
        padding: 0 1;
        content-align: left middle;
        color: #b9d2e3;
    }
    #body {
        height: 1fr;
        padding: 1;
    }
    #nav {
        width: 30;
        min-width: 24;
        border: solid #204258;
        background: #0b141d;
    }
    .panel-title {
        height: 1;
        padding: 0 1;
        background: #102131;
        color: #7ce0c9;
        text-style: bold;
    }
    #actions {
        height: 1fr;
        border: none;
        background: #0b141d;
        color: #dbe8f2;
        padding: 0;
    }
    #actions > .option-list--option {
        color: #c2d4e2;
        padding: 0 1;
        text-wrap: nowrap;
        text-overflow: ellipsis;
        text-style: none;
    }
    #actions > .option-list--option-highlighted {
        background: #173549;
        color: #f4fbff;
        text-style: none;
    }
    #actions:focus > .option-list--option-highlighted {
        background: #0f766e;
        color: #faffff;
        text-style: none;
    }
    #context {
        width: 1fr;
        padding-left: 1;
    }
    #action-panel {
        height: 9;
        min-height: 6;
        border: solid #204258;
        background: #0b141d;
    }
    #snapshot-panel {
        height: 1fr;
        min-height: 10;
        border: solid #204258;
        background: #0b141d;
        margin-top: 1;
    }
    #activity-panel {
        height: 12;
        min-height: 6;
        border: solid #204258;
        background: #0b141d;
        margin-top: 1;
    }
    #details {
        height: 1fr;
        border: none;
        padding: 0 1;
        color: #dbe8f2;
    }
    #preview-scroll {
        height: 1fr;
        padding: 0;
        background: #0b141d;
    }
    #preview-scroll:focus, #preview-scroll:focus-within {
        background: #0d1a26;
    }
    #preview {
        width: 1fr;
        height: auto;
        border: none;
        padding: 0 1;
        color: #dbe8f2;
    }
    #log {
        height: 1fr;
        border: none;
        padding: 0 1;
        color: #dbe8f2;
        background: #0b141d;
    }
    #log:focus {
        background: #0d1a26;
    }
    /* centralized green focus border for any panel that owns focus */
    #nav:focus-within,
    #action-panel:focus-within,
    #snapshot-panel:focus-within,
    #activity-panel:focus-within {
        border: heavy #1ad48f;
    }
    #confirm-dialog, #form-dialog, #feature-dialog, #menu-dialog {
        width: 72;
        max-width: 96%;
        height: auto;
        max-height: 92%;
        padding: 1 2;
        border: solid #2ea7a0;
        background: #0b1a26;
    }
    #feature-dialog {
        height: 80%;
    }
    #menu-dialog {
        height: auto;
    }
    #menu-list {
        height: auto;
        max-height: 16;
        border: solid #2b4b63;
        background: #05101a;
        padding: 0 1;
        margin-bottom: 1;
    }
    #menu-list:focus {
        border: solid #2ea7a0;
    }
    #menu-list > .option-list--option {
        color: #c2d4e2;
        padding: 0 1;
    }
    #menu-list > .option-list--option-highlighted {
        background: #173549;
        color: #f4fbff;
    }
    #menu-list:focus > .option-list--option-highlighted {
        background: #0f766e;
        color: #faffff;
    }
    #menu-title {
        height: 2;
        padding-bottom: 1;
        text-style: bold;
        color: #f4fbff;
    }
    #menu-help {
        height: auto;
        padding-bottom: 1;
        color: #9fb4c4;
    }
    #menu-buttons {
        height: 3;
        align-horizontal: right;
        padding-top: 1;
    }
    #confirm-buttons, #form-buttons, #feature-buttons {
        height: 3;
        align-horizontal: right;
        padding-top: 1;
    }
    #form-title, #feature-title, #confirm-label {
        height: 2;
        padding-bottom: 1;
        text-style: bold;
        color: #f4fbff;
    }
    #feature-help {
        height: auto;
        padding-bottom: 1;
        color: #9fb4c4;
    }
    #form-fields {
        height: auto;
        max-height: 24;
        border: none;
        padding: 0 1 0 0;
        background: #0b1a26;
    }
    #feature-list {
        height: 1fr;
    }
    .form-row {
        height: 5;
        padding: 0;
        margin-bottom: 1;
    }
    .form-label {
        height: 1;
        padding-bottom: 0;
        color: #9fb4c4;
    }
    Input {
        border: solid #2b4b63;
        background: #05101a;
        color: #eaf2f8;
    }
    Input:focus {
        border: solid #2ea7a0;
        background: #0d2030;
        color: #ffffff;
    }
    Button {
        min-width: 10;
        padding: 0 2;
        background: #14202b;
        color: #e7f0f7;
        border: none;
        text-style: bold;
        margin: 0 1 0 0;
    }
    Button:hover {
        background: #1d3444;
        color: #ffffff;
    }
    Button:focus {
        background: #2ea7a0;
        color: #081018;
    }
    #feature-list {
        border: solid #2b4b63;
        background: #05101a;
        padding: 0 1;
    }
    #feature-list:focus {
        border: solid #2ea7a0;
    }
    #feature-list > .selection-list--button {
        color: #c2d4e2;
        background: #05101a;
    }
    #feature-list > .selection-list--button-highlighted {
        color: #f4fbff;
        background: #173549;
    }
    #feature-list > .selection-list--button-selected {
        color: #7ce0c9;
        background: #05101a;
    }
    #feature-list > .selection-list--button-selected-highlighted {
        color: #f8fffe;
        background: #0f766e;
    }
    #bottombar {
        dock: bottom;
        height: 1;
        background: #102a3e;
    }
    #connectionbar {
        width: 1fr;
        padding: 0 1;
        background: #102a3e;
        color: #7ce0c9;
        text-overflow: ellipsis;
    }
    #keybar {
        width: auto;
        padding: 0 1;
        background: #102a3e;
        color: #c2d4e2;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("r", "refresh_preview", "Refresh"),
        Binding("enter", "run_selected", "Run"),
        Binding("greater_than_sign", "grow_nav", "Wider nav", show=False),
        Binding("less_than_sign", "shrink_nav", "Narrower nav", show=False),
        Binding("plus", "grow_activity", "Taller log", show=False),
        Binding("equals_sign", "grow_activity", "Taller log", show=False),
        Binding("minus", "shrink_activity", "Shorter log", show=False),
        Binding("ctrl+l", "clear_log", "Clear log", show=False),
    ]

    _NAV_WIDTH_MIN = 22
    _NAV_WIDTH_MAX = 48
    _ACTIVITY_HEIGHT_MIN = 6
    _ACTIVITY_HEIGHT_MAX = 24

    def __init__(
        self,
        *,
        title_text: str,
        actions: list[TUIAction],
        snapshot_provider: Callable[..., str],
        connection_provider: Callable[[], str] | None = None,
        connection_interval: float = 60.0,
    ) -> None:
        super().__init__()
        self.title = title_text
        self.actions_data = actions
        self.snapshot_provider = snapshot_provider
        self.connection_provider = connection_provider
        self.connection_interval = max(5.0, float(connection_interval))
        self.controller = TerminalUI(self)
        self._ignored_initial_highlight = False
        self._nav_width = 30
        self._activity_height = 12
        self._last_status: str | None = None
        self._last_connection: str | None = None
        self._last_details: str | None = None
        self._last_preview: str | None = None
        self._last_details_width: int = 0

    def compose(self) -> ComposeResult:
        with Vertical(id="topbar"):
            yield Static(self.title, id="titlebar", markup=False)
            yield Static("", id="status", markup=False)
        with Horizontal(id="body"):
            with Vertical(id="nav"):
                yield Static("Commands", id="actions-title", classes="panel-title")
                yield OptionList(*[action.title for action in self.actions_data], id="actions")
            with Vertical(id="context"):
                with Vertical(id="action-panel"):
                    yield Static("Selected", id="details-title", classes="panel-title")
                    yield Static("", id="details", markup=False)
                with Vertical(id="snapshot-panel"):
                    yield Static("Snapshot", id="preview-title", classes="panel-title")
                    with VerticalScroll(id="preview-scroll"):
                        yield Static("", id="preview", markup=False)
                with Vertical(id="activity-panel"):
                    yield Static("Activity", id="log-title", classes="panel-title")
                    yield RichLog(id="log", wrap=True, highlight=True, markup=False)
        with Horizontal(id="bottombar"):
            yield Static("Connection: not checked", id="connectionbar", markup=False)
            yield Static(
                "Tab cycle panels  ↑/↓ move  Enter run  r refresh  < > resize nav  - + resize log  Ctrl-L clear  q quit",
                id="keybar",
                markup=False,
            )

    def on_mount(self) -> None:
        actions = self.query_one("#actions", OptionList)
        actions.highlighted = 0
        actions.focus()
        self.refresh_preview()
        self._update_action_details()
        self.set_status("Ready. Enter runs the selected action.")
        self.set_timer(0.1, self.refresh_connection_status, name="connection-status-initial")
        self.set_interval(self.connection_interval, self.refresh_connection_status, name="connection-status")

    def set_status(self, text: str) -> None:
        if text == self._last_status:
            return
        self._last_status = text
        self.query_one("#status", Static).update(text)

    def append_log(self, text: str) -> None:
        log = self.query_one("#log", RichLog)
        for line in text.splitlines() or [""]:
            log.write(line)

    def set_connection_status(self, text: str) -> None:
        if text == self._last_connection:
            return
        self._last_connection = text
        try:
            self.query_one("#connectionbar", Static).update(text)
        except Exception:
            return

    def _modal_open(self) -> bool:
        return len(self.screen_stack) > 1

    def _update_action_details(self) -> None:
        action = self._current_action()
        details = self.query_one("#details", Static)
        width = max(24, details.size.width - 4 if details.size.width else 52)
        wrapped = textwrap.wrap(
            action.description,
            width=width,
            break_long_words=False,
            break_on_hyphens=False,
        ) or [action.description]
        rendered = "\n".join([action.title, "", *wrapped])
        if rendered == self._last_details and width == self._last_details_width:
            return
        self._last_details = rendered
        self._last_details_width = width
        details.update(rendered)

    def refresh_preview(self) -> None:
        preview = self.query_one("#preview", Static)
        width = max(40, preview.size.width - 2 if preview.size.width else 70)
        try:
            rendered = self.snapshot_provider(width)
        except TypeError:
            rendered = self.snapshot_provider()
        if rendered == self._last_preview:
            return
        self._last_preview = rendered
        preview.update(rendered)

    async def refresh_connection_status(self) -> None:
        if self.connection_provider is None:
            self.set_connection_status("Connection: no checker configured")
            return
        try:
            line = await asyncio.to_thread(self.connection_provider)
        except Exception as exc:
            line = f"Connection: check failed ({exc})"
        self.set_connection_status(line)

    def _current_action(self) -> TUIAction:
        option_list = self.query_one("#actions", OptionList)
        index = option_list.highlighted or 0
        return self.actions_data[index]

    def _select_action(self, index: int) -> TUIAction:
        option_list = self.query_one("#actions", OptionList)
        safe_index = max(0, min(index, len(self.actions_data) - 1))
        option_list.highlighted = safe_index
        return self.actions_data[safe_index]

    async def _execute_action(self, action: TUIAction) -> None:
        self.set_status(f"Running: {action.title}")
        stream = io.StringIO()
        result: int | None = None
        try:
            with redirect_stdout(stream), redirect_stderr(stream):
                maybe_result = action.run(self.controller)
                if inspect.isawaitable(maybe_result):
                    result = await maybe_result
                else:
                    result = maybe_result
        except Exception as exc:  # pragma: no cover - defensive UI guard
            self.append_log(f"{action.title} failed: {exc}")
            self.refresh_preview()
            self.set_status(f"{action.title} failed")
            return
        output = stream.getvalue().strip()
        if output:
            self.append_log(output)
        self.refresh_preview()
        self.set_status(f"{action.title} complete ({result})")

    async def action_run_selected(self) -> None:
        if self._modal_open():
            return
        await self._execute_action(self._current_action())

    def action_refresh_preview(self) -> None:
        if self._modal_open():
            return
        self.refresh_preview()
        self.set_status("Snapshot refreshed")
        self.set_timer(0.1, self.refresh_connection_status, name="connection-status-manual")

    def action_cursor_down(self) -> None:
        if self._modal_open():
            return
        option_list = self.query_one("#actions", OptionList)
        option_list.action_cursor_down()
        self._update_action_details()
        self.set_status(self._current_action().title)

    def action_cursor_up(self) -> None:
        if self._modal_open():
            return
        option_list = self.query_one("#actions", OptionList)
        option_list.action_cursor_up()
        self._update_action_details()
        self.set_status(self._current_action().title)

    def on_option_list_option_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        if event.option_list.id != "actions" or self._modal_open():
            return
        if not self._ignored_initial_highlight:
            self._ignored_initial_highlight = True
            if event.option_index == 0:
                return
        self._select_action(event.option_index)
        self._update_action_details()
        self.set_status(self._current_action().title)

    async def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option_list.id != "actions" or self._modal_open():
            return
        await self._execute_action(self._select_action(event.option_index))

    def _set_nav_width(self, width: int) -> None:
        bounded = max(self._NAV_WIDTH_MIN, min(self._NAV_WIDTH_MAX, width))
        if bounded == self._nav_width:
            return
        self._nav_width = bounded
        try:
            self.query_one("#nav").styles.width = bounded
        except Exception:
            return
        self.set_status(f"Nav width {bounded}")

    def _set_activity_height(self, height: int) -> None:
        bounded = max(self._ACTIVITY_HEIGHT_MIN, min(self._ACTIVITY_HEIGHT_MAX, height))
        if bounded == self._activity_height:
            return
        self._activity_height = bounded
        try:
            self.query_one("#activity-panel").styles.height = bounded
        except Exception:
            return
        self.set_status(f"Activity height {bounded}")

    def action_grow_nav(self) -> None:
        if self._modal_open():
            return
        self._set_nav_width(self._nav_width + 2)

    def action_shrink_nav(self) -> None:
        if self._modal_open():
            return
        self._set_nav_width(self._nav_width - 2)

    def action_grow_activity(self) -> None:
        if self._modal_open():
            return
        self._set_activity_height(self._activity_height + 2)

    def action_shrink_activity(self) -> None:
        if self._modal_open():
            return
        self._set_activity_height(self._activity_height - 2)

    def action_clear_log(self) -> None:
        if self._modal_open():
            return
        try:
            self.query_one("#log", RichLog).clear()
        except Exception:
            return
        self.set_status("Activity cleared")


def launch_tui(
    *,
    title: str,
    actions: list[TUIAction],
    snapshot_provider: Callable[[], str],
    connection_provider: Callable[[], str] | None = None,
) -> int:
    app = OperatorApp(
        title_text=title,
        actions=actions,
        snapshot_provider=snapshot_provider,
        connection_provider=connection_provider,
    )
    return int(app.run() or 0)
