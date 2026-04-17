"""Textual-based terminal UI for operator workflows."""

from __future__ import annotations

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
            Label(self.message, id="confirm-label"),
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
                    Label(field.label, classes="form-label"),
                    Input(value=field.value, password=field.password, id=f"field-{field.key}"),
                    classes="form-row",
                )
            )
        yield Vertical(
            Label(self.title_text, id="form-title"),
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
            Label(self.title_text, id="feature-title"),
            Label(
                self.help_text or "Use space to toggle an item. Save applies the current selection.",
                id="feature-help",
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

    async def confirm(self, message: str) -> bool:
        return bool(await self.app.push_screen_wait(ConfirmScreen(message)))

    async def form(self, title: str, fields: list[FormField]) -> dict[str, str] | None:
        return await self.app.push_screen_wait(FormScreen(title, fields))

    async def multi_select(
        self,
        title: str,
        options: list[str],
        selected: list[str] | tuple[str, ...],
        *,
        help_text: str = "",
    ) -> list[str] | None:
        return await self.app.push_screen_wait(
            MultiSelectScreen(title, options, selected, help_text=help_text)
        )

    def append_log(self, text: str) -> None:
        self.app.append_log(text)


class OperatorApp(App[int]):
    CSS = """
    Screen {
        layout: vertical;
        background: #081018;
        color: #dbe8f2;
    }
    ConfirmScreen, FormScreen, MultiSelectScreen {
        align: center middle;
    }
    #titlebar {
        dock: top;
        height: 1;
        padding: 0 1;
        background: #0d1822;
        color: #f4fbff;
        text-style: bold;
    }
    #status {
        dock: top;
        height: 1;
        border: none;
        background: #0b141d;
        padding: 0 1;
        content-align: left middle;
        color: #dbe8f2;
    }
    #body {
        height: 1fr;
        padding: 1;
    }
    #nav {
        width: 30;
        min-width: 24;
        border: solid #183140;
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
        color: #9fb4c4;
        padding: 0 1;
        text-wrap: nowrap;
        text-overflow: ellipsis;
    }
    #actions > .option-list--option-highlighted {
        background: #173549;
        color: #f4fbff;
        text-style: bold;
    }
    #actions:focus {
        border: solid #2ea7a0;
    }
    #actions:focus > .option-list--option-highlighted {
        background: #0f766e;
        color: #faffff;
        text-style: bold;
    }
    #context {
        width: 1fr;
        padding-left: 1;
    }
    #action-panel {
        height: 8;
        min-height: 5;
        border: solid #183140;
        background: #0b141d;
    }
    #snapshot-panel {
        height: 1fr;
        min-height: 6;
        border: solid #183140;
        background: #0b141d;
        margin-top: 1;
    }
    #activity-panel {
        height: 10;
        min-height: 5;
        border: solid #183140;
        background: #0b141d;
        margin-top: 1;
    }
    #details {
        height: 1fr;
        border: none;
        padding: 1;
        color: #dbe8f2;
    }
    #preview {
        height: 1fr;
        border: none;
        padding: 1;
        color: #dbe8f2;
    }
    #log {
        height: 1fr;
        border: none;
        padding: 1;
        color: #dbe8f2;
    }
    #confirm-dialog, #form-dialog, #feature-dialog {
        width: 68;
        max-width: 100%;
        height: auto;
        padding: 1 2;
        border: solid #29475d;
        background: #0b141d;
    }
    #confirm-buttons, #form-buttons, #feature-buttons {
        height: auto;
        align-horizontal: right;
        padding-top: 1;
    }
    #form-title, #feature-title, #confirm-label {
        padding-bottom: 1;
        text-style: bold;
        color: #f4fbff;
    }
    #feature-help {
        padding-bottom: 1;
        color: #9fb4c4;
    }
    #form-fields, #feature-list {
        max-height: 18;
    }
    .form-row {
        padding-bottom: 1;
    }
    .form-label {
        padding-bottom: 0;
        color: #9fb4c4;
    }
    Input {
        border: solid #29475d;
        background: #081018;
        color: #e7f0f7;
    }
    Input:focus {
        border: solid #2ea7a0;
        background: #0d1822;
        color: #ffffff;
    }
    Button {
        min-width: 9;
        padding: 0 2;
        background: #14202b;
        color: #e7f0f7;
        border: none;
        text-style: bold;
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
        border: solid #193243;
        background: #081018;
        padding: 0 1;
    }
    #feature-list:focus {
        border: solid #2ea7a0;
    }
    #feature-list > .selection-list--button {
        color: #9fb4c4;
        background: #081018;
    }
    #feature-list > .selection-list--button-highlighted {
        color: #f4fbff;
        background: #153042;
    }
    #feature-list > .selection-list--button-selected {
        color: #7ce0c9;
        background: #081018;
    }
    #feature-list > .selection-list--button-selected-highlighted {
        color: #f8fffe;
        background: #0f766e;
    }
    #footerbar {
        dock: bottom;
        height: 1;
        padding: 0 1;
        background: #0d1822;
        color: #9fb4c4;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("r", "refresh_preview", "Refresh"),
        Binding("enter", "run_selected", "Run"),
        Binding("j", "cursor_down", "Down"),
        Binding("k", "cursor_up", "Up"),
    ]

    def __init__(self, *, title_text: str, actions: list[TUIAction], snapshot_provider: Callable[..., str]) -> None:
        super().__init__()
        self.title = title_text
        self.actions_data = actions
        self.snapshot_provider = snapshot_provider
        self.controller = TerminalUI(self)
        self._ignored_initial_highlight = False

    def compose(self) -> ComposeResult:
        yield Static(self.title, id="titlebar")
        yield Static("", id="status")
        with Horizontal(id="body"):
            with Vertical(id="nav"):
                yield Static("Commands", id="actions-title", classes="panel-title")
                yield OptionList(*[action.title for action in self.actions_data], id="actions")
            with Vertical(id="context"):
                with Vertical(id="action-panel"):
                    yield Static("Selected", id="details-title", classes="panel-title")
                    yield Static("", id="details")
                with Vertical(id="snapshot-panel"):
                    yield Static("Snapshot", id="preview-title", classes="panel-title")
                    yield Static("", id="preview")
                with Vertical(id="activity-panel"):
                    yield Static("Activity", id="log-title", classes="panel-title")
                    yield RichLog(id="log", wrap=True, highlight=True, markup=False)
        yield Static("j/k or arrows move | Enter run | r refresh | q quit", id="footerbar")

    def on_mount(self) -> None:
        actions = self.query_one("#actions", OptionList)
        actions.highlighted = 0
        actions.focus()
        self.refresh_preview()
        self._update_action_details()
        self.set_status("Ready. Enter runs the selected action.")

    def set_status(self, text: str) -> None:
        self.query_one("#status", Static).update(text)

    def append_log(self, text: str) -> None:
        log = self.query_one("#log", RichLog)
        for line in text.splitlines() or [""]:
            log.write(line)

    def _modal_open(self) -> bool:
        return len(self.screen_stack) > 1

    def _update_action_details(self) -> None:
        action = self._current_action()
        details = self.query_one("#details", Static)
        wrapped = textwrap.wrap(
            action.description,
            width=max(24, details.size.width - 4 if details.size.width else 52),
            break_long_words=False,
            break_on_hyphens=False,
        ) or [action.description]
        details.update("\n".join([action.title, "", *wrapped]))

    def refresh_preview(self) -> None:
        preview = self.query_one("#preview", Static)
        width = max(40, preview.size.width - 2 if preview.size.width else 70)
        try:
            rendered = self.snapshot_provider(width)
        except TypeError:
            rendered = self.snapshot_provider()
        preview.update(rendered)

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


def launch_tui(
    *,
    title: str,
    actions: list[TUIAction],
    snapshot_provider: Callable[[], str],
) -> int:
    app = OperatorApp(title_text=title, actions=actions, snapshot_provider=snapshot_provider)
    return int(app.run() or 0)
