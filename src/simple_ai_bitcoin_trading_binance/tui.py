"""Textual-based terminal UI for operator workflows."""

from __future__ import annotations

import asyncio
import inspect
import io
import textwrap
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, field as dataclass_field
from typing import Awaitable, Callable

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, OptionList, RichLog, Static
from textual.widgets.option_list import Option


@dataclass(frozen=True)
class TUIAction:
    key: str
    title: str
    description: str
    run: Callable[["TerminalUI"], Awaitable[int | None] | int | None]
    enabled: Callable[[], bool] | None = None
    disabled_reason: Callable[[], str] | str = "Locked until Binance credentials validate."
    aliases: tuple[str, ...] = dataclass_field(default_factory=tuple)

    def is_enabled(self) -> bool:
        if self.enabled is None:
            return True
        return bool(self.enabled())

    def lock_reason(self) -> str:
        if callable(self.disabled_reason):
            return str(self.disabled_reason())
        return str(self.disabled_reason)


@dataclass(frozen=True)
class FormField:
    key: str
    label: str
    value: str = ""
    password: bool = False


def _bounded_index(highlighted: int | None, count: int) -> int:
    if count <= 0 or highlighted is None:
        return 0
    return max(0, min(int(highlighted), count - 1))


class ConfirmScreen(ModalScreen[bool]):
    BINDINGS = [
        Binding("escape", "dismiss_false", "Cancel", show=False, priority=True),
        Binding("enter", "activate_focused", "Choose", show=False, priority=True),
    ]

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

    def on_mount(self) -> None:
        self.query_one("#cancel", Button).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "confirm")

    def action_activate_focused(self) -> None:
        self.dismiss(getattr(self.focused, "id", "") == "confirm")

    def action_dismiss_false(self) -> None:
        self.dismiss(False)


class FormScreen(ModalScreen[dict[str, str] | None]):
    BINDINGS = [
        Binding("escape", "dismiss_none", "Cancel", show=False, priority=True),
        Binding("enter", "activate_focused", "Choose", show=False, priority=True),
        Binding("ctrl+s", "save", "Save", show=False, priority=True),
    ]

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

    def _submit_field_id(self, current_id: str) -> None:
        ids = [f"field-{field.key}" for field in self.fields]
        if current_id not in ids:
            self.dismiss(self._values())
            return
        index = ids.index(current_id)
        if index >= len(ids) - 1:
            self.dismiss(self._values())
            return
        self.query_one(f"#{ids[index + 1]}", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self._submit_field_id(event.input.id or "")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "save":
            self.dismiss(self._values())
        else:
            self.dismiss(None)

    def action_save(self) -> None:
        self.dismiss(self._values())

    def action_activate_focused(self) -> None:
        focused_id = getattr(self.focused, "id", "")
        if focused_id == "cancel":
            self.dismiss(None)
            return
        if focused_id == "save":
            self.action_save()
            return
        self._submit_field_id(focused_id)

    def action_dismiss_none(self) -> None:
        self.dismiss(None)


class MenuScreen(ModalScreen[str | None]):
    """Modal list-picker used for settings hubs and similar routers."""

    AUTO_FOCUS = "#menu-list"

    BINDINGS = [
        Binding("escape", "dismiss_none", "Cancel", show=False, priority=True),
        Binding("up", "cursor_up", "Up", show=False, priority=True),
        Binding("down", "cursor_down", "Down", show=False, priority=True),
        Binding("k", "cursor_up", "Up", show=False, priority=True),
        Binding("j", "cursor_down", "Down", show=False, priority=True),
        Binding("ctrl+p", "cursor_up", "Up", show=False, priority=True),
        Binding("ctrl+n", "cursor_down", "Down", show=False, priority=True),
        Binding("pageup", "page_up", "Page up", show=False, priority=True),
        Binding("pagedown", "page_down", "Page down", show=False, priority=True),
        Binding("home", "first", "First", show=False, priority=True),
        Binding("end", "last", "Last", show=False, priority=True),
        Binding("enter", "select_highlighted", "Open", show=False, priority=True),
        Binding("space", "select_highlighted", "Open", show=False, priority=True),
        *[
            Binding(str(index), f"select_index({index - 1})", f"Select {index}", show=False, priority=True)
            for index in range(1, 10)
        ],
    ]

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
        self._highlighted = 0

    def compose(self) -> ComposeResult:
        help_text = self.help_text or "Select an item with Up/Down and press Enter."
        help_text = f"{help_text}\nKeys: Up/Down or j/k move, 1-9 select, Enter open, Escape close."
        yield Vertical(
            Label(self.title_text, id="menu-title", markup=False),
            Static(
                help_text,
                id="menu-help",
                markup=False,
            ),
            VerticalScroll(
                *[
                    Static(self._menu_row_text(index), id=f"menu-row-{index}", classes="menu-row", markup=False)
                    for index in range(len(self.options))
                ],
                id="menu-list",
            ),
            Horizontal(
                Button("Close", id="close"),
                id="menu-buttons",
            ),
            id="menu-dialog",
        )

    def on_mount(self) -> None:
        self._highlighted = 0
        self._sync_rows()
        self._focus_menu_list()
        self.call_later(self._focus_menu_list)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(None)

    def _menu_list(self):
        return self.query_one("#menu-list")

    def _focus_menu_list(self) -> None:
        menu_list = self._menu_list()
        self.set_focus(menu_list)
        menu_list.focus()

    def _menu_row_text(self, index: int) -> str:
        marker = ">" if index == self._highlighted else " "
        _key, label = self.options[index]
        return f"{marker} {index + 1}. {label}"

    def _sync_rows(self) -> None:
        menu_list = self._menu_list()
        setattr(menu_list, "highlighted", self._highlighted if self.options else None)
        for index in range(len(self.options)):
            row = self.query_one(f"#menu-row-{index}", Static)
            row.update(self._menu_row_text(index))
            row.set_class(index == self._highlighted, "menu-row-highlighted")
        if self.options:
            row = self.query_one(f"#menu-row-{self._highlighted}", Static)
            scroll_visible = getattr(row, "scroll_visible", None)
            if callable(scroll_visible):
                scroll_visible(animate=False)

    def _highlighted_index(self) -> int:
        return _bounded_index(self._highlighted, len(self.options))

    def _set_highlighted_index(self, index: int) -> None:
        if not self.options:
            return
        self._highlighted = max(0, min(index, len(self.options) - 1))
        self._sync_rows()
        self._focus_menu_list()

    def action_cursor_down(self) -> None:
        self._set_highlighted_index(self._highlighted_index() + 1)

    def action_cursor_up(self) -> None:
        self._set_highlighted_index(self._highlighted_index() - 1)

    def action_page_down(self) -> None:
        self._set_highlighted_index(self._highlighted_index() + 5)

    def action_page_up(self) -> None:
        self._set_highlighted_index(self._highlighted_index() - 5)

    def action_first(self) -> None:
        self._set_highlighted_index(0)

    def action_last(self) -> None:
        self._set_highlighted_index(len(self.options) - 1)

    def action_select_index(self, index: int) -> None:
        if index < 0 or index >= len(self.options):
            return
        key, _label = self.options[index]
        self.dismiss(key)

    def action_select_highlighted(self) -> None:
        if getattr(self.focused, "id", "") == "close":
            self.dismiss(None)
            return
        if not self.options:
            self.dismiss(None)
            return
        self.action_select_index(self._highlighted_index())

    def action_dismiss_none(self) -> None:
        self.dismiss(None)


class MultiSelectScreen(ModalScreen[list[str] | None]):
    AUTO_FOCUS = "#feature-list"

    BINDINGS = [
        Binding("escape", "dismiss_none", "Cancel", show=False, priority=True),
        Binding("up", "cursor_up", "Up", show=False, priority=True),
        Binding("down", "cursor_down", "Down", show=False, priority=True),
        Binding("k", "cursor_up", "Up", show=False, priority=True),
        Binding("j", "cursor_down", "Down", show=False, priority=True),
        Binding("ctrl+p", "cursor_up", "Up", show=False, priority=True),
        Binding("ctrl+n", "cursor_down", "Down", show=False, priority=True),
        Binding("pageup", "page_up", "Page up", show=False, priority=True),
        Binding("pagedown", "page_down", "Page down", show=False, priority=True),
        Binding("home", "first", "First", show=False, priority=True),
        Binding("end", "last", "Last", show=False, priority=True),
        Binding("space", "toggle_highlighted", "Toggle", show=False, priority=True),
        Binding("enter", "activate_focused", "Toggle", show=False, priority=True),
        Binding("ctrl+s", "save", "Save", show=False, priority=True),
        *[
            Binding(str(index), f"toggle_index({index - 1})", f"Toggle {index}", show=False, priority=True)
            for index in range(1, 10)
        ],
    ]

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
        self._highlighted = 0

    def compose(self) -> ComposeResult:
        help_text = self.help_text or "Use space to toggle an item. Save applies the current selection."
        help_text = f"{help_text}\nKeys: Up/Down or j/k move, 1-9 toggle, Ctrl-S save, Escape cancel."
        yield Vertical(
            Label(self.title_text, id="feature-title", markup=False),
            Static(
                help_text,
                id="feature-help",
                markup=False,
            ),
            VerticalScroll(
                *[
                    Static(
                        self._feature_row_text(index),
                        id=f"feature-row-{index}",
                        classes="feature-row",
                        markup=False,
                    )
                    for index in range(len(self.options))
                ],
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
        self._highlighted = 0
        self._sync_rows()
        self._focus_feature_list()
        self.call_later(self._focus_feature_list)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "all":
            self.selected = set(self.options)
            self._sync_rows()
            return
        if event.button.id == "none":
            self.selected.clear()
            self._sync_rows()
            return
        if event.button.id == "save":
            self.action_save()
            return
        self.dismiss(None)

    def _feature_list(self):
        return self.query_one("#feature-list")

    def _focus_feature_list(self) -> None:
        feature_list = self._feature_list()
        self.set_focus(feature_list)
        feature_list.focus()

    def _feature_row_text(self, index: int) -> str:
        marker = ">" if index == self._highlighted else " "
        option = self.options[index]
        checked = "x" if option in self.selected else " "
        return f"{marker} {index + 1}. [{checked}] {option}"

    def _selected_values(self) -> list[str]:
        return [option for option in self.options if option in self.selected]

    def _sync_rows(self) -> None:
        feature_list = self._feature_list()
        setattr(feature_list, "highlighted", self._highlighted if self.options else None)
        setattr(feature_list, "selected", self._selected_values())
        for index, option in enumerate(self.options):
            row = self.query_one(f"#feature-row-{index}", Static)
            row.update(self._feature_row_text(index))
            row.set_class(index == self._highlighted, "feature-row-highlighted")
            row.set_class(option in self.selected, "feature-row-selected")

    def _highlighted_index(self) -> int:
        return _bounded_index(self._highlighted, len(self.options))

    def _set_highlighted_index(self, index: int) -> None:
        if not self.options:
            return
        self._highlighted = max(0, min(index, len(self.options) - 1))
        self._sync_rows()
        highlighted_row = self.query_one(f"#feature-row-{self._highlighted}", Static)
        if hasattr(highlighted_row, "scroll_visible"):
            highlighted_row.scroll_visible(animate=False)
        self._focus_feature_list()

    def action_cursor_down(self) -> None:
        self._set_highlighted_index(self._highlighted_index() + 1)

    def action_cursor_up(self) -> None:
        self._set_highlighted_index(self._highlighted_index() - 1)

    def action_page_down(self) -> None:
        self._set_highlighted_index(self._highlighted_index() + 5)

    def action_page_up(self) -> None:
        self._set_highlighted_index(self._highlighted_index() - 5)

    def action_first(self) -> None:
        self._set_highlighted_index(0)

    def action_last(self) -> None:
        self._set_highlighted_index(len(self.options) - 1)

    def action_toggle_highlighted(self) -> None:
        if not self.options:
            return
        self.action_toggle_index(self._highlighted_index())

    def action_toggle_index(self, index: int) -> None:
        if index < 0 or index >= len(self.options):
            return
        self._highlighted = index
        option = self.options[index]
        if option in self.selected:
            self.selected.remove(option)
        else:
            self.selected.add(option)
        self._sync_rows()
        self._focus_feature_list()

    def action_activate_focused(self) -> None:
        focused_id = getattr(self.focused, "id", "")
        if focused_id == "all":
            self.selected = set(self.options)
            self._sync_rows()
            return
        if focused_id == "none":
            self.selected.clear()
            self._sync_rows()
            return
        if focused_id == "save":
            self.action_save()
            return
        if focused_id == "cancel":
            self.dismiss(None)
            return
        self.action_toggle_highlighted()

    def action_save(self) -> None:
        self.dismiss(self._selected_values())

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

        maybe_mount = self.app.push_screen(screen, callback=_settle)
        if inspect.isawaitable(maybe_mount):
            await maybe_mount
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

    @staticmethod
    async def run_blocking(func, *args, **kwargs):
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
    #actions > .option-list--option-disabled {
        color: #617181;
        text-style: italic;
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
    .menu-row {
        color: #c2d4e2;
        padding: 0 1;
        height: 1;
    }
    .menu-row-highlighted {
        background: #173549;
        color: #f4fbff;
    }
    #menu-list:focus .menu-row-highlighted {
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
    .feature-row {
        color: #c2d4e2;
        background: #05101a;
        height: 1;
    }
    .feature-row-highlighted {
        color: #f4fbff;
        background: #173549;
    }
    .feature-row-selected {
        color: #7ce0c9;
        background: #05101a;
    }
    .feature-row-selected.feature-row-highlighted {
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
        Binding("up", "cursor_up", "Up", show=False, priority=True),
        Binding("down", "cursor_down", "Down", show=False, priority=True),
        Binding("k", "cursor_up", "Up", show=False),
        Binding("j", "cursor_down", "Down", show=False),
        Binding("ctrl+p", "cursor_up", "Up", show=False),
        Binding("ctrl+n", "cursor_down", "Down", show=False),
        Binding("pageup", "page_up", "Page up", show=False),
        Binding("pagedown", "page_down", "Page down", show=False),
        Binding("home", "first_action", "First", show=False),
        Binding("end", "last_action", "Last", show=False),
        Binding("enter", "run_selected", "Run", priority=True),
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
        self._last_details_title: str | None = None
        self._last_details: str | None = None
        self._last_preview: str | None = None
        self._last_details_width: int = 0
        self._action_task: asyncio.Task[None] | None = None

    def compose(self) -> ComposeResult:
        with Vertical(id="topbar"):
            yield Static(self.title, id="titlebar", markup=False)
            yield Static("", id="status", markup=False)
        with Horizontal(id="body"):
            with Vertical(id="nav"):
                yield Static("Main menu", id="actions-title", classes="panel-title")
                yield OptionList(
                    *[
                        Option(self._action_option_label(action), id=action.key, disabled=not action.is_enabled())
                        for action in self.actions_data
                    ],
                    id="actions",
                )
            with Vertical(id="context"):
                with Vertical(id="action-panel"):
                    yield Static(self.actions_data[0].title if self.actions_data else "Command", id="details-title", classes="panel-title")
                    yield Static("", id="details", markup=False)
                with Vertical(id="snapshot-panel"):
                    yield Static("Dashboard snapshot", id="preview-title", classes="panel-title")
                    with VerticalScroll(id="preview-scroll", can_focus=False):
                        yield Static("", id="preview", markup=False)
                with Vertical(id="activity-panel"):
                    yield Static("Activity log", id="log-title", classes="panel-title")
                    log = RichLog(id="log", wrap=True, highlight=True, markup=False)
                    log.can_focus = False
                    yield log
        with Horizontal(id="bottombar"):
            yield Static("Connection: not checked", id="connectionbar", markup=False)
            yield Static(
                "Up/Down select  Enter run  r refresh snapshot  < > command width  - + log height  Ctrl-L clear  q quit",
                id="keybar",
                markup=False,
            )

    def on_mount(self) -> None:
        actions = self.query_one("#actions", OptionList)
        actions.highlighted = self._first_enabled_action_index()
        actions.focus()
        self.refresh_action_availability()
        self.refresh_preview()
        self._update_action_details()
        self.set_status("Ready. Up/Down selects a command; Enter runs it.")
        self.set_timer(0.1, self.refresh_connection_status, name="connection-status-initial")
        self.set_interval(self.connection_interval, self.refresh_connection_status, name="connection-status")

    def on_key(self, event) -> None:
        if self._modal_open() or event.key not in {"tab", "shift+tab"}:
            return
        self.query_one("#actions", OptionList).focus()
        event.prevent_default()
        event.stop()

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

    def _active_modal_screen(self):
        if not self._modal_open():
            return None
        try:
            screen = self.screen_stack[-1]
        except Exception:
            return None
        return None if screen is self else screen

    def _call_modal_action(self, name: str) -> bool:
        screen = self._active_modal_screen()
        if screen is None:
            return False
        action = getattr(screen, f"action_{name}", None)
        if not callable(action):
            return False
        action()
        return True

    def _update_action_details(self) -> None:
        action = self._current_action()
        try:
            title = self.query_one("#details-title", Static)
        except Exception:
            title = None
        if title is not None and action.title != self._last_details_title:
            self._last_details_title = action.title
            title.update(action.title)
        details = self.query_one("#details", Static)
        width = max(24, details.size.width - 4 if details.size.width else 52)
        wrapped = textwrap.wrap(
            action.description,
            width=width,
            break_long_words=False,
            break_on_hyphens=False,
        ) or [action.description]
        if action.is_enabled():
            rendered = "\n".join(wrapped)
        else:
            rendered = "\n".join([*wrapped, "", f"Locked: {action.lock_reason()}"])
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
        self.refresh_action_availability()

    def _current_action(self) -> TUIAction:
        option_list = self.query_one("#actions", OptionList)
        index = _bounded_index(option_list.highlighted, len(self.actions_data))
        return self.actions_data[index]

    @staticmethod
    def _action_option_label(action: TUIAction) -> str:
        return action.title if action.is_enabled() else f"{action.title}  (locked)"

    def _first_enabled_action_index(self) -> int:
        for index, action in enumerate(self.actions_data):
            if action.is_enabled():
                return index
        return 0

    def _nearest_enabled_index(self, index: int, *, direction: int = 1) -> int:
        if not self.actions_data:
            return 0
        bounded = max(0, min(index, len(self.actions_data) - 1))
        if self.actions_data[bounded].is_enabled():
            return bounded
        step = 1 if direction >= 0 else -1
        cursor = bounded + step
        while 0 <= cursor < len(self.actions_data):
            if self.actions_data[cursor].is_enabled():
                return cursor
            cursor += step
        cursor = bounded - step
        while 0 <= cursor < len(self.actions_data):
            if self.actions_data[cursor].is_enabled():
                return cursor
            cursor -= step
        return bounded

    def refresh_action_availability(self) -> None:
        try:
            option_list = self.query_one("#actions", OptionList)
        except Exception:
            return
        previous_index = _bounded_index(option_list.highlighted, len(self.actions_data))
        for index, action in enumerate(self.actions_data):
            enabled = action.is_enabled()
            try:
                option_list.replace_option_prompt_at_index(index, self._action_option_label(action))
                if enabled:
                    option_list.enable_option_at_index(index)
                else:
                    option_list.disable_option_at_index(index)
            except Exception:
                return
        safe_index = self._nearest_enabled_index(previous_index)
        if option_list.highlighted != safe_index:
            option_list.highlighted = safe_index
        self._update_action_details()

    def _select_action(self, index: int, *, direction: int = 1) -> TUIAction:
        option_list = self.query_one("#actions", OptionList)
        safe_index = self._nearest_enabled_index(index, direction=direction)
        if option_list.highlighted != safe_index:
            option_list.highlighted = safe_index
        option_list.focus()
        return self.actions_data[safe_index]

    async def _execute_action(self, action: TUIAction) -> None:
        if not action.is_enabled():
            reason = action.lock_reason()
            self.append_log(reason)
            self.set_status(reason)
            return
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
        self.refresh_action_availability()
        self.refresh_preview()
        if isinstance(result, int) and result != 0:
            self.set_status(f"{action.title} failed ({result})")
        else:
            self.set_status(f"{action.title} complete ({result})")

    def _execute_action_in_background(self, action: TUIAction) -> None:
        if self._action_task is not None and not self._action_task.done():
            self.set_status("Another action is already running.")
            return

        async def runner() -> None:
            try:
                await self._execute_action(action)
            finally:
                self._action_task = None

        self._action_task = asyncio.create_task(runner())

    async def action_run_selected(self) -> None:
        if self._modal_open():
            if self._call_modal_action("activate_focused"):
                return
            self._call_modal_action("select_highlighted")
            return
        action = self._current_action()
        if not action.is_enabled():
            reason = action.lock_reason()
            self.append_log(reason)
            self.set_status(reason)
            return
        if self.is_running:
            self._execute_action_in_background(action)
        else:
            await self._execute_action(action)

    def action_refresh_preview(self) -> None:
        if self._modal_open():
            return
        self.refresh_preview()
        self.set_status("Dashboard snapshot refreshed")
        self.set_timer(0.1, self.refresh_connection_status, name="connection-status-manual")

    def _set_current_action_index(self, index: int) -> None:
        current = _bounded_index(self.query_one("#actions", OptionList).highlighted, len(self.actions_data))
        direction = 1 if index >= current else -1
        action = self._select_action(index, direction=direction)
        self._update_action_details()
        self.set_status(action.title)

    def action_cursor_down(self) -> None:
        if self._call_modal_action("cursor_down"):
            return
        if self._modal_open():
            return
        current = _bounded_index(self.query_one("#actions", OptionList).highlighted, len(self.actions_data))
        self._set_current_action_index(current + 1)

    def action_cursor_up(self) -> None:
        if self._call_modal_action("cursor_up"):
            return
        if self._modal_open():
            return
        current = _bounded_index(self.query_one("#actions", OptionList).highlighted, len(self.actions_data))
        self._set_current_action_index(current - 1)

    def action_page_down(self) -> None:
        if self._call_modal_action("page_down"):
            return
        if self._modal_open():
            return
        current = _bounded_index(self.query_one("#actions", OptionList).highlighted, len(self.actions_data))
        self._set_current_action_index(current + 5)

    def action_page_up(self) -> None:
        if self._call_modal_action("page_up"):
            return
        if self._modal_open():
            return
        current = _bounded_index(self.query_one("#actions", OptionList).highlighted, len(self.actions_data))
        self._set_current_action_index(current - 5)

    def action_first_action(self) -> None:
        if self._call_modal_action("first"):
            return
        if self._modal_open():
            return
        self._set_current_action_index(self._first_enabled_action_index())

    def action_last_action(self) -> None:
        if self._call_modal_action("last"):
            return
        if self._modal_open():
            return
        for index in range(len(self.actions_data) - 1, -1, -1):
            if self.actions_data[index].is_enabled():
                self._set_current_action_index(index)
                return
        self._set_current_action_index(len(self.actions_data) - 1)

    def on_option_list_option_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        if event.option_list.id != "actions" or self._modal_open():
            return
        if not self._ignored_initial_highlight:
            self._ignored_initial_highlight = True
            if event.option_index == 0:
                return
        try:
            if event.option_list.highlighted != event.option_index:
                return
            self._update_action_details()
        except Exception:
            return

    async def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option_list.id != "actions" or self._modal_open():
            return
        try:
            action = self._select_action(event.option_index)
        except Exception:
            return
        if not action.is_enabled():
            reason = action.lock_reason()
            self.append_log(reason)
            self.set_status(reason)
            return
        if self.is_running:
            self._execute_action_in_background(action)
        else:
            await self._execute_action(action)

    def _set_nav_width(self, width: int) -> None:
        bounded = max(self._NAV_WIDTH_MIN, min(self._NAV_WIDTH_MAX, width))
        if bounded == self._nav_width:
            return
        self._nav_width = bounded
        try:
            self.query_one("#nav").styles.width = bounded
        except Exception:
            return
        self.set_status(f"Command list width {bounded}")

    def _set_activity_height(self, height: int) -> None:
        bounded = max(self._ACTIVITY_HEIGHT_MIN, min(self._ACTIVITY_HEIGHT_MAX, height))
        if bounded == self._activity_height:
            return
        self._activity_height = bounded
        try:
            self.query_one("#activity-panel").styles.height = bounded
        except Exception:
            return
        self.set_status(f"Activity log height {bounded}")

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
        self.set_status("Activity log cleared")


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
