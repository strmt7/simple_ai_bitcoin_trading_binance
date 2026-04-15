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
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Footer, Header, Input, Label, OptionList, RichLog, Static


@dataclass(frozen=True)
class TUIAction:
    key: str
    title: str
    description: str
    run: Callable[["TerminalUI"], Awaitable[int | None] | int | None]


class PromptScreen(ModalScreen[str | None]):
    def __init__(self, label: str, default: str = "", *, password: bool = False) -> None:
        super().__init__()
        self.label = label
        self.default = default
        self.password = password

    def compose(self) -> ComposeResult:
        yield Vertical(
            Label(self.label, id="prompt-label"),
            Input(value=self.default, password=self.password, id="prompt-input"),
            Horizontal(
                Button("Submit", variant="primary", id="submit"),
                Button("Cancel", id="cancel"),
                id="prompt-buttons",
            ),
            id="prompt-dialog",
        )

    def on_mount(self) -> None:
        self.query_one("#prompt-input", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.dismiss(event.value)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "submit":
            self.dismiss(self.query_one("#prompt-input", Input).value)
        else:
            self.dismiss(None)


class ConfirmScreen(ModalScreen[bool]):
    def __init__(self, message: str) -> None:
        super().__init__()
        self.message = message

    def compose(self) -> ComposeResult:
        yield Vertical(
            Label(self.message, id="confirm-label"),
            Horizontal(
                Button("Confirm", variant="error", id="confirm"),
                Button("Cancel", id="cancel"),
                id="confirm-buttons",
            ),
            id="confirm-dialog",
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "confirm")


class TerminalUI:
    def __init__(self, app: "OperatorApp") -> None:
        self.app = app

    async def prompt(self, label: str, default: str = "") -> str:
        value = await self.app.push_screen_wait(PromptScreen(label, default))
        return default if value is None else str(value).strip() or default

    async def secret(self, label: str, default: str = "") -> str:
        value = await self.app.push_screen_wait(PromptScreen(label, default, password=True))
        return default if value is None else str(value).strip()

    async def confirm(self, message: str) -> bool:
        return bool(await self.app.push_screen_wait(ConfirmScreen(message)))

    def append_log(self, text: str) -> None:
        self.app.append_log(text)


class OperatorApp(App[int]):
    CSS = """
    Screen {
        layout: vertical;
    }
    #body {
        height: 1fr;
    }
    #actions {
        width: 28;
        border: solid $primary;
    }
    #right {
        width: 1fr;
    }
    #details {
        height: 6;
        border: solid $warning;
        padding: 0 1;
    }
    #preview {
        height: 1fr;
        border: solid $accent;
        padding: 0 1;
    }
    #status {
        height: 1;
        border: solid $secondary;
        padding: 0 1;
    }
    #log {
        height: 8;
        border: solid $success;
    }
    #prompt-dialog, #confirm-dialog {
        width: 70;
        height: auto;
        padding: 1 2;
        border: heavy $primary;
        background: $panel;
        align: center middle;
    }
    #prompt-buttons, #confirm-buttons {
        height: auto;
        align-horizontal: right;
        padding-top: 1;
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

    def compose(self) -> ComposeResult:
        yield Header()
        yield Horizontal(
            OptionList(*[f"{action.key}. {action.title}" for action in self.actions_data], id="actions"),
            Vertical(
                Static("", id="status"),
                Static("", id="details"),
                Static("", id="preview"),
                RichLog(id="log", wrap=True, highlight=True, markup=False),
                id="right",
            ),
            id="body",
        )
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#actions", OptionList).highlighted = 0
        self.refresh_preview()
        self._update_action_details()
        self.set_status("Ready")

    def set_status(self, text: str) -> None:
        self.query_one("#status", Static).update(text)

    def append_log(self, text: str) -> None:
        log = self.query_one("#log", RichLog)
        for line in text.splitlines() or [""]:
            log.write(line)

    def _update_action_details(self) -> None:
        action = self._current_action()
        details = self.query_one("#details", Static)
        wrapped = textwrap.wrap(
            action.description,
            width=max(24, details.size.width - 4 if details.size.width else 52),
            break_long_words=False,
            break_on_hyphens=False,
        ) or [action.description]
        details.update("\n".join([action.title, *wrapped]))

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
        await self._execute_action(self._current_action())

    def action_refresh_preview(self) -> None:
        self.refresh_preview()
        self.set_status("Refreshed")

    def action_cursor_down(self) -> None:
        option_list = self.query_one("#actions", OptionList)
        option_list.action_cursor_down()
        self._update_action_details()
        self.set_status(self._current_action().title)

    def action_cursor_up(self) -> None:
        option_list = self.query_one("#actions", OptionList)
        option_list.action_cursor_up()
        self._update_action_details()
        self.set_status(self._current_action().title)

    async def on_option_list_option_selected(self, _event: OptionList.OptionSelected) -> None:
        await self._execute_action(self._current_action())


def launch_tui(
    *,
    title: str,
    actions: list[TUIAction],
    snapshot_provider: Callable[[], str],
) -> int:
    app = OperatorApp(title_text=title, actions=actions, snapshot_provider=snapshot_provider)
    return int(app.run() or 0)
