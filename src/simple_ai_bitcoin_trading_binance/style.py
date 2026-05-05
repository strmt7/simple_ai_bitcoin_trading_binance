"""ANSI styling helpers for the interactive shell and rich CLI output.

The module is stdlib-only and degrades gracefully when stdout is not a TTY
(continuous integration, `tee`, piped output). A single environment variable,
``NO_COLOR``, forces plain text rendering in line with the NO_COLOR convention.
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from typing import Iterable

RESET = "\x1b[0m"
BOLD = "\x1b[1m"
DIM = "\x1b[2m"
ITALIC = "\x1b[3m"
UNDERLINE = "\x1b[4m"

_COLORS: dict[str, str] = {
    "black": "\x1b[30m",
    "red": "\x1b[31m",
    "green": "\x1b[32m",
    "yellow": "\x1b[33m",
    "blue": "\x1b[34m",
    "magenta": "\x1b[35m",
    "cyan": "\x1b[36m",
    "white": "\x1b[37m",
    "bright_black": "\x1b[90m",
    "bright_red": "\x1b[91m",
    "bright_green": "\x1b[92m",
    "bright_yellow": "\x1b[93m",
    "bright_blue": "\x1b[94m",
    "bright_magenta": "\x1b[95m",
    "bright_cyan": "\x1b[96m",
    "bright_white": "\x1b[97m",
}


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004


def _enable_windows_virtual_terminal(stream) -> bool:
    """Enable ANSI escape handling for classic Windows consoles when possible."""

    fileno = getattr(stream, "fileno", None)
    if fileno is None:
        return False
    try:
        fd = int(fileno())
    except (OSError, TypeError, ValueError):
        return False
    try:
        import ctypes
        import msvcrt
    except ImportError:
        return False
    handle = msvcrt.get_osfhandle(fd)
    mode = ctypes.c_ulong()
    kernel32 = ctypes.windll.kernel32
    if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
        return False
    updated = mode.value | _ENABLE_VIRTUAL_TERMINAL_PROCESSING
    if updated == mode.value:
        return True
    return bool(kernel32.SetConsoleMode(handle, updated))


def supports_unicode(stream=None, sample: str = "▸❯┌─┐│└┘├┤") -> bool:
    """Return True when ``stream`` can encode the shell's display glyphs."""

    target = stream if stream is not None else sys.stdout
    encoding = getattr(target, "encoding", None)
    if not encoding:
        return True
    try:
        sample.encode(str(encoding))
    except (LookupError, UnicodeEncodeError):
        return False
    return True


def supports_ansi_terminal(stream=None) -> bool:
    """Return True when the given stream can run an ANSI/VT terminal UI."""

    target = stream if stream is not None else sys.stdout
    isatty = getattr(target, "isatty", None)
    if isatty is None:
        return False
    try:
        if not bool(isatty()):
            return False
    except (ValueError, OSError):
        return False
    if os.name == "nt":
        return _enable_windows_virtual_terminal(target)
    if os.environ.get("TERM", "").strip().lower() == "dumb":
        return False
    return True


def supports_color(stream=None) -> bool:
    """Return True when the given stream accepts ANSI color styling."""

    if os.environ.get("NO_COLOR", "").strip():
        return False
    if os.environ.get("FORCE_COLOR", "").strip():
        return True
    return supports_ansi_terminal(stream)


@dataclass(frozen=True)
class Palette:
    """Colors used throughout the shell."""

    primary: str = "cyan"
    accent: str = "bright_cyan"
    ok: str = "green"
    warn: str = "yellow"
    bad: str = "red"
    muted: str = "bright_black"
    heading: str = "bright_white"


def color(text: str, name: str, *, enabled: bool = True) -> str:
    """Wrap ``text`` with the ANSI color named ``name``.

    Unknown color names are returned without styling.  When ``enabled`` is
    False the original text is returned untouched, which keeps logs safe for
    non-TTY consumers.
    """

    if not enabled:
        return text
    code = _COLORS.get(name)
    if code is None:
        return text
    return f"{code}{text}{RESET}"


def bold(text: str, *, enabled: bool = True) -> str:
    if not enabled:
        return text
    return f"{BOLD}{text}{RESET}"


def dim(text: str, *, enabled: bool = True) -> str:
    if not enabled:
        return text
    return f"{DIM}{text}{RESET}"


def strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences — useful for width calculations."""

    return _ANSI_RE.sub("", text)


def visible_len(text: str) -> int:
    """Return the visible character width of ``text`` after stripping ANSI."""

    return len(strip_ansi(text))


def hrule(width: int, char: str = "─") -> str:
    """Return a horizontal rule of ``width`` cells using ``char``."""

    if width <= 0:
        return ""
    if not char:
        char = "─"
    single = char[0]
    return single * width


def pad_visible(text: str, width: int) -> str:
    """Right-pad ``text`` so its visible width reaches ``width`` cells."""

    visible = visible_len(text)
    if visible >= width:
        return text
    return text + " " * (width - visible)


def frame(
    title: str,
    lines: Iterable[str],
    *,
    width: int = 72,
    enabled: bool = True,
    palette: Palette | None = None,
    unicode_enabled: bool = True,
) -> list[str]:
    """Render a single-line-border box around ``lines`` with a title header.

    The returned list is suitable for ``"\n".join`` rendering or for the TUI
    snapshot provider.  ``width`` includes the border cells.
    """

    palette = palette or Palette()
    width = max(10, int(width))
    inner = width - 2
    header_text = f" {title} "
    if visible_len(header_text) > inner:
        header_text = header_text[: max(0, inner)]
    padded_header = pad_visible(header_text, inner)
    styled_header = bold(color(padded_header, palette.heading, enabled=enabled), enabled=enabled)

    border = ("┌", "─", "┐", "│", "├", "┤", "└", "┘") if unicode_enabled else ("+", "-", "+", "|", "+", "+", "+", "+")
    top = border[0] + border[1] * inner + border[2]
    bot = border[6] + border[1] * inner + border[7]
    rows: list[str] = [top, f"{border[3]}{styled_header}{border[3]}"]
    rows.append(border[4] + border[1] * inner + border[5])
    for raw in lines:
        text = str(raw)
        if visible_len(text) > inner:
            # hard truncate on visible width; drop the last color reset to avoid
            # leaking half a color escape into the border
            truncated: list[str] = []
            running = 0
            for chunk in re.split(r"(\x1b\[[0-9;]*m)", text):
                if not chunk:
                    continue
                if chunk.startswith("\x1b"):
                    truncated.append(chunk)
                    continue
                # invariant: when we enter this branch, running < inner, so
                # ``remaining`` is always positive.  That's why there is no
                # dead ``remaining <= 0`` guard here — the post-append break
                # below is the only exit from the visible-character budget.
                remaining = inner - running
                truncated.append(chunk[:remaining])
                running += len(chunk[:remaining])
                if running >= inner:
                    break
            text = "".join(truncated) + RESET
            text = pad_visible(text, inner)
        else:
            text = pad_visible(text, inner)
        rows.append(f"{border[3]}{text}{border[3]}")
    rows.append(bot)
    return rows


def ok(text: str, *, enabled: bool = True, palette: Palette | None = None) -> str:
    palette = palette or Palette()
    return color(text, palette.ok, enabled=enabled)


def warn(text: str, *, enabled: bool = True, palette: Palette | None = None) -> str:
    palette = palette or Palette()
    return color(text, palette.warn, enabled=enabled)


def bad(text: str, *, enabled: bool = True, palette: Palette | None = None) -> str:
    palette = palette or Palette()
    return color(text, palette.bad, enabled=enabled)


def muted(text: str, *, enabled: bool = True, palette: Palette | None = None) -> str:
    palette = palette or Palette()
    return color(text, palette.muted, enabled=enabled)
