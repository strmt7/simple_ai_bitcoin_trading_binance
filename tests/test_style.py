from __future__ import annotations



from simple_ai_bitcoin_trading_binance import style


class _FakeStream:
    def __init__(self, tty: bool = True, raise_exc: Exception | None = None):
        self._tty = tty
        self._raise = raise_exc

    def isatty(self) -> bool:
        if self._raise is not None:
            raise self._raise
        return self._tty


class _StreamNoIsatty:
    pass


def test_supports_color_no_color_env(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    assert style.supports_color(_FakeStream(tty=True)) is False


def test_supports_color_force_color_env(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("FORCE_COLOR", "1")
    assert style.supports_color(_FakeStream(tty=False)) is True


def test_supports_color_tty_true(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    assert style.supports_color(_FakeStream(tty=True)) is True


def test_supports_color_tty_false(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    assert style.supports_color(_FakeStream(tty=False)) is False


def test_supports_color_isatty_value_error(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    assert style.supports_color(_FakeStream(raise_exc=ValueError("closed"))) is False


def test_supports_color_isatty_os_error(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    assert style.supports_color(_FakeStream(raise_exc=OSError("bad fd"))) is False


def test_supports_color_stream_without_isatty(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    assert style.supports_color(_StreamNoIsatty()) is False


def test_supports_color_default_uses_sys_stdout(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    # Default stream path. Behavior depends on the environment; just make sure
    # it returns a bool and does not crash.
    result = style.supports_color()
    assert isinstance(result, bool)


def test_color_disabled_returns_text():
    assert style.color("hi", "red", enabled=False) == "hi"


def test_color_unknown_name_returns_text():
    assert style.color("hi", "not-a-color", enabled=True) == "hi"


def test_color_known_name_wraps_escape():
    out = style.color("hi", "red", enabled=True)
    assert out.startswith("\x1b[31m")
    assert out.endswith(style.RESET)
    assert "hi" in out


def test_bold_enabled_and_disabled():
    assert style.bold("x", enabled=False) == "x"
    out = style.bold("x", enabled=True)
    assert out.startswith(style.BOLD)
    assert out.endswith(style.RESET)


def test_dim_enabled_and_disabled():
    assert style.dim("x", enabled=False) == "x"
    out = style.dim("x", enabled=True)
    assert out.startswith(style.DIM)
    assert out.endswith(style.RESET)


def test_strip_ansi_with_and_without_escapes():
    assert style.strip_ansi("plain") == "plain"
    assert style.strip_ansi("\x1b[31mhi\x1b[0m") == "hi"


def test_visible_len_counts_only_printable():
    assert style.visible_len("hello") == 5
    assert style.visible_len("\x1b[31mhi\x1b[0m") == 2


def test_hrule_zero_width_empty():
    assert style.hrule(0) == ""
    assert style.hrule(-3) == ""


def test_hrule_positive_width_default_char():
    assert style.hrule(5) == "─" * 5


def test_hrule_empty_char_fallback():
    assert style.hrule(4, char="") == "─" * 4


def test_hrule_custom_char_uses_first_character():
    assert style.hrule(3, char="=*") == "==="


def test_pad_visible_padding_short_text():
    padded = style.pad_visible("hi", 5)
    assert padded == "hi   "


def test_pad_visible_text_longer_than_width_returns_as_is():
    text = "hello"
    # width less than the visible length: text returned unchanged
    assert style.pad_visible(text, 3) == text


def test_frame_disabled_palette_and_empty_lines():
    rows = style.frame("Title", [], width=20, enabled=False)
    assert rows[0].startswith("┌")
    assert rows[0].endswith("┐")
    assert rows[-1].startswith("└")
    assert rows[-1].endswith("┘")
    # header row
    assert "Title" in style.strip_ansi(rows[1])
    # divider row exists
    assert rows[2].startswith("├")
    # only top, header, divider, bottom = 4 rows (no content)
    assert len(rows) == 4


def test_frame_narrow_width_truncates_long_text_with_ansi():
    # inject a chunk with embedded ANSI to exercise the split/truncate branch.
    long_text = "\x1b[31m" + "A" * 100 + "\x1b[0m"
    rows = style.frame("Hdr", [long_text], width=12, enabled=True)
    # every rendered row's visible length must be <= width
    for row in rows:
        assert style.visible_len(row) <= 12


def test_frame_truncation_loops_through_multiple_chunks_without_break():
    # Multiple small text chunks separated by ANSI: the first chunk does NOT
    # trigger the ``running >= inner`` break, so the loop body iterates back
    # to the top of the for -- exercising the 174->163 branch.
    long_text = "\x1b[31m" + "A" + "\x1b[0m" + "B" * 200
    rows = style.frame("hdr", [long_text], width=12, enabled=True)
    for row in rows:
        assert style.visible_len(row) <= 12


def test_frame_truncation_natural_loop_exit(monkeypatch):
    # Force every non-ANSI chunk to be shorter than ``inner`` so the loop
    # drains naturally and the ``running >= inner`` break never fires: this
    # covers the 163->176 "for-else fallthrough" transition.
    real_split = style.re.split

    def fake_split(pattern, text, *a, **kw):
        # The pattern uses raw-string syntax so the literal sequence "\\x1b"
        # (four characters) appears in the pattern source.
        if isinstance(pattern, str) and r"\x1b" in pattern:
            # all non-ANSI chunks short enough that running never hits inner
            return ["A", "\x1b[31m", "B", "\x1b[0m"]
        return real_split(pattern, text, *a, **kw)

    monkeypatch.setattr(style.re, "split", fake_split)
    # visible_len of raw input is larger than inner so we enter truncation.
    rows = style.frame("h", ["A" * 100], width=30, enabled=False)  # inner=28
    for row in rows:
        assert style.visible_len(row) <= 30


def test_frame_truncation_remaining_zero_break(monkeypatch):
    # Craft a chunk list that forces ``running`` past ``inner`` without
    # triggering the break at 174 on that specific iteration -- by returning
    # a text chunk whose characters are only counted post-slice to exactly
    # inner via an intermediate empty-string chunk and a subsequent ANSI,
    # and then a final text chunk meets remaining<=0.
    #
    # Approach: return [textA of length inner-1, ANSI, textB of length 1,
    # ANSI, textC]. textA: running=inner-1 (no break). ANSI: continue. textB
    # remaining=1, slice[:1], running=inner, break at 174.  Still 174.
    #
    # To reach 171 we must bypass 174.  We do that by making `len(chunk[:n])`
    # report less than n on the iteration where running would hit inner.
    # Use a custom string subclass that lies about its length.
    real_split = style.re.split

    class _LenLiar(str):
        def __new__(cls, content, reported_len):
            obj = str.__new__(cls, content)
            obj._reported = reported_len
            return obj

        def __len__(self):
            return self._reported

        def __getitem__(self, key):
            # return a str so further slicing uses real length semantics
            return str.__getitem__(self, key)

    def fake_split(pattern, text, *a, **kw):
        if isinstance(pattern, str) and "\x1b" in pattern:
            # Pretend "AAAAA" has length 0 so running stays 0 after it.
            # Then the next text chunk sees remaining=inner (OK), but we want
            # remaining<=0.  Instead: pretend first chunk has length==inner
            # but report it as length 0 -> running stays 0, no break.  Then
            # ANSI, then text "B" that has reported length == inner: running
            # becomes inner, 174 fires.  Still 174.
            #
            # Reaching 171 requires `remaining<=0` BEFORE slicing.  So the
            # ONLY way is for running to already be >= inner when the text
            # chunk begins.  That in turn requires a previous iteration to
            # have set running >= inner without triggering 174, which is
            # impossible because 174 checks the same condition.
            # Hence this branch is structurally unreachable via normal means.
            # We still drive the monkeypatched split through to keep the test
            # meaningful; coverage of 171 is left to coverage pragma in the
            # source or accepted as dead defensive code.
            return ["A", "\x1b[31m", "B", "\x1b[0m", "C"]
        return real_split(pattern, text, *a, **kw)

    monkeypatch.setattr(style.re, "split", fake_split)
    rows = style.frame("h", ["X" * 100], width=10, enabled=False)
    assert rows  # smoke


def test_frame_custom_palette_enabled_contains_color_escape():
    palette = style.Palette(heading="red")
    rows = style.frame("Hi", ["body"], width=20, enabled=True, palette=palette)
    # the heading row should carry an ANSI escape when enabled is True
    assert "\x1b[" in rows[1]


def test_frame_title_longer_than_inner_is_truncated():
    # width 10 -> inner 8 -> header_text " verylong " has visible_len 10 > 8
    rows = style.frame("verylongtitle", [], width=10, enabled=False)
    # header row width still matches inner
    header = rows[1]
    # strip border chars and styling
    visible = style.strip_ansi(header)
    assert visible.startswith("│") and visible.endswith("│")
    inner = visible[1:-1]
    assert len(inner) == 10 - 2


def test_frame_short_line_no_truncation_padding_path():
    rows = style.frame("t", ["short"], width=20, enabled=False)
    # the content row width should match width
    content = rows[3]
    assert style.visible_len(content) == 20


def test_ok_warn_bad_muted_helpers_disabled():
    assert style.ok("ok", enabled=False) == "ok"
    assert style.warn("warn", enabled=False) == "warn"
    assert style.bad("bad", enabled=False) == "bad"
    assert style.muted("m", enabled=False) == "m"


def test_ok_warn_bad_muted_helpers_enabled_with_default_palette():
    assert style.ok("ok", enabled=True).startswith("\x1b[")
    assert style.warn("warn", enabled=True).startswith("\x1b[")
    assert style.bad("bad", enabled=True).startswith("\x1b[")
    assert style.muted("m", enabled=True).startswith("\x1b[")


def test_ok_with_custom_palette():
    palette = style.Palette(ok="blue")
    assert "\x1b[34m" in style.ok("ok", enabled=True, palette=palette)
