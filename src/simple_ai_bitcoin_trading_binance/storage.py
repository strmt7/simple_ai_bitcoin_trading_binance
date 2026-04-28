"""Small filesystem persistence helpers."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def write_json_atomic(
    path: str | Path,
    payload: Any,
    *,
    indent: int | None = 2,
    sort_keys: bool = False,
    mode: int | None = None,
) -> None:
    """Write JSON through a same-directory temporary file, then replace."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=indent, sort_keys=sort_keys)
            handle.write("\n")
        if mode is not None:
            os.chmod(tmp_path, mode)
        os.replace(tmp_path, target)
        if mode is not None:
            os.chmod(target, mode)
    except Exception:
        try:
            tmp_path.unlink()
        except OSError:
            pass
        raise
