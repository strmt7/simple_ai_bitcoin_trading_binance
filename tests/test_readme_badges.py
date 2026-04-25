from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KARPATHY_BADGE_URL = (
    "https://img.shields.io/static/v1?label=&message=andrej-karpathy-skills"
    "&color=555&logo=github&logoColor=white"
)


def test_readme_badge_block_matches_generator() -> None:
    result = subprocess.run(
        [sys.executable, "tools/update_readme_badges.py", "--check"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_readme_uses_github_logo_static_karpathy_badge() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "<!-- BEGIN GENERATED BADGES -->" in readme
    assert "<!-- END GENERATED BADGES -->" in readme
    assert "[![andrej-karpathy-skills](" in readme
    assert KARPATHY_BADGE_URL in readme
    assert "https://github.com/forrestchang/andrej-karpathy-skills" in readme
    assert "https://img.shields.io/badge/andrej-karpathy-skills" not in readme
