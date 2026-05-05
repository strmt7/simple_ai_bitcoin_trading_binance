from __future__ import annotations

import argparse
import importlib.util
import subprocess
from pathlib import Path

import pytest


def _load_push_helper():
    helper_path = Path(__file__).resolve().parents[1] / "tools" / "push_with_pat.py"
    spec = importlib.util.spec_from_file_location("push_with_pat", helper_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_pat_helper_only_allows_expected_https_github_remote() -> None:
    helper = _load_push_helper()
    assert helper._github_owner_repo("https://github.com/strmt7/simple_ai_bitcoin_trading_binance.git") == (
        "strmt7",
        "simple_ai_bitcoin_trading_binance",
    )
    assert helper._github_owner_repo("http://github.com/strmt7/simple_ai_bitcoin_trading_binance.git") is None
    assert helper._github_owner_repo("git@github.com:strmt7/simple_ai_bitcoin_trading_binance.git") is None
    helper._validate_allowed_remote("https://github.com/strmt7/simple_ai_bitcoin_trading_binance.git")
    with pytest.raises(SystemExit, match="unexpected remote"):
        helper._validate_allowed_remote("https://github.com/other/simple_ai_bitcoin_trading_binance.git")


def test_pat_helper_resolves_named_remote_before_token_read(monkeypatch) -> None:
    helper = _load_push_helper()
    monkeypatch.setattr(helper.shutil, "which", lambda _name: "git")

    def runner(command, **kwargs):
        assert kwargs["env"].get("GITHUB_TOKEN") is None
        assert kwargs["capture_output"] is True
        return subprocess.CompletedProcess(command, 0, stdout="https://github.com/evil/repo.git\n")

    args = argparse.Namespace(
        remote="origin",
        refspec="main",
        username="x-access-token",
        token_env="GITHUB_TOKEN",
        dry_run=True,
    )
    with pytest.raises(SystemExit, match="unexpected remote"):
        helper.run_push(args, env={}, runner=runner, token_reader=lambda _prompt: (_ for _ in ()).throw(AssertionError))
