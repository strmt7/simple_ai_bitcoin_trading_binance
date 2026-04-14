from __future__ import annotations

from pathlib import Path

from simple_ai_bitcoin_trading_binance.config import load_runtime, prompt_runtime, save_runtime
from simple_ai_bitcoin_trading_binance.types import RuntimeConfig


def test_save_and_load_runtime(tmp_path: Path, monkeypatch) -> None:
    # isolate config path by patching home
    monkeypatch.setenv("HOME", str(tmp_path))
    cfg = RuntimeConfig(api_key="a", api_secret="b", symbol="BTCUSDC", interval="15m", dry_run=True)
    save_runtime(cfg)
    loaded = load_runtime()
    assert loaded.api_key == "a"
    assert loaded.api_secret == "b"


def test_prompt_runtime_updates(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    inputs = iter(["spot", "BTCUSDC", "15m", "n", "n", "n"])
    def fake_input(prompt: str) -> str:
        return next(inputs)

    responses = iter(["api_key", "api_secret"])
    out = prompt_runtime(RuntimeConfig(), key_getter=fake_input, secret_getter=lambda _: next(responses))
    assert out.symbol == "BTCUSDC"
    assert out.interval == "15m"
    assert not out.testnet
    assert out.api_key == "api_key"
    assert out.api_secret == "api_secret"
    assert not out.dry_run
