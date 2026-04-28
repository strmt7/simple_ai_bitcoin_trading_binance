from __future__ import annotations

from pathlib import Path

from simple_ai_bitcoin_trading_binance.config import (
    _read_config_json,
    load_runtime,
    load_strategy,
    prompt_runtime,
    save_runtime,
    save_strategy,
)
from simple_ai_bitcoin_trading_binance.config import (
    config_paths,
)
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


def test_prompt_runtime_rejects_invalid_market_and_non_btc_symbol(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    current = RuntimeConfig(symbol="BTCUSDC", market_type="futures")

    inputs = iter(["margin", "ETHUSDC", "1h", "y", "y", "y"])

    def fake_input(_prompt: str) -> str:
        return next(inputs)

    responses = iter(["", ""])
    out = prompt_runtime(current, key_getter=fake_input, secret_getter=lambda _: next(responses))
    assert out.market_type == "futures"
    assert out.symbol == "BTCUSDC"
    assert out.interval == "1h"
    assert out.testnet is True
    assert out.dry_run is True
    assert out.validate_account is True


def test_prompt_runtime_preserves_existing_credentials_on_blank_or_whitespace(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    current = RuntimeConfig(
        api_key="persisted-key",
        api_secret="persisted-secret",
        max_rate_calls_per_minute=321,
        recv_window_ms=8000,
        compute_backend="auto",
        managed_usdc=42.0,
        managed_btc=0.5,
    )

    inputs = iter(["", "", "", "y", "y", "y"])

    def fake_input(_prompt: str) -> str:
        return next(inputs)

    responses = iter(["   ", "\t"])
    out = prompt_runtime(current, key_getter=fake_input, secret_getter=lambda _: next(responses))
    assert out.api_key == "persisted-key"
    assert out.api_secret == "persisted-secret"
    assert out.max_rate_calls_per_minute == 321
    assert out.recv_window_ms == 8000
    assert out.compute_backend == "auto"
    assert out.managed_usdc == 42.0
    assert out.managed_btc == 0.5


def test_load_runtime_ignores_invalid_json_payload(tmp_path: Path, monkeypatch) -> None:
    runtime_path = tmp_path / ".simple_ai_trading" / "runtime.json"
    runtime_path.parent.mkdir(parents=True, exist_ok=True)
    runtime_path.write_text("{bad json", encoding="utf-8")
    monkeypatch.setenv("HOME", str(tmp_path))

    loaded = load_runtime()
    assert loaded == RuntimeConfig()


def test_load_runtime_supports_payload_overrides(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    loaded = load_runtime({"api_key": "x", "api_secret": "y", "max_rate_calls_per_minute": 5})
    assert loaded.api_key == "x"
    assert loaded.api_secret == "y"
    assert loaded.max_rate_calls_per_minute == 5


def test_runtime_public_dict_redacts_credentials() -> None:
    payload = RuntimeConfig(api_key="x", api_secret="y").public_dict()
    assert payload["api_key"] == "<redacted>"
    assert payload["api_secret"] == "<redacted>"


def test_load_runtime_forces_supported_symbol(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    loaded = load_runtime({"symbol": "ETHUSDC"})
    assert loaded.symbol == "BTCUSDC"


def test_load_strategy_coerces_feature_windows(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    cfg_file = config_paths()["strategy"]
    cfg_file.parent.mkdir(parents=True, exist_ok=True)
    cfg_file.write_text('{"feature_windows": [6, 18], "risk_per_trade": 0.001, "enabled_features": ["momentum_1", "rsi"]}', encoding="utf-8")
    loaded = load_strategy()
    assert loaded.feature_windows == (6, 18)
    assert loaded.enabled_features == ("momentum_1", "rsi")


def test_load_strategy_coerces_invalid_feature_windows(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    cfg_file = config_paths()["strategy"]
    cfg_file.parent.mkdir(parents=True, exist_ok=True)
    cfg_file.write_text('{"feature_windows": [12], "risk_per_trade": 0.002}', encoding="utf-8")
    loaded = load_strategy()
    assert loaded.feature_windows == (10, 40)
    assert loaded.risk_per_trade == 0.002


def test_load_strategy_feature_windows_conversion_fails_and_falls_back(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    cfg_file = config_paths()["strategy"]
    cfg_file.parent.mkdir(parents=True, exist_ok=True)
    cfg_file.write_text('{"feature_windows": ["bad", "worse"], "risk_per_trade": 0.004}', encoding="utf-8")
    loaded = load_strategy()
    assert loaded.feature_windows == (10, 40)
    assert loaded.risk_per_trade == 0.004


def test_load_strategy_invalid_enabled_features_falls_back_to_default(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    cfg_file = config_paths()["strategy"]
    cfg_file.parent.mkdir(parents=True, exist_ok=True)
    cfg_file.write_text('{"enabled_features": ["not-real"]}', encoding="utf-8")
    loaded = load_strategy()
    assert "momentum_1" in loaded.enabled_features


def test_read_config_json_rejects_non_dict_payload(tmp_path: Path) -> None:
    path = tmp_path / "payload.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")
    assert _read_config_json(path) == {}
    assert _read_config_json(tmp_path / "missing.json") == {}


def test_save_strategy_and_read_back(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    save_strategy(load_strategy())
    loaded = load_strategy()
    assert loaded == load_strategy()
