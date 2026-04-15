from __future__ import annotations

from simple_ai_bitcoin_trading_binance.dashboard import (
    DashboardSnapshot,
    load_artifact_preview,
    render_dashboard,
)


def test_render_dashboard_includes_all_sections() -> None:
    snapshot = DashboardSnapshot(
        runtime={
            "symbol": "BTCUSDC",
            "interval": "15m",
            "market_type": "spot",
            "testnet": True,
            "dry_run": True,
            "validate_account": True,
            "api_key": "<redacted>",
            "api_secret": "<redacted>",
            "max_rate_calls_per_minute": 1100,
        },
        strategy={
            "signal_threshold": 0.58,
            "label_threshold": 0.001,
            "risk_per_trade": 0.01,
            "max_position_pct": 0.2,
            "stop_loss_pct": 0.02,
            "take_profit_pct": 0.03,
            "cooldown_minutes": 5,
            "max_trades_per_day": 24,
            "model_lookback": 250,
            "training_epochs": 250,
            "feature_windows": [10, 40],
        },
        artifacts=["train_run.json command=train symbol=BTCUSDC market=spot ts=1"],
        account_lines=["market=spot testnet=True", "BTC: free=1.0 locked=0.0"],
        notes=["note a", "note b"],
    )
    text = render_dashboard(snapshot)
    assert "Session" in text
    assert "Model" in text
    assert "Account" in text
    assert "Recent artifacts" in text
    assert "BTCUSDC" in text
    assert "note a" in text


def test_load_artifact_preview_handles_unreadable_and_valid_json(tmp_path) -> None:
    valid = tmp_path / "valid.json"
    valid.write_text('{"command":"backtest","timestamp":123,"runtime":{"symbol":"BTCUSDC","market_type":"spot"}}', encoding="utf-8")
    invalid = tmp_path / "invalid.json"
    invalid.write_text("{bad", encoding="utf-8")
    assert load_artifact_preview(valid).startswith("valid.json command=backtest")
    assert load_artifact_preview(invalid) == "invalid.json [unreadable]"
