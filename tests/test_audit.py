from __future__ import annotations

import pytest

from simple_ai_bitcoin_trading_binance.api import Candle
from simple_ai_bitcoin_trading_binance.audit import (
    _dominant_interval_ms,
    _gap_count,
    _max_latest_feature_delta,
    build_audit_report,
    render_audit_report,
)
from simple_ai_bitcoin_trading_binance.features import feature_signature, make_rows
from simple_ai_bitcoin_trading_binance.model import serialize_model, train
from simple_ai_bitcoin_trading_binance.types import RuntimeConfig, StrategyConfig


def _candles(count: int = 140, *, gap: bool = False, duplicate: bool = False) -> list[Candle]:
    rows: list[Candle] = []
    for i in range(count):
        offset = 60_000 if gap and i >= count // 2 else 0
        open_time = i * 60_000 + offset
        close = 100.0 + i * 0.2 + (i % 5) * 0.01
        rows.append(
            Candle(
                open_time=open_time,
                open=close,
                high=close * 1.002,
                low=close * 0.998,
                close=close,
                volume=1.0 + (i % 3),
                close_time=open_time + 59_000,
            )
        )
    if duplicate and rows:
        rows.append(rows[-1])
    return rows


def test_interval_gap_and_short_feature_delta_helpers() -> None:
    assert _dominant_interval_ms([_candles(1)[0]]) is None
    assert _gap_count([_candles(1)[0]]) == 0
    assert _gap_count(_candles(20, gap=True)) == 1
    assert _max_latest_feature_delta(_candles(10), StrategyConfig()) is None


def test_audit_report_good_path_and_render(tmp_path) -> None:
    strategy = StrategyConfig()
    candles = _candles(220)
    rows = make_rows(
        candles,
        strategy.feature_windows[0],
        strategy.feature_windows[1],
        label_threshold=strategy.label_threshold,
        enabled_features=strategy.enabled_features,
    )
    model = train(
        rows,
        epochs=3,
        feature_signature=feature_signature(
            strategy.feature_windows[0],
            strategy.feature_windows[1],
            strategy.label_threshold,
            enabled_features=strategy.enabled_features,
        ),
    )
    model.validation_size = 5
    model.decision_threshold = 0.61
    path = tmp_path / "model.json"
    serialize_model(model, path)

    report = build_audit_report(
        candles,
        RuntimeConfig(),
        strategy,
        model_path=path,
    )

    assert report.ok is True
    assert report.raw_candles == len(candles)
    assert report.clean_candles == len(candles)
    assert report.feature_rows == len(rows)
    assert report.max_feature_delta == pytest.approx(0.0)
    rendered = render_audit_report(report)
    assert "Local operator audit" in rendered
    assert "[ok] model artifact" in rendered
    assert "threshold=0.610" in rendered

    no_model_report = build_audit_report(candles, RuntimeConfig(), strategy, model_path=None)
    assert no_model_report.ok is True

    short_report = build_audit_report(_candles(5), RuntimeConfig(), strategy, model_path=None)
    assert short_report.ok is False
    assert any(check.label == "feature stability" and check.status == "warn" for check in short_report.checks)


def test_audit_report_warns_and_fixes_for_bad_local_state(tmp_path, monkeypatch) -> None:
    import simple_ai_bitcoin_trading_binance.audit as audit_mod

    monkeypatch.setattr(audit_mod, "_max_latest_feature_delta", lambda *_args, **_kwargs: 0.25)
    strategy = StrategyConfig(
        max_open_positions=0,
        max_trades_per_day=0,
        risk_per_trade=0.20,
        max_position_pct=0.90,
        max_drawdown_limit=0.75,
    )
    runtime = RuntimeConfig(symbol="BTCUSDC", testnet=False, dry_run=False, market_type="weird")

    report = build_audit_report(
        _candles(140, gap=True, duplicate=True),
        runtime,
        strategy,
        model_path=tmp_path / "missing.json",
    )

    assert report.ok is False
    statuses = {(check.label, check.status) for check in report.checks}
    assert ("safety target", "fix") in statuses
    assert ("market type", "fix") in statuses
    assert ("model artifact", "fix") in statuses
    assert ("feature stability", "warn") in statuses
    assert report.duplicate_open_times == 1
    assert report.gap_count == 1


def test_audit_report_marks_incompatible_model_as_fix(tmp_path) -> None:
    bad_model = tmp_path / "bad.json"
    bad_model.write_text(
        '{"feature_version":"v0","feature_dim":1,"weights":[0.0],"bias":0.0,'
        '"epochs":1,"feature_means":[0.0],"feature_stds":[1.0]}',
        encoding="utf-8",
    )

    report = build_audit_report(
        _candles(),
        RuntimeConfig(),
        StrategyConfig(),
        model_path=bad_model,
    )

    assert report.ok is False
    assert any(check.label == "model artifact" and check.status == "fix" for check in report.checks)
