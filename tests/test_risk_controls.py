from __future__ import annotations

from simple_ai_bitcoin_trading_binance.risk_controls import (
    assess_entry_risk,
    build_risk_policy_report,
    render_risk_policy_report,
)
from simple_ai_bitcoin_trading_binance.types import RuntimeConfig, StrategyConfig


def test_risk_policy_report_allows_default_paper_and_renders_summary() -> None:
    report = build_risk_policy_report(RuntimeConfig(), StrategyConfig(), effective_dry_run=True)

    assert report.allowed is True
    assert report.block_count == 0
    assert report.warning_count >= 1
    assert report.notional_cap_pct == 0.01
    assert report.max_loss_per_trade_pct == 0.0002
    assert report.checks[0].asdict()["label"] == "symbol"
    rendered = render_risk_policy_report(report)
    assert "Risk policy report" in rendered
    assert "allowed=True" in rendered


def test_risk_policy_blocks_mainnet_live_missing_credentials_and_zero_cash(tmp_path) -> None:
    report = build_risk_policy_report(
        RuntimeConfig(testnet=False, dry_run=False, managed_usdc=0.0),
        StrategyConfig(risk_per_trade=0.0, max_position_pct=0.0),
        effective_dry_run=False,
        model_path=tmp_path / "missing-model.json",
    )

    labels = {check.label: check.status for check in report.checks}
    assert report.allowed is False
    assert report.block_count >= 5
    assert labels["execution environment"] == "block"
    assert labels["order credentials"] == "block"
    assert labels["managed USDC"] == "block"
    assert labels["risk per trade"] == "block"
    assert labels["model path"] == "block"
    assert report.asdict()["allowed"] is False


def test_risk_policy_warns_on_aggressive_but_possible_settings(tmp_path) -> None:
    model_path = tmp_path / "model.json"
    model_path.write_text("{}", encoding="utf-8")
    report = build_risk_policy_report(
        RuntimeConfig(market_type="futures", api_key="k", api_secret="s", dry_run=False),
        StrategyConfig(
            leverage=80.0,
            risk_per_trade=0.04,
            max_position_pct=0.90,
            max_drawdown_limit=0.75,
            slippage_bps=150.0,
            taker_fee_bps=125.0,
        ),
        effective_dry_run=False,
        model_path=model_path,
    )

    assert report.allowed is True
    assert report.leverage == 80.0
    assert report.warning_count >= 6
    assert any(check.label == "effective leverage" and check.status == "warn" for check in report.checks)

    disabled_drawdown = build_risk_policy_report(
        RuntimeConfig(),
        StrategyConfig(max_drawdown_limit=0.0),
        effective_dry_run=True,
    )
    assert any(check.label == "drawdown stop" and check.status == "warn" for check in disabled_drawdown.checks)

    strategy = StrategyConfig()
    strategy.leverage = float("nan")
    strategy.risk_per_trade = object()
    coerced = build_risk_policy_report(
        RuntimeConfig(market_type="futures"),
        strategy,
        effective_dry_run=True,
    )
    assert coerced.leverage == 1.0
    assert any(check.label == "risk per trade" and check.status == "block" for check in coerced.checks)


def test_entry_risk_decision_explains_each_block() -> None:
    allowed = assess_entry_risk(
        direction=1,
        position_side=0,
        max_open_positions=1,
        max_daily_trades=2,
        daily_trade_count=0,
        cash=1000.0,
        price=50_000.0,
        drawdown=0.0,
        drawdown_limit=0.2,
    )
    assert allowed.allowed is True
    assert allowed.code == "allowed"

    cases = [
        (dict(direction=0), "no_signal"),
        (dict(position_side=1), "position_open"),
        (dict(max_open_positions=0), "max_open_positions"),
        (dict(max_daily_trades=2, daily_trade_count=2), "trade_cap"),
        (dict(cash=0.0), "cash"),
        (dict(price=0.0), "price"),
        (dict(drawdown=0.2, drawdown_limit=0.2), "drawdown"),
        (dict(price=float("nan")), "nonfinite"),
        (dict(cash=object()), "nonfinite"),
    ]
    base = dict(
        direction=1,
        position_side=0,
        max_open_positions=1,
        max_daily_trades=2,
        daily_trade_count=0,
        cash=1000.0,
        price=50_000.0,
        drawdown=0.0,
        drawdown_limit=0.2,
    )
    for override, code in cases:
        decision = assess_entry_risk(**{**base, **override})
        assert decision.allowed is False
        assert decision.code == code
        assert decision.asdict()["metrics"]
