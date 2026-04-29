"""Local risk policy checks shared by operator preflight and live runs."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .types import RuntimeConfig, StrategyConfig


@dataclass(frozen=True)
class RiskCheck:
    status: str
    label: str
    detail: str
    metric: float | int | str | None = None
    limit: float | int | str | None = None

    def asdict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class RiskPolicyReport:
    checks: tuple[RiskCheck, ...]
    effective_dry_run: bool
    leverage: float
    notional_cap_pct: float
    max_loss_per_trade_pct: float

    @property
    def allowed(self) -> bool:
        return all(check.status != "block" for check in self.checks)

    @property
    def warning_count(self) -> int:
        return sum(1 for check in self.checks if check.status == "warn")

    @property
    def block_count(self) -> int:
        return sum(1 for check in self.checks if check.status == "block")

    def asdict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["allowed"] = self.allowed
        payload["warning_count"] = self.warning_count
        payload["block_count"] = self.block_count
        return payload


@dataclass(frozen=True)
class EntryRiskDecision:
    allowed: bool
    code: str
    detail: str
    metrics: dict[str, object]

    def asdict(self) -> dict[str, object]:
        return asdict(self)


def _check(
    status: str,
    label: str,
    detail: str,
    *,
    metric: float | int | str | None = None,
    limit: float | int | str | None = None,
) -> RiskCheck:
    return RiskCheck(status, label, detail, metric=metric, limit=limit)


def _environment(runtime: RuntimeConfig) -> str:
    if getattr(runtime, "demo", False):
        return "demo"
    return "testnet" if runtime.testnet else "mainnet"


def _effective_leverage(strategy: StrategyConfig, market_type: str, requested: float | None = None) -> float:
    if market_type != "futures":
        return 1.0
    raw = float(strategy.leverage if requested is None else requested)
    if not math.isfinite(raw):
        return 1.0
    return max(1.0, min(125.0, raw))


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return parsed if math.isfinite(parsed) else default


def _metric_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return float("nan")


def build_risk_policy_report(
    runtime: RuntimeConfig,
    strategy: StrategyConfig,
    *,
    effective_dry_run: bool | None = None,
    leverage: float | None = None,
    model_path: str | Path | None = None,
) -> RiskPolicyReport:
    """Return deterministic local risk checks without network access."""

    dry_run = bool(runtime.dry_run if effective_dry_run is None else effective_dry_run)
    effective_leverage = _effective_leverage(strategy, runtime.market_type, leverage)
    notional_cap_pct = min(
        max(0.0, _finite(strategy.risk_per_trade)) * effective_leverage,
        max(0.0, _finite(strategy.max_position_pct)) * effective_leverage,
        1.0,
    )
    max_loss_per_trade_pct = notional_cap_pct * max(0.0, _finite(strategy.stop_loss_pct))
    checks: list[RiskCheck] = []

    checks.append(_check("ok" if runtime.symbol == "BTCUSDC" else "block", "symbol", runtime.symbol))
    checks.append(
        _check(
            "ok" if runtime.market_type in {"spot", "futures"} else "block",
            "market type",
            runtime.market_type,
        )
    )
    environment = _environment(runtime)
    safe_endpoint = runtime.testnet or getattr(runtime, "demo", False)
    checks.append(
        _check(
            "ok" if safe_endpoint else "block",
            "execution environment",
            f"{environment} endpoint",
        )
    )
    if dry_run:
        checks.append(_check("ok", "order mode", "paper/dry-run"))
    else:
        has_credentials = bool(runtime.api_key and runtime.api_secret)
        checks.append(
            _check(
                "ok" if has_credentials else "block",
                "order credentials",
                "configured" if has_credentials else "missing API key/secret",
            )
        )

    cash = _finite(getattr(runtime, "managed_usdc", 0.0))
    checks.append(_check("ok" if cash > 0.0 else "block", "managed USDC", f"{cash:.2f}", metric=cash, limit=">0"))
    checks.append(
        _check(
            "ok" if effective_leverage <= 25.0 else "warn",
            "effective leverage",
            f"{effective_leverage:.1f}x",
            metric=effective_leverage,
            limit=25.0,
        )
    )
    risk_per_trade = _finite(strategy.risk_per_trade)
    checks.append(
        _check(
            "block" if risk_per_trade <= 0.0 else ("ok" if risk_per_trade <= 0.02 else "warn"),
            "risk per trade",
            f"{risk_per_trade:.2%}",
            metric=risk_per_trade,
            limit=0.02,
        )
    )
    max_position = _finite(strategy.max_position_pct)
    checks.append(
        _check(
            "block" if max_position <= 0.0 else ("ok" if max_position <= 0.50 else "warn"),
            "max position",
            f"{max_position:.2%}",
            metric=max_position,
            limit=0.50,
        )
    )
    checks.append(
        _check(
            "ok" if notional_cap_pct <= 0.50 else "warn",
            "entry notional cap",
            f"{notional_cap_pct:.2%} of equity",
            metric=notional_cap_pct,
            limit=0.50,
        )
    )
    stop_loss = _finite(strategy.stop_loss_pct)
    take_profit = _finite(strategy.take_profit_pct)
    checks.append(
        _check(
            "warn" if stop_loss <= 0.0 else "ok",
            "stop loss",
            f"{stop_loss:.2%}",
            metric=stop_loss,
            limit=">0",
        )
    )
    checks.append(
        _check(
            "warn" if take_profit <= 0.0 else "ok",
            "take profit",
            f"{take_profit:.2%}",
            metric=take_profit,
            limit=">0",
        )
    )
    checks.append(
        _check(
            "ok" if max_loss_per_trade_pct <= 0.02 else "warn",
            "estimated loss at stop",
            f"{max_loss_per_trade_pct:.2%} of equity",
            metric=max_loss_per_trade_pct,
            limit=0.02,
        )
    )

    if strategy.max_trades_per_day <= 0:
        checks.append(_check("warn", "daily trade cap", "disabled", metric=0, limit=">0"))
    else:
        checks.append(
            _check("ok", "daily trade cap", str(strategy.max_trades_per_day), metric=strategy.max_trades_per_day)
        )
    drawdown_limit = _finite(strategy.max_drawdown_limit)
    if drawdown_limit <= 0.0:
        checks.append(_check("warn", "drawdown stop", "disabled", metric=0.0, limit=">0"))
    else:
        checks.append(
            _check(
                "ok" if drawdown_limit <= 0.50 else "warn",
                "drawdown stop",
                f"{drawdown_limit:.2%}",
                metric=drawdown_limit,
                limit=0.50,
            )
        )
    slippage_bps = _finite(strategy.slippage_bps)
    checks.append(
        _check(
            "ok" if slippage_bps <= 100.0 else "warn",
            "slippage assumption",
            f"{slippage_bps:.1f} bps",
            metric=slippage_bps,
            limit=100.0,
        )
    )
    taker_fee_bps = _finite(strategy.taker_fee_bps)
    checks.append(
        _check(
            "ok" if taker_fee_bps <= 100.0 else "warn",
            "fee assumption",
            f"{taker_fee_bps:.1f} bps",
            metric=taker_fee_bps,
            limit=100.0,
        )
    )
    if strategy.external_signals_enabled:
        checks.append(
            _check(
                "ok",
                "external signal quorum",
                f"enabled min_providers={strategy.external_signal_min_providers}",
                metric=strategy.external_signal_min_providers,
            )
        )
    else:
        checks.append(_check("warn", "external signal quorum", "disabled"))

    if model_path is not None:
        path = Path(model_path)
        checks.append(
            _check(
                "ok" if dry_run or path.exists() else "block",
                "model path",
                str(path) if path.exists() else f"missing {path}",
            )
        )

    return RiskPolicyReport(
        checks=tuple(checks),
        effective_dry_run=dry_run,
        leverage=effective_leverage,
        notional_cap_pct=notional_cap_pct,
        max_loss_per_trade_pct=max_loss_per_trade_pct,
    )


def assess_entry_risk(
    *,
    direction: int,
    position_side: int,
    max_open_positions: int,
    max_daily_trades: int,
    daily_trade_count: int,
    cash: float,
    price: float,
    drawdown: float,
    drawdown_limit: float,
) -> EntryRiskDecision:
    metrics: dict[str, Any] = {
        "direction": int(direction),
        "position_side": int(position_side),
        "max_open_positions": int(max_open_positions),
        "max_daily_trades": int(max_daily_trades),
        "daily_trade_count": int(daily_trade_count),
        "cash": _metric_float(cash),
        "price": _metric_float(price),
        "drawdown": _metric_float(drawdown),
        "drawdown_limit": _metric_float(drawdown_limit),
    }
    if not all(math.isfinite(float(metrics[key])) for key in ("cash", "price", "drawdown", "drawdown_limit")):
        return EntryRiskDecision(False, "nonfinite", "non-finite risk input", metrics)
    if direction == 0:
        return EntryRiskDecision(False, "no_signal", "no actionable entry signal", metrics)
    if position_side != 0:
        return EntryRiskDecision(False, "position_open", "position already open", metrics)
    if max_open_positions <= 0:
        return EntryRiskDecision(False, "max_open_positions", "entry disabled by max_open_positions", metrics)
    if max_daily_trades > 0 and daily_trade_count >= max_daily_trades:
        return EntryRiskDecision(False, "trade_cap", "daily trade cap reached", metrics)
    if cash <= 0.0:
        return EntryRiskDecision(False, "cash", "no managed cash available", metrics)
    if price <= 0.0:
        return EntryRiskDecision(False, "price", "invalid market price", metrics)
    if drawdown_limit > 0.0 and drawdown >= drawdown_limit:
        return EntryRiskDecision(False, "drawdown", "drawdown limit already reached", metrics)
    return EntryRiskDecision(True, "allowed", "entry risk checks passed", metrics)


def render_risk_policy_report(report: RiskPolicyReport) -> str:
    lines = [
        "Risk policy report",
        (
            f"allowed={report.allowed} warnings={report.warning_count} blocks={report.block_count} "
            f"dry_run={report.effective_dry_run} leverage={report.leverage:.1f}x "
            f"notional_cap={report.notional_cap_pct:.2%} stop_loss_equity={report.max_loss_per_trade_pct:.2%}"
        ),
    ]
    for check in report.checks:
        lines.append(f"[{check.status}] {check.label}: {check.detail}")
    return "\n".join(lines)
