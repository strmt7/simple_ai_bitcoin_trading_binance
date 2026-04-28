"""Local operator audit for data, model, and risk readiness."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .api import Candle
from .features import make_rows
from .market_data import clean_candles
from .model import ModelFeatureMismatchError, ModelLoadError, load_model, model_decision_threshold
from .types import RuntimeConfig, StrategyConfig


@dataclass(frozen=True)
class AuditCheck:
    """A single readiness or quality check."""

    status: str
    label: str
    detail: str


@dataclass(frozen=True)
class AuditReport:
    """Structured result returned by ``build_audit_report``."""

    checks: tuple[AuditCheck, ...]
    raw_candles: int
    clean_candles: int
    feature_rows: int
    duplicate_open_times: int
    gap_count: int
    max_feature_delta: float | None

    @property
    def ok(self) -> bool:
        return all(check.status != "fix" for check in self.checks)


def _check(status: str, label: str, detail: str) -> AuditCheck:
    return AuditCheck(status=status, label=label, detail=detail)


def _duplicate_open_times(candles: Sequence[Candle]) -> int:
    seen: set[int] = set()
    duplicates = 0
    for candle in candles:
        if candle.open_time in seen:
            duplicates += 1
        else:
            seen.add(candle.open_time)
    return duplicates


def _dominant_interval_ms(candles: Sequence[Candle]) -> int | None:
    ordered = sorted(candles, key=lambda candle: candle.open_time)
    diffs = [
        int(right.open_time - left.open_time)
        for left, right in zip(ordered, ordered[1:])
        if right.open_time > left.open_time
    ]
    if not diffs:
        return None
    diffs.sort()
    return diffs[len(diffs) // 2]


def _gap_count(candles: Sequence[Candle]) -> int:
    interval = _dominant_interval_ms(candles)
    if interval is None or interval <= 0:
        return 0
    ordered = sorted(candles, key=lambda candle: candle.open_time)
    return sum(
        1
        for left, right in zip(ordered, ordered[1:])
        if right.open_time - left.open_time > interval * 1.5
    )


def _max_latest_feature_delta(candles: Sequence[Candle], strategy: StrategyConfig) -> float | None:
    rows_full = make_rows(
        candles,
        strategy.feature_windows[0],
        strategy.feature_windows[1],
        label_threshold=strategy.label_threshold,
        enabled_features=strategy.enabled_features,
    )
    if not rows_full:
        return None
    long_window = max(strategy.feature_windows)
    tail_size = max(80, long_window * 4)
    if len(candles) <= tail_size:
        return 0.0
    rows_tail = make_rows(
        list(candles)[-tail_size:],
        strategy.feature_windows[0],
        strategy.feature_windows[1],
        label_threshold=strategy.label_threshold,
        enabled_features=strategy.enabled_features,
    )
    if not rows_tail or rows_tail[-1].timestamp != rows_full[-1].timestamp:  # pragma: no cover - guarded by tail sizing
        return None
    return max(
        abs(left - right)
        for left, right in zip(rows_full[-1].features, rows_tail[-1].features)
    )


def build_audit_report(
    candles: Sequence[Candle],
    runtime: RuntimeConfig,
    strategy: StrategyConfig,
    *,
    model_path: Path | None = None,
) -> AuditReport:
    """Build a no-network audit of the current local operator state."""

    raw_count = len(candles)
    duplicates = _duplicate_open_times(candles)
    cleaned = clean_candles(candles)
    gaps = _gap_count(cleaned)
    rows = make_rows(
        cleaned,
        strategy.feature_windows[0],
        strategy.feature_windows[1],
        label_threshold=strategy.label_threshold,
        enabled_features=strategy.enabled_features,
    )
    max_delta = _max_latest_feature_delta(cleaned, strategy)

    checks: list[AuditCheck] = []
    checks.append(_check("ok" if runtime.symbol == "BTCUSDC" else "fix", "symbol", runtime.symbol))
    checks.append(_check("ok" if runtime.testnet else "fix", "safety target", "testnet enabled" if runtime.testnet else "testnet disabled"))
    checks.append(_check("ok" if runtime.dry_run else "warn", "default execution", "paper mode" if runtime.dry_run else "authenticated testnet live by default"))
    checks.append(_check("ok" if runtime.market_type in {"spot", "futures"} else "fix", "market type", runtime.market_type))
    checks.append(_check("ok" if raw_count > 0 else "fix", "raw candles", str(raw_count)))
    checks.append(_check("ok" if len(cleaned) == raw_count else "warn", "candle cleaning", f"clean={len(cleaned)} raw={raw_count} duplicates={duplicates}"))
    checks.append(_check("ok" if gaps == 0 else "warn", "time gaps", f"{gaps} gap(s) detected"))
    checks.append(_check("ok" if rows else "fix", "feature rows", str(len(rows))))
    if max_delta is None:
        checks.append(_check("warn", "feature stability", "insufficient comparable tail window"))
    else:
        status = "ok" if max_delta <= 1e-9 else "warn"
        checks.append(_check(status, "feature stability", f"latest-row max_delta={max_delta:.3g}"))

    if strategy.max_open_positions <= 0:
        checks.append(_check("warn", "open-position gate", "max_open_positions=0 disables entries"))
    else:
        checks.append(_check("ok", "open-position gate", f"max_open_positions={strategy.max_open_positions}"))
    if strategy.max_trades_per_day <= 0:
        checks.append(_check("warn", "daily trade cap", "disabled"))
    else:
        checks.append(_check("ok", "daily trade cap", str(strategy.max_trades_per_day)))
    checks.append(_check("ok" if strategy.risk_per_trade <= 0.05 else "warn", "risk per trade", f"{strategy.risk_per_trade:.2%}"))
    checks.append(_check("ok" if strategy.max_position_pct <= 0.50 else "warn", "max position", f"{strategy.max_position_pct:.2%}"))
    checks.append(_check("ok" if strategy.max_drawdown_limit <= 0.50 else "warn", "drawdown limit", f"{strategy.max_drawdown_limit:.2%}"))

    if model_path is not None:
        if not model_path.exists():
            checks.append(_check("fix", "model artifact", f"missing {model_path}"))
        else:
            try:
                from .features import feature_signature

                model = load_model(
                    model_path,
                    expected_feature_version=strategy.feature_version,
                    expected_feature_signature=feature_signature(
                        strategy.feature_windows[0],
                        strategy.feature_windows[1],
                        strategy.label_threshold,
                        feature_version=strategy.feature_version,
                        enabled_features=strategy.enabled_features,
                    ),
                    expected_feature_dim=None,
                )
            except (OSError, ValueError, ModelLoadError, ModelFeatureMismatchError) as exc:
                checks.append(_check("fix", "model artifact", f"not usable: {exc}"))
            else:
                threshold = model_decision_threshold(model, strategy.signal_threshold)
                checks.append(
                    _check(
                        "ok",
                        "model artifact",
                        f"dim={model.feature_dim} threshold={threshold:.3f} validation={model.validation_size}",
                    )
                )

    return AuditReport(
        checks=tuple(checks),
        raw_candles=raw_count,
        clean_candles=len(cleaned),
        feature_rows=len(rows),
        duplicate_open_times=duplicates,
        gap_count=gaps,
        max_feature_delta=max_delta,
    )


def render_audit_report(report: AuditReport) -> str:
    """Render ``AuditReport`` for CLI/TUI use."""

    lines = ["Local operator audit"]
    for check in report.checks:
        lines.append(f"[{check.status}] {check.label}: {check.detail}")
    lines.extend(
        [
            "",
            "Next steps:",
            "- fix any [fix] item before paper or authenticated testnet execution",
            "- investigate [warn] items before increasing risk or loop duration",
            "- rerun prepare, evaluate, backtest, doctor, and audit after changing strategy settings",
        ]
    )
    return "\n".join(lines)
