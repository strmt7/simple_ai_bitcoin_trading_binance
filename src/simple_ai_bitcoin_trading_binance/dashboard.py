"""Dashboard rendering helpers for the operator console."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DashboardSnapshot:
    runtime: dict[str, Any]
    strategy: dict[str, Any]
    artifacts: list[str]
    account_lines: list[str]
    notes: list[str]


def _box(title: str, lines: list[str], *, width: int = 88) -> str:
    inner = max(20, width - 4)
    top = f"+- {title[: inner - 2].ljust(inner - 2, '-')} -+"
    body = [f"| {line[:inner].ljust(inner)} |" for line in lines] or [f"| {'':{inner}} |"]
    bottom = "+" + ("-" * (width - 2)) + "+"
    return "\n".join([top, *body, bottom])


def _runtime_lines(runtime: dict[str, Any]) -> list[str]:
    return [
        f"symbol={runtime.get('symbol', '-')} interval={runtime.get('interval', '-')} market={runtime.get('market_type', '-')}",
        f"testnet={runtime.get('testnet', '-')} paper_default={runtime.get('dry_run', '-')} validate_account={runtime.get('validate_account', '-')}",
        f"api_key={'loaded' if runtime.get('api_key') else 'missing'} api_secret={'loaded' if runtime.get('api_secret') else 'missing'}",
        f"max_rate_calls_per_minute={runtime.get('max_rate_calls_per_minute', '-')}",
    ]


def _strategy_lines(strategy: dict[str, Any]) -> list[str]:
    feature_windows = strategy.get("feature_windows", ["-", "-"])
    return [
        f"threshold={strategy.get('signal_threshold', '-')} label_threshold={strategy.get('label_threshold', '-')}",
        f"risk={strategy.get('risk_per_trade', '-')} max_position={strategy.get('max_position_pct', '-')}",
        f"stop={strategy.get('stop_loss_pct', '-')} take={strategy.get('take_profit_pct', '-')}",
        f"cooldown={strategy.get('cooldown_minutes', '-')} max_trades_per_day={strategy.get('max_trades_per_day', '-')}",
        f"lookback={strategy.get('model_lookback', '-')} epochs={strategy.get('training_epochs', '-')} feature_windows={feature_windows}",
    ]


def _artifact_lines(artifacts: list[str]) -> list[str]:
    return artifacts or ["No recent artifacts found under data/."]


def _account_lines(lines: list[str]) -> list[str]:
    return lines or ["No account data loaded."]


def render_dashboard(snapshot: DashboardSnapshot, *, width: int = 88) -> str:
    sections = [
        _box("Runtime", _runtime_lines(snapshot.runtime), width=width),
        _box("Strategy", _strategy_lines(snapshot.strategy), width=width),
        _box("Account", _account_lines(snapshot.account_lines), width=width),
        _box("Recent artifacts", _artifact_lines(snapshot.artifacts), width=width),
        _box("Actions", [
            "Use menu for guided actions or subcommands for automation.",
            "Safe authenticated testnet execution is explicit; secrets stay redacted in visible output.",
            *snapshot.notes,
        ], width=width),
    ]
    return "\n\n".join(sections)


def load_artifact_preview(path: Path) -> str:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return f"{path.name} [unreadable]"
    if not isinstance(payload, dict):
        return f"{path.name} [non-object]"
    command = payload.get("command", "json")
    timestamp = payload.get("timestamp", "-")
    runtime = payload.get("runtime", {})
    symbol = payload.get("symbol") or runtime.get("symbol", "-")
    market = payload.get("market") or runtime.get("market_type", "-")
    return f"{path.name} command={command} symbol={symbol} market={market} ts={timestamp}"
