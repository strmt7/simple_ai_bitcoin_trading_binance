"""Live-run artifact helpers."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping

from .risk_controls import RiskPolicyReport
from .types import StrategyConfig


def build_live_run_payload(
    *,
    runtime_public: Mapping[str, Any],
    strategy: StrategyConfig,
    steps_total: int,
    market: str,
    symbol: str,
    model_path: Path,
    model: object | None,
    starting_cash: float,
    external_signal_cache: Path,
    risk_policy: RiskPolicyReport,
) -> dict[str, object]:
    return {
        "command": "live",
        "timestamp": int(time.time()),
        "runtime": dict(runtime_public),
        "strategy": strategy.asdict(),
        "steps_total": int(steps_total),
        "market": market,
        "symbol": symbol,
        "model_path": str(model_path),
        "events": [],
        "model_signature": str(getattr(model, "feature_signature", "")) or None,
        "starting_cash": float(starting_cash),
        "external_signal_cache": str(external_signal_cache),
        "risk_policy": risk_policy.asdict(),
    }
