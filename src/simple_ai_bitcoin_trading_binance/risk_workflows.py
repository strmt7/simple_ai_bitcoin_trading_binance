"""CLI-facing risk policy workflows."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable

from .risk_controls import build_risk_policy_report, render_risk_policy_report
from .types import RuntimeConfig, StrategyConfig


def command_risk(
    args: argparse.Namespace,
    *,
    load_runtime_fn: Callable[[], RuntimeConfig],
    load_strategy_fn: Callable[[], StrategyConfig],
) -> int:
    if getattr(args, "paper", False) and getattr(args, "live", False):
        print("Choose either --paper or --live, not both.")
        return 2
    runtime = load_runtime_fn()
    strategy = load_strategy_fn()
    if getattr(args, "live", False):
        effective_dry_run = False
    elif getattr(args, "paper", False):
        effective_dry_run = True
    else:
        effective_dry_run = bool(runtime.dry_run)
    leverage = getattr(args, "leverage", None)
    if leverage is not None and runtime.market_type != "futures":
        leverage = 1.0
    report = build_risk_policy_report(
        runtime,
        strategy,
        effective_dry_run=effective_dry_run,
        leverage=leverage,
        model_path=getattr(args, "model", None),
    )
    if getattr(args, "json", False):
        print(json.dumps(report.asdict(), indent=2, sort_keys=True))
    else:
        print(render_risk_policy_report(report))
    return 0 if report.allowed else 2
