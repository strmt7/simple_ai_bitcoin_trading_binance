"""Objective functions for ranking backtest / walk-forward results.

Three presets ship out of the box — ``conservative``, ``default``, ``risky`` —
each expressing a different risk-adjusted view of a candidate strategy.  They
are pure functions of a ``BacktestResult`` so they can be reused from the CLI,
the training suite, and the autonomous loop's sanity gates without import loops.

The scorers all return values in roughly the same range so the training suite
can compare across them when the user wants to see "how would the conservative
pick score under the risky lens".
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

from .backtest import BacktestResult


@dataclass(frozen=True)
class ObjectiveSpec:
    """A named scoring function with operator-facing metadata."""

    name: str
    label: str
    summary: str
    long_description: str
    scorer: Callable[[BacktestResult], float]
    min_closed_trades: int = 3
    max_drawdown_rejection: float = 1.0  # 1.0 = never reject on drawdown alone
    training: "ObjectiveTraining | None" = None

    def score(self, result: BacktestResult) -> float:
        return float(self.scorer(result))

    def accepts(self, result: BacktestResult) -> bool:
        """Return False when a candidate fails hard gates — used by tuning."""

        if result.closed_trades < max(0, int(self.min_closed_trades)):
            return False
        if self.max_drawdown_rejection < 1.0 and result.max_drawdown > self.max_drawdown_rejection:
            return False
        if result.stopped_by_drawdown and self.max_drawdown_rejection < 0.5:
            return False
        return True


@dataclass(frozen=True)
class ObjectiveTraining:
    """Per-objective hyperparameter + strategy defaults used by the training suite.

    These presets describe the intent of each objective — for example, the
    conservative profile trains on fewer epochs with stronger L2 regularization
    because it preferences model stability over ultimate score, while the risky
    profile pushes more epochs and a softer penalty to chase marginal gains.
    """

    epochs: int
    learning_rate: float
    l2_penalty: float
    signal_threshold: float
    stop_loss_pct: float
    take_profit_pct: float
    risk_per_trade: float
    max_position_pct: float
    max_trades_per_day: int
    leverage: float
    cooldown_minutes: int
    calibrate_threshold: bool
    walk_forward_train: int
    walk_forward_test: int
    walk_forward_step: int
    polynomial_degree: int = 2
    polynomial_top_features: int = 6
    extra_lookback_windows: tuple[int, ...] = (5, 20, 60)


def _safe(value: float, default: float = 0.0) -> float:
    return default if not math.isfinite(value) else float(value)


def _return_ratio(result: BacktestResult) -> float:
    if result.starting_cash <= 0:
        return 0.0
    return _safe(result.realized_pnl / result.starting_cash)


def _conservative_scorer(result: BacktestResult) -> float:
    """Reward steady, low-drawdown returns with a modest preference for win rate.

    Formula (roughly Calmar-like):
        return_ratio − 3·max_drawdown − 0.005·trades + 0.1·win_rate
    plus a hard penalty for runs that tripped the drawdown circuit breaker.
    """

    penalty = 0.25 if result.stopped_by_drawdown else 0.0
    return (
        _return_ratio(result)
        - 3.0 * _safe(result.max_drawdown)
        - 0.005 * float(result.closed_trades)
        + 0.10 * _safe(result.win_rate)
        - penalty
    )


def _default_scorer(result: BacktestResult) -> float:
    """Balanced risk-adjusted return scorer.

    Formula:
        return_ratio − 1.5·max_drawdown − 0.0005·trades + 0.15·win_rate
    """

    penalty = 0.10 if result.stopped_by_drawdown else 0.0
    return (
        _return_ratio(result)
        - 1.5 * _safe(result.max_drawdown)
        - 0.0005 * float(result.closed_trades)
        + 0.15 * _safe(result.win_rate)
        - penalty
    )


def _risky_scorer(result: BacktestResult) -> float:
    """Reward raw return more heavily and tolerate bigger drawdowns / more trades."""

    penalty = 0.05 if result.stopped_by_drawdown else 0.0
    return (
        1.25 * _return_ratio(result)
        - 0.8 * _safe(result.max_drawdown)
        + 0.05 * _safe(result.win_rate)
        - penalty
    )


CONSERVATIVE = ObjectiveSpec(
    name="conservative",
    label="Conservative",
    summary="Prioritize capital preservation, reject high drawdown, prefer few high-quality trades.",
    long_description=(
        "Training uses extra regularization and calibrated thresholds. Strategy "
        "defaults favor small position sizes, longer cooldowns, and a higher signal "
        "threshold so the bot stays out of coin-flip regimes."
    ),
    scorer=_conservative_scorer,
    min_closed_trades=5,
    max_drawdown_rejection=0.15,
    training=ObjectiveTraining(
        epochs=400,
        learning_rate=0.02,
        l2_penalty=5e-3,
        signal_threshold=0.66,
        stop_loss_pct=0.010,
        take_profit_pct=0.022,
        risk_per_trade=0.005,
        max_position_pct=0.10,
        max_trades_per_day=8,
        leverage=1.0,
        cooldown_minutes=15,
        calibrate_threshold=True,
        walk_forward_train=400,
        walk_forward_test=90,
        walk_forward_step=30,
        polynomial_degree=2,
        polynomial_top_features=5,
        extra_lookback_windows=(10, 30, 90),
    ),
)

DEFAULT = ObjectiveSpec(
    name="default",
    label="Default",
    summary="Balanced risk-adjusted return — the middle preset for most operators.",
    long_description=(
        "Balanced training budget and strategy defaults. Targets a middle-of-the-road "
        "Sharpe-like profile on walk-forward evaluation."
    ),
    scorer=_default_scorer,
    min_closed_trades=3,
    max_drawdown_rejection=0.25,
    training=ObjectiveTraining(
        epochs=600,
        learning_rate=0.03,
        l2_penalty=1e-3,
        signal_threshold=0.58,
        stop_loss_pct=0.018,
        take_profit_pct=0.030,
        risk_per_trade=0.010,
        max_position_pct=0.20,
        max_trades_per_day=16,
        leverage=1.5,
        cooldown_minutes=7,
        calibrate_threshold=True,
        walk_forward_train=500,
        walk_forward_test=120,
        walk_forward_step=40,
        polynomial_degree=2,
        polynomial_top_features=7,
        extra_lookback_windows=(5, 20, 60),
    ),
)

RISKY = ObjectiveSpec(
    name="risky",
    label="Risky",
    summary="Chase higher absolute returns; tolerates larger drawdowns and more trades.",
    long_description=(
        "Uses a longer training budget with a softer L2 and a lower signal threshold so "
        "the model takes more shots. Strategy defaults include a bigger max-position and "
        "shorter cooldown. Keep this preset on testnet."
    ),
    scorer=_risky_scorer,
    min_closed_trades=2,
    max_drawdown_rejection=0.45,
    training=ObjectiveTraining(
        epochs=900,
        learning_rate=0.04,
        l2_penalty=3e-4,
        signal_threshold=0.53,
        stop_loss_pct=0.028,
        take_profit_pct=0.045,
        risk_per_trade=0.018,
        max_position_pct=0.30,
        max_trades_per_day=32,
        leverage=2.5,
        cooldown_minutes=3,
        calibrate_threshold=True,
        walk_forward_train=600,
        walk_forward_test=150,
        walk_forward_step=50,
        polynomial_degree=3,
        polynomial_top_features=9,
        extra_lookback_windows=(3, 15, 45, 120),
    ),
)


_REGISTRY: dict[str, ObjectiveSpec] = {
    CONSERVATIVE.name: CONSERVATIVE,
    DEFAULT.name: DEFAULT,
    RISKY.name: RISKY,
}
_ALIASES: dict[str, str] = {
    "balanced": DEFAULT.name,
}


def available_objectives() -> tuple[str, ...]:
    return tuple(_REGISTRY.keys())


def get_objective(name: str) -> ObjectiveSpec:
    """Look up an objective by case-insensitive name."""

    key = str(name).strip().lower()
    if not key:
        raise ValueError("Objective name cannot be empty")
    key = _ALIASES.get(key, key)
    if key not in _REGISTRY:
        allowed = ", ".join(_REGISTRY)
        raise ValueError(f"Unknown objective {name!r}. Known: {allowed}")
    return _REGISTRY[key]


def describe_objectives() -> list[dict[str, str]]:
    """Return a display-friendly description for every registered objective."""

    return [
        {
            "name": spec.name,
            "label": spec.label,
            "summary": spec.summary,
            "long_description": spec.long_description,
        }
        for spec in _REGISTRY.values()
    ]


def rank_candidates(
    candidates: list[tuple[dict[str, object], BacktestResult]],
    objective: ObjectiveSpec,
) -> list[dict[str, object]]:
    """Sort ``candidates`` by objective score, annotate with reasons for rejection.

    Each entry is ``(params_dict, backtest_result)``.  Rejections are kept in
    the output but scored as ``-inf`` so they sink to the bottom; the reason is
    placed under ``"reject_reason"``.
    """

    ranked: list[dict[str, object]] = []
    for params, result in candidates:
        score = objective.score(result)
        accepted = objective.accepts(result)
        reject_reason: str | None = None
        if not accepted:
            reasons: list[str] = []
            if result.closed_trades < objective.min_closed_trades:
                reasons.append(f"closed_trades<{objective.min_closed_trades}")
            if objective.max_drawdown_rejection < 1.0 and result.max_drawdown > objective.max_drawdown_rejection:
                reasons.append(f"max_drawdown>{objective.max_drawdown_rejection}")
            if result.stopped_by_drawdown and objective.max_drawdown_rejection < 0.5:
                reasons.append("stopped_by_drawdown")
            reject_reason = "; ".join(reasons) or "hard-gate-failed"
        ranked.append({
            "params": params,
            "score": score if accepted else float("-inf"),
            "raw_score": score,
            "result": result,
            "accepted": accepted,
            "reject_reason": reject_reason,
        })
    def _rank_score(entry: dict[str, object]) -> float:
        value = entry["score"]
        return float(value) if isinstance(value, (int, float)) else float("-inf")

    ranked.sort(key=_rank_score, reverse=True)
    return ranked
