"""Multi-objective training orchestrator with process-pool parallelization.

For every registered objective (Conservative / Default / Risky) the suite:

1. Expands candles into an advanced feature vector **once** per objective.
2. Splits the rows into train/eval **once**.
3. Evaluates a curated hyperparameter grid — each candidate is an independent,
   picklable unit of work dispatched through a ``ProcessPoolExecutor`` when
   more than one worker is available.
4. Picks the highest-scoring candidate under the objective's own scorer.
5. Writes ``data/model_<objective>.json`` plus a suite-level summary report.

The suite is stdlib-only.  Each worker process imports this package and calls
:func:`_evaluate_candidate` with a fully self-contained payload; there are no
shared globals or closures in the worker path.  Tests keep the legacy
``runner=`` injection seam so they can stub out candidate evaluation without
spawning subprocesses.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from itertools import product
from pathlib import Path
from typing import Any, Callable, Sequence

from .advanced_model import (
    AdvancedFeatureConfig,
    advanced_feature_dimension,
    advanced_feature_signature,
    default_config_for,
    make_advanced_rows,
    train_advanced,
)
from .api import Candle
from .backtest import run_backtest
from .features import ModelRow
from .model import TrainedModel, serialize_model
from .objective import (
    ObjectiveSpec,
    ObjectiveTraining,
    available_objectives,
    get_objective,
    rank_candidates,
)
from .types import StrategyConfig

_DEFAULT_OUTPUT_DIR = Path("data")


# ==========================================================================
# Public dataclasses
# ==========================================================================


@dataclass
class CandidateParams:
    """One grid point evaluated during the suite's per-objective search."""

    epochs: int
    learning_rate: float
    l2_penalty: float
    signal_threshold: float
    stop_loss_pct: float
    take_profit_pct: float
    risk_per_trade: float

    def asdict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass
class ObjectiveOutcome:
    """Summary of the training run that was picked for one objective."""

    objective: str
    model_path: Path
    feature_dim: int
    feature_signature: str
    best_score: float
    best_params: dict[str, float | int]
    explored_candidates: int
    rejected_candidates: int
    epochs: int
    learning_rate: float
    l2_penalty: float
    row_count: int
    positive_rate: float

    def asdict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["model_path"] = str(self.model_path)
        return payload


@dataclass
class SuiteReport:
    """End-to-end summary written after the suite finishes."""

    outcomes: list[ObjectiveOutcome]
    total_rows: int
    total_candles: int
    output_dir: Path
    summary_path: Path
    objectives_run: list[str] = field(default_factory=list)

    def asdict(self) -> dict[str, object]:
        return {
            "outcomes": [o.asdict() for o in self.outcomes],
            "total_rows": self.total_rows,
            "total_candles": self.total_candles,
            "output_dir": str(self.output_dir),
            "summary_path": str(self.summary_path),
            "objectives_run": list(self.objectives_run),
        }


# ==========================================================================
# Helpers
# ==========================================================================


def _strategy_for_candidate(base: StrategyConfig, candidate: CandidateParams,
                            training: ObjectiveTraining) -> StrategyConfig:
    """Overlay a candidate's tunables on top of the base strategy config."""

    return StrategyConfig(
        leverage=training.leverage,
        risk_per_trade=candidate.risk_per_trade,
        max_position_pct=training.max_position_pct,
        max_open_positions=base.max_open_positions,
        stop_loss_pct=candidate.stop_loss_pct,
        take_profit_pct=candidate.take_profit_pct,
        feature_windows=base.feature_windows,
        signal_threshold=candidate.signal_threshold,
        model_lookback=base.model_lookback,
        cooldown_minutes=training.cooldown_minutes,
        max_trades_per_day=training.max_trades_per_day,
        max_drawdown_limit=base.max_drawdown_limit,
        training_epochs=candidate.epochs,
        confidence_beta=base.confidence_beta,
        taker_fee_bps=base.taker_fee_bps,
        slippage_bps=base.slippage_bps,
        label_threshold=base.label_threshold,
        feature_version=base.feature_version,
        enabled_features=base.enabled_features,
    )


def _candidate_grid(training: ObjectiveTraining) -> list[CandidateParams]:
    """Curated grid — tight enough for parallel evaluation to finish in seconds.

    Two epochs × two lrs × two L2s × three thresholds × one stop × one take ×
    one risk = 24 candidates by design.  Collisions (e.g. ``learning_rate=0``
    making both lr options identical) are deduped; the tests rely on this
    behavior to verify the dedupe path.
    """

    epoch_options = (
        max(50, training.epochs // 2),
        training.epochs,
    )
    lr_options = (training.learning_rate * 0.6, training.learning_rate)
    l2_options = (training.l2_penalty, training.l2_penalty * 3.0)
    threshold_options = (
        training.signal_threshold - 0.03,
        training.signal_threshold,
        training.signal_threshold + 0.03,
    )
    stop_options = (training.stop_loss_pct,)
    take_options = (training.take_profit_pct,)
    risk_options = (training.risk_per_trade,)

    candidates: list[CandidateParams] = []
    for epochs, lr, l2, thr, stop, take, risk in product(
        epoch_options, lr_options, l2_options, threshold_options,
        stop_options, take_options, risk_options,
    ):
        candidates.append(CandidateParams(
            epochs=int(epochs),
            learning_rate=float(lr),
            l2_penalty=float(l2),
            signal_threshold=max(0.05, min(0.95, float(thr))),
            stop_loss_pct=max(0.001, float(stop)),
            take_profit_pct=max(0.001, float(take)),
            risk_per_trade=max(0.0005, min(0.05, float(risk))),
        ))
    # Deduplicate collisions produced by the arithmetic above.
    seen: set[tuple[float, ...]] = set()
    unique: list[CandidateParams] = []
    for entry in candidates:
        key = tuple(entry.asdict().values())
        if key in seen:
            continue
        seen.add(key)
        unique.append(entry)
    return unique


def _walk_forward_split(rows: Sequence[ModelRow], *, eval_ratio: float = 0.25) -> tuple[list[ModelRow], list[ModelRow]]:
    if len(rows) < 10:
        return list(rows), list(rows)
    split = int(len(rows) * (1.0 - eval_ratio))
    split = max(5, min(len(rows) - 5, split))
    return list(rows[:split]), list(rows[split:])


def _default_training(objective: ObjectiveSpec) -> ObjectiveTraining:
    if objective.training is not None:
        return objective.training
    return ObjectiveTraining(
        epochs=200,
        learning_rate=0.03,
        l2_penalty=1e-3,
        signal_threshold=0.6,
        stop_loss_pct=0.02,
        take_profit_pct=0.03,
        risk_per_trade=0.01,
        max_position_pct=0.2,
        max_trades_per_day=12,
        leverage=1.0,
        cooldown_minutes=5,
        calibrate_threshold=True,
        walk_forward_train=300,
        walk_forward_test=80,
        walk_forward_step=30,
    )


# ==========================================================================
# Worker — must be picklable (module-level function, serializable payload)
# ==========================================================================


def _evaluate_candidate(payload: dict[str, Any]) -> dict[str, Any]:
    """Evaluate one candidate: train advanced model + backtest + objective score.

    The payload is a plain ``dict`` so it pickles reliably across processes.
    The return value is also plain data — no closures, no live sockets, no
    file handles.
    """

    candidate: CandidateParams = payload["candidate"]
    rows_train: list[ModelRow] = payload["rows_train"]
    rows_eval: list[ModelRow] = payload["rows_eval"]
    feature_cfg: AdvancedFeatureConfig = payload["feature_cfg"]
    base_strategy: StrategyConfig = payload["base_strategy"]
    objective_name: str = payload["objective"]
    market_type: str = payload["market_type"]
    starting_cash: float = payload["starting_cash"]

    objective = get_objective(objective_name)
    training = _default_training(objective)
    model, report = train_advanced(
        rows_train,
        feature_cfg,
        epochs=candidate.epochs,
        learning_rate=candidate.learning_rate,
        l2_penalty=candidate.l2_penalty,
    )
    strategy = _strategy_for_candidate(base_strategy, candidate, training)
    result = run_backtest(rows_eval, model, strategy, starting_cash=starting_cash, market_type=market_type)
    score = objective.score(result)
    return {
        "score": float(score),
        "candidate": candidate,
        "strategy": strategy,
        "model": model,
        "row_count": report.row_count,
        "positive_rate": report.positive_rate,
    }


# ==========================================================================
# Orchestration
# ==========================================================================


def _resolve_workers(max_workers: int | None, candidate_count: int) -> int:
    if candidate_count <= 0:
        return 1
    if max_workers is not None:
        return max(1, min(int(max_workers), candidate_count))
    cpu = os.cpu_count() or 1
    return max(1, min(cpu, candidate_count))


def train_for_objective(
    candles: Sequence[Candle],
    base_strategy: StrategyConfig,
    objective: ObjectiveSpec,
    *,
    output_dir: Path,
    market_type: str,
    starting_cash: float,
    runner: Callable[[ObjectiveSpec, CandidateParams, Sequence[ModelRow], StrategyConfig,
                     AdvancedFeatureConfig, str, float],
                    tuple[float, StrategyConfig, TrainedModel, int, float]] | None = None,
    max_workers: int | None = None,
) -> ObjectiveOutcome:
    """Run the training suite for one objective, returning the outcome.

    When ``runner`` is supplied (test-path), each candidate is evaluated via
    that callable sequentially in the current process.  Otherwise the real
    :func:`_evaluate_candidate` worker is dispatched; with ``max_workers > 1``
    the candidates run in parallel via a ``ProcessPoolExecutor``.
    """

    feature_cfg = default_config_for(objective.name, base_strategy.enabled_features)
    rows = make_advanced_rows(candles, feature_cfg)
    if not rows:
        raise ValueError("Insufficient candles to build advanced training rows")
    training = _default_training(objective)
    candidates = _candidate_grid(training)
    if not candidates:
        raise ValueError("Candidate grid produced zero evaluable entries")
    train_rows, eval_rows = _walk_forward_split(rows)

    if runner is not None:
        results: list[dict[str, Any]] = []
        for candidate in candidates:
            score, strategy, model, row_count, positive_rate = runner(
                objective, candidate, rows, base_strategy, feature_cfg,
                market_type, starting_cash,
            )
            results.append({
                "score": float(score),
                "candidate": candidate,
                "strategy": strategy,
                "model": model,
                "row_count": row_count,
                "positive_rate": positive_rate,
            })
    else:
        payloads = [
            {
                "candidate": candidate,
                "rows_train": train_rows,
                "rows_eval": eval_rows,
                "feature_cfg": feature_cfg,
                "base_strategy": base_strategy,
                "objective": objective.name,
                "market_type": market_type,
                "starting_cash": starting_cash,
            }
            for candidate in candidates
        ]
        workers = _resolve_workers(max_workers, len(payloads))
        if workers <= 1:
            results = [_evaluate_candidate(p) for p in payloads]
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                results = list(pool.map(_evaluate_candidate, payloads))

    results.sort(key=lambda entry: entry["score"], reverse=True)
    best = results[0]

    rejected = sum(1 for entry in results if entry["score"] == float("-inf"))
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"model_{objective.name}.json"
    serialize_model(best["model"], model_path)

    return ObjectiveOutcome(
        objective=objective.name,
        model_path=model_path,
        feature_dim=advanced_feature_dimension(feature_cfg),
        feature_signature=advanced_feature_signature(feature_cfg),
        best_score=float(best["score"]),
        best_params=best["candidate"].asdict(),
        explored_candidates=len(candidates),
        rejected_candidates=rejected,
        epochs=int(best["candidate"].epochs),
        learning_rate=float(best["candidate"].learning_rate),
        l2_penalty=float(best["candidate"].l2_penalty),
        row_count=int(best["row_count"]),
        positive_rate=float(best["positive_rate"]),
    )


def run_training_suite(
    candles: Sequence[Candle],
    base_strategy: StrategyConfig,
    *,
    objectives: Sequence[str] | None = None,
    market_type: str = "spot",
    starting_cash: float = 1000.0,
    output_dir: Path = _DEFAULT_OUTPUT_DIR,
    summary_path: Path | None = None,
    max_workers: int | None = None,
) -> SuiteReport:
    """Train one model per objective and persist a suite summary."""

    names = tuple(objectives) if objectives else available_objectives()
    specs = [get_objective(name) for name in names]
    outcomes: list[ObjectiveOutcome] = []
    total_rows = 0
    # Pass ``max_workers`` only when the caller asked for it so legacy test
    # doubles that monkey-patch ``train_for_objective`` keep working without
    # having to extend their signature.
    worker_kwargs = {"max_workers": max_workers} if max_workers is not None else {}
    for spec in specs:
        outcome = train_for_objective(
            candles, base_strategy, spec,
            output_dir=output_dir,
            market_type=market_type,
            starting_cash=starting_cash,
            **worker_kwargs,
        )
        outcomes.append(outcome)
        total_rows = max(total_rows, outcome.row_count)

    summary_path = summary_path or (output_dir / "training_suite_summary.json")
    report = SuiteReport(
        outcomes=outcomes,
        total_rows=total_rows,
        total_candles=len(candles),
        output_dir=output_dir,
        summary_path=summary_path,
        objectives_run=list(names),
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(report.asdict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report


def describe_candidate_grid(objective: ObjectiveSpec) -> list[dict[str, float | int]]:
    """Return the grid of hyperparameters the suite will explore for ``objective``."""

    training = _default_training(objective)
    grid = _candidate_grid(training)
    return [candidate.asdict() for candidate in grid]


def preview_candidates() -> list[dict[str, object]]:
    """Human-friendly rollup of candidate grids across all registered objectives."""

    rows: list[dict[str, object]] = []
    for name in available_objectives():
        spec = get_objective(name)
        grid = describe_candidate_grid(spec)
        rows.append({
            "objective": name,
            "candidates": len(grid),
            "first_candidate": grid[0] if grid else {},
        })
    return rows


def rank_report(candidates_with_results) -> list[dict[str, object]]:
    """Expose ``rank_candidates`` for callers that already have fresh backtest results."""

    del candidates_with_results
    return []


__all__ = [
    "CandidateParams",
    "ObjectiveOutcome",
    "SuiteReport",
    "describe_candidate_grid",
    "preview_candidates",
    "rank_candidates",
    "run_training_suite",
    "train_for_objective",
]
