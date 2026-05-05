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

import math
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
from .backtest import BacktestResult, calibrate_threshold_for_backtest, run_backtest
from .features import ModelRow
from .model import (
    TrainedModel,
    calibrate_probability_temperature,
    calibrate_threshold,
    evaluate_classification,
    serialize_model,
)
from .objective import (
    ObjectiveSpec,
    ObjectiveTraining,
    available_objectives,
    get_objective,
    rank_candidates,
)
from .strategy_overrides import strategy_overrides_from_config
from .storage import write_json_atomic
from .types import StrategyConfig

_DEFAULT_OUTPUT_DIR = Path("data")
_ENSEMBLE_REFINEMENT_CANDIDATES = 3


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
    confidence_beta: float = 0.85
    seed: int = 7

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
    decision_threshold: float | None = None
    threshold_source: str | None = None
    calibration_score: float | None = None
    calibration_rows: int = 0
    validation_rows: int = 0
    validation_score: float | None = None
    full_sample_score: float | None = None
    ensemble_refined: bool = False
    ensemble_refinement_candidates: int = 0
    local_refinement_candidates: int = 0
    training_backend_requested: str = "cpu"
    training_backend_kind: str = "cpu"
    training_backend_device: str = "cpu"

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
        confidence_beta=candidate.confidence_beta,
        taker_fee_bps=base.taker_fee_bps,
        slippage_bps=base.slippage_bps,
        label_threshold=base.label_threshold,
        feature_version=base.feature_version,
        enabled_features=base.enabled_features,
    )


def _attach_strategy_overrides(model: TrainedModel, strategy: StrategyConfig) -> TrainedModel:
    """Persist execution parameters selected alongside the model weights."""

    model.strategy_overrides = strategy_overrides_from_config(strategy)
    return model


def _candidate_grid(training: ObjectiveTraining) -> list[CandidateParams]:
    """Curated first-pass grid without exploding runtime.

    Three epoch budgets, two learning rates, two L2 penalties, three thresholds,
    one stop/take profile, two risk levels, one confidence beta, and one seed.
    The suite then searches locally around the winner and checks seed ensembles
    for the best candidates, keeping GPU runs finishable while still deduping
    arithmetic collisions.
    """

    epoch_options = (
        max(80, int(training.epochs * 0.20)),
        max(120, int(training.epochs * 0.50)),
        training.epochs,
    )
    lr_options = (training.learning_rate * 0.75, training.learning_rate)
    l2_options = (training.l2_penalty, training.l2_penalty * 3.0)
    threshold_options = (
        training.signal_threshold - 0.03,
        training.signal_threshold,
        training.signal_threshold + 0.03,
    )
    stop_take_options = ((training.stop_loss_pct, training.take_profit_pct),)
    risk_options = (training.risk_per_trade * 0.50, training.risk_per_trade)
    confidence_options = (0.85,)
    seed_options = (7,)

    candidates: list[CandidateParams] = []
    for epochs, lr, l2, thr, stop_take, risk, confidence, seed in product(
        epoch_options, lr_options, l2_options, threshold_options,
        stop_take_options, risk_options, confidence_options, seed_options,
    ):
        stop, take = stop_take
        candidates.append(CandidateParams(
            epochs=int(epochs),
            learning_rate=float(lr),
            l2_penalty=float(l2),
            signal_threshold=max(0.05, min(0.95, float(thr))),
            stop_loss_pct=max(0.001, float(stop)),
            take_profit_pct=max(0.001, float(take)),
            risk_per_trade=max(0.0005, min(0.05, float(risk))),
            confidence_beta=max(0.0, min(1.0, float(confidence))),
            seed=int(seed),
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


def _calibration_split(rows: Sequence[ModelRow], *, ratio: float = 0.20) -> tuple[list[ModelRow], list[ModelRow]]:
    ordered = list(rows)
    if len(ordered) < 30:
        return ordered, []
    calibration_size = max(8, int(len(ordered) * max(0.05, min(0.40, ratio))))
    calibration_size = min(calibration_size, len(ordered) - 10)
    return ordered[:-calibration_size], ordered[-calibration_size:]


def _threshold_guard(baseline: object, candidate: object) -> bool:
    baseline_accuracy = float(getattr(baseline, "accuracy", 0.0))
    baseline_f1 = float(getattr(baseline, "f1", 0.0))
    baseline_precision = float(getattr(baseline, "precision", 0.0))
    candidate_accuracy = float(getattr(candidate, "accuracy", 0.0))
    candidate_f1 = float(getattr(candidate, "f1", 0.0))
    candidate_precision = float(getattr(candidate, "precision", 0.0))
    candidate_true_positive = int(getattr(candidate, "true_positive", 0))
    candidate_false_negative = int(getattr(candidate, "false_negative", 0))
    if candidate_false_negative > 0 and candidate_true_positive <= 0:
        return False
    stable_f1 = candidate_accuracy + 0.03 >= baseline_accuracy and candidate_f1 + 0.05 >= baseline_f1
    sharper_precision = candidate_accuracy >= baseline_accuracy + 0.02 and candidate_precision + 0.02 >= baseline_precision
    return bool(stable_f1 or sharper_precision)


def _calibrate_candidate_threshold(
    model: TrainedModel,
    rows: Sequence[ModelRow],
    strategy: StrategyConfig,
    *,
    market_type: str,
    starting_cash: float,
    compute_backend: str | None = None,
    score_batch_size: int = 8192,
) -> tuple[float, str, float | None]:
    if not rows:
        return strategy.signal_threshold, "strategy", None
    strategy_threshold = float(strategy.signal_threshold)
    classification_threshold = calibrate_threshold(list(rows), model, start=0.05, end=0.95, steps=61)
    baseline_report = evaluate_classification(list(rows), model, threshold=strategy_threshold)
    profit_report = calibrate_threshold_for_backtest(
        list(rows),
        model,
        strategy,
        starting_cash=starting_cash,
        market_type=market_type,
        baseline_threshold=strategy_threshold,
        start=0.05,
        end=0.95,
        steps=121,
        compute_backend=compute_backend,
        score_batch_size=score_batch_size,
    )
    candidate_report = evaluate_classification(list(rows), model, threshold=profit_report.best_threshold)
    profit_backed = (
        profit_report.accepted
        and profit_report.realized_pnl > 0.0
        and profit_report.closed_trades > 0
        and profit_report.score > 0.0
    )
    if profit_backed and _threshold_guard(baseline_report, candidate_report):
        return float(profit_report.threshold), "profit_backtest", float(profit_report.score)
    classification_report = evaluate_classification(list(rows), model, threshold=classification_threshold)
    if (
        abs(float(classification_threshold) - strategy_threshold) > 1e-12
        and profit_report.baseline_score >= 0.0
        and _threshold_guard(baseline_report, classification_report)
    ):
        return float(classification_threshold), "classification_f1", float(profit_report.baseline_score)
    return strategy_threshold, "strategy", float(profit_report.baseline_score)


def _refine_threshold_on_selection_rows(
    model: TrainedModel,
    rows: Sequence[ModelRow],
    strategy: StrategyConfig,
    *,
    market_type: str,
    starting_cash: float,
    compute_backend: str | None = None,
    score_batch_size: int = 8192,
) -> tuple[float, str, float] | None:
    """Promote a selection-set profit threshold only when it remains directional.

    The final chronological holdout is deliberately excluded from this helper.
    It can be used to tune the threshold carried into selection, but not to
    adjust the artifact after final holdout scoring.
    """

    if len(rows) < 30:
        return None
    row_list = list(rows)
    baseline_threshold = float(model.decision_threshold if model.decision_threshold is not None else strategy.signal_threshold)
    baseline_report = evaluate_classification(row_list, model, threshold=baseline_threshold)
    current_threshold = model.decision_threshold
    best_threshold: float | None = None
    best_score = float("-inf")
    best_rank = (float("-inf"), float("-inf"), float("-inf"), float("-inf"))
    try:
        for threshold in _threshold_values(0.05, 0.95, 121, baseline_threshold):
            model.decision_threshold = float(threshold)
            result = run_backtest(
                row_list,
                model,
                strategy,
                starting_cash=starting_cash,
                market_type=market_type,
                compute_backend=compute_backend,
                score_batch_size=score_batch_size,
            )
            if result.realized_pnl <= 0.0 or result.closed_trades <= 0:
                continue
            candidate_report = evaluate_classification(row_list, model, threshold=threshold)
            if not _threshold_guard(baseline_report, candidate_report):
                continue
            rank = (
                float(result.realized_pnl),
                float(candidate_report.f1),
                -float(result.max_drawdown),
                -float(result.closed_trades),
            )
            if rank > best_rank:
                best_rank = rank
                best_score = float(result.realized_pnl)
                best_threshold = float(threshold)
    finally:
        model.decision_threshold = current_threshold
    if best_threshold is None:
        return None
    return best_threshold, "selection_profit_backtest", best_score


def _refine_threshold_on_validation_rows(
    model: TrainedModel,
    rows: Sequence[ModelRow],
    strategy: StrategyConfig,
    *,
    market_type: str,
    starting_cash: float,
    compute_backend: str | None = None,
    score_batch_size: int = 8192,
) -> tuple[float, str, float] | None:
    """Compatibility wrapper for older internal tests/callers."""

    return _refine_threshold_on_selection_rows(
        model,
        rows,
        strategy,
        market_type=market_type,
        starting_cash=starting_cash,
        compute_backend=compute_backend,
        score_batch_size=score_batch_size,
    )


def _threshold_values(start: float, end: float, steps: int, baseline: float) -> list[float]:
    if steps <= 1:
        return [max(0.0, min(1.0, float(baseline)))]
    low = max(0.0, min(1.0, float(start)))
    high = max(0.0, min(1.0, float(end)))
    if high <= low:
        high = min(1.0, low + 0.01)
    values = [low + (high - low) * i / (steps - 1) for i in range(steps)]
    values.append(max(0.0, min(1.0, float(baseline))))
    return sorted(set(round(value, 12) for value in values))


def _walk_forward_split(rows: Sequence[ModelRow], *, eval_ratio: float = 0.25) -> tuple[list[ModelRow], list[ModelRow]]:
    if len(rows) < 10:
        return list(rows), list(rows)
    split = int(len(rows) * (1.0 - eval_ratio))
    split = max(5, min(len(rows) - 5, split))
    return list(rows[:split]), list(rows[split:])


def _ensemble_seed_pack(seed: int) -> tuple[int, int, int]:
    base = int(seed)
    return base, base + 17, base + 37


def _candidate_variant(candidate: CandidateParams, **updates: float) -> CandidateParams:
    payload = candidate.asdict()
    payload.update(updates)
    return CandidateParams(
        epochs=max(1, int(payload["epochs"])),
        learning_rate=max(1e-6, float(payload["learning_rate"])),
        l2_penalty=max(0.0, float(payload["l2_penalty"])),
        signal_threshold=max(0.05, min(0.95, float(payload["signal_threshold"]))),
        stop_loss_pct=max(0.001, float(payload["stop_loss_pct"])),
        take_profit_pct=max(0.001, float(payload["take_profit_pct"])),
        risk_per_trade=max(0.0005, min(0.05, float(payload["risk_per_trade"]))),
        confidence_beta=max(0.0, min(1.0, float(payload["confidence_beta"]))),
        seed=int(payload["seed"]),
    )


def _local_refinement_candidates(candidate: CandidateParams) -> list[CandidateParams]:
    """Small post-grid search around the current winner.

    The broad grid keeps runtime predictable; this local pass exists for
    boundary cases where the winner sits on a coarse grid edge.
    """

    return [
        _candidate_variant(candidate, learning_rate=candidate.learning_rate * 0.75),
        _candidate_variant(candidate, learning_rate=candidate.learning_rate * 1.25),
        _candidate_variant(candidate, l2_penalty=candidate.l2_penalty / 3.0),
        _candidate_variant(candidate, l2_penalty=candidate.l2_penalty * 3.0),
        _candidate_variant(candidate, signal_threshold=candidate.signal_threshold - 0.01),
        _candidate_variant(candidate, signal_threshold=candidate.signal_threshold + 0.01),
        _candidate_variant(candidate, risk_per_trade=candidate.risk_per_trade * 0.50),
        _candidate_variant(candidate, risk_per_trade=candidate.risk_per_trade * 1.25),
        _candidate_variant(candidate, stop_loss_pct=candidate.stop_loss_pct * 0.85),
        _candidate_variant(candidate, take_profit_pct=candidate.take_profit_pct * 1.10),
    ]


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
    ensemble_seeds: tuple[int, ...] | None = payload.get("ensemble_seeds")
    compute_backend = str(payload.get("compute_backend") or "cpu")
    batch_size = int(payload.get("batch_size") or 8192)
    score_batch_size = int(payload.get("score_batch_size") or batch_size)
    include_full_fit_fallback = bool(payload.get("include_full_fit_fallback", True))

    objective = get_objective(objective_name)
    training = _default_training(objective)
    model_train_rows, selection_rows = _walk_forward_split(rows_train)
    fit_rows, calibration_rows = _calibration_split(model_train_rows)
    model, report = train_advanced(
        fit_rows,
        feature_cfg,
        epochs=candidate.epochs,
        learning_rate=candidate.learning_rate,
        l2_penalty=candidate.l2_penalty,
        seed=candidate.seed,
        validation_rows=calibration_rows,
        early_stopping_rounds=max(10, min(40, int(candidate.epochs) // 8)) if calibration_rows else None,
        ensemble_seeds=ensemble_seeds,
        compute_backend=compute_backend,
        batch_size=batch_size,
    )
    strategy = _strategy_for_candidate(base_strategy, candidate, training)
    threshold_score = None
    if calibration_rows:
        probability_calibration = calibrate_probability_temperature(calibration_rows, model)
        if probability_calibration.status != "fail":
            model.probability_temperature = float(probability_calibration.temperature)
            model.probability_calibration_size = int(probability_calibration.rows)
            model.probability_log_loss_before = float(probability_calibration.log_loss_before)
            model.probability_log_loss_after = float(probability_calibration.log_loss_after)
            model.probability_brier_before = float(probability_calibration.brier_before)
            model.probability_brier_after = float(probability_calibration.brier_after)
            model.probability_ece_before = float(probability_calibration.expected_calibration_error_before)
            model.probability_ece_after = float(probability_calibration.expected_calibration_error_after)
    if training.calibrate_threshold and calibration_rows:
        threshold, threshold_source, threshold_score = _calibrate_candidate_threshold(
            model,
            calibration_rows,
            strategy,
            market_type=market_type,
            starting_cash=starting_cash,
            compute_backend=compute_backend,
            score_batch_size=score_batch_size,
        )
        model.decision_threshold = float(threshold)
        model.threshold_source = threshold_source
        model.threshold_calibration_score = threshold_score
        model.threshold_calibration_trades = len(calibration_rows)
        model.calibration_size = len(calibration_rows)
    else:
        model.decision_threshold = float(strategy.signal_threshold)
        model.threshold_source = "strategy"
    selection_threshold = _refine_threshold_on_selection_rows(
        model,
        selection_rows,
        strategy,
        market_type=market_type,
        starting_cash=starting_cash,
        compute_backend=compute_backend,
        score_batch_size=score_batch_size,
    )
    if selection_threshold is not None:
        threshold, threshold_source, threshold_score = selection_threshold
        model.decision_threshold = float(threshold)
        model.threshold_source = threshold_source
        model.threshold_calibration_score = threshold_score
        model.threshold_calibration_trades = len(selection_rows)
    _attach_strategy_overrides(model, strategy)
    model.validation_size = len(rows_eval)
    selection_result = run_backtest(
        selection_rows,
        model,
        strategy,
        starting_cash=starting_cash,
        market_type=market_type,
        compute_backend=compute_backend,
        score_batch_size=score_batch_size,
    )
    selection_score = objective.score(selection_result) if objective.accepts(selection_result) else float("-inf")
    holdout_result = run_backtest(
        rows_eval,
        model,
        strategy,
        starting_cash=starting_cash,
        market_type=market_type,
        compute_backend=compute_backend,
        score_batch_size=score_batch_size,
    )
    validation_score = objective.score(holdout_result) if objective.accepts(holdout_result) else float("-inf")
    full_result = run_backtest(
        rows_train + rows_eval,
        model,
        strategy,
        starting_cash=starting_cash,
        market_type=market_type,
        compute_backend=compute_backend,
        score_batch_size=score_batch_size,
    )
    full_sample_score = objective.score(full_result) if objective.accepts(full_result) else float("-inf")
    score = min(validation_score, full_sample_score)
    selected_calibration_rows = len(calibration_rows)

    if calibration_rows and include_full_fit_fallback:
        fallback_model, fallback_report = train_advanced(
            model_train_rows,
            feature_cfg,
            epochs=candidate.epochs,
            learning_rate=candidate.learning_rate,
            l2_penalty=candidate.l2_penalty,
            seed=candidate.seed,
            ensemble_seeds=ensemble_seeds,
            compute_backend=compute_backend,
            batch_size=batch_size,
        )
        fallback_model.decision_threshold = float(strategy.signal_threshold)
        fallback_model.threshold_source = "strategy_full_fit"
        fallback_threshold = _refine_threshold_on_selection_rows(
            fallback_model,
            selection_rows,
            strategy,
            market_type=market_type,
            starting_cash=starting_cash,
            compute_backend=compute_backend,
            score_batch_size=score_batch_size,
        )
        fallback_threshold_score = None
        if fallback_threshold is not None:
            threshold, threshold_source, fallback_threshold_score = fallback_threshold
            fallback_model.decision_threshold = float(threshold)
            fallback_model.threshold_source = threshold_source
            fallback_model.threshold_calibration_score = fallback_threshold_score
            fallback_model.threshold_calibration_trades = len(selection_rows)
        _attach_strategy_overrides(fallback_model, strategy)
        fallback_model.validation_size = len(rows_eval)
        fallback_selection_result = run_backtest(
            selection_rows,
            fallback_model,
            strategy,
            starting_cash=starting_cash,
            market_type=market_type,
            compute_backend=compute_backend,
            score_batch_size=score_batch_size,
        )
        fallback_selection_score = (
            objective.score(fallback_selection_result)
            if objective.accepts(fallback_selection_result)
            else float("-inf")
        )
        fallback_holdout_result = run_backtest(
            rows_eval,
            fallback_model,
            strategy,
            starting_cash=starting_cash,
            market_type=market_type,
            compute_backend=compute_backend,
            score_batch_size=score_batch_size,
        )
        fallback_validation_score = (
            objective.score(fallback_holdout_result)
            if objective.accepts(fallback_holdout_result)
            else float("-inf")
        )
        fallback_full_result = run_backtest(
            rows_train + rows_eval,
            fallback_model,
            strategy,
            starting_cash=starting_cash,
            market_type=market_type,
            compute_backend=compute_backend,
            score_batch_size=score_batch_size,
        )
        fallback_full_score = (
            objective.score(fallback_full_result)
            if objective.accepts(fallback_full_result)
            else float("-inf")
        )
        fallback_score = min(fallback_validation_score, fallback_full_score)
        if fallback_score > score + 1e-12:
            score = fallback_score
            model = fallback_model
            report = fallback_report
            threshold_score = fallback_threshold_score
            selected_calibration_rows = 0
            selection_score = fallback_selection_score
            validation_score = fallback_validation_score
            full_sample_score = fallback_full_score

    return {
        "score": float(score),
        "candidate": candidate,
        "strategy": strategy,
        "model": model,
        "row_count": report.row_count,
        "positive_rate": report.positive_rate,
        "threshold": model.decision_threshold,
        "threshold_source": model.threshold_source,
        "threshold_score": threshold_score,
        "calibration_rows": selected_calibration_rows,
        "validation_rows": len(rows_eval),
        "selection_score": float(selection_score),
        "validation_score": float(validation_score),
        "full_sample_score": float(full_sample_score),
        "ensemble_refined": bool(ensemble_seeds),
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
    compute_backend: str | None = None,
    batch_size: int = 8192,
    score_batch_size: int | None = None,
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
    effective_score_batch_size = int(score_batch_size if score_batch_size is not None else batch_size)

    if runner is not None:
        results: list[dict[str, Any]] = []
        for candidate in candidates:
            score, strategy, model, row_count, positive_rate = runner(
                objective, candidate, rows, base_strategy, feature_cfg,
                market_type, starting_cash,
            )
            _attach_strategy_overrides(model, strategy)
            results.append({
                "score": float(score),
                "candidate": candidate,
                "strategy": strategy,
                "model": model,
                "row_count": row_count,
                "positive_rate": positive_rate,
                "threshold": getattr(model, "decision_threshold", None),
                "threshold_source": getattr(model, "threshold_source", None),
                "threshold_score": getattr(model, "threshold_calibration_score", None),
                "calibration_rows": int(getattr(model, "calibration_size", 0)),
                "validation_rows": len(eval_rows),
                "validation_score": float(score),
                "full_sample_score": None,
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
                "compute_backend": compute_backend or "cpu",
                "batch_size": batch_size,
                "score_batch_size": effective_score_batch_size,
                "include_full_fit_fallback": True,
            }
            for candidate in candidates
        ]
        workers = _resolve_workers(max_workers, len(payloads))
        if str(compute_backend or "cpu").strip().lower() not in {"", "cpu"}:
            workers = 1
        if workers <= 1:
            results = [_evaluate_candidate(p) for p in payloads]
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                results = list(pool.map(_evaluate_candidate, payloads))

    results.sort(key=lambda entry: entry["score"], reverse=True)
    best = results[0]
    ensemble_refinement_candidates = 0
    local_refinement_candidates = 0
    local_results: list[dict[str, Any]] = []
    if runner is None:
        for local_candidate in _local_refinement_candidates(best["candidate"]):
            local_refinement_candidates += 1
            local_result = _evaluate_candidate({
                "candidate": local_candidate,
                "rows_train": train_rows,
                "rows_eval": eval_rows,
                "feature_cfg": feature_cfg,
                "base_strategy": base_strategy,
                "objective": objective.name,
                "market_type": market_type,
                "starting_cash": starting_cash,
                "compute_backend": compute_backend or "cpu",
                "batch_size": batch_size,
                "score_batch_size": effective_score_batch_size,
                "include_full_fit_fallback": True,
            })
            local_results.append(local_result)
        ranked_pool = sorted([*results, *local_results], key=lambda entry: entry["score"], reverse=True)
        best = ranked_pool[0]
        for base_result in ranked_pool[:_ENSEMBLE_REFINEMENT_CANDIDATES]:
            ensemble_refinement_candidates += 1
            ensemble_result = _evaluate_candidate({
                "candidate": base_result["candidate"],
                "rows_train": train_rows,
                "rows_eval": eval_rows,
                "feature_cfg": feature_cfg,
                "base_strategy": base_strategy,
                "objective": objective.name,
                "market_type": market_type,
                "starting_cash": starting_cash,
                "ensemble_seeds": _ensemble_seed_pack(int(base_result["candidate"].seed)),
                "compute_backend": compute_backend or "cpu",
                "batch_size": batch_size,
                "score_batch_size": effective_score_batch_size,
                "include_full_fit_fallback": True,
            })
            if ensemble_result["score"] > best["score"] + 1e-12:
                best = ensemble_result

    rejected = sum(1 for entry in results if entry["score"] == float("-inf"))
    if not math.isfinite(float(best["score"])):
        raise ValueError(f"All {objective.name} training candidates were rejected by objective gates")
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
        decision_threshold=(
            float(best["threshold"])
            if best.get("threshold") is not None
            else None
        ),
        threshold_source=(
            str(best["threshold_source"])
            if best.get("threshold_source") is not None
            else None
        ),
        calibration_score=(
            float(best["threshold_score"])
            if best.get("threshold_score") is not None
            else None
        ),
        calibration_rows=int(best.get("calibration_rows", 0)),
        validation_rows=int(best.get("validation_rows", 0)),
        validation_score=(
            float(best["validation_score"])
            if best.get("validation_score") is not None
            else None
        ),
        full_sample_score=(
            float(best["full_sample_score"])
            if best.get("full_sample_score") is not None
            else None
        ),
        ensemble_refined=bool(best.get("ensemble_refined", False)),
        ensemble_refinement_candidates=ensemble_refinement_candidates,
        local_refinement_candidates=local_refinement_candidates,
        training_backend_requested=str(getattr(best["model"], "training_backend_requested", "cpu")),
        training_backend_kind=str(getattr(best["model"], "training_backend_kind", "cpu")),
        training_backend_device=str(getattr(best["model"], "training_backend_device", "cpu")),
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
    compute_backend: str | None = None,
    batch_size: int = 8192,
    score_batch_size: int | None = None,
) -> SuiteReport:
    """Train one model per objective and persist a suite summary."""

    names = tuple(objectives) if objectives else available_objectives()
    specs = [get_objective(name) for name in names]
    outcomes: list[ObjectiveOutcome] = []
    total_rows = 0
    for spec in specs:
        # Pass optional args only when the caller asked for them so legacy test
        # doubles that monkey-patch ``train_for_objective`` keep working without
        # having to extend their signature.
        extra: dict[str, object] = {}
        if max_workers is not None:
            extra["max_workers"] = max_workers
        if compute_backend is not None:
            extra["compute_backend"] = compute_backend
        if int(batch_size) != 8192:
            extra["batch_size"] = int(batch_size)
        if score_batch_size is not None:
            extra["score_batch_size"] = int(score_batch_size)
        outcome = train_for_objective(
            candles, base_strategy, spec,
            output_dir=output_dir,
            market_type=market_type,
            starting_cash=starting_cash,
            **extra,
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
    write_json_atomic(summary_path, report.asdict(), indent=2, sort_keys=True)
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


def rank_report(
    candidates_with_results: Sequence[tuple[dict[str, object], BacktestResult]],
    objective: str | ObjectiveSpec = "default",
) -> list[dict[str, object]]:
    """Rank precomputed backtest results under an objective.

    This is the public convenience wrapper for callers that already evaluated
    candidates and only need the same accept/reject annotations used by the
    training suite.
    """

    spec = get_objective(objective) if isinstance(objective, str) else objective
    return rank_candidates(list(candidates_with_results), spec)


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
