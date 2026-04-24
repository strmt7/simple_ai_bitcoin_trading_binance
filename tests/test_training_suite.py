"""Comprehensive unit tests for the multi-objective training suite."""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Sequence

import pytest

from simple_ai_bitcoin_trading_binance import training_suite
from simple_ai_bitcoin_trading_binance.advanced_model import AdvancedFeatureConfig
from simple_ai_bitcoin_trading_binance.api import Candle
from simple_ai_bitcoin_trading_binance.backtest import BacktestResult
from simple_ai_bitcoin_trading_binance.features import ModelRow
from simple_ai_bitcoin_trading_binance.model import TrainedModel
from simple_ai_bitcoin_trading_binance.objective import (
    ObjectiveSpec,
    ObjectiveTraining,
    get_objective,
)
from simple_ai_bitcoin_trading_binance.training_suite import (
    CandidateParams,
    ObjectiveOutcome,
    SuiteReport,
    _candidate_grid,
    _default_training,
    _strategy_for_candidate,
    _walk_forward_split,
    describe_candidate_grid,
    preview_candidates,
    rank_report,
    run_training_suite,
    train_for_objective,
)
from simple_ai_bitcoin_trading_binance.types import StrategyConfig


# ----- helpers --------------------------------------------------------------


def _synthetic_candles(n: int = 500, base: float = 100.0) -> list[Candle]:
    candles: list[Candle] = []
    price = base
    for i in range(n):
        open_ = price
        close = price * (1.0 + 0.0005 * math.sin(i / 5.0) + 0.0002)
        high = max(open_, close) * 1.002
        low = min(open_, close) * 0.998
        candles.append(Candle(
            open_time=i * 60_000,
            open=open_,
            high=high,
            low=low,
            close=close,
            volume=1.0 + (i % 5) * 0.1,
            close_time=i * 60_000 + 60_000,
        ))
        price = close
    return candles


def _fake_trained_model(dim: int = 4) -> TrainedModel:
    return TrainedModel(
        weights=[0.0] * dim,
        bias=0.0,
        feature_dim=dim,
        epochs=1,
        feature_means=[0.0] * dim,
        feature_stds=[1.0] * dim,
    )


def _rows(n: int) -> list[ModelRow]:
    return [ModelRow(timestamp=i, close=1.0, features=(0.1, 0.2), label=i % 2) for i in range(n)]


# ----- CandidateParams ------------------------------------------------------


def test_candidate_params_asdict_keys() -> None:
    params = CandidateParams(
        epochs=10, learning_rate=0.01, l2_penalty=0.001,
        signal_threshold=0.6, stop_loss_pct=0.02, take_profit_pct=0.03,
        risk_per_trade=0.01,
    )
    d = params.asdict()
    expected_keys = {
        "epochs", "learning_rate", "l2_penalty",
        "signal_threshold", "stop_loss_pct", "take_profit_pct", "risk_per_trade",
    }
    assert set(d.keys()) == expected_keys


# ----- _candidate_grid ------------------------------------------------------


def test_candidate_grid_returns_unique_deduped_list() -> None:
    training = get_objective("default").training
    grid = _candidate_grid(training)
    assert len(grid) > 0
    # dedupe check: no two entries share identical tuple of values
    tuples = [tuple(c.asdict().values()) for c in grid]
    assert len(tuples) == len(set(tuples))
    # the grid should include variation in epochs/lr/threshold
    epoch_set = {c.epochs for c in grid}
    lr_set = {c.learning_rate for c in grid}
    threshold_set = {c.signal_threshold for c in grid}
    assert len(epoch_set) >= 2
    assert len(lr_set) >= 2
    assert len(threshold_set) >= 2


def test_candidate_grid_dedupes_colliding_entries() -> None:
    """Force a collision by zeroing the learning rate so the two lr options collapse."""

    colliding = ObjectiveTraining(
        epochs=200,
        learning_rate=0.0,  # * 0.6 and * 1.0 both equal 0.0 -> dedup
        l2_penalty=1e-3,
        signal_threshold=0.5,
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
    grid = _candidate_grid(colliding)
    # All candidates distinct after dedup
    tuples = [tuple(c.asdict().values()) for c in grid]
    assert len(tuples) == len(set(tuples))
    # With lr collapsing, we expect fewer entries than the raw Cartesian product
    # (2*1*2*3*2*2*2 = 96) instead of the full 192.
    assert len(grid) <= 96


# ----- _walk_forward_split --------------------------------------------------


def test_walk_forward_split_small_rows_returns_copies() -> None:
    rows = _rows(5)
    train, test = _walk_forward_split(rows)
    assert train == rows
    assert test == rows
    # independent lists
    assert train is not rows


def test_walk_forward_split_large_rows_splits_properly() -> None:
    rows = _rows(100)
    train, test = _walk_forward_split(rows, eval_ratio=0.25)
    assert len(train) + len(test) == len(rows)
    assert len(test) >= 5


# ----- _default_training fallback ------------------------------------------


def test_default_training_with_missing_metadata() -> None:
    # ObjectiveSpec requires a scorer; provide a no-op lambda
    spec = ObjectiveSpec(
        name="custom",
        label="Custom",
        summary="s",
        long_description="d",
        scorer=lambda _r: 0.0,
        training=None,
    )
    training = _default_training(spec)
    assert isinstance(training, ObjectiveTraining)
    assert training.epochs == 200


def test_default_training_uses_metadata_when_present() -> None:
    spec = get_objective("default")
    training = _default_training(spec)
    assert training is spec.training


# ----- _strategy_for_candidate ---------------------------------------------


def test_strategy_for_candidate_applies_overlays() -> None:
    base = StrategyConfig()
    params = CandidateParams(
        epochs=77, learning_rate=0.05, l2_penalty=0.002,
        signal_threshold=0.7, stop_loss_pct=0.05, take_profit_pct=0.06,
        risk_per_trade=0.02,
    )
    training = get_objective("default").training
    strat = _strategy_for_candidate(base, params, training)
    assert isinstance(strat, StrategyConfig)
    assert strat.training_epochs == 77
    assert strat.signal_threshold == pytest.approx(0.7)
    assert strat.stop_loss_pct == pytest.approx(0.05)
    assert strat.take_profit_pct == pytest.approx(0.06)
    assert strat.risk_per_trade == pytest.approx(0.02)
    assert strat.leverage == training.leverage
    assert strat.cooldown_minutes == training.cooldown_minutes


# ----- train_for_objective: happy path with fake runner --------------------


def _make_result(**overrides) -> BacktestResult:
    defaults = dict(
        starting_cash=1000.0, ending_cash=1050.0, realized_pnl=50.0,
        win_rate=0.6, trades=5, max_drawdown=0.02, closed_trades=5,
        gross_exposure=100.0, total_fees=0.1, stopped_by_drawdown=False,
        max_exposure=100.0, trades_per_day_cap_hit=0,
    )
    defaults.update(overrides)
    return BacktestResult(**defaults)


def test_train_for_objective_happy_with_fake_runner(tmp_path: Path) -> None:
    candles = _synthetic_candles(n=200)
    strategy = StrategyConfig()
    objective = get_objective("default")

    scores_cycle = iter([0.1, 0.9, float("-inf"), 0.5])

    def runner(_obj, candidate, rows, base, feat_cfg, market, cash):
        try:
            score = next(scores_cycle)
        except StopIteration:
            score = 0.0
        return score, base, _fake_trained_model(feat_cfg.polynomial_top_features), 42, 0.5

    outcome = train_for_objective(
        candles, strategy, objective,
        output_dir=tmp_path,
        market_type="spot",
        starting_cash=1000.0,
        runner=runner,
    )
    assert isinstance(outcome, ObjectiveOutcome)
    model_file = tmp_path / f"model_{objective.name}.json"
    assert model_file.exists()
    # outcome.asdict conversion covered
    assert outcome.asdict()["model_path"] == str(model_file)
    # rejected counts entries scored as -inf
    assert outcome.rejected_candidates >= 1


def test_train_for_objective_insufficient_candles(tmp_path: Path) -> None:
    objective = get_objective("default")
    with pytest.raises(ValueError, match="Insufficient candles"):
        train_for_objective(
            [],
            StrategyConfig(),
            objective,
            output_dir=tmp_path,
            market_type="spot",
            starting_cash=1000.0,
            runner=lambda *a, **k: (0.0, StrategyConfig(), _fake_trained_model(), 0, 0.0),
        )


def test_train_for_objective_empty_candidate_grid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    objective = get_objective("default")
    candles = _synthetic_candles(n=200)

    # Monkeypatch the module-level _candidate_grid to return []
    monkeypatch.setattr(training_suite, "_candidate_grid", lambda training: [])

    def runner(*a, **k):  # pragma: no cover - should not be called
        raise AssertionError("runner should not be invoked for empty grid")

    with pytest.raises(ValueError, match="Candidate grid produced zero"):
        train_for_objective(
            candles, StrategyConfig(), objective,
            output_dir=tmp_path,
            market_type="spot",
            starting_cash=1000.0,
            runner=runner,
        )


# ----- run_training_suite --------------------------------------------------


def test_run_training_suite_with_explicit_objectives(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    candles = _synthetic_candles(n=100)
    strat = StrategyConfig()

    def fake_train_for_objective(
        candles, base_strategy, objective, *,
        output_dir, market_type, starting_cash, runner=None,
    ):
        # write a placeholder model file so layout matches production
        model_path = output_dir / f"model_{objective.name}.json"
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path.write_text("{}", encoding="utf-8")
        return ObjectiveOutcome(
            objective=objective.name,
            model_path=model_path,
            feature_dim=4,
            feature_signature="sig",
            best_score=0.5,
            best_params={"epochs": 1},
            explored_candidates=1,
            rejected_candidates=0,
            epochs=1,
            learning_rate=0.01,
            l2_penalty=0.0,
            row_count=50,
            positive_rate=0.5,
        )

    monkeypatch.setattr(training_suite, "train_for_objective", fake_train_for_objective)

    report = run_training_suite(
        candles, strat,
        objectives=["default", "conservative"],
        market_type="spot",
        starting_cash=1000.0,
        output_dir=tmp_path,
    )
    assert isinstance(report, SuiteReport)
    assert {o.objective for o in report.outcomes} == {"default", "conservative"}
    summary = tmp_path / "training_suite_summary.json"
    assert summary.exists()
    data = json.loads(summary.read_text(encoding="utf-8"))
    assert data["total_candles"] == len(candles)
    assert set(data["objectives_run"]) == {"default", "conservative"}


def test_run_training_suite_default_objectives_and_summary_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    candles = _synthetic_candles(n=80)
    strat = StrategyConfig()

    def fake_train_for_objective(
        candles, base_strategy, objective, *,
        output_dir, market_type, starting_cash, runner=None,
    ):
        model_path = output_dir / f"model_{objective.name}.json"
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path.write_text("{}", encoding="utf-8")
        return ObjectiveOutcome(
            objective=objective.name,
            model_path=model_path,
            feature_dim=3,
            feature_signature="sig",
            best_score=1.0,
            best_params={"epochs": 1},
            explored_candidates=1,
            rejected_candidates=0,
            epochs=1,
            learning_rate=0.01,
            l2_penalty=0.0,
            row_count=60,
            positive_rate=0.5,
        )

    monkeypatch.setattr(training_suite, "train_for_objective", fake_train_for_objective)

    summary = tmp_path / "custom_summary.json"
    report = run_training_suite(
        candles, strat,
        objectives=None,
        output_dir=tmp_path,
        summary_path=summary,
    )
    assert summary.exists()
    # All three default objectives should be covered
    assert len(report.outcomes) >= 3
    payload = json.loads(summary.read_text(encoding="utf-8"))
    assert payload["summary_path"] == str(summary)


# ----- describe_candidate_grid + preview_candidates ------------------------


def test_describe_candidate_grid_keys() -> None:
    grid = describe_candidate_grid(get_objective("default"))
    assert len(grid) > 0
    for item in grid:
        assert "epochs" in item and "learning_rate" in item


def test_preview_candidates_shape() -> None:
    rows = preview_candidates()
    assert len(rows) >= 3
    for row in rows:
        assert "objective" in row
        assert "candidates" in row
        assert "first_candidate" in row


# ----- rank_report placeholder ---------------------------------------------


def test_rank_report_returns_empty_list() -> None:
    assert rank_report([("whatever", "any")]) == []


# ----- real-runner smoke test for train_for_objective ----------------------


def test_train_for_objective_real_runner_small_dataset(tmp_path: Path) -> None:
    """Exercises the real ``_run_candidate`` path end-to-end with a small dataset.

    We shrink the grid by monkey-patching ``_candidate_grid`` to a single
    lightweight candidate so the test stays fast but still traverses
    ``make_advanced_rows`` + ``train_advanced`` + ``run_backtest``.
    """

    candles = _synthetic_candles(n=260)
    objective = get_objective("default")

    single_candidate = CandidateParams(
        epochs=2,
        learning_rate=0.05,
        l2_penalty=1e-4,
        signal_threshold=0.55,
        stop_loss_pct=0.02,
        take_profit_pct=0.03,
        risk_per_trade=0.01,
    )

    # Swap in a tiny grid (monkeypatch via direct assignment on the module).
    original = training_suite._candidate_grid
    training_suite._candidate_grid = lambda _t: [single_candidate]
    try:
        outcome = train_for_objective(
            candles, StrategyConfig(), objective,
            output_dir=tmp_path,
            market_type="spot",
            starting_cash=1000.0,
        )
    finally:
        training_suite._candidate_grid = original

    assert (tmp_path / f"model_{objective.name}.json").exists()
    assert outcome.feature_dim > 0
    assert outcome.row_count > 0
