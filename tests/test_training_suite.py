"""Comprehensive unit tests for the multi-objective training suite."""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from simple_ai_bitcoin_trading_binance import training_suite
from simple_ai_bitcoin_trading_binance.advanced_model import default_config_for
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
    _calibration_split,
    _candidate_grid,
    _default_training,
    _ensemble_seed_pack,
    _calibrate_candidate_threshold,
    _evaluate_candidate,
    _local_refinement_candidates,
    _refine_threshold_on_selection_rows,
    _strategy_for_candidate,
    _threshold_guard,
    _threshold_values,
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
        risk_per_trade=0.01, confidence_beta=0.9, seed=11,
    )
    d = params.asdict()
    expected_keys = {
        "epochs", "learning_rate", "l2_penalty",
        "signal_threshold", "stop_loss_pct", "take_profit_pct", "risk_per_trade",
        "confidence_beta", "seed",
    }
    assert set(d.keys()) == expected_keys


# ----- _candidate_grid ------------------------------------------------------


def test_candidate_grid_returns_unique_deduped_list() -> None:
    training = get_objective("default").training
    grid = _candidate_grid(training)
    assert len(grid) == 72
    # dedupe check: no two entries share identical tuple of values
    tuples = [tuple(c.asdict().values()) for c in grid]
    assert len(tuples) == len(set(tuples))
    # the grid should include variation in epochs/lr/threshold
    epoch_set = {c.epochs for c in grid}
    lr_set = {c.learning_rate for c in grid}
    l2_set = {c.l2_penalty for c in grid}
    threshold_set = {c.signal_threshold for c in grid}
    confidence_set = {c.confidence_beta for c in grid}
    seed_set = {c.seed for c in grid}
    assert len(epoch_set) >= 2
    assert len(lr_set) >= 2
    assert len(l2_set) >= 2
    assert len(threshold_set) >= 2
    assert confidence_set == {0.85}
    assert seed_set == {7}


def test_candidate_grid_dedupes_colliding_entries() -> None:
    """Force a collision by zeroing the learning rate so the two lr options collapse."""

    colliding = ObjectiveTraining(
        epochs=200,
        learning_rate=0.0,  # both learning-rate options equal 0.0 -> dedup
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
    assert len(grid) == 36


# ----- calibration helpers --------------------------------------------------


def test_calibration_split_small_and_large_rows() -> None:
    train_small, calibration_small = _calibration_split(_rows(12))
    assert len(train_small) == 12
    assert calibration_small == []

    train_large, calibration_large = _calibration_split(_rows(100), ratio=0.2)
    assert len(train_large) == 80
    assert len(calibration_large) == 20
    assert train_large[-1].timestamp == 79
    assert calibration_large[0].timestamp == 80


def test_threshold_guard_accepts_stable_or_sharper_candidates() -> None:
    class Report:
        def __init__(self, *, accuracy: float, f1: float, precision: float):
            self.accuracy = accuracy
            self.f1 = f1
            self.precision = precision

    baseline = Report(accuracy=0.60, f1=0.50, precision=0.45)
    stable = Report(accuracy=0.58, f1=0.46, precision=0.20)
    sharper = Report(accuracy=0.63, f1=0.10, precision=0.44)
    rejected = Report(accuracy=0.50, f1=0.30, precision=0.20)
    zero_hit = SimpleNamespace(accuracy=0.90, f1=0.0, precision=0.0, true_positive=0, false_negative=3)

    assert _threshold_guard(baseline, stable) is True
    assert _threshold_guard(baseline, sharper) is True
    assert _threshold_guard(baseline, rejected) is False
    assert _threshold_guard(baseline, zero_hit) is False


def test_threshold_values_clamps_single_step_and_expands_inverted_range() -> None:
    assert _threshold_values(0.2, 0.8, 1, 1.4) == [1.0]

    values = _threshold_values(0.7, 0.6, 3, 0.65)

    assert values == sorted(values)
    assert values[0] == 0.65
    assert values[-1] == 0.71


def test_calibrate_candidate_threshold_without_rows_uses_strategy_threshold() -> None:
    model = _fake_trained_model(2)
    strategy = StrategyConfig(signal_threshold=0.62)

    threshold, source, score = _calibrate_candidate_threshold(
        model,
        [],
        strategy,
        market_type="spot",
        starting_cash=1000.0,
    )

    assert threshold == pytest.approx(0.62)
    assert source == "strategy"
    assert score is None


def test_calibrate_candidate_threshold_accepts_profit_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    reports = [
        SimpleNamespace(accuracy=0.60, f1=0.50, precision=0.45),
        SimpleNamespace(accuracy=0.63, f1=0.46, precision=0.44),
    ]
    observed_calibration_kwargs: dict[str, object] = {}

    def fake_profit_calibration(*a, **k):
        observed_calibration_kwargs.update(k)
        return SimpleNamespace(
            accepted=True,
            threshold=0.72,
            best_threshold=0.72,
            score=4.25,
            realized_pnl=4.25,
            closed_trades=3,
        )

    monkeypatch.setattr(training_suite, "calibrate_threshold", lambda *a, **k: 0.41)
    monkeypatch.setattr(training_suite, "evaluate_classification", lambda *a, **k: reports.pop(0))
    monkeypatch.setattr(training_suite, "calibrate_threshold_for_backtest", fake_profit_calibration)

    threshold, source, score = _calibrate_candidate_threshold(
        _fake_trained_model(2),
        _rows(12),
        StrategyConfig(signal_threshold=0.60),
        market_type="spot",
        starting_cash=1000.0,
        compute_backend="directml",
        score_batch_size=64,
    )

    assert threshold == pytest.approx(0.72)
    assert source == "profit_backtest"
    assert score == pytest.approx(4.25)
    assert observed_calibration_kwargs["compute_backend"] == "directml"
    assert observed_calibration_kwargs["score_batch_size"] == 64


def test_calibrate_candidate_threshold_rejects_weak_profit_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    reports = [
        SimpleNamespace(accuracy=0.60, f1=0.50, precision=0.45),
        SimpleNamespace(accuracy=0.50, f1=0.20, precision=0.20),
        SimpleNamespace(accuracy=0.52, f1=0.25, precision=0.22),
    ]

    monkeypatch.setattr(training_suite, "calibrate_threshold", lambda *a, **k: 0.43)
    monkeypatch.setattr(training_suite, "evaluate_classification", lambda *a, **k: reports.pop(0))
    monkeypatch.setattr(
        training_suite,
        "calibrate_threshold_for_backtest",
        lambda *a, **k: SimpleNamespace(
            accepted=True,
            threshold=0.81,
            best_threshold=0.81,
            score=1.5,
            baseline_score=-0.5,
            realized_pnl=1.5,
            closed_trades=2,
        ),
    )

    threshold, source, score = _calibrate_candidate_threshold(
        _fake_trained_model(2),
        _rows(12),
        StrategyConfig(signal_threshold=0.60),
        market_type="spot",
        starting_cash=1000.0,
    )

    assert threshold == pytest.approx(0.60)
    assert source == "strategy"
    assert score == pytest.approx(-0.5)


def test_calibrate_candidate_threshold_rejects_unprofitable_profit_report(monkeypatch: pytest.MonkeyPatch) -> None:
    reports = [
        SimpleNamespace(accuracy=0.60, f1=0.50, precision=0.45),
        SimpleNamespace(accuracy=0.63, f1=0.46, precision=0.44),
        SimpleNamespace(accuracy=0.63, f1=0.46, precision=0.44),
    ]
    monkeypatch.setattr(training_suite, "calibrate_threshold", lambda *a, **k: 0.43)
    monkeypatch.setattr(training_suite, "evaluate_classification", lambda *a, **k: reports.pop(0))
    monkeypatch.setattr(
        training_suite,
        "calibrate_threshold_for_backtest",
        lambda *a, **k: SimpleNamespace(
            accepted=True,
            threshold=0.81,
            best_threshold=0.81,
            score=-50.0,
            baseline_score=-100.0,
            realized_pnl=0.0,
            closed_trades=0,
        ),
    )

    threshold, source, score = _calibrate_candidate_threshold(
        _fake_trained_model(2),
        _rows(12),
        StrategyConfig(signal_threshold=0.60),
        market_type="spot",
        starting_cash=1000.0,
    )

    assert threshold == pytest.approx(0.60)
    assert source == "strategy"
    assert score == pytest.approx(-100.0)


def test_calibrate_candidate_threshold_accepts_safe_classification_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reports = [
        SimpleNamespace(accuracy=0.60, f1=0.50, precision=0.45),
        SimpleNamespace(accuracy=0.58, f1=0.46, precision=0.44),
        SimpleNamespace(accuracy=0.58, f1=0.46, precision=0.44),
    ]

    monkeypatch.setattr(training_suite, "calibrate_threshold", lambda *a, **k: 0.43)
    monkeypatch.setattr(training_suite, "evaluate_classification", lambda *a, **k: reports.pop(0))
    monkeypatch.setattr(
        training_suite,
        "calibrate_threshold_for_backtest",
        lambda *a, **k: SimpleNamespace(
            accepted=False,
            threshold=0.60,
            best_threshold=0.43,
            score=0.2,
            baseline_score=0.2,
            realized_pnl=0.0,
            closed_trades=0,
        ),
    )

    threshold, source, score = _calibrate_candidate_threshold(
        _fake_trained_model(2),
        _rows(12),
        StrategyConfig(signal_threshold=0.60),
        market_type="spot",
        starting_cash=1000.0,
    )

    assert threshold == pytest.approx(0.43)
    assert source == "classification_f1"
    assert score == pytest.approx(0.2)


def test_refine_threshold_on_selection_rows_promotes_positive_profit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[tuple[str | None, int]] = []

    def fake_evaluate(*_args, **kwargs):
        threshold = float(kwargs.get("threshold", 0.5))
        if threshold <= 0.05:
            return SimpleNamespace(accuracy=0.60, f1=0.35, precision=0.30, true_positive=3, false_negative=2)
        return SimpleNamespace(accuracy=0.40, f1=0.20, precision=0.20, true_positive=1, false_negative=3)

    def fake_run_backtest(_rows, model, *_args, **_kwargs):
        observed.append((_kwargs.get("compute_backend"), _kwargs.get("score_batch_size")))
        if float(model.decision_threshold or 0.0) <= 0.05:
            return SimpleNamespace(realized_pnl=1.75, closed_trades=3, max_drawdown=0.01)
        return SimpleNamespace(realized_pnl=-0.05, closed_trades=2, max_drawdown=0.02)

    monkeypatch.setattr(training_suite, "evaluate_classification", fake_evaluate)
    monkeypatch.setattr(training_suite, "run_backtest", fake_run_backtest)

    refined = _refine_threshold_on_selection_rows(
        _fake_trained_model(2),
        _rows(40),
        StrategyConfig(signal_threshold=0.60),
        market_type="spot",
        starting_cash=1000.0,
        compute_backend="directml",
        score_batch_size=64,
    )

    assert refined is not None
    threshold, source, score = refined
    assert threshold == pytest.approx(0.05)
    assert source == "selection_profit_backtest"
    assert score == pytest.approx(1.75)
    assert observed
    assert all(item == ("directml", 64) for item in observed)


def test_refine_threshold_on_selection_rows_rejects_losing_profit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        training_suite,
        "run_backtest",
        lambda *_a, **_k: SimpleNamespace(realized_pnl=-0.01, closed_trades=2, max_drawdown=0.01),
    )

    refined = _refine_threshold_on_selection_rows(
        _fake_trained_model(2),
        _rows(40),
        StrategyConfig(signal_threshold=0.60),
        market_type="spot",
        starting_cash=1000.0,
    )

    assert refined is None


def test_refine_threshold_validation_wrapper_delegates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    def fake_refine(*_args, **kwargs):
        observed.update(kwargs)
        return 0.5, "selection_profit_backtest", 1.0

    monkeypatch.setattr(training_suite, "_refine_threshold_on_selection_rows", fake_refine)

    refined = training_suite._refine_threshold_on_validation_rows(
        _fake_trained_model(2),
        _rows(40),
        StrategyConfig(signal_threshold=0.60),
        market_type="spot",
        starting_cash=1000.0,
        compute_backend="directml",
        score_batch_size=12,
    )

    assert refined == (0.5, "selection_profit_backtest", 1.0)
    assert observed["compute_backend"] == "directml"
    assert observed["score_batch_size"] == 12


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
        risk_per_trade=0.02, confidence_beta=0.77,
    )
    training = get_objective("default").training
    strat = _strategy_for_candidate(base, params, training)
    assert isinstance(strat, StrategyConfig)
    assert strat.training_epochs == 77
    assert strat.signal_threshold == pytest.approx(0.7)
    assert strat.stop_loss_pct == pytest.approx(0.05)
    assert strat.take_profit_pct == pytest.approx(0.06)
    assert strat.risk_per_trade == pytest.approx(0.02)
    assert strat.confidence_beta == pytest.approx(0.77)
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


def test_evaluate_candidate_without_calibration_uses_strategy_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _fake_trained_model(2)

    def fake_train_advanced(rows, cfg, **kwargs):
        assert kwargs["seed"] == 11
        assert kwargs["validation_rows"] == []
        assert kwargs["early_stopping_rounds"] is None
        return model, SimpleNamespace(row_count=len(rows), positive_rate=0.5)

    monkeypatch.setattr(training_suite, "train_advanced", fake_train_advanced)
    monkeypatch.setattr(training_suite, "run_backtest", lambda *a, **k: _make_result())

    candidate = CandidateParams(
        epochs=4,
        learning_rate=0.01,
        l2_penalty=0.001,
        signal_threshold=0.64,
        stop_loss_pct=0.02,
        take_profit_pct=0.03,
        risk_per_trade=0.01,
        seed=11,
    )
    result = _evaluate_candidate({
        "candidate": candidate,
        "rows_train": _rows(20),
        "rows_eval": _rows(8),
        "feature_cfg": default_config_for("default", ()),
        "base_strategy": StrategyConfig(),
        "objective": "default",
        "market_type": "spot",
        "starting_cash": 1000.0,
    })

    assert result["threshold"] == pytest.approx(0.64)
    assert result["threshold_source"] == "strategy"
    assert result["calibration_rows"] == 0
    assert model.strategy_overrides["signal_threshold"] == pytest.approx(0.64)
    assert model.strategy_overrides["risk_per_trade"] == pytest.approx(0.01)
    assert model.strategy_overrides["take_profit_pct"] == pytest.approx(0.03)


def test_evaluate_candidate_failed_probability_calibration_continues(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"train": 0}

    def fake_train_advanced(rows, cfg, **kwargs):
        calls["train"] += 1
        if calls["train"] == 1:
            assert kwargs["seed"] == 11
            assert len(kwargs["validation_rows"]) > 0
            assert kwargs["early_stopping_rounds"] is not None
        else:
            assert kwargs["seed"] == 11
            assert "validation_rows" not in kwargs
        return _fake_trained_model(2), SimpleNamespace(row_count=len(rows), positive_rate=0.5)

    monkeypatch.setattr(training_suite, "train_advanced", fake_train_advanced)
    monkeypatch.setattr(
        training_suite,
        "calibrate_probability_temperature",
        lambda *a, **k: SimpleNamespace(status="fail"),
    )
    monkeypatch.setattr(
        training_suite,
        "_calibrate_candidate_threshold",
        lambda *a, **k: (0.67, "classification_f1", 2.5),
    )
    monkeypatch.setattr(training_suite, "run_backtest", lambda *a, **k: _make_result())

    candidate = CandidateParams(
        epochs=80,
        learning_rate=0.01,
        l2_penalty=0.001,
        signal_threshold=0.64,
        stop_loss_pct=0.02,
        take_profit_pct=0.03,
        risk_per_trade=0.01,
        seed=11,
    )
    result = _evaluate_candidate({
        "candidate": candidate,
        "rows_train": _rows(50),
        "rows_eval": _rows(10),
        "feature_cfg": default_config_for("default", ()),
        "base_strategy": StrategyConfig(),
        "objective": "default",
        "market_type": "spot",
        "starting_cash": 1000.0,
    })

    assert result["threshold"] == pytest.approx(0.67)
    assert result["threshold_source"] == "classification_f1"
    assert result["threshold_score"] == pytest.approx(2.5)
    assert result["calibration_rows"] > 0


def test_evaluate_candidate_selects_full_fit_when_calibration_regresses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"train": 0, "backtest": 0}

    def fake_train_advanced(rows, cfg, **kwargs):
        calls["train"] += 1
        assert kwargs["seed"] == 11
        model = _fake_trained_model(2)
        return model, SimpleNamespace(row_count=len(rows), positive_rate=0.5)

    def fake_run_backtest(*args, **kwargs):
        calls["backtest"] += 1
        if calls["backtest"] == 1:
            return _make_result(ending_cash=1000.0, realized_pnl=0.0, win_rate=0.2)
        if calls["backtest"] in {2, 3}:
            return _make_result(ending_cash=1010.0, realized_pnl=10.0, win_rate=0.4)
        return _make_result(ending_cash=1075.0, realized_pnl=75.0, win_rate=0.8)

    monkeypatch.setattr(training_suite, "train_advanced", fake_train_advanced)
    monkeypatch.setattr(
        training_suite,
        "calibrate_probability_temperature",
        lambda *a, **k: SimpleNamespace(status="fail"),
    )
    monkeypatch.setattr(
        training_suite,
        "_calibrate_candidate_threshold",
        lambda *a, **k: (0.67, "classification_f1", 2.5),
    )
    monkeypatch.setattr(training_suite, "run_backtest", fake_run_backtest)

    candidate = CandidateParams(
        epochs=80,
        learning_rate=0.01,
        l2_penalty=0.001,
        signal_threshold=0.64,
        stop_loss_pct=0.02,
        take_profit_pct=0.03,
        risk_per_trade=0.01,
        seed=11,
    )
    result = _evaluate_candidate({
        "candidate": candidate,
        "rows_train": _rows(50),
        "rows_eval": _rows(10),
        "feature_cfg": default_config_for("default", ()),
        "base_strategy": StrategyConfig(),
        "objective": "default",
        "market_type": "spot",
        "starting_cash": 1000.0,
    })

    assert calls["train"] == 2
    assert calls["backtest"] == 6
    assert result["threshold"] == pytest.approx(0.64)
    assert result["threshold_source"] == "strategy_full_fit"
    assert result["threshold_score"] is None
    assert result["calibration_rows"] == 0
    assert result["validation_score"] > 0.0
    assert result["full_sample_score"] > 0.0
    assert result["model"].strategy_overrides["signal_threshold"] == pytest.approx(0.64)


def test_evaluate_candidate_keeps_final_holdout_out_of_threshold_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    threshold_rows: list[list[int]] = []
    backtest_calls: list[tuple[list[int], str | None, int | None]] = []

    def fake_train_advanced(rows, cfg, **kwargs):
        return _fake_trained_model(2), SimpleNamespace(row_count=len(rows), positive_rate=0.5)

    def fake_refine(model, rows, strategy, **kwargs):
        del model, strategy, kwargs
        threshold_rows.append([row.timestamp for row in rows])
        return None

    def fake_run_backtest(rows, _model, _strategy, **kwargs):
        timestamps = [row.timestamp for row in rows]
        backtest_calls.append((
            timestamps,
            kwargs.get("compute_backend"),
            kwargs.get("score_batch_size"),
        ))
        if timestamps and min(timestamps) >= 1_000:
            return _make_result(realized_pnl=-10.0, closed_trades=5)
        return _make_result(realized_pnl=50.0, closed_trades=5)

    monkeypatch.setattr(training_suite, "train_advanced", fake_train_advanced)
    monkeypatch.setattr(
        training_suite,
        "calibrate_probability_temperature",
        lambda *a, **k: SimpleNamespace(status="fail"),
    )
    monkeypatch.setattr(
        training_suite,
        "_calibrate_candidate_threshold",
        lambda *a, **k: (0.64, "classification_f1", 1.0),
    )
    monkeypatch.setattr(training_suite, "_refine_threshold_on_selection_rows", fake_refine)
    monkeypatch.setattr(training_suite, "run_backtest", fake_run_backtest)

    candidate = CandidateParams(
        epochs=80,
        learning_rate=0.01,
        l2_penalty=0.001,
        signal_threshold=0.64,
        stop_loss_pct=0.02,
        take_profit_pct=0.03,
        risk_per_trade=0.01,
        seed=11,
    )
    result = _evaluate_candidate({
        "candidate": candidate,
        "rows_train": _rows(80),
        "rows_eval": [
            ModelRow(timestamp=1_000 + i, close=1.0, features=(0.1, 0.2), label=i % 2)
            for i in range(12)
        ],
        "feature_cfg": default_config_for("default", ()),
        "base_strategy": StrategyConfig(),
        "objective": "default",
        "market_type": "spot",
        "starting_cash": 1000.0,
        "compute_backend": "directml",
        "batch_size": 16,
        "score_batch_size": 32,
        "include_full_fit_fallback": False,
    })

    assert threshold_rows
    assert max(threshold_rows[0]) < 1_000
    assert result["score"] == float("-inf")
    assert result["validation_score"] == float("-inf")
    assert backtest_calls
    assert all(call[1:] == ("directml", 32) for call in backtest_calls)


def test_evaluate_candidate_can_skip_full_fit_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"train": 0}

    def fake_train_advanced(rows, cfg, **kwargs):
        calls["train"] += 1
        model = _fake_trained_model(2)
        return model, SimpleNamespace(row_count=len(rows), positive_rate=0.5)

    monkeypatch.setattr(training_suite, "train_advanced", fake_train_advanced)
    monkeypatch.setattr(
        training_suite,
        "calibrate_probability_temperature",
        lambda *a, **k: SimpleNamespace(status="fail"),
    )
    monkeypatch.setattr(
        training_suite,
        "_calibrate_candidate_threshold",
        lambda *a, **k: (0.67, "classification_f1", 2.5),
    )
    monkeypatch.setattr(training_suite, "run_backtest", lambda *a, **k: _make_result())

    candidate = CandidateParams(
        epochs=80,
        learning_rate=0.01,
        l2_penalty=0.001,
        signal_threshold=0.64,
        stop_loss_pct=0.02,
        take_profit_pct=0.03,
        risk_per_trade=0.01,
        seed=11,
    )
    result = _evaluate_candidate({
        "candidate": candidate,
        "rows_train": _rows(50),
        "rows_eval": _rows(10),
        "feature_cfg": default_config_for("default", ()),
        "base_strategy": StrategyConfig(),
        "objective": "default",
        "market_type": "spot",
        "starting_cash": 1000.0,
        "include_full_fit_fallback": False,
    })

    assert calls["train"] == 1
    assert result["calibration_rows"] > 0
    assert result["threshold_source"] == "classification_f1"


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
    assert outcome.validation_rows >= 0
    assert outcome.validation_score is not None


def test_train_for_objective_rejects_all_rejected_candidates(tmp_path: Path) -> None:
    candles = _synthetic_candles(n=200)
    strategy = StrategyConfig()
    objective = get_objective("default")

    def runner(_obj, candidate, rows, base, feat_cfg, market, cash):
        return float("-inf"), base, _fake_trained_model(feat_cfg.polynomial_top_features), 42, 0.5

    with pytest.raises(ValueError, match="All default training candidates were rejected"):
        train_for_objective(
            candles,
            strategy,
            objective,
            output_dir=tmp_path,
            market_type="spot",
            starting_cash=1000.0,
            runner=runner,
        )
    assert not (tmp_path / f"model_{objective.name}.json").exists()


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


def test_train_for_objective_promotes_better_seed_ensemble(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    objective = get_objective("default")
    candidate = CandidateParams(
        epochs=2,
        learning_rate=0.05,
        l2_penalty=1e-4,
        signal_threshold=0.55,
        stop_loss_pct=0.02,
        take_profit_pct=0.03,
        risk_per_trade=0.01,
        seed=11,
    )
    monkeypatch.setattr(training_suite, "_candidate_grid", lambda _training: [candidate])
    monkeypatch.setattr(training_suite, "_local_refinement_candidates", lambda _candidate: [])

    def fake_evaluate(payload):
        refined = bool(payload.get("ensemble_seeds"))
        return {
            "score": 2.0 if refined else 1.0,
            "candidate": payload["candidate"],
            "strategy": StrategyConfig(),
            "model": _fake_trained_model(),
            "row_count": 10,
            "positive_rate": 0.5,
            "threshold": 0.55,
            "threshold_source": "strategy",
            "threshold_score": None,
            "calibration_rows": 0,
            "validation_rows": 5,
            "validation_score": 2.0 if refined else 1.0,
            "full_sample_score": 2.0 if refined else 1.0,
            "ensemble_refined": refined,
        }

    monkeypatch.setattr(training_suite, "_evaluate_candidate", fake_evaluate)

    outcome = train_for_objective(
        _synthetic_candles(n=220),
        StrategyConfig(),
        objective,
        output_dir=tmp_path,
        market_type="spot",
        starting_cash=1000.0,
        max_workers=1,
    )

    assert _ensemble_seed_pack(11) == (11, 28, 48)
    assert outcome.best_score == 2.0
    assert outcome.ensemble_refined is True
    assert outcome.ensemble_refinement_candidates == 1


def test_train_for_objective_gpu_backend_forces_sequential_workers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    default_objective = get_objective("default")
    objective = ObjectiveSpec(
        name="default",
        label="Smoke",
        summary="Lenient smoke objective",
        long_description="Lenient real-runner smoke objective for tiny synthetic datasets.",
        scorer=lambda result: 1.0 + result.realized_pnl / max(1.0, result.starting_cash),
        min_closed_trades=0,
        min_realized_pnl=None,
        min_edge_vs_buy_hold=None,
        max_drawdown_rejection=1.0,
        training=default_objective.training,
    )
    candidate = CandidateParams(
        epochs=2,
        learning_rate=0.05,
        l2_penalty=1e-4,
        signal_threshold=0.55,
        stop_loss_pct=0.02,
        take_profit_pct=0.03,
        risk_per_trade=0.01,
        seed=7,
    )
    monkeypatch.setattr(training_suite, "_candidate_grid", lambda _training: [candidate])
    monkeypatch.setattr(training_suite, "_local_refinement_candidates", lambda _candidate: [])
    observed: list[tuple[str, int, int]] = []

    def fake_evaluate(payload):
        observed.append((payload["compute_backend"], payload["batch_size"], payload["score_batch_size"]))
        return {
            "score": 1.0,
            "candidate": payload["candidate"],
            "strategy": StrategyConfig(),
            "model": _fake_trained_model(),
            "row_count": 10,
            "positive_rate": 0.5,
            "threshold": 0.55,
            "threshold_source": "strategy",
            "threshold_score": None,
            "calibration_rows": 0,
            "validation_rows": 5,
            "validation_score": 1.0,
            "full_sample_score": 1.0,
            "ensemble_refined": False,
        }

    monkeypatch.setattr(training_suite, "_evaluate_candidate", fake_evaluate)
    outcome = train_for_objective(
        _synthetic_candles(n=220),
        StrategyConfig(),
        objective,
        output_dir=tmp_path,
        market_type="spot",
        starting_cash=1000.0,
        max_workers=4,
        compute_backend="directml",
        batch_size=64,
        score_batch_size=32,
    )
    assert outcome.training_backend_kind == "cpu"
    assert observed
    assert all(item == ("directml", 64, 32) for item in observed)


def test_train_for_objective_rejects_weaker_seed_ensemble(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    objective = get_objective("default")
    candidate = CandidateParams(
        epochs=2,
        learning_rate=0.05,
        l2_penalty=1e-4,
        signal_threshold=0.55,
        stop_loss_pct=0.02,
        take_profit_pct=0.03,
        risk_per_trade=0.01,
        seed=7,
    )
    monkeypatch.setattr(training_suite, "_candidate_grid", lambda _training: [candidate])
    monkeypatch.setattr(training_suite, "_local_refinement_candidates", lambda _candidate: [])

    def fake_evaluate(payload):
        refined = bool(payload.get("ensemble_seeds"))
        return {
            "score": 0.5 if refined else 1.5,
            "candidate": payload["candidate"],
            "strategy": StrategyConfig(),
            "model": _fake_trained_model(),
            "row_count": 10,
            "positive_rate": 0.5,
            "threshold": 0.55,
            "threshold_source": "strategy",
            "threshold_score": None,
            "calibration_rows": 0,
            "validation_rows": 5,
            "validation_score": 0.5 if refined else 1.5,
            "full_sample_score": 0.5 if refined else 1.5,
            "ensemble_refined": refined,
        }

    monkeypatch.setattr(training_suite, "_evaluate_candidate", fake_evaluate)

    outcome = train_for_objective(
        _synthetic_candles(n=220),
        StrategyConfig(),
        objective,
        output_dir=tmp_path,
        market_type="spot",
        starting_cash=1000.0,
        max_workers=1,
    )

    assert outcome.best_score == 1.5
    assert outcome.ensemble_refined is False
    assert outcome.ensemble_refinement_candidates == 1


def test_train_for_objective_promotes_better_local_refinement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    objective = get_objective("default")
    candidate = CandidateParams(
        epochs=2,
        learning_rate=0.05,
        l2_penalty=1e-4,
        signal_threshold=0.55,
        stop_loss_pct=0.02,
        take_profit_pct=0.03,
        risk_per_trade=0.01,
        seed=7,
    )
    monkeypatch.setattr(training_suite, "_candidate_grid", lambda _training: [candidate])

    def fake_evaluate(payload):
        refined = bool(payload.get("ensemble_seeds"))
        candidate_ = payload["candidate"]
        score = 2.0 if candidate_.risk_per_trade < candidate.risk_per_trade and not refined else 1.0
        return {
            "score": score,
            "candidate": candidate_,
            "strategy": StrategyConfig(),
            "model": _fake_trained_model(),
            "row_count": 10,
            "positive_rate": 0.5,
            "threshold": 0.55,
            "threshold_source": "strategy",
            "threshold_score": None,
            "calibration_rows": 0,
            "validation_rows": 5,
            "validation_score": score,
            "full_sample_score": score,
            "ensemble_refined": refined,
        }

    monkeypatch.setattr(training_suite, "_evaluate_candidate", fake_evaluate)

    outcome = train_for_objective(
        _synthetic_candles(n=220),
        StrategyConfig(),
        objective,
        output_dir=tmp_path,
        market_type="spot",
        starting_cash=1000.0,
        max_workers=1,
    )

    assert len(_local_refinement_candidates(candidate)) == 10
    assert outcome.best_score == 2.0
    assert outcome.best_params["risk_per_trade"] == pytest.approx(0.005)
    assert outcome.local_refinement_candidates == 10
    assert outcome.ensemble_refined is False


def test_train_for_objective_checks_top_candidates_for_seed_ensembles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    objective = get_objective("default")
    candidates = [
        CandidateParams(epochs=2, learning_rate=0.05, l2_penalty=1e-4,
                        signal_threshold=0.55, stop_loss_pct=0.02,
                        take_profit_pct=0.03, risk_per_trade=0.01, seed=7),
        CandidateParams(epochs=2, learning_rate=0.05, l2_penalty=1e-4,
                        signal_threshold=0.55, stop_loss_pct=0.02,
                        take_profit_pct=0.03, risk_per_trade=0.01, seed=11),
        CandidateParams(epochs=2, learning_rate=0.05, l2_penalty=1e-4,
                        signal_threshold=0.55, stop_loss_pct=0.02,
                        take_profit_pct=0.03, risk_per_trade=0.01, seed=13),
        CandidateParams(epochs=2, learning_rate=0.05, l2_penalty=1e-4,
                        signal_threshold=0.55, stop_loss_pct=0.02,
                        take_profit_pct=0.03, risk_per_trade=0.01, seed=17),
    ]
    monkeypatch.setattr(training_suite, "_candidate_grid", lambda _training: candidates)
    monkeypatch.setattr(training_suite, "_local_refinement_candidates", lambda _candidate: [])

    base_scores = {7: 3.0, 11: 2.0, 13: 1.0, 17: 0.5}
    ensemble_scores = {7: 2.5, 11: 4.0, 13: 1.1}
    evaluated: list[tuple[int, bool]] = []

    def fake_evaluate(payload):
        seed = int(payload["candidate"].seed)
        refined = bool(payload.get("ensemble_seeds"))
        evaluated.append((seed, refined))
        score = ensemble_scores[seed] if refined else base_scores[seed]
        return {
            "score": score,
            "candidate": payload["candidate"],
            "strategy": StrategyConfig(),
            "model": _fake_trained_model(),
            "row_count": 10,
            "positive_rate": 0.5,
            "threshold": 0.55,
            "threshold_source": "strategy",
            "threshold_score": None,
            "calibration_rows": 0,
            "validation_rows": 5,
            "validation_score": score,
            "full_sample_score": score,
            "ensemble_refined": refined,
        }

    monkeypatch.setattr(training_suite, "_evaluate_candidate", fake_evaluate)

    outcome = train_for_objective(
        _synthetic_candles(n=220),
        StrategyConfig(),
        objective,
        output_dir=tmp_path,
        market_type="spot",
        starting_cash=1000.0,
        max_workers=1,
    )

    assert outcome.best_score == 4.0
    assert outcome.best_params["seed"] == 11
    assert outcome.ensemble_refined is True
    assert outcome.ensemble_refinement_candidates == 3
    assert evaluated[-3:] == [(7, True), (11, True), (13, True)]


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


def test_run_training_suite_forwards_optional_gpu_args(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[dict[str, object]] = []

    def fake_train_for_objective(candles, base_strategy, objective, **kwargs):
        observed.append(kwargs)
        return ObjectiveOutcome(
            objective=objective.name,
            model_path=tmp_path / f"{objective.name}.json",
            feature_dim=1,
            feature_signature="sig",
            best_score=1.0,
            best_params={"epochs": 1},
            explored_candidates=1,
            rejected_candidates=0,
            epochs=1,
            learning_rate=0.01,
            l2_penalty=0.0,
            row_count=1,
            positive_rate=0.5,
        )

    monkeypatch.setattr(training_suite, "train_for_objective", fake_train_for_objective)
    run_training_suite(
        _synthetic_candles(n=20),
        StrategyConfig(),
        objectives=["default"],
        output_dir=tmp_path,
        compute_backend="directml",
        batch_size=64,
        score_batch_size=32,
    )
    assert observed[0]["compute_backend"] == "directml"
    assert observed[0]["batch_size"] == 64
    assert observed[0]["score_batch_size"] == 32


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


# ----- rank_report ---------------------------------------------------------


def test_rank_report_ranks_precomputed_backtests() -> None:
    ranked = rank_report([
        ({"name": "low"}, _make_result(realized_pnl=5.0, closed_trades=4, win_rate=0.25)),
        ({"name": "high"}, _make_result(realized_pnl=60.0, closed_trades=4, win_rate=0.75)),
        ({"name": "reject"}, _make_result(realized_pnl=80.0, closed_trades=0, win_rate=0.0)),
    ])

    assert ranked[0]["params"] == {"name": "high"}
    assert ranked[0]["accepted"] is True
    assert ranked[-1]["params"] == {"name": "reject"}
    assert ranked[-1]["accepted"] is False
    assert ranked[-1]["reject_reason"] == "closed_trades<3"


# ----- real-runner smoke test for train_for_objective ----------------------


def test_train_for_objective_real_runner_small_dataset(tmp_path: Path) -> None:
    """Exercises the real ``_run_candidate`` path end-to-end with a small dataset.

    We shrink the grid by monkey-patching ``_candidate_grid`` to a single
    lightweight candidate so the test stays fast but still traverses
    ``make_advanced_rows`` + ``train_advanced`` + ``run_backtest``.
    """

    candles = _synthetic_candles(n=260)
    default_objective = get_objective("default")
    objective = ObjectiveSpec(
        name="smoke",
        label="Smoke",
        summary="Lenient smoke objective",
        long_description="Lenient real-runner smoke objective for tiny synthetic datasets.",
        scorer=lambda result: 1.0 + result.realized_pnl / max(1.0, result.starting_cash),
        min_closed_trades=0,
        min_realized_pnl=None,
        min_edge_vs_buy_hold=None,
        max_drawdown_rejection=1.0,
        training=default_objective.training,
    )

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
    original_get_objective = training_suite.get_objective
    training_suite._candidate_grid = lambda _t: [single_candidate]
    training_suite.get_objective = lambda _name: objective
    try:
        outcome = train_for_objective(
            candles, StrategyConfig(), objective,
            output_dir=tmp_path,
            market_type="spot",
            starting_cash=1000.0,
        )
    finally:
        training_suite._candidate_grid = original
        training_suite.get_objective = original_get_objective

    assert (tmp_path / f"model_{objective.name}.json").exists()
    assert outcome.feature_dim > 0
    assert outcome.row_count > 0
