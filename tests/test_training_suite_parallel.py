"""Coverage tests for the parallelization + worker path of training_suite."""

from __future__ import annotations



from simple_ai_bitcoin_trading_binance import training_suite
from simple_ai_bitcoin_trading_binance.api import Candle
from simple_ai_bitcoin_trading_binance.config import load_strategy
from simple_ai_bitcoin_trading_binance.objective import get_objective


def _candles(n: int = 240) -> list[Candle]:
    out = []
    for i in range(n):
        price = 100.0 + (i % 7) * 0.25 + (i * 0.01)
        out.append(Candle(
            open_time=i * 60_000,
            open=price,
            high=price + 0.3,
            low=price - 0.3,
            close=price + 0.05,
            volume=1.0 + (i % 3),
            close_time=i * 60_000 + 59_000,
        ))
    return out


def test_resolve_workers_branches():
    # Empty candidates → 1 worker no matter what
    assert training_suite._resolve_workers(8, 0) == 1
    # Explicit max_workers clamps to candidate count
    assert training_suite._resolve_workers(8, 3) == 3
    # max_workers=None → uses os.cpu_count() floor 1
    assert training_suite._resolve_workers(None, 4) >= 1
    # Negative / zero max_workers clamps to 1
    assert training_suite._resolve_workers(0, 5) == 1


def test_evaluate_candidate_end_to_end():
    strategy = load_strategy()
    feature_cfg = training_suite.default_config_for("default", strategy.enabled_features)
    rows = training_suite.make_advanced_rows(_candles(240), feature_cfg)
    assert rows
    train_rows, eval_rows = training_suite._walk_forward_split(rows)
    candidate = training_suite.CandidateParams(
        epochs=5,
        learning_rate=0.05,
        l2_penalty=1e-3,
        signal_threshold=0.55,
        stop_loss_pct=0.02,
        take_profit_pct=0.03,
        risk_per_trade=0.01,
    )
    outcome = training_suite._evaluate_candidate({
        "candidate": candidate,
        "rows_train": train_rows,
        "rows_eval": eval_rows,
        "feature_cfg": feature_cfg,
        "base_strategy": strategy,
        "objective": "default",
        "market_type": "spot",
        "starting_cash": 1000.0,
    })
    assert "score" in outcome
    assert outcome["model"].feature_dim == len(rows[0].features)


def test_train_for_objective_serial_path(tmp_path, monkeypatch):
    """Real ``_evaluate_candidate`` path with max_workers=1 (serial)."""

    # Trim grid to 2 candidates so the test finishes fast.
    monkeypatch.setattr(
        training_suite,
        "_candidate_grid",
        lambda training: [
            training_suite.CandidateParams(
                epochs=3, learning_rate=0.05, l2_penalty=1e-3,
                signal_threshold=0.55, stop_loss_pct=0.02,
                take_profit_pct=0.03, risk_per_trade=0.01,
            ),
            training_suite.CandidateParams(
                epochs=3, learning_rate=0.04, l2_penalty=1e-3,
                signal_threshold=0.58, stop_loss_pct=0.02,
                take_profit_pct=0.03, risk_per_trade=0.01,
            ),
        ],
    )
    strategy = load_strategy()
    outcome = training_suite.train_for_objective(
        _candles(240),
        strategy,
        get_objective("default"),
        output_dir=tmp_path,
        market_type="spot",
        starting_cash=1000.0,
        max_workers=1,
    )
    assert outcome.objective == "default"
    assert outcome.model_path.exists()


def test_train_for_objective_parallel_path_via_mocked_pool(tmp_path, monkeypatch):
    """Max_workers>1 uses ProcessPoolExecutor; mock the pool for determinism."""

    class _FakePool:
        def __init__(self, *_args, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def map(self, func, payloads):
            return [func(p) for p in payloads]

    monkeypatch.setattr(training_suite, "ProcessPoolExecutor", _FakePool)
    monkeypatch.setattr(
        training_suite,
        "_candidate_grid",
        lambda training: [
            training_suite.CandidateParams(
                epochs=3, learning_rate=0.05, l2_penalty=1e-3,
                signal_threshold=0.55, stop_loss_pct=0.02,
                take_profit_pct=0.03, risk_per_trade=0.01,
            ),
            training_suite.CandidateParams(
                epochs=3, learning_rate=0.04, l2_penalty=1e-3,
                signal_threshold=0.58, stop_loss_pct=0.02,
                take_profit_pct=0.03, risk_per_trade=0.01,
            ),
        ],
    )
    strategy = load_strategy()
    outcome = training_suite.train_for_objective(
        _candles(240),
        strategy,
        get_objective("default"),
        output_dir=tmp_path,
        market_type="spot",
        starting_cash=1000.0,
        max_workers=4,
    )
    assert outcome.model_path.exists()


def test_run_training_suite_passes_max_workers(tmp_path, monkeypatch):
    """``run_training_suite`` forwards ``max_workers`` only when provided."""

    captured: list[dict[str, object]] = []

    fake_outcome = training_suite.ObjectiveOutcome(
        objective="default",
        model_path=tmp_path / "m.json",
        feature_dim=10,
        feature_signature="sig",
        best_score=0.1,
        best_params={},
        explored_candidates=1,
        rejected_candidates=0,
        epochs=3,
        learning_rate=0.05,
        l2_penalty=1e-3,
        row_count=42,
        positive_rate=0.5,
    )

    def fake_train_for_objective(*args, max_workers=None, **kwargs):
        captured.append({"max_workers": max_workers})
        return fake_outcome

    monkeypatch.setattr(training_suite, "train_for_objective", fake_train_for_objective)
    strategy = load_strategy()
    training_suite.run_training_suite(
        [], strategy, objectives=["default"],
        output_dir=tmp_path, summary_path=tmp_path / "s.json",
        max_workers=3,
    )
    assert captured[0]["max_workers"] == 3

    captured.clear()
    training_suite.run_training_suite(
        [], strategy, objectives=["default"],
        output_dir=tmp_path, summary_path=tmp_path / "s.json",
    )
    # max_workers default None means not forwarded
    assert captured[0]["max_workers"] is None
