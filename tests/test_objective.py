"""Branch-coverage tests for the objective scoring module."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from simple_ai_bitcoin_trading_binance.backtest import BacktestResult
from simple_ai_bitcoin_trading_binance import objective as obj


def _result(**overrides) -> BacktestResult:
    base = dict(
        starting_cash=1000.0,
        ending_cash=1100.0,
        realized_pnl=100.0,
        win_rate=0.6,
        trades=10,
        max_drawdown=0.05,
        closed_trades=10,
        gross_exposure=500.0,
        total_fees=1.0,
        stopped_by_drawdown=False,
        max_exposure=500.0,
        trades_per_day_cap_hit=0,
    )
    base.update(overrides)
    return BacktestResult(**base)


def test_available_and_describe():
    names = obj.available_objectives()
    assert set(names) == {"conservative", "default", "risky"}
    described = obj.describe_objectives()
    assert {entry["name"] for entry in described} == {"conservative", "default", "risky"}
    assert all("summary" in entry for entry in described)


def test_get_objective_lookups():
    assert obj.get_objective("DEFAULT").name == "default"
    assert obj.get_objective("balanced").name == "default"
    with pytest.raises(ValueError):
        obj.get_objective("")
    with pytest.raises(ValueError):
        obj.get_objective("nonesuch")


def test_scorer_paths_basic():
    result = _result()
    assert obj.CONSERVATIVE.score(result) < obj.RISKY.score(result)
    assert obj.DEFAULT.accepts(result) is True


def test_stopped_by_drawdown_penalty_paths():
    r = _result(stopped_by_drawdown=True, closed_trades=10, max_drawdown=0.10)
    for spec in (obj.CONSERVATIVE, obj.DEFAULT, obj.RISKY):
        # scorer still returns a float; acceptance may differ
        assert isinstance(spec.score(r), float)


def test_conservative_rejects_on_drawdown():
    r = _result(max_drawdown=0.5)
    assert obj.CONSERVATIVE.accepts(r) is False


def test_conservative_rejects_on_stopped_by_drawdown():
    r = _result(stopped_by_drawdown=True)
    assert obj.CONSERVATIVE.accepts(r) is False


def test_min_closed_trades_gate():
    r = _result(closed_trades=1)
    assert obj.CONSERVATIVE.accepts(r) is False


def test_safe_and_return_ratio_non_finite():
    assert obj._safe(float("nan")) == 0.0
    assert obj._safe(1.2) == pytest.approx(1.2)
    r = _result(starting_cash=0.0, realized_pnl=10.0)
    assert obj._return_ratio(r) == 0.0
    r2 = _result(realized_pnl=float("nan"))
    # scorer should absorb NaN → 0 via _safe
    assert math.isfinite(obj.CONSERVATIVE.score(r2))


def test_max_drawdown_rejection_one_never_rejects_on_drawdown():
    # Manually craft an objective with rejection=1.0 — never rejects on drawdown
    spec = replace(obj.DEFAULT, max_drawdown_rejection=1.0, min_closed_trades=0)
    assert spec.accepts(_result(max_drawdown=0.99)) is True


def test_rank_candidates_orders_accepted_first():
    winning = _result(realized_pnl=500.0, max_drawdown=0.02, closed_trades=10)
    losing = _result(realized_pnl=-100.0, max_drawdown=0.30, closed_trades=1)
    ranked = obj.rank_candidates(
        [({"id": "loser"}, losing), ({"id": "winner"}, winning)],
        obj.CONSERVATIVE,
    )
    assert ranked[0]["params"]["id"] == "winner"
    assert ranked[0]["accepted"] is True
    assert ranked[1]["accepted"] is False
    assert ranked[1]["score"] == float("-inf")
    assert ranked[1]["reject_reason"]


def test_rank_candidates_records_stopped_by_drawdown_reason():
    bad = _result(stopped_by_drawdown=True, max_drawdown=0.3, closed_trades=5)
    ranked = obj.rank_candidates([({"id": "x"}, bad)], obj.CONSERVATIVE)
    assert "stopped_by_drawdown" in ranked[0]["reject_reason"]


def test_rank_candidates_hard_gate_fallback_reason():
    # Build a fake objective where the only rejection is a hard gate we can't
    # describe via the declared reasons — forces the "hard-gate-failed" default.
    class _Custom(obj.ObjectiveSpec):
        pass

    spec = replace(obj.DEFAULT, min_closed_trades=0, max_drawdown_rejection=1.0)
    # accepts is overridden implicitly by replace? No — but we can use a subclass
    def never(_):
        return False

    spec_obj = replace(spec)  # keep dataclass happy
    # attach an accepts-always-False shim by wrapping
    class _Wrap:
        def __init__(self, inner):
            self._inner = inner
            self.name = inner.name
            self.label = inner.label
            self.summary = inner.summary
            self.long_description = inner.long_description
            self.min_closed_trades = inner.min_closed_trades
            self.max_drawdown_rejection = inner.max_drawdown_rejection
            self.scorer = inner.scorer
            self.training = inner.training

        def score(self, result):
            return 0.0

        def accepts(self, _result):
            return False

    wrapped = _Wrap(spec_obj)
    ranked = obj.rank_candidates([({"id": "x"}, _result())], wrapped)
    assert ranked[0]["reject_reason"] == "hard-gate-failed"


def test_objective_training_dataclass_defaults_present():
    for spec in (obj.CONSERVATIVE, obj.DEFAULT, obj.RISKY):
        assert spec.training is not None
        assert spec.training.epochs > 0
