"""Independent backtesting panel — interval + time-window validation, no forced training.

This module is the operator surface for ad-hoc backtests.  It does **not**
train a model; callers pass a model path (or ``None`` for a zero-weight
baseline walk) and the panel loads it, applies it to the selected candles,
and writes a timestamped report under ``data/backtests/``.

The panel is intentionally decoupled from the training CLI so operators can
ask "how does my saved Conservative model perform on just 2026-02 at 5m?"
without re-running the fetch/train pipeline.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

from .advanced_model import default_config_for, make_advanced_rows
from .api import Candle
from .backtest import BacktestResult, run_backtest
from .features import ModelRow, make_rows
from .intervals import interval_minutes, supported_intervals, validate_interval
from .model import ModelLoadError, TrainedModel, load_model
from .objective import ObjectiveSpec, get_objective
from .types import StrategyConfig

_REPORT_DIR = Path("data/backtests")
_FILENAME_SANITIZE = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass
class BacktestRequest:
    """Parameters a user submits when launching a backtest.

    Every field is validated before the run starts so the user never sees a
    half-executed backtest because of a typo.
    """

    interval: str
    market_type: str = "spot"
    start_ms: int | None = None
    end_ms: int | None = None
    model_path: str | None = None
    data_path: str = "data/historical_btcusdc.json"
    starting_cash: float = 1000.0
    objective: str | None = None
    tag: str = ""
    notes: str = ""

    def validated_interval(self) -> str:
        return validate_interval(self.interval, self.market_type)


@dataclass
class BacktestReport:
    """The artifact written for every backtest run."""

    request: BacktestRequest
    result: BacktestResult
    rows_used: int
    candles_used: int
    started_at_ms: int
    finished_at_ms: int
    model_loaded: bool
    model_path_resolved: str | None
    objective_score: float | None
    objective_accepts: bool | None
    tag: str
    filename: str

    def asdict(self) -> dict[str, object]:
        request_dict = asdict(self.request)
        result_dict = asdict(self.result)
        return {
            "request": request_dict,
            "result": result_dict,
            "rows_used": self.rows_used,
            "candles_used": self.candles_used,
            "started_at_ms": self.started_at_ms,
            "finished_at_ms": self.finished_at_ms,
            "duration_ms": self.finished_at_ms - self.started_at_ms,
            "model_loaded": self.model_loaded,
            "model_path_resolved": self.model_path_resolved,
            "objective": {
                "name": self.request.objective,
                "score": self.objective_score,
                "accepted": self.objective_accepts,
            },
            "tag": self.tag,
            "filename": self.filename,
        }


def parse_date_ms(value: str | None, *, end_of_day: bool = False) -> int | None:
    """Parse ``YYYY-MM-DD`` or ``YYYY-MM-DDTHH:MM:SS`` into a UTC ms timestamp.

    Returns ``None`` when ``value`` is falsy so optional fields map to None
    without the caller having to branch.
    """

    if not value:
        return None
    candidate = value.strip()
    formats = ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M")
    for fmt in formats:
        try:
            parsed = datetime.strptime(candidate, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        if fmt == "%Y-%m-%d" and end_of_day:
            parsed = parsed.replace(hour=23, minute=59, second=59)
        return int(parsed.timestamp() * 1000)
    raise ValueError(f"Cannot parse date {value!r}. Use YYYY-MM-DD or ISO 8601.")


def filter_candles(
    candles: Sequence[Candle],
    *,
    start_ms: int | None,
    end_ms: int | None,
) -> list[Candle]:
    """Return the subset of ``candles`` whose close-time lies inside the window."""

    low = 0 if start_ms is None else int(start_ms)
    high = 2 ** 63 - 1 if end_ms is None else int(end_ms)
    return [candle for candle in candles if low <= candle.close_time <= high]


def _sanitize(tag: str) -> str:
    trimmed = _FILENAME_SANITIZE.sub("-", tag.strip())
    return trimmed.strip("-")[:40]


def build_report_filename(request: BacktestRequest, *, ts_ms: int) -> str:
    """Deterministic, operator-friendly filename.

    Pattern::

        backtest_<tag>_<market>_<interval>_<YYYYMMDDHHMMSS>.json

    When a tag is missing, the objective name is used; when that is also
    missing, ``untagged`` is used so the filename still conveys the run.
    """

    base_tag = _sanitize(request.tag) or _sanitize(request.objective or "untagged")
    stamp = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc).strftime("%Y%m%d%H%M%S")
    interval = _sanitize(request.validated_interval())
    market = _sanitize(request.market_type)
    return f"backtest_{base_tag or 'untagged'}_{market}_{interval}_{stamp}.json"


def load_candles_from_json(path: str) -> list[Candle]:
    """Load candles previously persisted by ``command_fetch``."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected candle list in {path}")
    candles: list[Candle] = []
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        try:
            candles.append(Candle(
                open_time=int(entry["open_time"]),
                open=float(entry["open"]),
                high=float(entry["high"]),
                low=float(entry["low"]),
                close=float(entry["close"]),
                volume=float(entry.get("volume", 0.0)),
                close_time=int(entry["close_time"]),
            ))
        except (KeyError, TypeError, ValueError):
            continue
    return candles


def _zero_model(feature_dim: int) -> TrainedModel:
    """Return a neutral model that predicts 0.5 for every row.

    Useful as a baseline when the operator wants to see the strategy's
    trade-execution plumbing without any model influence.
    """

    return TrainedModel(
        weights=[0.0] * feature_dim,
        bias=0.0,
        feature_dim=feature_dim,
        epochs=0,
        feature_means=[0.0] * feature_dim,
        feature_stds=[1.0] * feature_dim,
    )


def _load_model_or_baseline(
    path: str | None,
    rows: Sequence[ModelRow],
    loader: Callable[[Path], TrainedModel],
) -> tuple[TrainedModel, bool, str | None]:
    if not path:
        dim = len(rows[0].features) if rows else 1
        return _zero_model(dim), False, None
    resolved = Path(path)
    try:
        return loader(resolved), True, str(resolved)
    except (FileNotFoundError, ModelLoadError):
        dim = len(rows[0].features) if rows else 1
        return _zero_model(dim), False, str(resolved)


def run_panel(
    request: BacktestRequest,
    strategy: StrategyConfig,
    *,
    candles_loader: Callable[[str], Sequence[Candle]] = load_candles_from_json,
    model_loader: Callable[[Path], TrainedModel] = load_model,
    report_dir: Path = _REPORT_DIR,
    clock=time.time,
) -> BacktestReport:
    """Execute one backtest and persist the report, returning the in-memory object."""

    request.validated_interval()  # fail fast on bad input
    objective: ObjectiveSpec | None = None
    if request.objective:
        objective = get_objective(request.objective)

    started_ms = int(clock() * 1000)
    candles = list(candles_loader(request.data_path))
    filtered = filter_candles(candles, start_ms=request.start_ms, end_ms=request.end_ms)
    # When the operator pins an objective, build the advanced feature rows tied
    # to that objective so a pre-trained advanced model validates cleanly.  The
    # base-feature path is preserved for operators backtesting the legacy model.
    if objective is not None:
        feature_cfg = default_config_for(objective.name, strategy.enabled_features)
        rows = make_advanced_rows(filtered, feature_cfg)
    else:
        rows = make_rows(
            filtered,
            strategy.feature_windows[0],
            strategy.feature_windows[1],
            lookahead=1,
            label_threshold=strategy.label_threshold,
            enabled_features=strategy.enabled_features,
        )
    model, loaded, resolved = _load_model_or_baseline(request.model_path, rows, model_loader)
    if rows and int(getattr(model, "feature_dim", 0)) != len(rows[0].features):
        raise ValueError(
            "model feature dimension "
            f"{getattr(model, 'feature_dim', 'unknown')} does not match panel rows "
            f"({len(rows[0].features)}). Use a model whose feature signature matches "
            "this panel request; objective runs need train-suite model_<objective>.json, "
            "while standard runs need train model artifacts with the same strategy features."
        )
    result = run_backtest(
        rows,
        model,
        strategy,
        starting_cash=request.starting_cash,
        market_type=request.market_type,
    )
    finished_ms = int(clock() * 1000)

    score = None
    accepts = None
    if objective is not None:
        score = objective.score(result)
        accepts = objective.accepts(result)

    report_dir.mkdir(parents=True, exist_ok=True)
    filename = build_report_filename(request, ts_ms=finished_ms)
    report = BacktestReport(
        request=request,
        result=result,
        rows_used=len(rows),
        candles_used=len(filtered),
        started_at_ms=started_ms,
        finished_at_ms=finished_ms,
        model_loaded=loaded,
        model_path_resolved=resolved,
        objective_score=score,
        objective_accepts=accepts,
        tag=request.tag,
        filename=filename,
    )
    (report_dir / filename).write_text(
        json.dumps(report.asdict(), indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    return report


@dataclass
class PanelListing:
    """A pointer to a historical report on disk."""

    path: Path
    tag: str
    interval: str
    market: str
    created_at: str

    def asdict(self) -> dict[str, str]:
        return {
            "path": str(self.path),
            "tag": self.tag,
            "interval": self.interval,
            "market": self.market,
            "created_at": self.created_at,
        }


def list_reports(report_dir: Path = _REPORT_DIR) -> list[PanelListing]:
    """Scan ``report_dir`` and return every backtest report in reverse-chron order."""

    if not report_dir.exists():
        return []
    listings: list[PanelListing] = []
    for path in sorted(report_dir.glob("backtest_*.json"), reverse=True):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        request = payload.get("request", {}) if isinstance(payload, dict) else {}
        listings.append(PanelListing(
            path=path,
            tag=str(request.get("tag") or ""),
            interval=str(request.get("interval") or ""),
            market=str(request.get("market_type") or ""),
            created_at=datetime.fromtimestamp(
                payload.get("finished_at_ms", 0) / 1000.0,
                tz=timezone.utc,
            ).isoformat(),
        ))
    return listings


def describe_supported_intervals(market_type: str) -> str:
    """Return a comma-separated listing of allowed intervals — used by help text."""

    return ", ".join(supported_intervals(market_type))


def estimated_candle_count(request: BacktestRequest) -> int:
    """How many candles the request's time window should yield on the exchange.

    Returns ``0`` when either bound is open — the UI shows an unconstrained hint
    in that case instead of a misleading number.
    """

    if request.start_ms is None or request.end_ms is None:
        return 0
    minutes = (request.end_ms - request.start_ms) // 60_000
    step = interval_minutes(request.validated_interval())
    if step <= 0 or minutes <= 0:
        return 0
    return int(minutes // step)
