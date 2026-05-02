"""Run a real-data signal soak and grade collected sources afterward."""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
import subprocess  # nosec B404
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

_JITTER_RANDOM = random.SystemRandom()


@dataclass(frozen=True)
class SignalIteration:
    """One signal polling iteration summary."""

    ok: bool
    fresh_count: int = 0
    provider_count: int = 0
    news_ai_status: str = "unknown"
    news_ai_latency_ms: int = 0
    news_backend: str = "unknown"
    reaction_required: bool = False
    elapsed_ms: int = 0
    error: str = ""


@dataclass(frozen=True)
class GradeSummary:
    """Source grading summary."""

    ok: bool
    graded_sources: int = 0
    ai_status: str = "unknown"
    ai_latency_ms: int = 0
    elapsed_ms: int = 0
    error: str = ""


def _timestamp() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _log(message: str) -> None:
    print(f"{_timestamp()} {message}", flush=True)


def _json_from_process(proc: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    payload = json.loads(proc.stdout)
    if not isinstance(payload, dict):
        raise ValueError("command did not return a JSON object")
    return payload


def _tail(text: str, limit: int = 500) -> str:
    return text[-limit:].replace("\r", "\\r").replace("\n", "\\n")


def _run_signal_iteration(args: argparse.Namespace) -> SignalIteration:
    started = time.time()
    cmd = [
        str(args.python),
        "-m",
        "simple_ai_bitcoin_trading_binance.cli",
        "signals",
        "--refresh",
        "--timeout",
        str(args.timeout_seconds),
        "--min-providers",
        str(args.min_providers),
        "--compute-backend",
        str(args.compute_backend),
        "--news-provider-limit",
        str(args.provider_limit),
        "--news-items-per-provider",
        str(args.news_items_per_provider),
        "--provider-parallelism",
        str(args.provider_parallelism),
        "--provider-jitter",
        str(args.provider_jitter_seconds),
        "--short-reaction-refresh",
        str(args.short_reaction_refresh_seconds),
        "--ollama-news",
        "--ollama-model",
        str(args.ollama_model),
        "--ollama-timeout",
        str(args.ollama_timeout_seconds),
        "--telemetry-db",
        str(args.db),
        "--cache",
        str(args.cache),
        "--json",
    ]
    try:
        proc = subprocess.run(  # nosec B603
            cmd,
            cwd=args.repo,
            text=True,
            capture_output=True,
            timeout=args.iteration_timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return SignalIteration(
            ok=False,
            elapsed_ms=int((time.time() - started) * 1000),
            error=f"timeout after {exc.timeout}s",
        )
    elapsed_ms = int((time.time() - started) * 1000)
    if proc.returncode != 0:
        return SignalIteration(
            ok=False,
            elapsed_ms=elapsed_ms,
            error=f"rc={proc.returncode} stderr={_tail(proc.stderr)} stdout={_tail(proc.stdout, 240)}",
        )
    try:
        payload = _json_from_process(proc)
    except Exception as exc:
        return SignalIteration(
            ok=False,
            elapsed_ms=elapsed_ms,
            error=f"json parse failed: {exc}; stdout={_tail(proc.stdout, 240)}",
        )
    return SignalIteration(
        ok=True,
        fresh_count=int(payload.get("fresh_count", 0)),
        provider_count=int(payload.get("provider_count", 0)),
        news_ai_status=str(payload.get("news_ai_status", "unknown")),
        news_ai_latency_ms=int(payload.get("news_ai_latency_ms", 0)),
        news_backend=str(payload.get("news_backend_kind", "unknown")),
        reaction_required=bool(payload.get("reaction_required", False)),
        elapsed_ms=elapsed_ms,
    )


def _run_source_grades(args: argparse.Namespace) -> GradeSummary:
    started = time.time()
    cmd = [
        str(args.python),
        "-m",
        "simple_ai_bitcoin_trading_binance.cli",
        "source-grades",
        "--db",
        str(args.db),
        "--window-hours",
        str(args.grade_window_hours),
        "--ollama",
        "--ollama-model",
        str(args.ollama_model),
        "--ollama-timeout",
        str(args.grade_ollama_timeout_seconds),
        "--json",
    ]
    try:
        proc = subprocess.run(  # nosec B603
            cmd,
            cwd=args.repo,
            text=True,
            capture_output=True,
            timeout=args.grade_timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return GradeSummary(
            ok=False,
            elapsed_ms=int((time.time() - started) * 1000),
            error=f"timeout after {exc.timeout}s",
        )
    elapsed_ms = int((time.time() - started) * 1000)
    if proc.returncode != 0:
        return GradeSummary(
            ok=False,
            elapsed_ms=elapsed_ms,
            error=f"rc={proc.returncode} stderr={_tail(proc.stderr)} stdout={_tail(proc.stdout, 240)}",
        )
    try:
        payload = _json_from_process(proc)
    except Exception as exc:
        return GradeSummary(
            ok=False,
            elapsed_ms=elapsed_ms,
            error=f"json parse failed: {exc}; stdout={_tail(proc.stdout, 240)}",
        )
    return GradeSummary(
        ok=True,
        graded_sources=int(payload.get("graded_sources", 0)),
        ai_status=str(payload.get("ai_status", "unknown")),
        ai_latency_ms=int(payload.get("ai_latency_ms", 0)),
        elapsed_ms=elapsed_ms,
    )


def _db_counts(db_path: Path) -> tuple[list[tuple[str, int]], int, int]:
    if not db_path.exists():
        return [], 0, 0
    recent_cutoff_ms = int((time.time() - 7200) * 1000)
    with sqlite3.connect(db_path) as conn:
        raw_rows = conn.execute(
            "SELECT kind, COUNT(*) FROM raw_observations GROUP BY kind ORDER BY kind"
        ).fetchall()
        grade_count = int(conn.execute("SELECT COUNT(*) FROM source_grades").fetchone()[0])
        recent_grade_count = int(
            conn.execute(
                "SELECT COUNT(*) FROM source_grades WHERE created_at_ms >= ?",
                (recent_cutoff_ms,),
            ).fetchone()[0]
        )
    return [(str(kind), int(count)) for kind, count in raw_rows], grade_count, recent_grade_count


def _path(value: str) -> Path:
    return Path(value)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run real external-signal polling for a fixed duration and grade telemetry sources."
    )
    parser.add_argument("--repo", type=_path, default=Path.cwd())
    parser.add_argument("--python", type=_path, default=Path(sys.executable))
    parser.add_argument("--db", type=_path, default=Path("data/trading_telemetry.sqlite"))
    parser.add_argument("--cache", type=_path, default=Path("data/signals/hour_soak_external_signals.json"))
    parser.add_argument("--duration-seconds", type=float, default=3660.0)
    parser.add_argument("--interval-seconds", type=float, default=60.0)
    parser.add_argument("--interval-jitter-seconds", type=float, default=10.0)
    parser.add_argument("--iteration-timeout-seconds", type=float, default=90.0)
    parser.add_argument("--grade-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--compute-backend", default="auto")
    parser.add_argument("--provider-limit", type=int, default=30)
    parser.add_argument("--min-providers", type=int, default=30)
    parser.add_argument("--news-items-per-provider", type=int, default=3)
    parser.add_argument("--provider-parallelism", type=int, default=12)
    parser.add_argument("--provider-jitter-seconds", type=float, default=0.25)
    parser.add_argument("--timeout-seconds", type=float, default=3.0)
    parser.add_argument("--short-reaction-refresh-seconds", type=int, default=20)
    parser.add_argument("--ollama-model", default="gemma4:e4b")
    parser.add_argument("--ollama-timeout-seconds", type=float, default=20.0)
    parser.add_argument("--grade-ollama-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--grade-window-hours", type=float, default=1.25)
    parser.add_argument("--skip-grading", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.repo = args.repo.resolve()
    args.db = (args.repo / args.db).resolve() if not args.db.is_absolute() else args.db
    args.cache = (args.repo / args.cache).resolve() if not args.cache.is_absolute() else args.cache
    args.python = (args.repo / args.python).resolve() if not args.python.is_absolute() else args.python
    args.db.parent.mkdir(parents=True, exist_ok=True)
    args.cache.parent.mkdir(parents=True, exist_ok=True)

    started = time.time()
    deadline = started + max(0.0, float(args.duration_seconds))
    iteration = 0
    successes = 0
    failures = 0
    min_fresh: int | None = None
    max_latency_ms = 0
    _log(
        "START real-data soak "
        f"repo={args.repo} db={args.db} cache={args.cache} "
        f"duration_seconds={args.duration_seconds}"
    )
    while time.time() < deadline:
        iteration += 1
        result = _run_signal_iteration(args)
        if result.ok:
            successes += 1
            min_fresh = result.fresh_count if min_fresh is None else min(min_fresh, result.fresh_count)
            max_latency_ms = max(max_latency_ms, result.news_ai_latency_ms)
            _log(
                "ITER "
                f"{iteration} ok providers={result.fresh_count}/{result.provider_count} "
                f"ai={result.news_ai_status} ai_ms={result.news_ai_latency_ms} "
                f"backend={result.news_backend} reaction={result.reaction_required} "
                f"elapsed_ms={result.elapsed_ms}"
            )
        else:
            failures += 1
            _log(f"ITER {iteration} failed elapsed_ms={result.elapsed_ms} error={result.error}")
        remaining = deadline - time.time()
        if remaining <= 0:
            break
        delay = float(args.interval_seconds) + _JITTER_RANDOM.uniform(
            -float(args.interval_jitter_seconds),
            float(args.interval_jitter_seconds),
        )
        time.sleep(max(5.0, min(delay, remaining)))

    grade = GradeSummary(ok=True, error="skipped") if args.skip_grading else _run_source_grades(args)
    if args.skip_grading:
        _log("GRADING skipped")
    elif grade.ok:
        _log(
            "GRADING ok "
            f"graded_sources={grade.graded_sources} ai={grade.ai_status} "
            f"ai_ms={grade.ai_latency_ms} elapsed_ms={grade.elapsed_ms}"
        )
    else:
        _log(f"GRADING failed elapsed_ms={grade.elapsed_ms} error={grade.error}")

    raw_rows, grade_count, recent_grade_count = _db_counts(args.db)
    _log(
        "DB_COUNTS "
        f"raw_observations={raw_rows} source_grades={grade_count} "
        f"recent_source_grades={recent_grade_count}"
    )
    _log(
        "END real-data soak "
        f"iterations={iteration} successes={successes} failures={failures} "
        f"min_fresh={min_fresh} max_news_ai_latency_ms={max_latency_ms} "
        f"runtime_seconds={int(time.time() - started)}"
    )
    if successes < 1:
        return 2
    if not grade.ok:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
