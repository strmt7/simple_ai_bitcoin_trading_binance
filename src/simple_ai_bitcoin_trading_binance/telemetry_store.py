"""Append-only raw observation and source grading store."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence, cast


@dataclass(frozen=True)
class RawObservation:
    id: int
    observed_at_ms: int
    kind: str
    source: str
    symbol: str
    horizon: str
    score: float | None
    confidence: float | None
    payload: dict[str, object] | list[object]

    def asdict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SourceGrade:
    id: int
    created_at_ms: int
    source: str
    horizon: str
    window_start_ms: int
    window_end_ms: int
    grade: int
    sample_count: int
    model: str
    reason: str
    evidence: dict[str, object]

    def asdict(self) -> dict[str, object]:
        return asdict(self)


class TradingTelemetryStore:
    """SQLite WAL store for replayable provider/model observations."""

    def __init__(self, path: str | Path = "data/trading_telemetry.sqlite") -> None:
        self.path = Path(path)
        self._conn: sqlite3.Connection | None = None

    def connect(self) -> sqlite3.Connection:
        if self._conn is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self.path)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._init_schema()
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "TradingTelemetryStore":
        self.connect()
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        self.close()

    @staticmethod
    def _now_ms() -> int:
        return int(time.time() * 1000)

    def _init_schema(self) -> None:
        conn = self.connect()
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS raw_observations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                observed_at_ms INTEGER NOT NULL,
                kind TEXT NOT NULL,
                source TEXT NOT NULL,
                symbol TEXT NOT NULL,
                horizon TEXT NOT NULL,
                score REAL,
                confidence REAL,
                payload_hash TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at_ms INTEGER NOT NULL,
                UNIQUE(kind, source, observed_at_ms, payload_hash)
            );
            CREATE INDEX IF NOT EXISTS idx_raw_observations_lookup
                ON raw_observations(kind, source, observed_at_ms);
            CREATE INDEX IF NOT EXISTS idx_raw_observations_symbol
                ON raw_observations(symbol, horizon, observed_at_ms);

            CREATE TABLE IF NOT EXISTS source_grades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at_ms INTEGER NOT NULL,
                source TEXT NOT NULL,
                horizon TEXT NOT NULL,
                window_start_ms INTEGER NOT NULL,
                window_end_ms INTEGER NOT NULL,
                grade INTEGER NOT NULL,
                sample_count INTEGER NOT NULL,
                model TEXT NOT NULL,
                reason TEXT NOT NULL,
                evidence_json TEXT NOT NULL,
                UNIQUE(source, horizon, window_start_ms, window_end_ms, model)
            );
            CREATE INDEX IF NOT EXISTS idx_source_grades_lookup
                ON source_grades(source, horizon, window_end_ms);
            """
        )
        conn.commit()

    @staticmethod
    def _payload_json(payload: Mapping[str, object] | Sequence[object]) -> str:
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)

    def record_observation(
        self,
        *,
        kind: str,
        source: str,
        payload: Mapping[str, object] | Sequence[object],
        observed_at_ms: int | None = None,
        symbol: str = "BTCUSDC",
        horizon: str = "",
        score: float | None = None,
        confidence: float | None = None,
    ) -> int:
        payload_json = self._payload_json(payload)
        payload_hash = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
        observed = self._now_ms() if observed_at_ms is None else int(observed_at_ms)
        created = self._now_ms()
        cursor = self.connect().execute(
            """
            INSERT OR IGNORE INTO raw_observations (
                observed_at_ms, kind, source, symbol, horizon, score, confidence,
                payload_hash, payload_json, created_at_ms
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                observed,
                str(kind),
                str(source),
                str(symbol or "BTCUSDC").upper(),
                str(horizon or ""),
                score,
                confidence,
                payload_hash,
                payload_json,
                created,
            ),
        )
        self.connect().commit()
        if cursor.rowcount > 0 and cursor.lastrowid:
            return int(cast(int, cursor.lastrowid))
        row = self.connect().execute(
            """
            SELECT id FROM raw_observations
            WHERE kind = ? AND source = ? AND observed_at_ms = ? AND payload_hash = ?
            """,
            (str(kind), str(source), observed, payload_hash),
        ).fetchone()
        return int(row["id"]) if row is not None else 0

    def record_signal_report(self, report: object, *, raw_payloads: Sequence[object] = ()) -> int:
        inserted = 0
        components = getattr(report, "components", [])
        for component in components:
            payload = component.asdict() if hasattr(component, "asdict") else dict(component)
            inserted += 1 if self.record_observation(
                kind="external_signal_component",
                source=str(payload.get("provider") or "unknown"),
                payload=payload,
                observed_at_ms=int(float(payload.get("known_at_ms") or getattr(report, "known_at_ms", 0) or self._now_ms())),
                symbol=str(payload.get("source_symbol") or "BTCUSDC"),
                horizon=str(payload.get("horizon") or ""),
                score=float(payload.get("score")) if payload.get("score") is not None else None,
                confidence=float(payload.get("urgency")) if payload.get("urgency") is not None else None,
            ) else 0
        for index, payload in enumerate(raw_payloads):
            if isinstance(payload, Mapping):
                source = str(payload.get("provider") or payload.get("source") or f"raw_{index}")
                record_payload: Mapping[str, object] | Sequence[object] = payload
            elif isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
                source = f"raw_{index}"
                record_payload = payload
            else:
                source = f"raw_{index}"
                record_payload = {"value": str(payload)}
            inserted += 1 if self.record_observation(
                kind="raw_provider_payload",
                source=source,
                payload=record_payload,
                observed_at_ms=getattr(report, "known_at_ms", self._now_ms()),
            ) else 0
        return inserted

    def recent_observations(
        self,
        *,
        since_ms: int,
        limit: int = 500,
        kind: str | None = None,
    ) -> list[RawObservation]:
        params: list[object] = [int(since_ms)]
        query = """
            SELECT id, observed_at_ms, kind, source, symbol, horizon, score, confidence, payload_json
            FROM raw_observations
            WHERE observed_at_ms >= ?
            """
        if kind:
            query += " AND kind = ?"
            params.append(str(kind))
        query += " ORDER BY observed_at_ms DESC, id DESC LIMIT ?"
        params.append(max(1, int(limit)))
        rows = self.connect().execute(query, params).fetchall()
        return [
            RawObservation(
                id=int(row["id"]),
                observed_at_ms=int(row["observed_at_ms"]),
                kind=str(row["kind"]),
                source=str(row["source"]),
                symbol=str(row["symbol"]),
                horizon=str(row["horizon"]),
                score=None if row["score"] is None else float(row["score"]),
                confidence=None if row["confidence"] is None else float(row["confidence"]),
                payload=cast(dict[str, object] | list[object], json.loads(str(row["payload_json"]))),
            )
            for row in rows
        ]

    def source_rollups(self, *, since_ms: int, until_ms: int) -> list[dict[str, object]]:
        rows = self.connect().execute(
            """
            SELECT source, horizon, COUNT(*) AS sample_count,
                   AVG(COALESCE(score, 0.0)) AS avg_score,
                   AVG(ABS(COALESCE(score, 0.0))) AS avg_abs_score,
                   AVG(COALESCE(confidence, 0.0)) AS avg_confidence,
                   SUM(CASE WHEN kind = 'raw_provider_payload' THEN 1 ELSE 0 END) AS raw_records,
                   SUM(CASE WHEN kind = 'external_signal_component' THEN 1 ELSE 0 END) AS component_records
            FROM raw_observations
            WHERE observed_at_ms >= ? AND observed_at_ms <= ?
            GROUP BY source, horizon
            ORDER BY sample_count DESC, source ASC
            """,
            (int(since_ms), int(until_ms)),
        ).fetchall()
        return [
            {
                "source": str(row["source"]),
                "horizon": str(row["horizon"] or "medium"),
                "sample_count": int(row["sample_count"]),
                "avg_score": float(row["avg_score"] or 0.0),
                "avg_abs_score": float(row["avg_abs_score"] or 0.0),
                "avg_confidence": float(row["avg_confidence"] or 0.0),
                "raw_records": int(row["raw_records"] or 0),
                "component_records": int(row["component_records"] or 0),
            }
            for row in rows
        ]

    def record_source_grade(
        self,
        *,
        source: str,
        horizon: str,
        window_start_ms: int,
        window_end_ms: int,
        grade: int,
        sample_count: int,
        model: str,
        reason: str,
        evidence: Mapping[str, object],
    ) -> SourceGrade:
        bounded_grade = max(0, min(10, int(grade)))
        created_at_ms = self._now_ms()
        evidence_payload = dict(evidence)
        cursor = self.connect().execute(
            """
            INSERT OR REPLACE INTO source_grades (
                created_at_ms, source, horizon, window_start_ms, window_end_ms,
                grade, sample_count, model, reason, evidence_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                created_at_ms,
                str(source),
                str(horizon or "medium"),
                int(window_start_ms),
                int(window_end_ms),
                bounded_grade,
                max(0, int(sample_count)),
                str(model),
                str(reason)[:240],
                self._payload_json(evidence_payload),
            ),
        )
        self.connect().commit()
        return SourceGrade(
            id=int(cast(int, cursor.lastrowid)),
            created_at_ms=created_at_ms,
            source=str(source),
            horizon=str(horizon or "medium"),
            window_start_ms=int(window_start_ms),
            window_end_ms=int(window_end_ms),
            grade=bounded_grade,
            sample_count=max(0, int(sample_count)),
            model=str(model),
            reason=str(reason)[:240],
            evidence=evidence_payload,
        )

    def recent_grades(self, *, limit: int = 50) -> list[SourceGrade]:
        rows = self.connect().execute(
            """
            SELECT id, created_at_ms, source, horizon, window_start_ms, window_end_ms,
                   grade, sample_count, model, reason, evidence_json
            FROM source_grades
            ORDER BY created_at_ms DESC, id DESC
            LIMIT ?
            """,
            (max(1, int(limit)),),
        ).fetchall()
        return [
            SourceGrade(
                id=int(row["id"]),
                created_at_ms=int(row["created_at_ms"]),
                source=str(row["source"]),
                horizon=str(row["horizon"]),
                window_start_ms=int(row["window_start_ms"]),
                window_end_ms=int(row["window_end_ms"]),
                grade=int(row["grade"]),
                sample_count=int(row["sample_count"]),
                model=str(row["model"]),
                reason=str(row["reason"]),
                evidence=cast(dict[str, object], json.loads(str(row["evidence_json"]))),
            )
            for row in rows
        ]

    def latest_source_grades(
        self,
        *,
        max_age_ms: int | None = None,
        now_ms: int | None = None,
    ) -> dict[tuple[str, str], SourceGrade]:
        params: list[object] = []
        query = """
            SELECT id, created_at_ms, source, horizon, window_start_ms, window_end_ms,
                   grade, sample_count, model, reason, evidence_json
            FROM source_grades
            """
        if max_age_ms is not None:
            reference_ms = self._now_ms() if now_ms is None else int(now_ms)
            query += " WHERE created_at_ms >= ?"
            params.append(reference_ms - max(0, int(max_age_ms)))
        query += " ORDER BY created_at_ms DESC, window_end_ms DESC, id DESC"
        rows = self.connect().execute(query, params).fetchall()
        latest: dict[tuple[str, str], SourceGrade] = {}
        for row in rows:
            key = (str(row["source"]), str(row["horizon"]))
            if key in latest:
                continue
            latest[key] = SourceGrade(
                id=int(row["id"]),
                created_at_ms=int(row["created_at_ms"]),
                source=str(row["source"]),
                horizon=str(row["horizon"]),
                window_start_ms=int(row["window_start_ms"]),
                window_end_ms=int(row["window_end_ms"]),
                grade=int(row["grade"]),
                sample_count=int(row["sample_count"]),
                model=str(row["model"]),
                reason=str(row["reason"]),
                evidence=cast(dict[str, object], json.loads(str(row["evidence_json"]))),
            )
        return latest
