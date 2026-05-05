"""Append-only raw observation and source grading store."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence, cast

_UNATTRIBUTED_RAW_SOURCE = "unattributed_raw_payload"
_DEFAULT_MAX_PAYLOAD_BYTES = 64 * 1024
_MAX_FUTURE_TIMESTAMP_SKEW_MS = 5 * 60_000
_OUTCOME_KINDS = ("external_signal_outcome", "signal_outcome", "source_outcome")


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

    def __init__(
        self,
        path: str | Path = "data/trading_telemetry.sqlite",
        *,
        max_payload_bytes: int = _DEFAULT_MAX_PAYLOAD_BYTES,
    ) -> None:
        self.path = Path(path)
        self.max_payload_bytes = max(256, int(max_payload_bytes))
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
            CREATE TABLE IF NOT EXISTS raw_payload_blobs (
                payload_hash TEXT PRIMARY KEY,
                payload_json TEXT NOT NULL,
                payload_bytes INTEGER NOT NULL,
                created_at_ms INTEGER NOT NULL
            );

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
        try:
            columns = {
                str(row["name"])
                for row in conn.execute("PRAGMA table_info(raw_payload_blobs)").fetchall()
            }
            if columns and "payload_bytes" not in columns:
                conn.execute("ALTER TABLE raw_payload_blobs ADD COLUMN payload_bytes INTEGER NOT NULL DEFAULT 0")
                conn.commit()
        except sqlite3.Error:  # pragma: no cover - legacy migration best effort
            conn.rollback()

    @staticmethod
    def _payload_json(payload: Mapping[str, object] | Sequence[object]) -> str:
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)

    def _bounded_payload_json(self, payload: Mapping[str, object] | Sequence[object]) -> tuple[str, str]:
        payload_json = self._payload_json(payload)
        payload_size = len(payload_json.encode("utf-8"))
        payload_hash = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
        blob_path = self._blob_path(payload_hash)
        try:
            blob_path.parent.mkdir(parents=True, exist_ok=True)
            if not blob_path.exists():
                blob_path.write_text(payload_json, encoding="utf-8")
        except OSError:  # pragma: no cover - blob files are a replay enhancement, not a trading blocker
            pass
        try:
            self.connect().execute(
                """
                INSERT OR IGNORE INTO raw_payload_blobs (
                    payload_hash, payload_json, payload_bytes, created_at_ms
                )
                VALUES (?, ?, ?, ?)
                """,
                (payload_hash, payload_json, payload_size, self._now_ms()),
            )
            self.connect().commit()
        except sqlite3.Error:  # pragma: no cover - telemetry blobs must never block observations
            self.connect().rollback()
        if payload_size <= self.max_payload_bytes:
            return payload_json, payload_hash
        base_payload: dict[str, object] = {
            "payload_truncated": True,
            "payload_bytes": payload_size,
            "payload_sha256": payload_hash,
            "payload_preview": "",
        }
        base_json = self._payload_json(base_payload)
        preview_budget = max(0, self.max_payload_bytes - len(base_json.encode("utf-8")) - 16)
        preview = payload_json[:preview_budget]
        bounded_payload = {**base_payload, "payload_preview": preview}
        bounded_json = self._payload_json(bounded_payload)
        while len(bounded_json.encode("utf-8")) > self.max_payload_bytes and preview:
            preview = preview[: max(0, len(preview) - 128)]
            bounded_payload["payload_preview"] = preview
            bounded_json = self._payload_json(bounded_payload)
        return bounded_json, payload_hash

    def load_payload_blob(self, payload_hash: str) -> Mapping[str, object] | Sequence[object] | None:
        row = self.connect().execute(
            "SELECT payload_json FROM raw_payload_blobs WHERE payload_hash = ?",
            (str(payload_hash),),
        ).fetchone()
        if row is None:
            blob_path = self._blob_path(str(payload_hash))
            if blob_path.exists():
                try:
                    return cast(Mapping[str, object] | Sequence[object], json.loads(blob_path.read_text(encoding="utf-8")))
                except (OSError, json.JSONDecodeError):
                    return None
            return None
        try:
            return cast(Mapping[str, object] | Sequence[object], json.loads(str(row["payload_json"])))
        except json.JSONDecodeError:
            return None

    def _blob_path(self, payload_hash: str) -> Path:
        safe_hash = "".join(ch for ch in str(payload_hash).lower() if ch in "0123456789abcdef")
        if len(safe_hash) < 8:
            safe_hash = hashlib.sha256(str(payload_hash).encode("utf-8")).hexdigest()
        return self.path.parent / f"{self.path.stem}_raw_payloads" / safe_hash[:2] / f"{safe_hash}.json"

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
        payload_json, payload_hash = self._bounded_payload_json(payload)
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
        for payload in raw_payloads:
            raw_horizon = ""
            raw_score: float | None = None
            raw_confidence: float | None = None
            observed_at_ms = int(getattr(report, "known_at_ms", self._now_ms()))
            if isinstance(payload, Mapping):
                source = str(payload.get("provider") or payload.get("source") or _UNATTRIBUTED_RAW_SOURCE)
                record_payload: Mapping[str, object] | Sequence[object] = payload
                raw_horizon = str(payload.get("horizon") or "")
                observed_value = payload.get("known_at_ms") or payload.get("observed_at_ms")
                if observed_value is not None:
                    try:
                        observed_at_ms = int(float(observed_value))
                    except (TypeError, ValueError):
                        observed_at_ms = int(getattr(report, "known_at_ms", self._now_ms()))
                if payload.get("score") is not None:
                    try:
                        raw_score = float(payload.get("score"))
                    except (TypeError, ValueError):
                        raw_score = None
                confidence_value = payload.get("urgency") if payload.get("urgency") is not None else payload.get("confidence")
                if confidence_value is not None:
                    try:
                        raw_confidence = float(confidence_value)
                    except (TypeError, ValueError):
                        raw_confidence = None
            elif isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
                source = _UNATTRIBUTED_RAW_SOURCE
                record_payload = payload
            else:
                source = _UNATTRIBUTED_RAW_SOURCE
                record_payload = {"value": str(payload)}
            inserted += 1 if self.record_observation(
                kind="raw_provider_payload",
                source=source,
                payload=record_payload,
                observed_at_ms=observed_at_ms,
                horizon=raw_horizon,
                score=raw_score,
                confidence=raw_confidence,
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

    def prune_raw_observations(
        self,
        *,
        before_ms: int | None = None,
        keep_latest: int | None = None,
    ) -> int:
        """Delete old raw observations by timestamp and/or keep only the newest rows."""
        deleted = 0
        conn = self.connect()
        if before_ms is not None:
            cursor = conn.execute("DELETE FROM raw_observations WHERE observed_at_ms < ?", (int(before_ms),))
            deleted += max(0, int(cursor.rowcount))
        if keep_latest is not None:
            keep = max(0, int(keep_latest))
            cursor = conn.execute(
                """
                DELETE FROM raw_observations
                WHERE id NOT IN (
                    SELECT id FROM raw_observations
                    ORDER BY observed_at_ms DESC, id DESC
                    LIMIT ?
                )
                """,
                (keep,),
            )
            deleted += max(0, int(cursor.rowcount))
        conn.commit()
        return deleted

    def source_rollups(self, *, since_ms: int, until_ms: int) -> list[dict[str, object]]:
        outcome_placeholders = ",".join("?" for _kind in _OUTCOME_KINDS)
        rows = self.connect().execute(
            f"""
            SELECT source, horizon, COUNT(*) AS sample_count,
                   AVG(COALESCE(score, 0.0)) AS avg_score,
                   AVG(ABS(COALESCE(score, 0.0))) AS avg_abs_score,
                   AVG(COALESCE(confidence, 0.0)) AS avg_confidence,
                   SUM(CASE WHEN kind = 'raw_provider_payload' THEN 1 ELSE 0 END) AS raw_records,
                   SUM(CASE WHEN kind = 'external_signal_component' THEN 1 ELSE 0 END) AS component_records
            FROM raw_observations
            WHERE observed_at_ms >= ? AND observed_at_ms <= ?
              AND source <> ?
              AND source NOT GLOB 'raw_[0-9]*'
              AND kind NOT IN ({outcome_placeholders})
            GROUP BY source, horizon
            ORDER BY sample_count DESC, source ASC
            """,
            (int(since_ms), int(until_ms), _UNATTRIBUTED_RAW_SOURCE, *_OUTCOME_KINDS),
        ).fetchall()
        outcome_rows = self.connect().execute(
            f"""
            SELECT source, horizon, score, payload_json
            FROM raw_observations
            WHERE observed_at_ms >= ? AND observed_at_ms <= ?
              AND source <> ?
              AND source NOT GLOB 'raw_[0-9]*'
              AND kind IN ({outcome_placeholders})
            """,
            (int(since_ms), int(until_ms), _UNATTRIBUTED_RAW_SOURCE, *_OUTCOME_KINDS),
        ).fetchall()
        outcomes: dict[tuple[str, str], dict[str, float]] = {}
        for row in outcome_rows:
            key = (str(row["source"]), str(row["horizon"] or "medium"))
            payload: object
            try:
                payload = json.loads(str(row["payload_json"]))
            except json.JSONDecodeError:
                payload = {}
            correct: bool | None = None
            if isinstance(payload, Mapping):
                for field in ("direction_correct", "correct", "hit"):
                    if isinstance(payload.get(field), bool):
                        correct = bool(payload[field])
                        break
                if correct is None:
                    prediction = payload.get("prediction_score", payload.get("score", row["score"]))
                    realized = payload.get("realized_return", payload.get("future_return"))
                    try:
                        prediction_value = float(prediction)
                        realized_value = float(realized)
                        if prediction_value != 0.0 and realized_value != 0.0:
                            correct = (prediction_value > 0.0) == (realized_value > 0.0)
                    except (TypeError, ValueError, OverflowError):
                        correct = None
            if correct is None:
                continue
            bucket = outcomes.setdefault(key, {"total": 0.0, "hits": 0.0})
            bucket["total"] += 1.0
            bucket["hits"] += 1.0 if correct else 0.0
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
                "outcome_records": int(outcomes.get((str(row["source"]), str(row["horizon"] or "medium")), {}).get("total", 0.0)),
                "directional_accuracy": (
                    None
                    if not outcomes.get((str(row["source"]), str(row["horizon"] or "medium")), {}).get("total", 0.0)
                    else outcomes[(str(row["source"]), str(row["horizon"] or "medium"))]["hits"]
                    / outcomes[(str(row["source"]), str(row["horizon"] or "medium"))]["total"]
                ),
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
        """Return the newest grade per source/horizon, optionally age-limited."""
        params: list[object] = []
        clauses: list[str] = []
        query = """
            SELECT id, created_at_ms, source, horizon, window_start_ms, window_end_ms,
                   grade, sample_count, model, reason, evidence_json
            FROM source_grades
            """
        if max_age_ms is not None or now_ms is not None:
            reference_ms = self._now_ms() if now_ms is None else int(now_ms)
            upper_ms = reference_ms + _MAX_FUTURE_TIMESTAMP_SKEW_MS
            clauses.extend(["created_at_ms <= ?", "window_end_ms <= ?"])
            params.extend([upper_ms, upper_ms])
            if max_age_ms is not None:
                lower_ms = reference_ms - max(0, int(max_age_ms))
                clauses.extend(["created_at_ms >= ?", "window_end_ms >= ?"])
                params.extend([lower_ms, lower_ms])
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
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
