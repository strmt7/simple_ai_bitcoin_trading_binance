from __future__ import annotations

import json
import sqlite3

import pytest

from simple_ai_bitcoin_trading_binance import source_grading
from simple_ai_bitcoin_trading_binance.source_grading import grade_sources, render_source_grade_run
from simple_ai_bitcoin_trading_binance.telemetry_store import TradingTelemetryStore


NOW_MS = 1_700_000_000_000


def test_telemetry_store_roundtrip_and_grading_with_ai(tmp_path) -> None:
    db = tmp_path / "telemetry.sqlite"
    with TradingTelemetryStore(db) as store:
        assert store.connect() is store.connect()
        first = store.record_observation(
            kind="external_signal_component",
            source="cointelegraph",
            payload={"provider": "cointelegraph", "score": 0.7},
            observed_at_ms=NOW_MS,
            horizon="short",
            score=0.7,
            confidence=0.9,
        )
        duplicate = store.record_observation(
            kind="external_signal_component",
            source="cointelegraph",
            payload={"provider": "cointelegraph", "score": 0.7},
            observed_at_ms=NOW_MS,
            horizon="short",
            score=0.7,
            confidence=0.9,
        )
        store.record_observation(
            kind="raw_provider_payload",
            source="cointelegraph",
            payload={"raw_xml": "<rss />"},
            observed_at_ms=NOW_MS,
            horizon="short",
        )
        assert store.record_signal_report(
            type("Report", (), {"known_at_ms": NOW_MS, "components": [{"provider": "dict_component", "score": 0.1}]})(),
            raw_payloads=[
                {
                    "provider": "raw_feed",
                    "horizon": "short",
                    "score": 0.2,
                    "urgency": 0.4,
                    "payload": {"headline": "Bitcoin ETF flows rise"},
                },
                {"provider": "raw_bad", "score": "bad", "urgency": "bad"},
                ["list"],
                "scalar",
            ],
        ) == 5
        observations = store.recent_observations(since_ms=NOW_MS - 1, limit=10)
        filtered = store.recent_observations(since_ms=NOW_MS - 1, kind="external_signal_component")
        rollups = store.source_rollups(since_ms=NOW_MS - 1, until_ms=NOW_MS + 1)
        store.close()
        store.close()
    assert duplicate == first
    assert len(observations) == 7
    assert observations[0].asdict()["kind"] in {"external_signal_component", "raw_provider_payload"}
    assert all(item.kind == "external_signal_component" for item in filtered)
    assert rollups[0]["source"] == "cointelegraph"
    assert rollups[0]["sample_count"] == 2
    rollup_by_source = {str(item["source"]): item for item in rollups}
    assert set(rollup_by_source) == {"cointelegraph", "dict_component", "raw_bad", "raw_feed"}
    assert rollup_by_source["raw_feed"]["horizon"] == "short"
    assert rollup_by_source["raw_feed"]["avg_score"] == pytest.approx(0.2)
    assert rollup_by_source["raw_feed"]["avg_confidence"] == pytest.approx(0.4)

    def post_json(_url: str, payload: dict[str, object], _timeout: float):
        assert payload["model"] == "gemma4:e4b"
        assert payload["keep_alive"] == "30m"
        assert payload["think"] is False
        assert payload["options"]["num_ctx"] == 1024
        assert payload["options"]["num_predict"] == 128
        assert "cointelegraph|short" in str(payload["messages"])
        return {
            "message": {
                "content": json.dumps(
                    {
                        "grades": {
                            "cointelegraph|short": 9,
                            "dict_component|medium": 5,
                            "raw_bad|medium": 4,
                            "raw_feed|short": 7,
                        }
                    }
                )
            }
        }

    run = grade_sources(
        db_path=db,
        window_hours=1,
        model="gemma4:e4b",
        post_json=post_json,
        now_ms=NOW_MS + 1,
    )
    assert run.status == "ok"
    assert run.ai_status == "ok"
    cointelegraph = [grade for grade in run.grades if grade.source == "cointelegraph"][0]
    assert cointelegraph.grade == 9
    assert all(not (grade.source.startswith("raw_") and grade.source[4:].isdigit()) for grade in run.grades)
    assert "unattributed_raw_payload" not in {grade.source for grade in run.grades}
    assert run.asdict()["graded_sources"] >= 1
    assert cointelegraph.asdict()["source"] == "cointelegraph"
    with TradingTelemetryStore(db) as store:
        recent_grades = store.recent_grades(limit=2)
    assert len(recent_grades) == 2
    assert recent_grades[0].model == "gemma4:e4b"
    assert "cointelegraph" in render_source_grade_run(run)


def test_source_rollups_include_directional_outcomes(tmp_path) -> None:
    db = tmp_path / "telemetry.sqlite"
    with TradingTelemetryStore(db) as store:
        store.record_observation(
            kind="external_signal_component",
            source="cointelegraph",
            payload={"provider": "cointelegraph", "score": 0.7},
            observed_at_ms=NOW_MS,
            horizon="short",
            score=0.7,
            confidence=0.9,
        )
        store.record_observation(
            kind="raw_provider_payload",
            source="cointelegraph",
            payload={"raw_xml": "<rss />"},
            observed_at_ms=NOW_MS,
            horizon="short",
        )
        store.record_observation(
            kind="external_signal_outcome",
            source="cointelegraph",
            payload={"prediction_score": 0.7, "future_return": 0.01},
            observed_at_ms=NOW_MS + 1,
            horizon="short",
        )
        store.record_observation(
            kind="source_outcome",
            source="cointelegraph",
            payload={"direction_correct": False},
            observed_at_ms=NOW_MS + 2,
            horizon="short",
        )
        store.record_observation(
            kind="signal_outcome",
            source="cointelegraph",
            payload={"correct": True},
            observed_at_ms=NOW_MS + 3,
            horizon="short",
        )
        store.record_observation(
            kind="signal_outcome",
            source="cointelegraph",
            payload=["not-a-mapping"],
            observed_at_ms=NOW_MS + 4,
            horizon="short",
        )
        store.record_observation(
            kind="signal_outcome",
            source="cointelegraph",
            payload={"prediction_score": "bad", "future_return": 0.01},
            observed_at_ms=NOW_MS + 5,
            horizon="short",
        )
        store.record_observation(
            kind="signal_outcome",
            source="cointelegraph",
            payload={"prediction_score": 0.0, "future_return": 0.01},
            observed_at_ms=NOW_MS + 6,
            horizon="short",
        )
        store.connect().execute(
            """
            INSERT INTO raw_observations (
                observed_at_ms, kind, source, symbol, horizon, score, confidence,
                payload_hash, payload_json, created_at_ms
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (NOW_MS + 7, "signal_outcome", "cointelegraph", "BTCUSDC", "short", None, None, "bad-json", "{", NOW_MS + 7),
        )
        store.connect().commit()
        rollup = store.source_rollups(since_ms=NOW_MS - 1, until_ms=NOW_MS + 10)[0]

    assert rollup["sample_count"] == 2
    assert rollup["outcome_records"] == 3
    assert rollup["directional_accuracy"] == pytest.approx(2 / 3)
    good_grade, good_reason = source_grading._heuristic_grade(rollup)
    bad_grade, _bad_reason = source_grading._heuristic_grade({**rollup, "directional_accuracy": 0.0})
    invalid_grade, invalid_reason = source_grading._heuristic_grade({**rollup, "directional_accuracy": object()})
    assert good_grade > bad_grade
    assert invalid_grade >= 0
    assert "directional_accuracy" not in invalid_reason
    assert "directional_accuracy=0.67" in good_reason


def test_telemetry_store_bounds_payloads_and_prunes(tmp_path) -> None:
    db = tmp_path / "bounded.sqlite"
    with TradingTelemetryStore(db, max_payload_bytes=512) as store:
        store.record_observation(
            kind="raw_provider_payload",
            source="large_feed",
            payload={"blob": "x" * 5_000},
            observed_at_ms=NOW_MS - 10_000,
        )
        store.record_observation(
            kind="raw_provider_payload",
            source="fresh_feed",
            payload={"value": "fresh"},
            observed_at_ms=NOW_MS,
        )
        large = [
            item
            for item in store.recent_observations(since_ms=NOW_MS - 20_000, limit=10)
            if item.source == "large_feed"
        ][0]
        assert isinstance(large.payload, dict)
        assert large.payload["payload_truncated"] is True
        assert int(large.payload["payload_bytes"]) > 512
        compact = json.dumps(large.payload, sort_keys=True, separators=(",", ":"))
        assert len(compact.encode("utf-8")) <= 512
        assert store.load_payload_blob(str(large.payload["payload_sha256"])) == {"blob": "x" * 5_000}
        bounded_json, payload_hash = store._bounded_payload_json({"blob": "x" * 50_000})
        assert len(bounded_json.encode("utf-8")) <= store.max_payload_bytes
        assert store.load_payload_blob(payload_hash) == {"blob": "x" * 50_000}
        assert store.load_payload_blob("missing") is None
        bad_blob = store._blob_path("bad")
        bad_blob.parent.mkdir(parents=True, exist_ok=True)
        bad_blob.write_text("not-json", encoding="utf-8")
        assert store.load_payload_blob("bad") is None
        store.connect().execute(
            "INSERT OR REPLACE INTO raw_payload_blobs(payload_hash, payload_json, payload_bytes, created_at_ms) "
            "VALUES (?, ?, ?, ?)",
            ("deadbeef", "{", 1, NOW_MS),
        )
        store.connect().commit()
        assert store.load_payload_blob("deadbeef") is None

        tiny_store = TradingTelemetryStore(tmp_path / "tiny.sqlite", max_payload_bytes=180)
        tiny_json, tiny_hash = tiny_store._bounded_payload_json({"blob": "x" * 5_000})
        assert len(tiny_json.encode("utf-8")) <= tiny_store.max_payload_bytes
        assert tiny_hash
        escaped_json, _escaped_hash = tiny_store._bounded_payload_json({"blob": '"' * 5_000})
        assert len(escaped_json.encode("utf-8")) <= tiny_store.max_payload_bytes

        assert store.prune_raw_observations(before_ms=NOW_MS - 1) == 1
        assert [item.source for item in store.recent_observations(since_ms=NOW_MS - 20_000, limit=10)] == ["fresh_feed"]
        store.record_observation(
            kind="raw_provider_payload",
            source="newest_feed",
            payload={"value": "newest"},
            observed_at_ms=NOW_MS + 1,
        )
        assert store.prune_raw_observations(keep_latest=1) == 1
        remaining = store.recent_observations(since_ms=NOW_MS - 20_000, limit=10)
    assert [item.source for item in remaining] == ["newest_feed"]


def test_telemetry_store_migrates_legacy_raw_payload_blob_table(tmp_path) -> None:
    db = tmp_path / "legacy.sqlite"
    conn = sqlite3.connect(db)
    try:
        conn.execute(
            "CREATE TABLE raw_payload_blobs ("
            "payload_hash TEXT PRIMARY KEY, "
            "payload_json TEXT NOT NULL, "
            "created_at_ms INTEGER NOT NULL)"
        )
        conn.commit()
    finally:
        conn.close()

    with TradingTelemetryStore(db) as store:
        columns = {str(row["name"]) for row in store.connect().execute("PRAGMA table_info(raw_payload_blobs)").fetchall()}

    assert "payload_bytes" in columns


def test_telemetry_raw_payload_bad_observed_time_falls_back_to_report_time(tmp_path) -> None:
    db = tmp_path / "bad-time.sqlite"
    with TradingTelemetryStore(db) as store:
        store.record_signal_report(
            type("Report", (), {"known_at_ms": NOW_MS, "components": []})(),
            raw_payloads=[{"provider": "bad_time", "known_at_ms": "not-a-number", "payload": {"x": 1}}],
        )
        observations = store.recent_observations(since_ms=NOW_MS - 1, limit=10)

    assert observations[0].source == "bad_time"
    assert observations[0].observed_at_ms == NOW_MS


def test_latest_source_grades_selects_latest_and_filters_stale(tmp_path) -> None:
    db = tmp_path / "latest-grades.sqlite"
    with TradingTelemetryStore(db) as store:
        old_duplicate = store.record_source_grade(
            source="coingecko_bitcoin",
            horizon="medium",
            window_start_ms=NOW_MS - 7_200_000,
            window_end_ms=NOW_MS - 3_600_000,
            grade=2,
            sample_count=4,
            model="heuristic",
            reason="old",
            evidence={},
        )
        fresh_duplicate = store.record_source_grade(
            source="coingecko_bitcoin",
            horizon="medium",
            window_start_ms=NOW_MS - 3_600_000,
            window_end_ms=NOW_MS - 1,
            grade=8,
            sample_count=9,
            model="gemma4:e4b",
            reason="fresh",
            evidence={"latency_ms": 42},
        )
        stale_only = store.record_source_grade(
            source="old_feed",
            horizon="long",
            window_start_ms=NOW_MS - 7_200_000,
            window_end_ms=NOW_MS - 3_600_000,
            grade=9,
            sample_count=2,
            model="heuristic",
            reason="stale",
            evidence={},
        )
        future_only = store.record_source_grade(
            source="future_feed",
            horizon="short",
            window_start_ms=NOW_MS,
            window_end_ms=NOW_MS + 3_600_000,
            grade=10,
            sample_count=1,
            model="heuristic",
            reason="future",
            evidence={},
        )
        stale_window = store.record_source_grade(
            source="stale_window",
            horizon="medium",
            window_start_ms=NOW_MS - 3_600_000,
            window_end_ms=NOW_MS - 10_000,
            grade=7,
            sample_count=1,
            model="heuristic",
            reason="stale window",
            evidence={},
        )
        store.connect().executemany(
            "UPDATE source_grades SET created_at_ms = ? WHERE id = ?",
            [
                (NOW_MS - 5_000, old_duplicate.id),
                (NOW_MS - 1_000, fresh_duplicate.id),
                (NOW_MS - 5_000, stale_only.id),
                (NOW_MS + 3_600_000, future_only.id),
                (NOW_MS - 1_000, stale_window.id),
            ],
        )
        store.connect().commit()
        latest = store.latest_source_grades()
        fresh = store.latest_source_grades(max_age_ms=2_000, now_ms=NOW_MS)

    assert latest[("coingecko_bitcoin", "medium")].grade == 8
    assert latest[("old_feed", "long")].grade == 9
    assert latest[("future_feed", "short")].grade == 10
    assert fresh[("coingecko_bitcoin", "medium")].grade == 8
    assert ("old_feed", "long") not in fresh
    assert ("future_feed", "short") not in fresh
    assert ("stale_window", "medium") not in fresh


def test_source_grading_empty_and_ollama_fallback(tmp_path) -> None:
    empty = grade_sources(db_path=tmp_path / "empty.sqlite", window_hours=1, ollama_enabled=False, now_ms=NOW_MS)
    assert empty.status == "empty"
    assert "no telemetry" in empty.warnings[0]

    db = tmp_path / "fallback.sqlite"
    with TradingTelemetryStore(db) as store:
        store.record_observation(
            kind="external_signal_component",
            source="internal_model",
            payload={"score": -0.2},
            observed_at_ms=NOW_MS,
            horizon="short",
            score=-0.2,
            confidence=0.4,
        )

    fallback = grade_sources(
        db_path=db,
        window_hours=1,
        ollama_enabled=True,
        post_json=lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("offline")),
        now_ms=NOW_MS + 1,
    )
    assert fallback.status == "warn"
    assert fallback.ai_status == "error"
    assert fallback.model == "heuristic"
    assert 0 <= fallback.grades[0].grade <= 10
    assert "warning:" in render_source_grade_run(fallback)
    heuristic = grade_sources(db_path=db, window_hours=1, ollama_enabled=False, now_ms=NOW_MS + 1)
    assert heuristic.ai_status == "disabled"
    assert heuristic.status == "ok"

    calls = {"count": 0}

    def post_json_missing(*_args, **_kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return {"response": json.dumps({"grades": {}})}
        return []

    partial = grade_sources(
        db_path=db,
        window_hours=1,
        ollama_enabled=True,
        post_json=post_json_missing,
        now_ms=NOW_MS + 2,
    )
    assert partial.status == "warn"
    assert partial.ai_status == "ok"
    assert partial.grades[0].model == "heuristic"
    assert "missed" in partial.warnings[0]


def test_source_grading_respects_ai_disable_and_rollup_budget(tmp_path) -> None:
    db = tmp_path / "budget.sqlite"
    with TradingTelemetryStore(db) as store:
        for index in range(4):
            store.record_observation(
                kind="external_signal_component",
                source=f"source_{index}",
                payload={"score": 0.1 * index},
                observed_at_ms=NOW_MS,
                horizon="short",
                score=0.1 * index,
                confidence=0.5,
            )

    disabled = grade_sources(
        db_path=db,
        window_hours=1,
        ollama_enabled=False,
        post_json=lambda *_args, **_kwargs: pytest.fail("Ollama should stay disabled"),
        now_ms=NOW_MS + 1,
    )
    assert disabled.ai_status == "disabled"
    assert disabled.status == "ok"
    assert {grade.model for grade in disabled.grades} == {"heuristic"}

    model_disabled = grade_sources(
        db_path=db,
        window_hours=1,
        model="off",
        ollama_enabled=True,
        post_json=lambda *_args, **_kwargs: pytest.fail("Ollama should stay model-disabled"),
        now_ms=NOW_MS + 1,
    )
    assert model_disabled.ai_status == "disabled"

    calls = {"count": 0}

    def post_json(_url: str, payload: dict[str, object], _timeout: float):
        calls["count"] += 1
        messages = str(payload["messages"])
        assert "source_0|short" in messages
        assert "source_1|short" in messages
        assert "source_2|short" not in messages
        return {"response": json.dumps({"grades": {"source_0|short": 8, "source_1|short": 7}})}

    budgeted = grade_sources(
        db_path=db,
        window_hours=1,
        model="gemma4:e4b",
        ollama_enabled=True,
        post_json=post_json,
        ai_rollup_limit=2,
        ai_max_batches=1,
        ai_max_single_fills=0,
        now_ms=NOW_MS + 2,
    )
    models = {grade.source: grade.model for grade in budgeted.grades}
    assert calls["count"] == 1
    assert budgeted.status == "warn"
    assert budgeted.ai_status == "ok"
    assert any("limited to 2 of 4" in warning for warning in budgeted.warnings)
    assert models["source_0"] == "gemma4:e4b"
    assert models["source_1"] == "gemma4:e4b"
    assert models["source_2"] == "heuristic"
    assert models["source_3"] == "heuristic"

    skipped = grade_sources(
        db_path=db,
        window_hours=1,
        model="gemma4:e4b",
        ollama_enabled=True,
        ollama_timeout_seconds=0.0,
        post_json=lambda *_args, **_kwargs: pytest.fail("Ollama should be skipped"),
        now_ms=NOW_MS + 3,
    )
    assert skipped.ai_status == "skipped"
    assert any("non-positive timeout" in warning for warning in skipped.warnings)

    zero_budget = grade_sources(
        db_path=db,
        window_hours=1,
        model="gemma4:e4b",
        ollama_enabled=True,
        post_json=lambda *_args, **_kwargs: pytest.fail("Ollama should be budget-skipped"),
        ai_rollup_limit=0,
        now_ms=NOW_MS + 4,
    )
    assert zero_budget.ai_status == "skipped"
    assert any("source horizon budget is zero" in warning for warning in zero_budget.warnings)


def test_source_grading_helper_edges(monkeypatch) -> None:
    assert isinstance(source_grading._now_ms(), int)
    assert source_grading._clamp_int("bad", 0, 10, 5) == 5
    assert source_grading._clamp_int(99, 0, 10, 5) == 10
    assert source_grading._ollama_grading_requested(enabled=True, model="off", timeout_seconds=1.0) == (
        False,
        "disabled",
    )
    assert source_grading._ollama_grading_requested(enabled=True, model="gemma4:e4b", timeout_seconds=0.0) == (
        False,
        "skipped",
    )
    assert source_grading._recover_grade_mapping("no grades") == {}
    assert source_grading._recover_grade_mapping('{"grades":{"x|medium":8,"y|short":10}}') == {
        "x|medium": 8,
        "y|short": 10,
    }
    assert source_grading._json_mapping_from_text("prefix {\"grades\": []} suffix") == {"grades": []}
    with pytest.raises(json.JSONDecodeError):
        source_grading._json_mapping_from_text("no json")
    with pytest.raises(ValueError, match="JSON object"):
        source_grading._json_mapping_from_text("[1]")
    rollups = [
        {
            "source": "x",
            "horizon": "medium",
            "sample_count": 1,
            "avg_score": 0.0,
            "avg_abs_score": 0.0,
            "avg_confidence": 0.0,
            "raw_records": 0,
            "component_records": 1,
        }
    ]
    with pytest.raises(ValueError, match="unexpected Ollama"):
        source_grading._ai_grades(
            rollups,
            model="gemma4:e4b",
            base_url="http://localhost:11434",
            timeout_seconds=1.0,
            post_json=lambda *_args, **_kwargs: [],
        )
    with pytest.raises(ValueError, match="missed grades"):
        source_grading._ai_grades(
            rollups,
            model="gemma4:e4b",
            base_url="http://localhost:11434",
            timeout_seconds=1.0,
            post_json=lambda *_args, **_kwargs: {"response": "{}"},
        )
    grades, _latency = source_grading._ai_grades(
        rollups,
        model="gemma4:e4b",
        base_url="http://localhost:11434/",
        timeout_seconds=1.0,
        post_json=lambda *_args, **_kwargs: {
            "response": json.dumps(
                {
                    "grades": [
                        "skip",
                        ["too-short"],
                        ["list_source", "", 7],
                        {"source": "", "grade": 1},
                        {"source": "x", "grade": "bad", "reason": ""},
                    ]
                }
            )
        },
    )
    assert grades[("x", "medium")] == (5, "AI grade")
    assert grades[("list_source", "medium")] == (7, "AI grade")
    single_key, single_grade, _latency = source_grading._ai_grade_single(
        {**rollups[0], "source": "single", "horizon": "short"},
        model="gemma4:e4b",
        base_url="http://localhost:11434",
        timeout_seconds=1.0,
        post_json=lambda *_args, **_kwargs: {"response": "grade: 6"},
    )
    assert single_key == ("single", "short")
    assert single_grade == (6, "AI single-source grade")
    with pytest.raises(ValueError, match="unexpected Ollama"):
        source_grading._ai_grade_single(
            rollups[0],
            model="gemma4:e4b",
            base_url="http://localhost:11434",
            timeout_seconds=1.0,
            post_json=lambda *_args, **_kwargs: [],
        )
    with pytest.raises(json.JSONDecodeError):
        source_grading._ai_grade_single(
            rollups[0],
            model="gemma4:e4b",
            base_url="http://localhost:11434",
            timeout_seconds=1.0,
            post_json=lambda *_args, **_kwargs: {"response": "not parseable"},
        )
    recovered, _latency = source_grading._ai_grades(
        rollups,
        model="gemma4:e4b",
        base_url="http://localhost:11434",
        timeout_seconds=1.0,
        post_json=lambda *_args, **_kwargs: {
            "response": '{"grades":{"x|medium":8,"list_source|medium":7,"bad":"'
        },
    )
    assert recovered[("x", "medium")] == (8, "AI grade (recovered JSON)")
    assert recovered[("list_source", "medium")] == (7, "AI grade (recovered JSON)")
    two_rollups = [
        rollups[0],
        {
            "source": "missing",
            "horizon": "short",
            "sample_count": 2,
            "avg_score": 0.1,
            "avg_abs_score": 0.1,
            "avg_confidence": 0.2,
            "raw_records": 1,
            "component_records": 1,
        },
    ]
    calls = {"count": 0}

    def post_json_partial(*_args, **_kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return {"response": json.dumps({"grades": {"x|medium": 8}})}
        return {"response": json.dumps({"grade": 9, "reason": "single fill"})}

    filled, _latency = source_grading._ai_grades(
        two_rollups,
        model="gemma4:e4b",
        base_url="http://localhost:11434",
        timeout_seconds=1.0,
        post_json=post_json_partial,
    )
    assert filled[("x", "medium")] == (8, "AI grade")
    assert filled[("missing", "short")] == (9, "single fill")
    calls["count"] = 0

    def post_json_single_failure(*_args, **_kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return {"response": json.dumps({"grades": {"x|medium": 8}})}
        return []

    partial_fill, _latency = source_grading._ai_grades(
        two_rollups,
        model="gemma4:e4b",
        base_url="http://localhost:11434",
        timeout_seconds=1.0,
        post_json=post_json_single_failure,
    )
    assert partial_fill == {("x", "medium"): (8, "AI grade")}
    many_rollups = [
        {**rollups[0], "source": f"source_{index}", "horizon": "medium"}
        for index in range(source_grading._AI_SINGLE_FILL_LIMIT + 2)
    ]
    single_calls = {"count": 0}

    def post_json_limited_singles(_url, payload, *_args, **_kwargs):
        single_calls["count"] += 1
        if payload["options"]["num_ctx"] == 1024:
            if single_calls["count"] == 1:
                return {"response": json.dumps({"grades": {"source_0|medium": 8}})}
            return {"response": json.dumps({"grades": {}})}
        return {"response": json.dumps({"grade": 6, "reason": "limited fill"})}

    limited, _latency = source_grading._ai_grades(
        many_rollups,
        model="gemma4:e4b",
        base_url="http://localhost:11434",
        timeout_seconds=1.0,
        post_json=post_json_limited_singles,
    )
    assert len(limited) == source_grading._AI_SINGLE_FILL_LIMIT + 1
    assert ("source_0", "medium") in limited
    assert (f"source_{source_grading._AI_SINGLE_FILL_LIMIT + 1}", "medium") not in limited
    mapped, _latency = source_grading._ai_grades(
        rollups,
        model="gemma4:e4b",
        base_url="http://localhost:11434",
        timeout_seconds=1.0,
        post_json=lambda *_args, **_kwargs: {
            "response": json.dumps({"grades": {"x|medium": 8, "bad-key": 1}})
        },
    )
    assert mapped == {("x", "medium"): (8, "AI grade")}
    positional, _latency = source_grading._ai_grades(
        two_rollups,
        model="gemma4:e4b",
        base_url="http://localhost:11434",
        timeout_seconds=1.0,
        post_json=lambda *_args, **_kwargs: {
            "response": json.dumps({"grades": {"a": 8, "b": 2}})
        },
        max_single_fills=0,
    )
    assert positional == {
        ("x", "medium"): (8, "AI grade (positional recovery)"),
        ("missing", "short"): (2, "AI grade (positional recovery)"),
    }
    batch_calls = {"count": 0}
    two_batch_rollups = [
        {**rollups[0], "source": f"batch_{index}", "horizon": "medium"}
        for index in range(source_grading._AI_GRADE_BATCH_SIZE + 1)
    ]

    def post_json_second_batch_timeout(_url, payload, *_args, **_kwargs):
        batch_calls["count"] += 1
        if batch_calls["count"] == 1:
            return {"response": json.dumps({"grades": {"batch_0|medium": 8}})}
        raise source_grading.RequestException("late timeout")

    partial_batches, _latency = source_grading._ai_grades(
        two_batch_rollups,
        model="gemma4:e4b",
        base_url="http://localhost:11434",
        timeout_seconds=1.0,
        post_json=post_json_second_batch_timeout,
        max_single_fills=0,
    )
    assert partial_batches == {("batch_0", "medium"): (8, "AI grade")}
    assert source_grading._ai_grades(
        [{**rollups[0], "source": "budget_stop"}],
        model="gemma4:e4b",
        base_url="http://localhost:11434",
        timeout_seconds=1.0,
        post_json=lambda *_args, **_kwargs: pytest.fail("batch budget should stop before request"),
        max_batches=0,
    ) == ({}, 0)

    clock = {"now": 0.0}
    budget_calls = {"count": 0}

    def post_json_exhausts_budget(*_args, **_kwargs):
        budget_calls["count"] += 1
        clock["now"] = 1.0
        return {"response": json.dumps({"grades": {}})}

    monkeypatch.setattr(source_grading.time, "monotonic", lambda: clock["now"])
    budgeted, _latency = source_grading._ai_grades(
        two_rollups,
        model="gemma4:e4b",
        base_url="http://localhost:11434",
        timeout_seconds=1.0,
        post_json=post_json_exhausts_budget,
        max_total_seconds=0.5,
    )
    assert budgeted == {}
    assert budget_calls["count"] == 1

    times = iter([0.0, 1.0])
    monkeypatch.setattr(source_grading.time, "monotonic", lambda: next(times))
    assert source_grading._ai_grades(
        two_rollups,
        model="gemma4:e4b",
        base_url="http://localhost:11434",
        timeout_seconds=1.0,
        post_json=lambda *_args, **_kwargs: pytest.fail("expired aggregate budget should skip batch"),
        max_total_seconds=0.5,
    ) == ({}, 0)
