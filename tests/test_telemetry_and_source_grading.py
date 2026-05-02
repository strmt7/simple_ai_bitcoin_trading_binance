from __future__ import annotations

import json

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
            raw_payloads=[["list"], "scalar"],
        ) == 3
        observations = store.recent_observations(since_ms=NOW_MS - 1, limit=10)
        filtered = store.recent_observations(since_ms=NOW_MS - 1, kind="external_signal_component")
        rollups = store.source_rollups(since_ms=NOW_MS - 1, until_ms=NOW_MS + 1)
        store.close()
        store.close()
    assert duplicate == first
    assert len(observations) == 5
    assert observations[0].asdict()["kind"] in {"external_signal_component", "raw_provider_payload"}
    assert all(item.kind == "external_signal_component" for item in filtered)
    assert rollups[0]["source"] == "cointelegraph"
    assert rollups[0]["sample_count"] == 2

    def post_json(_url: str, payload: dict[str, object], _timeout: float):
        assert payload["model"] == "gemma4:e4b"
        assert payload["keep_alive"] == "30m"
        assert payload["options"]["num_ctx"] == 2048
        assert payload["options"]["num_predict"] == 768
        assert "cointelegraph|short" in str(payload["prompt"])
        return {
            "response": json.dumps(
                {
                    "grades": {
                        "cointelegraph|short": 9,
                        "dict_component|medium": 5,
                        "raw_0|medium": 5,
                        "raw_1|medium": 5,
                    }
                }
            )
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
    assert run.asdict()["graded_sources"] >= 1
    assert cointelegraph.asdict()["source"] == "cointelegraph"
    with TradingTelemetryStore(db) as store:
        recent_grades = store.recent_grades(limit=2)
    assert len(recent_grades) == 2
    assert recent_grades[0].model == "gemma4:e4b"
    assert "cointelegraph" in render_source_grade_run(run)


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


def test_source_grading_helper_edges() -> None:
    assert isinstance(source_grading._now_ms(), int)
    assert source_grading._clamp_int("bad", 0, 10, 5) == 5
    assert source_grading._clamp_int(99, 0, 10, 5) == 10
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
