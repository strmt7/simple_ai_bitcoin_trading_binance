from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


_TOOL_PATH = Path(__file__).resolve().parents[1] / "tools" / "quality_metrics.py"
_SPEC = importlib.util.spec_from_file_location("quality_metrics", _TOOL_PATH)
assert _SPEC is not None and _SPEC.loader is not None
quality_metrics = importlib.util.module_from_spec(_SPEC)
sys.modules["quality_metrics"] = quality_metrics
_SPEC.loader.exec_module(quality_metrics)


def test_quality_metrics_counts_source_tests_complexity_and_cli_commands() -> None:
    blobs = [
        quality_metrics.FileBlob(
            "src/simple_ai_bitcoin_trading_binance/cli.py",
            "\n".join(
                [
                    "def build(subparsers):",
                    "    subparsers.add_parser('run')",
                    "    if subparsers:",
                    "        return True",
                    "    return False",
                ]
            ),
        ),
        quality_metrics.FileBlob(
            "src/simple_ai_bitcoin_trading_binance/model.py",
            "\n".join(
                [
                    "def score(value):",
                    "    return 1 if value > 0 else 0",
                ]
            ),
        ),
        quality_metrics.FileBlob("tests/test_cli.py", "def test_cli():\n    assert True\n"),
    ]

    metrics = quality_metrics.measure_blobs(blobs)

    assert metrics["source_files"] == 2
    assert metrics["test_files"] == 1
    assert metrics["source_lines"] == 7
    assert metrics["test_lines"] == 2
    assert metrics["largest_source_file"] == "src/simple_ai_bitcoin_trading_binance/cli.py"
    assert metrics["largest_source_file_lines"] == 5
    assert metrics["cli_lines"] == 5
    assert metrics["function_count"] == 2
    assert metrics["cli_command_count"] == 1
    assert metrics["max_cyclomatic_complexity"] >= 2
    assert metrics["parse_error_count"] == 0


def test_quality_metrics_reports_parse_errors_and_numeric_deltas() -> None:
    metrics = quality_metrics.measure_blobs(
        [
            quality_metrics.FileBlob("src/simple_ai_bitcoin_trading_binance/bad.py", "def broken(:\n"),
            quality_metrics.FileBlob("docs/ignore.py", "def ignored():\n    pass\n"),
        ]
    )
    comparison = quality_metrics.compare_metrics(
        {"source_files": 1, "source_lines": 10, "parse_errors": []},
        {"source_files": 2, "source_lines": 15, "parse_errors": []},
    )

    assert metrics["parse_error_count"] == 1
    assert metrics["parse_errors"] == ["src/simple_ai_bitcoin_trading_binance/bad.py:1"]
    assert comparison["source_files"]["delta"] == 1.0
    assert comparison["source_lines"]["percent"] == 50.0
