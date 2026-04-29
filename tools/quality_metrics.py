#!/usr/bin/env python3
"""Deterministic repository quality metrics for before/after refinement checks."""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PREFIX = "src/simple_ai_bitcoin_trading_binance/"
TEST_PREFIX = "tests/"
_CLI_COMMAND_RE = re.compile(r"\bsubparsers\.add_parser\(\s*[\"']([^\"']+)[\"']")
_FUNCTION_NODES = (ast.FunctionDef, ast.AsyncFunctionDef)
_DECISION_NODES = (
    ast.If,
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.ExceptHandler,
    ast.IfExp,
    ast.Assert,
    ast.comprehension,
)


@dataclass(frozen=True)
class FileBlob:
    path: str
    text: str


def _is_source_path(path: str) -> bool:
    return path.startswith(SRC_PREFIX) and path.endswith(".py")


def _is_test_path(path: str) -> bool:
    return path.startswith(TEST_PREFIX) and path.endswith(".py")


def _run_git(args: list[str]) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout


def _iter_working_tree_files() -> list[str]:
    files = []
    for prefix in (SRC_PREFIX, TEST_PREFIX):
        files.extend(path.relative_to(REPO_ROOT).as_posix() for path in (REPO_ROOT / prefix).rglob("*.py"))
    return sorted(path for path in files if _is_source_path(path) or _is_test_path(path))


def _iter_ref_files(ref: str) -> list[str]:
    paths = _run_git(["ls-tree", "-r", "--name-only", ref]).splitlines()
    return sorted(path for path in paths if _is_source_path(path) or _is_test_path(path))


def _load_blobs(ref: str | None) -> list[FileBlob]:
    if ref:
        return [
            FileBlob(path, _run_git(["show", f"{ref}:{path}"]))
            for path in _iter_ref_files(ref)
        ]
    return [
        FileBlob(path, (REPO_ROOT / path).read_text(encoding="utf-8"))
        for path in _iter_working_tree_files()
    ]


def _line_count(text: str) -> int:
    return len(text.splitlines())


def _function_complexity(node: ast.AST) -> int:
    complexity = 1
    for child in ast.walk(node):
        if isinstance(child, _DECISION_NODES):
            complexity += 1
        elif isinstance(child, ast.BoolOp):
            complexity += max(0, len(child.values) - 1)
        elif isinstance(child, ast.Match):
            complexity += len(child.cases)
    return complexity


def measure_blobs(blobs: Iterable[FileBlob]) -> dict[str, object]:
    source_files = [blob for blob in blobs if _is_source_path(blob.path)]
    test_files = [blob for blob in blobs if _is_test_path(blob.path)]
    source_line_counts = {blob.path: _line_count(blob.text) for blob in source_files}
    source_lines = sum(source_line_counts.values())
    test_lines = sum(_line_count(blob.text) for blob in test_files)
    function_lengths: list[int] = []
    complexities: list[int] = []
    parse_errors: list[str] = []
    cli_commands: set[str] = set()

    for blob in source_files:
        if blob.path.endswith("/cli.py"):
            cli_commands.update(_CLI_COMMAND_RE.findall(blob.text))
        try:
            tree = ast.parse(blob.text)
        except SyntaxError as exc:
            parse_errors.append(f"{blob.path}:{exc.lineno or 0}")
            continue
        for node in ast.walk(tree):
            if isinstance(node, _FUNCTION_NODES):
                end_lineno = getattr(node, "end_lineno", node.lineno)
                function_lengths.append(max(1, int(end_lineno) - int(node.lineno) + 1))
                complexities.append(_function_complexity(node))
    largest_source_path = max(source_line_counts, key=source_line_counts.get, default="")

    return {
        "source_files": len(source_files),
        "test_files": len(test_files),
        "source_lines": source_lines,
        "test_lines": test_lines,
        "test_to_source_line_ratio": round(test_lines / source_lines, 4) if source_lines else 0.0,
        "largest_source_file": largest_source_path,
        "largest_source_file_lines": source_line_counts.get(largest_source_path, 0),
        "cli_lines": source_line_counts.get(f"{SRC_PREFIX}cli.py", 0),
        "function_count": len(function_lengths),
        "avg_function_lines": round(mean(function_lengths), 2) if function_lengths else 0.0,
        "max_function_lines": max(function_lengths, default=0),
        "avg_cyclomatic_complexity": round(mean(complexities), 2) if complexities else 0.0,
        "max_cyclomatic_complexity": max(complexities, default=0),
        "cli_command_count": len(cli_commands),
        "parse_error_count": len(parse_errors),
        "parse_errors": parse_errors,
    }


def measure_repository(ref: str | None = None) -> dict[str, object]:
    return measure_blobs(_load_blobs(ref))


def compare_metrics(baseline: dict[str, object], current: dict[str, object]) -> dict[str, dict[str, float]]:
    comparison: dict[str, dict[str, float]] = {}
    for key, current_value in current.items():
        baseline_value = baseline.get(key)
        if isinstance(current_value, (int, float)) and isinstance(baseline_value, (int, float)):
            delta = float(current_value) - float(baseline_value)
            percent = (delta / float(baseline_value) * 100.0) if baseline_value else 0.0
            comparison[key] = {
                "baseline": float(baseline_value),
                "current": float(current_value),
                "delta": round(delta, 4),
                "percent": round(percent, 2),
            }
    return comparison


def _render_text(metrics: dict[str, object], comparison: dict[str, dict[str, float]] | None = None) -> str:
    lines = ["Repository quality metrics"]
    for key, value in metrics.items():
        if key == "parse_errors":
            continue
        lines.append(f"{key}: {value}")
    if comparison:
        lines.append("Comparison")
        for key, values in comparison.items():
            lines.append(
                f"{key}: baseline={values['baseline']:g} current={values['current']:g} "
                f"delta={values['delta']:g} percent={values['percent']:g}%"
            )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", help="measure a git ref instead of the working tree")
    parser.add_argument("--compare-ref", help="also compare against this git ref")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args(argv)

    current = measure_repository(args.ref)
    baseline = measure_repository(args.compare_ref) if args.compare_ref else None
    comparison = compare_metrics(baseline, current) if baseline is not None else None
    if args.json:
        payload: dict[str, object] = {"metrics": current}
        if comparison is not None:
            payload["comparison"] = comparison
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(_render_text(current, comparison))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
