# AGENTS

## Objective

Build and maintain a testnet-first BTCUSDC trading CLI that is conservative by default, fully test-covered, and safe for iterative development. Keep edits minimal, correct, and reproducible.

## Context-loading rules

Use this order before broad reads:

1. `README.md`
2. one nearest implementation file in `src/simple_ai_bitcoin_trading_binance/`
3. the matching test file in `tests/`
4. the closest repo-local skill in `.agents/skills/`

Do not expand to broad directory scans on first pass. Open more files only when the task cannot be completed safely with the above context.

## Mandatory constraints

- Never assume behavior from memory. Confirm by running tests or inspection of source.
- No network calls in tests unless explicitly stubbed.
- Preserve conservative defaults (`testnet`, `dry_run` behavior, BTCUSDC-only execution).
- Do not claim production readiness or profitability without reproducible evidence from test artifacts.
- Keep secrets out of prompts, logs, and history.
- Avoid unnecessary hardcoded host assumptions; prefer configuration or environment overrides when host selection can be made safely dynamic.
- Never print, serialize, commit, or echo raw credentials, API keys, secrets, tokens, or signed request material.
- Any runtime/config payload written to stdout, stderr, JSON artifacts, docs, or tests must use deterministic redaction for secret fields.
- Add or update tests whenever credential-handling code changes so that raw secret values are provably absent from outputs and artifacts.
- Keep changes in this repo scoped and avoid editing unrelated files.

## Verification minimum

- run `python3 -m pytest -q` after any behavior change.
- run focused regression tests matching the touched file(s) first.
- run `python3 -m coverage run --source=src/simple_ai_bitcoin_trading_binance -m pytest -q` before closing significant feature work, then inspect misses.
- for CLI behavior changes, run `python3 -m pytest -q tests/test_cli.py tests/test_cli_coverage.py`.
- for model or backtest changes, include both unit and coverage tests for that domain.

## Edge-case policy

- Every new branch should have a direct test assertion.
- Preserve exception behavior unless explicitly changing the contract.
- If the branch is error handling, test both normal and fallback paths.

## File map

- core: `src/simple_ai_bitcoin_trading_binance/`
- tests: `tests/`
- workflows: `.github/workflows/`
- agent process: `.agents/skills/` and this file

## Completion signal

- code changes are covered by unit tests and at least one validation run relevant to behavior.
- docs/tests/CI reflect any changed assumptions.
