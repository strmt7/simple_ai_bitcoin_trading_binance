---
name: python-patterns
description: Python patterns for this repo — pure-stdlib trading core, typed dataclasses, no hidden dependencies, deterministic I/O.
origin: adapted from ZMB-UZH/omero-docker-extended python-patterns (ECC v1.10.0)
---

# Python Patterns

Use this skill whenever you add or refactor Python in `src/simple_ai_bitcoin_trading_binance/`, `tests/`, or `tools/`.

## Ground rules

- The trading core is **pure stdlib + `requests` + `textual`**. Do not introduce numpy, pandas, scikit-learn, or any ML framework; if you feel you need one, stop and ask first.
- Every module exposes **frozen dataclasses** for data-in-motion (`Candle`, `ModelRow`, `BacktestResult`, `ClassificationReport`, …) and regular dataclasses for config. Keep that split.
- New public functions must carry type hints and a one-line docstring when the "why" is non-obvious. Do not document obvious behavior.
- Avoid hidden globals. Pass configuration explicitly. If you need a session-lived object (e.g. the shell state), inject it through the constructor.
- Prefer small files. Any module crossing ~300 logical lines should be split along clean responsibility lines (`api`, `features`, `model`, `backtest`, `shell`, `chart`, `style`, …).

## Trading-specific rules

- **Credential redaction is a contract**, not a formatting choice. Any payload that could be printed, logged, or persisted runs through the existing redaction helpers (`RuntimeConfig.public_dict`, `_redact_request_url`, `_redact_sensitive_text`). New printable surfaces that skip this are blocking bugs.
- **Deterministic artifacts**: JSON written under `data/` must use `sort_keys=True` and keep a stable top-level schema (`command`, `timestamp`, `runtime`, …). Never embed raw API keys.
- **No side effects at import time** — no network calls, no file writes, no environment variable mutation. Tests rely on import being free.
- **Safety defaults survive**: `testnet=True`, `dry_run=True`, BTCUSDC-only. A change that weakens any of these requires an explicit user opt-in path with a test.

## Formatting / lint

- Ruff is the lint + format baseline (see `pyproject.toml`). Run `ruff format` and `ruff check --fix` on touched files before committing.
- Python ≥ 3.11 is required. Use `X | Y` unions, `list[T]`, `dict[K, V]`, and `from __future__ import annotations` consistently.

## Don't

- Don't silently widen exception handling to make a test pass — catch only what you know you need to recover from.
- Don't add CLI output that isn't covered by a test. The repo runs `coverage --fail-under=100`; an uncovered branch will break CI, not just coverage drift.
- Don't stash config inside module-level mutable state — use `config_paths()` / `load_runtime()` each time.
