# Contributing

Small, correct, test-covered changes are preferred over sweeping rewrites.

## Ground rules

- Python ≥ 3.11.  Runtime dependencies are pinned in `pyproject.toml` — adding
  a new one needs a justification in the PR description.
- 100% branch coverage is enforced in CI.  Every new branch needs a direct
  test assertion.
- Testnet-first safety defaults are load-bearing.  A change that flips
  `testnet=True` or `dry_run=True` as a default is a contract break and must be
  flagged explicitly in the PR.
- Never commit secrets.  `.gitignore` excludes `.env` and the project configs
  are stored under `~/.config/simple_ai_bitcoin_trading_binance/` (mode 0600)
  at runtime.

## Local workflow

```bash
python3 -m venv .venv
.venv/bin/pip install -e .
.venv/bin/pip install pytest coverage ruff mypy

# Tests + coverage gate
.venv/bin/python -m coverage run --source=src/simple_ai_bitcoin_trading_binance --branch -m pytest -q
.venv/bin/python -m coverage report --fail-under=100

# Static checks on touched files
.venv/bin/ruff check src tests
.venv/bin/ruff format --check src tests
.venv/bin/mypy src
```

## Commit etiquette

- One logical change per commit.  Tests first (or alongside) the code.
- Subject line in imperative mood (`add`, `fix`, `refactor`), ≤ 72 chars.
- Body explains the *why* when it's non-obvious; the code already says the *what*.

## Pushing

If you need to push with a Personal Access Token, use the helper:

```bash
GITHUB_TOKEN=… python3 tools/push_with_pat.py origin my-branch
```

The helper serves your PAT to `git push` over a short-lived UNIX socket so it
never appears in `argv`, remotes, logs, or long-lived credential stores.
