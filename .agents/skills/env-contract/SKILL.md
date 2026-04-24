---
name: env-contract
description: Rules for environment-driven configuration — template-backed, typed, documented, never hard-coded.
origin: adapted from ZMB-UZH/omero-docker-extended env-contract-reviewer
---

# Env Contract

Use this skill whenever you add, rename, or remove an environment variable, or change how config is loaded.

## Rules

- All environment variables read by the trading CLI are listed in `.env.example` at the repo root. That file is the canonical tracked contract.
- Variables are trimmed (`.strip()`) before use. An empty string means "unset".
- Non-example env files (`.env`, `deploy.env`, etc.) are never created, edited, or read by the repo's tooling. They are developer-local, and git ignores them.
- **Never** duplicate a default in both Dockerfile `ARG` and `.env.example` and `docker-compose.yml`. Pick one source of truth (usually `.env.example`).
- **Never** `eval`, `source`, or shell-interpolate an env file. Parse it as data.

## Required change pattern

1. Add the variable to `.env.example` with a one-line comment describing purpose and default.
2. Load it via the existing helper (e.g. `os.getenv("NAME", "").strip()`), with an explicit default in the caller.
3. Update the README "Host overrides" or "Environment" section with the new variable.
4. If the variable affects behavior, add a test that sets it and asserts the observable effect.

## Currently tracked variables

| Name | Purpose | Trust boundary |
|---|---|---|
| `BINANCE_BASE_URL` | Override both spot and futures base hosts (proxy/testing). | Trusted for the current session only. |
| `BINANCE_SPOT_BASE_URL` | Override spot base host. | Trusted for the current session only. |
| `BINANCE_FUTURES_BASE_URL` | Override futures base host. | Trusted for the current session only. |

Adding a new entry to this table is part of the change, not a follow-up.

## Don't

- Don't read secrets from env at import time. Lazy-read inside the function that needs them, so tests can patch cleanly.
- Don't let an env override silently promote to the persisted config. Persisted config (`runtime.json`) is user-owned; env overrides are session-scoped.
