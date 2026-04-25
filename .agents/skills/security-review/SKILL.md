---
name: security-review
description: Security checklist for credential handling, request signing, artifact redaction, and order-path guards before any broad change lands.
origin: adapted from ZMB-UZH/omero-docker-extended security-review for a Binance testnet CLI
---

# Security Review

Use this skill whenever you touch `api.py`, `config.py`, `cli.py`, live-order paths, persisted artifacts, or anything that reads env variables or prompts for secrets.

## Read order

1. The change itself (`git diff`).
2. `AGENTS.md` — single-session rule, redaction contract, testnet-first defaults.
3. Existing redaction helpers: `_redact_request_url`, `_redact_sensitive_text`, `RuntimeConfig.public_dict`.
4. The nearest affected test file — make sure it asserts on **absence** of secrets in output/artifacts.

## Always check

- **Signed requests** never leak `signature`, `timestamp`, `recvWindow`, or the raw API key into logs, error messages, `last_request_info`, or JSON artifacts. If you build a new URL by hand, redact it before storing.
- **Order rejections** must be caught and recorded as structured artifacts (`order_error`), never allowed to bubble up as raw exceptions that might contain signed material in the traceback.
- **Emergency / close-out orders** on futures must use `reduceOnly=true` and market type `RESULT`. A close path that could increase exposure is a bug.
- **Live mode preflight** fails fast when credentials are missing or the loaded model is incompatible. Never silently fall back to paper when the user asked for live.
- **Environment overrides** (`BINANCE_BASE_URL`, `BINANCE_SPOT_BASE_URL`, `BINANCE_FUTURES_BASE_URL`) are trusted only within the session that reads them. Do not persist env-resolved hostnames back into config.
- **Config files** (`~/.config/simple_ai_bitcoin_trading_binance/runtime.json`) are written with mode `0o600`. A new writer must call `chmod` explicitly.

## Never

- Never print, echo, format, or serialize an API key or secret, even truncated.
- Never embed credentials in test fixtures that live on disk. If a test needs a value, generate an obviously-fake one inline.
- Never commit a real API key, PAT, or signed request payload. If you accidentally stage one, rotate it before pushing.
- Never catch `BaseException`. Narrow your `except` to the class you actually handle.

## Required tests on secret-handling changes

- A test that proves the raw secret is absent from every output surface your change touches (stdout, stderr, persisted JSON, `last_request_info`).
- A test for both success and failure branches — the redaction contract must hold when things go wrong, not only when they go right.
