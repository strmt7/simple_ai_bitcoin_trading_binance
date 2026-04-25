---
name: rate-limit-compliance
description: Binance rate-limit etiquette, backoff rules, and paging caps that must be preserved by any network-touching change.
origin: repo-local skill adapted from omero-docker-extended compliance-and-rate-limit
---

# Rate Limit Compliance

Use this skill whenever you touch `api.py`, `command_fetch`, `command_live`, or add a new network-touching command.

## Baseline

- The REST client throttles at `max_rate_calls_per_minute` (default 1100, hard-clamped to 2000).
- Retries follow exponential backoff with a `Retry-After` header override. Retryable statuses: `{418, 429, 500, 502, 503, 504}`. Retryable API codes: `{-1003, -1007}`.
- Signed endpoint calls stay sequential — do not parallelize them. Paging for klines is sequential for the same reason.

## Paging caps

- Spot `/api/v3/klines`: ≤ **1000** rows per request.
- USD-M futures `/fapi/v1/klines`: ≤ **1500** rows per request.
- `command_fetch --batch-size` must clamp to the correct ceiling based on `runtime.market_type`. Any new paging helper must clamp too.

## Concurrency

- Read-only public endpoints (`/klines`, `/ticker/price`, `/exchangeInfo`) can be batched sequentially without trouble. Do not introduce a thread pool or asyncio for them without a measured need.
- Signed endpoints are single-flight. If you add a new authenticated call, route it through `_request(..., signed=True)` so the rate-limiter and redaction both kick in.

## When you add a new endpoint

1. Add the retry-status / retry-code semantics explicitly; don't rely on default `_request` behavior if the endpoint uses different conventions.
2. Add a test covering:
   - happy path,
   - the specific retry status this endpoint might return,
   - malformed payload.
3. Make sure `last_request_info` is updated with a redacted URL — the TUI and `doctor --online` both rely on it.

## Don't

- Don't bypass the throttle with a direct `requests.get`. Go through `self.session` via `_request`.
- Don't escalate retries beyond `max_retries`. If an endpoint needs more than four retries, something is wrong with the endpoint or the call, not the retry count.
- Don't leak raw URLs into logs. Every `last_error` path uses `_redact_sensitive_text`.
