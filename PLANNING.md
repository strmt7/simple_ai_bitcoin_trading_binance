# Implementation Planning: BTCUSDC Testnet Trading CLI

## Purpose

This document defines the concrete roadmap to evolve
`simple_ai_bitcoin_trading_binance` from a functional prototype into a
reliable, fully test-covered, testnet-first trading stack with real leverage support
and repeatable model iteration.

Scope is intentionally constrained to:

- Symbol: `BTCUSDC` only
- Binance testnet-first behavior
- Spot and futures execution modes
- Conservative trading defaults with explicit user opt-in for risk changes

## Current Baseline (What already exists)

- CLI with commands: `configure`, `fetch`, `train`, `tune`, `backtest`, `evaluate`,
  `live`, `status`, `connect`, `strategy`.
- Minimal feature extraction (`features.py`) and pure-stdlib logistic model (`model.py`).
- Binance REST client with signed request support, leverage clamp, and rate-limit retry logic (`api.py`).
- Backtest engine with stop-loss/take-profit and drawdown cap (`backtest.py`).
- Persistence via local JSON for runtime/strategy and model artifacts.
- Comprehensive unit and coverage tests already present and passing.

## Problem Statement

The system currently executes the workflow but still needs stronger production-readiness in:

1. Data reliability (API consistency, rate-limit behavior, retries, resumability).
2. Model quality and validity pipeline (feature drift checks, reproducible experiments).
3. Live execution safety (position accounting, leverage validation, emergency controls).
4. Test coverage breadth on critical trading branches and failure states.
5. Operator visibility (experiment traceability, run metrics, and failure diagnostics).

## Non-Goals (for this roadmap)

- Any live money execution.
- Multi-symbol support.
- Dark pool/internal order-book feed support.
- Deep-learning frameworks in MVP (keep stdlib + deterministic dependencies).
- Web UI or mobile interface.

## Target Architecture (MVP + Growth Path)

```
CLI -> Strategy/Runtime Config -> Data/API Layer -> Features -> Model -> Risk Engine -> Executor -> Backtest/Live -> Observability
                                                           \
                                                            -> Experiment Records -> Artifact Store
```

### Components and ownership

- `cli.py`
  - Command orchestration, argument validation, guardrails, user-facing messages.
  - Add explicit safety workflow gates before any non-paper order submit.

- `api.py`
  - All exchange interactions remain here (REST testnet/live endpoints, retries, leverage bracket lookup).
  - Keep request telemetry as deterministic structured metadata for auditability.

- `features.py`
  - Deterministic feature pipeline with schema + versioning.
  - Add missing-value handling and drift detection counters.

- `model.py`
  - Deterministic training, calibration, walk-forward reporting, model metadata and checksums.

- `backtest.py`
  - Single source of truth for signal-to-execution simulation.
- `tests/`
  - Unit, property-like, and contract tests for every branch and failure path.
  - Recorded/replay fixtures for API and live-loop behavior.

## Strategic Pillars

1. **Reliable Data Pipeline**
   - Deterministic fetch windows, checkpointing, and resume handling.
2. **Adaptive Yet Safe Model**
   - Clear model lifecycle: train -> evaluate -> calibrate -> backtest -> optional live dry-run -> live loop.
3. **Strict Risk Control**
   - Hard circuit breakers before/after execution decisions.
4. **Operational Transparency**
   - Every run writes machine-readable run context and outcome.
5. **Test and Verification Coverage**
   - Expand beyond unit tests into branch, contract, and behavior-integration tests.

---

## Detailed Implementation Plan

## Phase 0 — Planning Execution Preparation (1-2 days)

### 0.1 Create explicit roadmap and execution contract
- Create this planning artifact and keep it updated as source of truth.
- Add a short changelog entry for each release milestone (in-code or docs note).

### 0.2 Consolidate safety defaults
- Verify defaults remain:
  - testnet enabled
  - paper mode default
  - BTCUSDC restriction enforced
  - order submission path blocked without explicit confirmation for non-paper mode.

Acceptance criteria:
- A scripted smoke run demonstrates `configure -> status` retains conservative defaults.
- `pytest` remains green before and after any code change.

## Phase 1 — API Hardening (1-2 weeks)

### 1.1 Free endpoint contract matrix
Map every Binance endpoint call currently used:
- `/api/v3/ping`, `/api/v3/time`, `/api/v3/exchangeInfo`, `/api/v3/klines`, `/api/v3/account`, `/api/v3/ticker/price`, `/api/v3/order`
- `/fapi/v1/ping`, `/fapi/v1/time`, `/fapi/v1/exchangeInfo`, `/fapi/v1/klines`, `/fapi/v2/account`, `/fapi/v1/ticker/price`, `/fapi/v1/order`, `/fapi/v1/leverage`, `/fapi/v1/leverageBracket`

### 1.2 Improve observability in `api.py`
- Emit a minimal, typed event object for request attempts containing:
  - timestamp, method, endpoint, retries, latency, status, exception/error class, retry decision.
- Add request/response size + payload validation assertions in tests.

### 1.3 Pagination + dedupe strategy for klines/history
- Implement:
  - stable end/start window progression,
  - overlap-safe dedupe by `open_time`,
  - strict row schema validation before persistence.
- Keep CLI `fetch` capable of incremental append mode.

### 1.4 Leverage integrity checks in futures path
- In futures mode: always call bracket retrieval once per session and cache max leverage.
- Clamp both runtime/configured and CLI override leverage to:
  - exchange cap
  - safe local cap (1..125)
  - runtime override precedence rules.

Acceptance criteria:
- New tests for HTTP status branches, empty payload branches, and bracket parse failure branches are passing.
- Manual scripted check:
  - if leverage exceeds exchange bracket, submit path uses capped leverage.

## Phase 2 — Data and Feature Pipeline Upgrades (1-2 weeks)

### 2.1 Feature set versioning
- Add explicit `feature_version` in `StrategyConfig` and saved rows.
- Persist `feature_version` in model metadata and reject load mismatch with clear message.

### 2.2 Feature validation and stability
- Add deterministic checks in `features.py`:
  - window lengths exactly as configured,
  - numeric type assertions,
  - bounded feature range guards,
  - NaN/Inf rejection mode.
- Add tests for pathological input (flat markets, zero-volume windows, invalid highs/lows).

### 2.3 Train/eval split integrity
- Enforce minimum split sizes and explicit warnings when calibration uses insufficient data.
- Add `--seed` input in train/evaluate flows for reproducibility.

Acceptance criteria:
- Every feature generation test has branch coverage for skip/invalid-row handling.
- A training run with mismatched feature schema fails fast with an actionable error.

## Phase 3 — Model and Tuning Roadmap (2-3 weeks)

### 3.1 Deterministic training + checkpoints
- Save per-run training metadata:
  - commit hash (if available), sample count, split window, seed, threshold calibration settings.
- Add `model_version` field to model JSON payload.

### 3.2 Improve calibration and class-weight strategy
- Keep current logistic model as default.
- Add alternate threshold optimization:
  - precision-recall frontier score,
  - minimum-trade floor constraint.
- Ensure calibration does not overfit when sample count is low.

### 3.3 Walk-forward strictness
- Extend `walk_forward_report` to emit:
  - fold-level confusion/threshold
  - fold-level returns/sharpe proxy
  - rejected folds when `train_window` or class imbalance invalid.

### 3.4 Tune command expansion
- Make tune evaluate candidates through:
  1. walk-forward pass,
  2. fixed-threshold backtest,
  3. risk penalty scoring.
- Persist best candidate + full candidate ranking JSON artifact.

Acceptance criteria:
- Re-run `train --calibrate-threshold` yields stable model hash with fixed seed.
- Tune output includes full ranking and reasons for rejection.

## Phase 4 — Execution Engine Safety (2-3 weeks)

### 4.1 Live mode risk rails
- Before each order:
  - check cash exposure, active positions, max-open, daily trades,
  - check drawdown lockout from previous runs.
- Add "pre-flight kill" if API key missing in non-dry mode.

### 4.2 Post-order control loop
- After fill simulation/execution:
  - validate position size and leverage used,
  - apply cooldown handling and emergency fallback when stop-loss is hit.

### 4.3 Paper/live parity
- Build a small execution contract test that validates:
  - paper order payload fields and quantities,
  - dry-run and live branches consume the same strategy preflight logic.

### 4.4 Live telemetry and persistence
- Emit `live_run_<timestamp>.json` with:
  - step counters,
  - signal probabilities/threshold used,
  - position updates and PnL progression,
  - exception snapshots (truncated stack + context).

Acceptance criteria:
- Non-paper mode attempts fail fast and safe when credentials missing.
- `live` command with `--steps 1 --sleep 0 --paper` completes deterministically under mocked APIs.

## Phase 5 — Backtest Credibility and Profitability Evidence (1-2 weeks)

### 5.1 Backtest realism improvements
- Ensure fees/slippage are parameterized and defaulted from strategy config.
- Add maker/taker split option (future extension).
- Add warm-up period exclusion from score.

### 5.2 Robust baseline comparison
- Add benchmark strategy baseline:
  - hold/no-trade,
  - moving-average crossover baseline,
  - fixed-dollar-size naive baseline.
- Require strategy candidates to clear baseline by configurable edge threshold on backtest and walk-forward.

### 5.3 Significance-style rejection rules
- Reject parameter sets with:
  - zero/near-zero trades,
  - max drawdown breach rate above threshold,
  - excessive variance in per-fold returns.

Acceptance criteria:
- Backtest pipeline emits per-run comparables and baseline delta metrics.
- Tune command can exclude low-edge candidates automatically.

## Phase 6 — Test/Verification Expansion (ongoing, every phase)

### 6.1 Unit and coverage expansion
- Add focused tests for:
  - every `_request` branch and API code-path, including non-dict JSON rows and malformed headers.
  - strategy constraint enforcement.
  - feature schema mismatch and drift conditions.
  - walk-forward fold failures and warnings.

### 6.2 Contract tests
- Add end-to-end CLI command tests for:
  - `configure`/`status` consistency,
  - `train -> evaluate -> backtest -> tune -> live --paper` path with full mocking.

### 6.3 Integration smoke harness
- Add scripted dry-run integration scenario using local fake exchange responses:
  - fetch success
  - retry then success
  - malformed responses
  - leverage clamp branch
  - emergency shutdown branch

Acceptance criteria:
- Coverage reports maintain fail-under thresholds.
- New tests capture previously untested branch regions in api/config/cli/backtest/model/features.

## Phase 7 — Release and Maintenance

### 7.1 Versioning and changelog
- Add short changelog section for each milestone.
- Bump artifact version in package metadata as needed.

### 7.2 Documentation updates
- Update README with:
  - risk assumptions,
  - how to fetch and use testnet API keys,
  - interpretation of leverage and walk-forward metrics.

### 7.3 CI hardening
- Add explicit test for zero-coverage regression via `.coverage` sanity checks.
- Keep codecov uploads only after passing core suite.

---

## Milestone Plan and Dependencies

### Milestone A (immediately after this plan)
- Finish API and leverage hardening.
- File scope: `src/simple_ai_bitcoin_trading_binance/api.py`, `tests/test_api*.py`, `tests/test_cli*.py`.

### Milestone B
- Feature/model pipeline upgrade and training reproducibility.
- File scope: `features.py`, `model.py`, `types.py`, `config.py`, related tests.

### Milestone C
- Live execution safety, telemetry, and emergency controls.
- File scope: `cli.py`, `backtest.py`, model/config tests.

### Milestone D
- Backtest reliability/benchmarking and full coverage expansion.
- File scope: `backtest.py`, `tests/test_backtest*.py`, `tests/test_model*.py`.

---

## Risks and Mitigations

- **Exchange API drift:** Add endpoint contract tests with captured fixtures and run weekly.
- **Testnet behavior mismatch:** Keep paper-first by default and treat real execution as opt-in.
- **Overfit signals:** Use walk-forward + baseline rejection + minimum trade constraints.
- **Model misuse after schema drift:** Enforce feature_version and compatibility checks before inference.
- **Operational drift and silent failures:** Structured request telemetry + run artifacts mandatory.

## Success Criteria

- Functional:
  - all existing tests pass,
  - new tests increase branch/exception branch coverage for targeted modules,
  - `live --paper` can complete deterministic mocked runs end-to-end.
- Technical:
  - deterministic training with fixed seeds,
  - leverage always clamps to effective exchange limits in futures mode,
  - explicit circuit breakers before any non-paper submit.
- Operational:
  - every run writes an audit artifact,
  - backtest and walk-forward output includes edge metrics and rejection reasons.

---

## Tracking and Execution Rule

No roadmap item is considered done until:
- its acceptance criteria tests pass,
- associated docs are updated,
- and the change is pushed with a short commit message including the milestone ID.

