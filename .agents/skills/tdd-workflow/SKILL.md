---
name: tdd-workflow
description: Red-green-refactor workflow adapted for a 100% branch-coverage trading CLI — narrow regression tests first, no coverage regressions.
origin: adapted from ZMB-UZH/omero-docker-extended tdd-workflow
---

# TDD Workflow

Use this skill whenever you add a feature, fix a bug, or change a behavior that users or tests can observe.

## The loop

1. **Name the contract** you're about to change in one sentence ("`shell` quits on `Ctrl-D` without error" / "`fetch --batch-size 2000` clamps to the exchange limit"). If you can't name it, you don't know the change yet.
2. **Write the failing regression test first** in `tests/` under the matching file. Target branches, not just happy paths. If the branch is error handling, assert both the failure and the recovery.
3. **Implement the smallest change** that moves the test from red to green. Do not take the opportunity to restructure unrelated code.
4. **Run targeted tests first** (`pytest -q tests/test_<module>.py`), then the full suite, then coverage with `--fail-under=100`.
5. **Stop at green.** Polish and refactoring is a separate commit with its own tests.

## Coverage is load-bearing

This repo runs `coverage report --fail-under=100` in CI. A new branch with no test assertion is a CI break, not a style issue. When you add code:

- Every new `if`, `elif`, `else`, `except`, `with`, and early `return` needs a test that exercises it.
- If a branch is genuinely unreachable, prove it — don't guard with `# pragma: no cover` unless you also leave a comment explaining why the branch exists.

## Tests are documentation

- Prefer fixtures that make the intent obvious (`fake_candles`, `trained_model_stub`) over opaque tuples.
- Assert on **public contract** (return values, persisted artifacts, printed lines). Avoid asserting on private attribute shape.
- Do not mock `BinanceClient` at a lower level than necessary. The existing tests stub at the HTTP seam — follow that pattern.

## Don't

- Don't weaken an existing assertion to get a test to pass after your change. If an assertion is wrong, fix it deliberately with an explanation in the commit.
- Don't use `time.sleep` in tests. Monkey-patch the clock or inject a sleep hook.
- Don't rely on network in tests. If you need HTTP behavior, stub `requests.Session.request` at the seam.
