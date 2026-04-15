# Verification Loop

## Principle

Prefer small, deterministic verification that directly confirms each assumption.

## Process

1. Run targeted tests first (same module).
2. Add/adjust tests for any new branch.
3. Run full suite before finishing broad changes.
4. Run coverage and confirm critical lanes have no uncovered new branches.
5. For CI-facing behavior, validate local workflow equivalents (same Python version and command shape used in `.github/workflows/ci.yml`).

## Rule

Do not report completion without showing the exact command/output outcome for each changed domain.
