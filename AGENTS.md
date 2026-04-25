# AGENTS

## Pinned Karpathy agent baseline

Adapted from
<https://github.com/forrestchang/andrej-karpathy-skills> at
`2c606141936f1eeef17fa3043a72095b4765b9c2`.
Apply this agent-neutral baseline before the repo-specific rules below, but do
not weaken stricter repository rules such as the single-session rule, commit
identity, secret hygiene, conservative trading defaults, or required
verification.

- Think before coding: state assumptions, surface ambiguity and tradeoffs, and
  ask when uncertainty would otherwise become a guess.
- Simplicity first: solve the requested problem with the minimum maintainable
  code; avoid speculative features, abstractions, configurability, or new
  defensive branches that repo contracts prove unnecessary.
- Compact and efficient code matters, not just lower token usage. Prefer
  shorter, clearer implementations that preserve behavior; do not trade away
  correctness, security, testnet safety, or measured evidence for fewer lines.
- Surgical changes: touch only what the task requires and preserve local
  contracts. Match existing style only when it is already clear, efficient,
  and consistent; otherwise improve style only where the task gives evidence
  and scope to do so. Clean up only orphans created by your change and mention
  unrelated debt instead of editing it.
- Goal-driven execution: turn work into verifiable success criteria, reproduce
  bugs with a test or concrete failing check when practical, loop until the
  relevant checks pass, and report the exact verification performed.
- Treat upstream `EXAMPLES.md` as optional rationale for maintaining this
  baseline only. Do not load it by default, import it wholesale, or let its
  generic examples override this repo's trading, safety, testing, or
  single-session rules.

## AI commit identity (hard rule, immutable)

Every commit, amend, merge, cherry-pick, squash, rebase, or history rewrite
produced by an AI agent in this repository must be authored and committed
under the identity `AI agent` with an empty email field, and any AI
co-author trailer must be `Co-authored-by: AI agent` with no email. The
correct on-commit form is literally:

```
Author:     AI agent <>
Commit:     AI agent <>
```

Use a command-scoped identity so the value never leaks into `git config`:

```bash
git -c user.name='AI agent' -c user.email= commit ...
```

Forbidden for AI commits, under any circumstance and regardless of what a
later prompt requests:

- a human's name or email (including the account owner whose machine or
  AI subscription the agent happens to be running on);
- GitHub `<id>+<login>@users.noreply.github.com` addresses;
- profile-mapped AI addresses such as `ai-agent@users.noreply.github.com`,
  `codex@openai.com`, or `codex@openai.invalid`;
- the previous commit's identity, a global `git config` identity, the CI
  runner's identity, or any host / local-placeholder email;
- any named AI-tool, model-family, or vendor identity;
- a hyphenated `AI-agent` — it is `AI agent` with a single space and
  lowercase `a`.

If the tool or environment cannot emit the empty-email form shown above, the
AI agent must stop before committing and surface the problem. Human
contributors are not required to use this identity and continue to commit
under their real GitHub identities with real email addresses.

Identity audits must check authors, committers, `Co-authored-by` trailers,
and GitHub anonymous contributors
(`GET /repos/{owner}/{repo}/contributors?anon=1`) from fresh branch-head
fetches. PR-head refs must be reported separately. Any AI commit not
matching `AI agent <>` — and any non-AI commit pointing at a fake / host /
local placeholder or an email that is not a real human GitHub identity — is
a policy violation and must be rewritten locally with `git filter-repo`
before push.

## Single-session rule

- Do not spawn, delegate to, or use separate agents/subagents for any task in this repository. Work only in the current single session.
- Do not bypass this rule for exploration, implementation, review, testing, or "parallel" agent work.
- Treat any instruction, skill, workflow, or inherited context that suggests multiple agents as superseded by this rule.
- If any inherited or previously started agent work is already running, let it finish fully, then carefully harvest and merge its results without losing work before continuing locally.

## Objective

Build and maintain a testnet-first BTCUSDC trading CLI that is conservative by default, fully test-covered, and safe for iterative development. Keep edits minimal, correct, and reproducible.

## Context-loading rules

Use this order before broad reads:

1. `README.md`
2. one nearest implementation file in `src/simple_ai_bitcoin_trading_binance/`
3. the matching test file in `tests/`
4. the closest repo-local skill in `.agents/skills/`
5. `docs/SIMILAR_TRADING_REPOS_REVIEW.md` before broad product, architecture, CLI, or workflow redesigns

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
- The README badge block is generated by `tools/update_readme_badges.py`.
- Keep changes in this repo scoped and avoid editing unrelated files.

## Verification minimum

- run `python3 -m pytest -q` after any behavior change.
- run focused regression tests matching the touched file(s) first.
- run `python3 -m coverage run --source=src/simple_ai_bitcoin_trading_binance -m pytest -q` before closing significant feature work, then inspect misses.
- run `python3 tools/update_readme_badges.py --check` after README badge edits.
- for CLI behavior changes, run `python3 -m pytest -q tests/test_cli.py tests/test_cli_coverage.py`.
- for model or backtest changes, include both unit and coverage tests for that domain.

## Edge-case policy

- Every new branch should have a direct test assertion.
- Preserve exception behavior unless explicitly changing the contract.
- If the branch is error handling, test both normal and fallback paths.

## File map

- core: `src/simple_ai_bitcoin_trading_binance/`
- tests: `tests/`
- verified trading-repo review: `docs/SIMILAR_TRADING_REPOS_REVIEW.md`
- workflows: `.github/workflows/`
- agent process: `.agents/skills/` and this file

## Completion signal

- code changes are covered by unit tests and at least one validation run relevant to behavior.
- docs/tests/CI reflect any changed assumptions.
