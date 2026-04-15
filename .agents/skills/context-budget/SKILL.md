# Context Budget

## Goal

Limit reads to the minimum high-signal surface required to implement the task correctly, then widen only when evidence is missing.

## Workflow

1. Read `AGENTS.md`.
2. Identify exactly which module and command path is affected.
3. Open one source file, one test file, and one skill that governs the area.
4. Implement with tight scope.
5. If uncertainty remains, open one more implementation file and one more test file only.

## Hard limits

- Avoid bulk reading unrelated directories and generated artifacts.
- Avoid opening entire modules for tangential concerns.
- Prefer `rg` for symbol search over opening large files.

## Exit condition

Before coding the next step, verify that remaining assumptions are either covered by
existing tests or addressed by newly added tests.
