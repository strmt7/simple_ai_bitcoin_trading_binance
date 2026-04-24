---
name: secret-hygiene
description: Practical rules for keeping API keys, PATs, and signed request material out of the repo, logs, and artifacts.
origin: repo-local skill inspired by ZMB-UZH/omero-docker-extended env-contract-reviewer
---

# Secret Hygiene

Use this skill any time secrets enter the session — user-supplied API keys, Git PATs, environment variables, or captured HTTP samples.

## Channels that must stay clean

- **Commit history**: no API keys, no PATs, no signed URLs. A committed secret must be rotated, not just reverted.
- **Runtime logs** (stdout, stderr, the TUI activity panel): all outgoing URLs flow through `_redact_request_url` before display.
- **Artifacts under `data/`**: persisted JSON uses `RuntimeConfig.public_dict()` for any embedded runtime snapshot.
- **Tests**: fixtures use obviously-fake values (`"fake-api-key"`, `"fake-secret"`). Never commit real testnet credentials even if they are "only" testnet.

## Workflow for user-supplied credentials

1. User provides a PAT or API key for a one-off operation.
2. Read it via `getpass.getpass` or from an env var you control. Do not splice it into `argv` or subprocess command lines.
3. When you hand it to Git, use the credential helper stub in `tools/push_with_pat.py` — the PAT is served over a socket to a scoped `GIT_ASKPASS`, never written to a file, remote URL, or `credential.helper`.
4. After the operation, `git remote -v` should not contain the PAT. `~/.git-credentials` should not exist (or should not mention github.com with the PAT).
5. If the PAT was pasted into the conversation, treat it as compromised the moment the task completes — tell the user to rotate it.

## Env-var overrides

`BINANCE_BASE_URL`, `BINANCE_SPOT_BASE_URL`, `BINANCE_FUTURES_BASE_URL` are advertised overrides. Any new env var you read:

- Has a matching `.env.example` entry.
- Is documented in the README Host overrides section.
- Is trimmed (`.strip()`) and treated as untrusted input.

## Quick self-check before committing

```bash
git diff --cached | grep -EiI 'ghp_|ghs_|github_pat_|AKIA|sk-[A-Za-z0-9]{20,}|-----BEGIN .* PRIVATE KEY-----'
git diff --cached | grep -E 'api_key|api_secret' | grep -v '<redacted>\|fake-\|BLANK'
```

A hit on either grep is a blocker.
