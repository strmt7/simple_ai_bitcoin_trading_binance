# Security

This project is **test-phase software** that trades on Binance **testnet** by
default.  Real-money execution is intentionally blocked in the current phase.
Any change that widens that scope must be accompanied by explicit operator
opt-in, tests that prove redaction still holds, and an update to this file.

## Threat model in scope

- Accidental credential disclosure via logs, artifacts, stdout/stderr, or git
  history.
- Accidental live-mode execution from a misconfigured strategy.
- Supply-chain drift from unexpected dependencies (we ship pure stdlib + the
  two listed deps — anything else is a red flag).
- Exchange API rate-limit abuse or order replay from loose loops.

## Out of scope

- Cryptographic attacks against Binance's HMAC implementation.
- Attacks that require root on the host — the app inherits whatever trust the
  host already has.
- Losses caused by genuinely adverse markets on testnet — testnet funds are not
  real money.

## Credential hygiene

- API keys + secrets live in `~/.config/simple_ai_bitcoin_trading_binance/runtime.json`
  with mode `0600`, created by `configure`.  They are read lazily; nothing
  imports the secret at module load time.
- Every outbound URL routed through `_redact_request_url` before any log, error
  message, or persisted artifact writes it.
- All `logging_ext` handlers apply a redaction filter that scrubs `ghp_*` /
  `github_pat_*` / `sk-*` tokens, signed query fields (`signature`,
  `timestamp`, `recvWindow`), and `X-MBX-APIKEY` headers.
- `RuntimeConfig.public_dict()` is the only acceptable source for runtime
  snapshots embedded in JSON artifacts.

## Reporting

For non-public issues, contact the repository owner privately with a clear
reproduction.  Do not file public issues that include a valid API key or PAT;
rotate the credential first.
