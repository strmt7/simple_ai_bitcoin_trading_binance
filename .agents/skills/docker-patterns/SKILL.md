---
name: docker-patterns
description: Containerization rules for this repo — pinned base image, non-root user, read-only FS, explicit volumes for config and data.
origin: adapted from ZMB-UZH/omero-docker-extended docker-patterns
---

# Docker Patterns

Use this skill whenever you touch `Dockerfile`, `docker-compose.yml`, `.dockerignore`, or any runtime-container entrypoint script.

## Image baseline

- Pin Python to a specific minor version (`python:3.12-slim-bookworm`). **Never** use `:latest` or a floating tag.
- The container runs a **non-root user** with UID/GID `10001`. No `apt` or `pip` operations at runtime.
- The final image should have zero build tools. Install build deps in a throwaway stage if needed, copy only the wheel / site-packages forward.
- Declare `ENTRYPOINT ["simple-ai-trading"]` so `docker run image <cmd>` works exactly like the local binary.

## Compose baseline

- Service names are lowercase kebab-case.
- Each service sets `security_opt: no-new-privileges:true`.
- Each service declares a `healthcheck` when it has a reachable endpoint or a meaningful readiness probe.
- Env defaults live in Dockerfile `ARG` or in `.env.example`, not inline in `docker-compose.yml`. Compose references `${VAR}` with a default via `${VAR:-fallback}` only when the fallback is safe for every environment.
- Config and data mount as **named volumes** or as bind mounts with explicit `ro` / `rw` flags. Never mount `/` or `$HOME`.

## What the simple-ai-trading container must not do

- Start a live loop by default. Default `CMD` is `shell` (interactive) or `menu`, not `live`.
- Accept a `BINANCE_LIVE=1` style env that flips `testnet=false`. Live real-money mode must remain an explicit CLI flag combination with `testnet=true`, not a container-level setting.
- Bake credentials into the image. Secrets come from mounted `~/.config/simple_ai_bitcoin_trading_binance/runtime.json` (mode 0600) or from env vars at runtime.

## Verification

```bash
docker build -t simple-ai-trading:dev .
docker run --rm simple-ai-trading:dev status
docker run --rm --entrypoint python simple-ai-trading:dev -m pytest -q
```

Any change to the runtime contract (entrypoint, user, volumes, healthcheck) is a change to operator expectations. Update the README and the `.env.example` in the same commit.
