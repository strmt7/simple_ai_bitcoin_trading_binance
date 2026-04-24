# syntax=docker/dockerfile:1.7
# ----------------------------------------------------------------------------
# Stage 1: builder — installs the package into a throwaway venv.
# ----------------------------------------------------------------------------
FROM python:3.12-slim-bookworm AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build
COPY pyproject.toml README.md LICENSE ./
COPY src ./src

RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install -q --upgrade pip \
    && /opt/venv/bin/pip install -q .

# ----------------------------------------------------------------------------
# Stage 2: runtime — minimal, non-root, read-only friendly.
# ----------------------------------------------------------------------------
FROM python:3.12-slim-bookworm AS runtime

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN groupadd --system --gid 10001 trader \
    && useradd --system --uid 10001 --gid trader --home-dir /home/trader --create-home trader

COPY --from=builder /opt/venv /opt/venv

USER trader
WORKDIR /home/trader

# Config + data are mounted by the operator (or docker-compose). The package
# itself ships no runtime state; it persists runtime + strategy JSON under
# ``~/.config/simple_ai_bitcoin_trading_binance`` inside the container.
VOLUME ["/home/trader/.config", "/home/trader/work/data"]

# Default to the interactive shell so ``docker run -it image`` feels natural.
ENTRYPOINT ["simple-ai-trading"]
CMD ["shell"]
