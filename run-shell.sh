#!/usr/bin/env sh
set -u

APP_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P) || exit 1
TRADER_SUBCOMMAND=shell

cd "$APP_DIR" || {
    echo "[simple-ai-trading] Failed to enter $APP_DIR." >&2
    exit 1
}

if [ -n "${PYTHONPATH:-}" ]; then
    PYTHONPATH="$APP_DIR/src:$PYTHONPATH"
else
    PYTHONPATH="$APP_DIR/src"
fi
export PYTHONPATH

try_exe() {
    candidate=$1
    label=$2
    shift 2
    if [ -x "$candidate" ]; then
        echo "[simple-ai-trading] using $label: $candidate"
        exec "$candidate" "$TRADER_SUBCOMMAND" "$@"
    fi
}

try_python() {
    candidate=$1
    label=$2
    shift 2
    if [ -x "$candidate" ] && "$candidate" -c 'import requests, textual; import simple_ai_bitcoin_trading_binance' >/dev/null 2>&1; then
        echo "[simple-ai-trading] using $label: $candidate -m simple_ai_bitcoin_trading_binance"
        exec "$candidate" -m simple_ai_bitcoin_trading_binance "$TRADER_SUBCOMMAND" "$@"
    fi
}

try_exe "$APP_DIR/.venv311/bin/simple-ai-trading" "local .venv311 console script" "$@"
try_exe "$APP_DIR/.venv/bin/simple-ai-trading" "local .venv console script" "$@"
try_exe "$APP_DIR/.venv311/Scripts/simple-ai-trading.exe" "local .venv311 Windows console script" "$@"
try_exe "$APP_DIR/.venv/Scripts/simple-ai-trading.exe" "local .venv Windows console script" "$@"

if command -v simple-ai-trading >/dev/null 2>&1; then
    TRADER_EXE=$(command -v simple-ai-trading)
    echo "[simple-ai-trading] using PATH console script: $TRADER_EXE"
    exec simple-ai-trading "$TRADER_SUBCOMMAND" "$@"
fi

try_python "$APP_DIR/.venv311/bin/python" "local .venv311 Python module" "$@"
try_python "$APP_DIR/.venv/bin/python" "local .venv Python module" "$@"
try_python "$APP_DIR/.venv311/Scripts/python.exe" "local .venv311 Windows Python module" "$@"
try_python "$APP_DIR/.venv/Scripts/python.exe" "local .venv Windows Python module" "$@"

if command -v python3.11 >/dev/null 2>&1 && python3.11 -c 'import requests, textual; import simple_ai_bitcoin_trading_binance' >/dev/null 2>&1; then
    echo "[simple-ai-trading] using PATH Python: python3.11 -m simple_ai_bitcoin_trading_binance"
    exec python3.11 -m simple_ai_bitcoin_trading_binance "$TRADER_SUBCOMMAND" "$@"
fi

if command -v python3 >/dev/null 2>&1 && python3 -c 'import sys; assert sys.version_info >= (3, 11); import requests, textual; import simple_ai_bitcoin_trading_binance' >/dev/null 2>&1; then
    echo "[simple-ai-trading] using PATH Python: python3 -m simple_ai_bitcoin_trading_binance"
    exec python3 -m simple_ai_bitcoin_trading_binance "$TRADER_SUBCOMMAND" "$@"
fi

cat >&2 <<EOF
[simple-ai-trading] Unable to launch "$TRADER_SUBCOMMAND".
Checked:
  $APP_DIR/.venv311/bin/simple-ai-trading
  $APP_DIR/.venv/bin/simple-ai-trading
  $APP_DIR/.venv311/Scripts/simple-ai-trading.exe
  $APP_DIR/.venv/Scripts/simple-ai-trading.exe
  simple-ai-trading on PATH
  $APP_DIR/.venv311/bin/python -m simple_ai_bitcoin_trading_binance
  $APP_DIR/.venv/bin/python -m simple_ai_bitcoin_trading_binance
  $APP_DIR/.venv311/Scripts/python.exe -m simple_ai_bitcoin_trading_binance
  $APP_DIR/.venv/Scripts/python.exe -m simple_ai_bitcoin_trading_binance
  python3.11 -m simple_ai_bitcoin_trading_binance
  python3 -m simple_ai_bitcoin_trading_binance with Python 3.11+

Create or repair a local environment with:
  python3.11 -m venv .venv311
  .venv311/bin/python -m pip install -e .
Or install the project so simple-ai-trading is available on PATH.
EOF
exit 1
