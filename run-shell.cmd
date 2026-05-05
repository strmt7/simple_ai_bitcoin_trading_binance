@echo off
setlocal
set "APP_DIR=%~dp0"
set "TRADER_SUBCOMMAND=shell"

cd /d "%APP_DIR%" || (
    echo [simple-ai-trading] Failed to enter "%APP_DIR%".
    exit /b 1
)

if defined PYTHONPATH (
    set "PYTHONPATH=%APP_DIR%src;%PYTHONPATH%"
) else (
    set "PYTHONPATH=%APP_DIR%src"
)

set "TRADER_EXE=%APP_DIR%.venv311\Scripts\simple-ai-trading.exe"
if exist "%TRADER_EXE%" goto run_local_exe

set "TRADER_EXE=%APP_DIR%.venv\Scripts\simple-ai-trading.exe"
if exist "%TRADER_EXE%" goto run_local_exe

where.exe simple-ai-trading >nul 2>nul
if not errorlevel 1 goto run_path_exe

set "TRADER_PY=%APP_DIR%.venv311\Scripts\python.exe"
if exist "%TRADER_PY%" (
    "%TRADER_PY%" -c "import requests, textual; import simple_ai_bitcoin_trading_binance" >nul 2>nul
    if not errorlevel 1 goto run_local_module
    set "LOCAL_PY_DIAG=%TRADER_PY% exists but cannot import the project dependencies."
)

set "TRADER_PY=%APP_DIR%.venv\Scripts\python.exe"
if exist "%TRADER_PY%" (
    "%TRADER_PY%" -c "import requests, textual; import simple_ai_bitcoin_trading_binance" >nul 2>nul
    if not errorlevel 1 goto run_local_module
    set "LOCAL_PY_DIAG=%TRADER_PY% exists but cannot import the project dependencies."
)

py -3.11 -c "import requests, textual; import simple_ai_bitcoin_trading_binance" >nul 2>nul
if not errorlevel 1 goto run_py311_module
py -3.11 -c "import sys" >nul 2>nul
if not errorlevel 1 set "PY311_DIAG=py -3.11 is available but cannot import the project dependencies."

python -c "import sys; sys.exit(1) if sys.version_info < (3, 11) else None; import requests, textual; import simple_ai_bitcoin_trading_binance" >nul 2>nul
if not errorlevel 1 goto run_python_module
python -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)" >nul 2>nul
if not errorlevel 1 set "PATH_PY_DIAG=python is 3.11+ but cannot import the project dependencies."

echo [simple-ai-trading] Unable to launch "%TRADER_SUBCOMMAND%".
echo Checked:
echo   "%APP_DIR%.venv311\Scripts\simple-ai-trading.exe"
echo   "%APP_DIR%.venv\Scripts\simple-ai-trading.exe"
echo   simple-ai-trading on PATH
echo   "%APP_DIR%.venv311\Scripts\python.exe" -m simple_ai_bitcoin_trading_binance
echo   "%APP_DIR%.venv\Scripts\python.exe" -m simple_ai_bitcoin_trading_binance
echo   py -3.11 -m simple_ai_bitcoin_trading_binance
echo   python -m simple_ai_bitcoin_trading_binance with Python 3.11+
if defined LOCAL_PY_DIAG echo Note: %LOCAL_PY_DIAG%
if defined PY311_DIAG echo Note: %PY311_DIAG%
if defined PATH_PY_DIAG echo Note: %PATH_PY_DIAG%
echo.
echo Create or repair a local environment with:
echo   py -3.11 -m venv .venv311
echo   .venv311\Scripts\python.exe -m pip install -e .
echo Or install the project so simple-ai-trading is available on PATH.
exit /b 1

:run_local_exe
echo [simple-ai-trading] using local console script: "%TRADER_EXE%"
"%TRADER_EXE%" %TRADER_SUBCOMMAND% %*
exit /b %ERRORLEVEL%

:run_path_exe
for /f "delims=" %%I in ('where.exe simple-ai-trading 2^>nul') do (
    set "TRADER_EXE=%%I"
    goto run_path_exe_resolved
)

:run_path_exe_resolved
echo [simple-ai-trading] using PATH console script: "%TRADER_EXE%"
simple-ai-trading %TRADER_SUBCOMMAND% %*
exit /b %ERRORLEVEL%

:run_local_module
echo [simple-ai-trading] using Python module: "%TRADER_PY%" -m simple_ai_bitcoin_trading_binance
"%TRADER_PY%" -m simple_ai_bitcoin_trading_binance %TRADER_SUBCOMMAND% %*
exit /b %ERRORLEVEL%

:run_py311_module
echo [simple-ai-trading] using Python launcher: py -3.11 -m simple_ai_bitcoin_trading_binance
py -3.11 -m simple_ai_bitcoin_trading_binance %TRADER_SUBCOMMAND% %*
exit /b %ERRORLEVEL%

:run_python_module
echo [simple-ai-trading] using PATH Python: python -m simple_ai_bitcoin_trading_binance
python -m simple_ai_bitcoin_trading_binance %TRADER_SUBCOMMAND% %*
exit /b %ERRORLEVEL%
