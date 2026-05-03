@echo off
setlocal
cd /d "%~dp0"

set "TRADER_EXE=%~dp0.venv311\Scripts\simple-ai-trading.exe"
if not exist "%TRADER_EXE%" (
    echo Missing "%TRADER_EXE%".
    echo Create the Python 3.11 virtual environment and install the project first.
    exit /b 1
)

"%TRADER_EXE%" shell %*
