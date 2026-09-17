@echo off
title Local LLM Studio
echo ============================================================
echo           LOCAL LLM STUDIO (Ollama + Local Model)
echo ============================================================
echo.
echo [1/3] Checking Python installation...
py --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python launcher 'py' was not found. Please ensure Python is installed and in PATH.
    pause
    exit /b 1
)

echo [2/3] Checking dependencies...
py -c "import fastapi, uvicorn, torch, transformers, httpx; print('Dependencies verified.')" >nul 2>&1
if %errorlevel% neq 0 (
    echo [!] Installing required packages...
    py -m pip install -r requirements.txt
)

echo [3/3] Starting Local LLM Studio...
py start_studio.py
pause
