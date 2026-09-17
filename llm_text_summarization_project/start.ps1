# Local LLM Studio - PowerShell Startup Script
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "         LOCAL LLM STUDIO (Ollama + Local Model)            " -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# 1. Check Python
Write-Host "[1/3] Checking Python installation..." -ForegroundColor Yellow
$pyVersion = py --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Python launcher 'py' was not found on PATH." -ForegroundColor Red
    exit 1
}
Write-Host "      Found: $pyVersion" -ForegroundColor Green

# 2. Check Ollama
Write-Host "[2/3] Checking Ollama Status..." -ForegroundColor Yellow
try {
    $ollamaCheck = Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -Method Get -TimeoutSec 3 -ErrorAction Stop
    $modelNames = $ollamaCheck.models | ForEach-Object { $_.name }
    Write-Host "      Ollama Service is running at http://localhost:11434" -ForegroundColor Green
    Write-Host "      Installed Ollama Models: $($modelNames -join ', ')" -ForegroundColor Green
} catch {
    Write-Host "      [!] Warning: Ollama is not running. Mode 1 will be offline until 'ollama serve' is executed." -ForegroundColor DarkYellow
}

# 3. Launch Application
Write-Host "[3/3] Launching Studio Backend & UI..." -ForegroundColor Yellow
py start_studio.py
