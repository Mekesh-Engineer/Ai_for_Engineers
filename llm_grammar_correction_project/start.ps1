# PowerShell startup script for Local LLM Studio
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Starting Local LLM Studio (Grammar Correction & Rewriting)" -ForegroundColor Green
Write-Host "  Dual-Mode: Ollama (Qwen 2.5 7B) + Project Local Models" -ForegroundColor Yellow
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
py start_studio.py
