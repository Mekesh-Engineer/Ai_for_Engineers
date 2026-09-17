#!/usr/bin/env python3
"""
Local LLM Studio - Launcher & Startup Orchestrator (Grammar Correction & Rewriting)
Checks environment, validates Ollama and Local models, launches FastAPI backend, and opens browser.
"""

import os
import sys
import time
import webbrowser
import threading
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

def print_banner():
    print(r"""
  +==============================================================+
  |              LOCAL LLM STUDIO — GRAMMAR CORRECTION           |
  |         Dual-Mode Local AI Workspace & Text Rewriting        |
  |    Mode 1: Ollama (Qwen 2.5 7B)  |  Mode 2: Local Models     |
  +==============================================================+
    """)

def check_ollama():
    """Verify Ollama availability and check for qwen2.5:7b."""
    print("[*] Checking Ollama Status at http://localhost:11434...")
    try:
        import httpx
        res = httpx.get("http://localhost:11434/api/tags", timeout=3.0)
        if res.status_code == 200:
            models = [m.get("name") for m in res.json().get("models", [])]
            print("    [+] Ollama Service: ONLINE")
            print(f"    [+] Discovered Ollama Models: {models}")
            if any("qwen2.5:7b" in m for m in models) or any("qwen" in m for m in models):
                print("    [+] Qwen 2.5 7B Model: READY")
            else:
                print("    [!] Warning: 'qwen2.5:7b' not listed. Run 'ollama pull qwen2.5:7b' if needed.")
            return True
        else:
            print(f"    [!] Ollama returned HTTP status {res.status_code}")
            return False
    except Exception as e:
        print(f"    [!] Ollama is not reachable ({e}).")
        print("        To use Ollama Mode, run 'ollama serve' in a separate terminal.")
        return False

def check_local_model():
    """Verify local model checkpoints in ./models/."""
    print("[*] Checking Project Local Model Storage (./models/)...")
    models_dir = PROJECT_ROOT / "models"
    safetensors = list(models_dir.rglob("*.safetensors"))
    pt_files = list(models_dir.rglob("*.pt"))
    bin_files = list(models_dir.rglob("*.bin"))
    
    total_weights = len(safetensors) + len(pt_files) + len(bin_files)
    if total_weights > 0:
        print(f"    [+] Local Model Weights Detected ({total_weights} files):")
        for f in (safetensors + pt_files + bin_files)[:3]:
            print(f"        - {f.relative_to(PROJECT_ROOT)}")
        return True
    else:
        print("    [!] No local model weight files found in ./models/.")
        return False

def open_browser_delayed(url: str, delay_seconds: float = 1.5):
    """Open web browser once backend is ready."""
    def _open():
        time.sleep(delay_seconds)
        print(f"[*] Opening Local LLM Studio in browser: {url}")
        webbrowser.open(url)
    
    t = threading.Thread(target=_open, daemon=True)
    t.start()

def main():
    print_banner()
    
    # 1. Check environments
    ollama_ready = check_ollama()
    local_ready = check_local_model()

    print("\n[+] System Readiness Summary:")
    print(f"    - Mode 1 (Ollama Qwen 2.5 7B):   {'READY' if ollama_ready else 'OFFLINE (Start Ollama)'}")
    print(f"    - Mode 2 (Project Local Model): {'READY' if local_ready else 'NOT DETECTED'}")
    print("    - Web Frontend & API Backend:   INITIALIZING...")

    host = "127.0.0.1"
    port = 8000
    url = f"http://{host}:{port}"

    open_browser_delayed(url, delay_seconds=1.5)

    import uvicorn
    print(f"\n[*] Starting Uvicorn server on {url} ... (Press Ctrl+C to stop)\n")
    uvicorn.run(
        "backend.main:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )

if __name__ == "__main__":
    main()
