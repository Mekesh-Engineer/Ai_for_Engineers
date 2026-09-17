# Local LLM Studio — Dual-Mode Local AI Workspace & Document Summarizer

**Local LLM Studio** is a self-contained, high-performance local AI workspace and document intelligence system built for engineers, researchers, and developers. It provides two independent operating modes powered by **Qwen 2.5 7B** (via Ollama) and local Hugging Face Transformers checkpoints.

---

## Architecture Overview

```text
                    LOCAL LLM STUDIO
                           │
             ┌─────────────┴─────────────┐
             │                           │
       OLLAMA MODE                LOCAL MODEL MODE
             │                           │
    localhost:11434                 ./models/
             │                           │
      Qwen 2.5 7B                  Local Model
             │                           │
             └─────────────┬─────────────┘
                           │
                    COMMON LLM API
                    (LLMProvider)
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
       CHAT          FILE SUMMARIZER    PROJECT ANALYZER
        │                  │                  │
        └──────────────────┼──────────────────┘
                           │
                  PROJECT-AWARE AI
                           │
              Interactive Web Interface (SPA)
```

---

## Operating Modes

### Mode 1 — Ollama Mode (Primary / High-Performance)
* Communicates directly with the local Ollama daemon via REST API at `http://localhost:11434`.
* Utilizes **`qwen2.5:7b`** (and any other custom personas discovered via `/api/tags`).
* Provides token-by-token **Server-Sent Events (SSE) streaming**, full GPU acceleration, and low-latency inference.
* Does **not** directly touch Ollama's internal file structures.

### Mode 2 — Project Local Model Mode (Standalone Checkpoint)
* Operates completely independently from Ollama.
* Auto-detects, validates, and loads local model checkpoints stored directly inside the `./models/` directory (e.g. `./models/saved_models/summarizer_model` with `model.safetensors`, `config.json`, and `tokenizer.json`).
* Uses Hugging Face Transformers with global in-memory process caching to avoid redundant weight reloads.

---

## Core Features

1. **Interactive AI Chat**:
   * Multi-turn conversations with history management.
   * Real-time SSE token streaming and abort/stop generation capability.
   * Full Markdown rendering with syntax highlighting and instant clipboard copying.
   * Dynamic system prompt profiles: *General Assistant, Coding Assistant, Debugging Assistant, Research Assistant, Project Analyst, Documentation Assistant, Engineering Assistant*.
   * **Project-Aware Chat**: Attach specific workspace files directly into the prompt context with interactive checkboxes.

2. **Hierarchical Document & File Summarizer**:
   * Multi-format parser supporting `.txt`, `.md`, `.py`, `.js`, `.ts`, `.html`, `.css`, `.json`, `.yaml`, `.yml`, `.csv`, `.pdf`, `.docx`.
   * Recursive token estimation and sliding window chunking.
   * Automatic **hierarchical multi-stage synthesis** for large documents to prevent context window overflow.
   * One-click specialized actions: *Summarize, Extract Key Points, Explain Code, Find Bugs, Ask Questions*.

3. **Project Workspace Analyzer & AI Debugger**:
   * Workspace directory tree scanner with ignore filters (`.git`, `node_modules`, `__pycache__`, binaries).
   * Static heuristic health checks (README documentation, dependency manifests, local checkpoints).
   * Comprehensive architectural report generation with component breakdowns and code quality recommendations.
   * **Diagnostic Error Debugger**: Input error messages, stack traces, and relevant files for root-cause analysis and automated code fixes.

4. **Model Control & Settings**:
   * Hot-swapping between Ollama Mode and Local Model Mode without restarting the application.
   * 5-point automated Ollama diagnostic test.
   * Local checkpoint validator and test generator.
   * System prompt custom profile creator.

---

## Quick Start Guide

### Prerequisites
* Python 3.10+ (Python 3.14 supported)
* (Optional for Mode 1) [Ollama](https://ollama.com/) with `qwen2.5:7b` installed:
  ```powershell
  ollama pull qwen2.5:7b
  ollama serve
  ```

### 1. One-Click Launch (Windows)

#### Option A: Using Batch Script
Double-click `start.bat` or run:
```cmd
start.bat
```

#### Option B: Using PowerShell
```powershell
.\start.ps1
```

#### Option C: Using Python
```powershell
py start_studio.py
```

The launcher will verify Ollama and local model availability, start the FastAPI server on `http://127.0.0.1:8000`, and automatically open your default web browser.

---

## REST API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | System health and backend connectivity status |
| `GET` | `/api/models` | List available Ollama models and local checkpoints |
| `POST` | `/api/models/test-ollama` | Run 5-stage Ollama connection and inference test |
| `POST` | `/api/models/test-local` | Validate and test local model in `./models/` |
| `POST` | `/api/models/switch` | Hot-swap active mode (`ollama` vs `local_model`) |
| `POST` | `/api/chat` | Non-streaming multi-turn chat |
| `POST` | `/api/chat/stream` | Server-Sent Events (SSE) streaming chat |
| `GET` | `/api/chat/conversations` | List conversation sessions |
| `POST` | `/api/files/upload` | Upload and extract text from document/code files |
| `POST` | `/api/files/summarize` | Single-pass or hierarchical chunk summarization |
| `GET` | `/api/project/tree` | Retrieve structured workspace directory tree |
| `GET` | `/api/project/files` | Scannable workspace files for context injection |
| `POST` | `/api/project/analyze` | Run architectural and code quality audit |
| `POST` | `/api/project/debug` | AI diagnostic debugging assistant |
| `GET` | `/api/settings` | Get runtime configurations and prompt profiles |
| `POST` | `/api/settings/profile` | Create and save custom system prompt profile |

---

## Running the Automated Test Suite

Run the full end-to-end studio test suite:
```powershell
py tests/test_studio_api.py
```

Run the Experiment 7 evaluation pipeline test suite:
```powershell
py verify_test_case.py
```

---

## Directory Structure

```text
llm_text_summarization_project/
├── backend/
│   ├── api/                  # FastAPI router endpoints (chat, models, files, project, settings)
│   ├── config/               # Settings manager
│   ├── services/             # LLMProvider, OllamaProvider, LocalModelProvider, FileService, Summarizer, Analyzer
│   ├── utils/                # Safe structured logger and security utilities
│   └── main.py               # FastAPI application entrypoint
├── config/
│   ├── api_config.json       # Optional external commercial API config
│   ├── model_config.json     # Model hyperparams & studio runtime settings
│   └── prompts.json          # System profiles & task templates
├── data/
│   ├── processed/            # Cleaned corpus and test splits
│   └── raw/                  # Raw dataset files
├── frontend/
│   ├── css/style.css         # Glassmorphism and custom styling
│   ├── js/                   # app.js, chat.js, files.js, project.js, settings.js
│   └── index.html            # SPA dashboard
├── models/
│   └── saved_models/         # Local Hugging Face checkpoints (model.safetensors, config.json)
├── results/                  # Summaries, metrics, evaluation reports, and charts
├── tests/
│   └── test_studio_api.py    # Automated Studio test suite
├── main.py                   # Experiment 7 CLI pipeline runner
├── verify_test_case.py       # Experiment 7 evaluation unit tests
├── start_studio.py           # Studio launcher
├── start.bat                 # Windows Batch launcher
├── start.ps1                 # Windows PowerShell launcher
└── requirements.txt          # Python dependencies
```
