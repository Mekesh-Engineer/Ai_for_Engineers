# Local LLM Studio — Troubleshooting & Diagnostics Guide

This document outlines resolutions for common scenarios when running Local LLM Studio in either **Ollama Mode** or **Project Local Model Mode**.

---

## 1. Ollama Connectivity Issues

### Symptom: `Ollama is not running or cannot be reached at http://localhost:11434`

**Root Cause**: The Ollama background daemon is stopped or listening on a different port.

**Resolution**:
1. Open a new terminal / PowerShell window and start the Ollama daemon:
   ```powershell
   ollama serve
   ```
2. Verify Ollama is running and responding:
   ```powershell
   ollama list
   ```
3. Test HTTP connectivity:
   ```powershell
   curl http://localhost:11434/api/tags
   ```
4. If Ollama runs on a non-default host or port (e.g. Docker or remote machine), set the environment variable:
   ```powershell
   $env:OLLAMA_BASE_URL="http://127.0.0.1:11434"
   ```
   or update it directly in the **Models & Settings** tab in the web UI.

---

## 2. Missing Qwen 2.5 7B Model in Ollama

### Symptom: `Model 'qwen2.5:7b' not found in installed models`

**Root Cause**: The Qwen 2.5 7B weights have not been downloaded to your local Ollama library.

**Resolution**:
1. Pull the model using the Ollama CLI:
   ```powershell
   ollama pull qwen2.5:7b
   ```
2. Confirm installation:
   ```powershell
   ollama list
   ```
3. In the Web UI (**Models & Settings** tab), click **[Refresh Models]** to update the model dropdown list.

---

## 3. Local Model Detection in `./models/`

### Symptom: `No compatible local model checkpoint found at ./models/`

**Root Cause**: Local Model Mode requires a valid Hugging Face Transformers model directory containing `config.json` and weight files (`model.safetensors` or `pytorch_model.bin`).

**Resolution**:
1. Verify that `./models/saved_models/summarizer_model/` contains:
   * `config.json`
   * `model.safetensors` (or `pytorch_model.bin`)
   * `tokenizer.json` and `tokenizer_config.json`
2. If the files are missing, run Experiment 7 training/export:
   ```powershell
   py main.py
   ```
   which automatically generates and persists the verified local model checkpoint to `./models/saved_models/summarizer_model`.

---

## 4. Port 8000 Already in Use

### Symptom: `[Errno 10048] error while attempting to bind on address ('127.0.0.1', 8000)`

**Root Cause**: Another application is currently using TCP port 8000.

**Resolution**:
1. Find the process using port 8000:
   ```powershell
   netstat -ano | findstr :8000
   ```
2. Terminate the process or change the port in `start_studio.py` / `config/model_config.json`:
   ```json
   "web_server": {
     "host": "127.0.0.1",
     "port": 8080
   }
   ```

---

## 5. Document Parsing (PDF / DOCX)

### Symptom: `Failed to parse PDF/DOCX document: ModuleNotFoundError`

**Root Cause**: Missing optional document processing packages.

**Resolution**:
Install `pypdf` and `python-docx`:
```powershell
py -m pip install pypdf python-docx
```

---

## 6. Context Window & Large Files

### Symptom: Large files or codebases taking significant processing time

**Resolution**:
* The studio incorporates **Hierarchical Chunking** for documents exceeding 2,500 estimated tokens.
* If a document is large, the studio automatically splits it into overlapping segments, generates section summaries, and aggregates them into a synthesized executive summary.
* In **Project-Aware Chat**, select only relevant files rather than checking the entire workspace to maintain fast response latency.
