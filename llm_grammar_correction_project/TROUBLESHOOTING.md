# Troubleshooting & Operational Guide

## Local LLM Studio — Grammar Correction & Text Rewriting

---

### 1. Ollama Connectivity Issues

#### Symptom:
Badge displays `Ollama Offline` or `Failed to connect to Ollama service`.

#### Resolution:
1. Ensure the Ollama daemon is running:
   ```powershell
   ollama serve
   ```
2. Verify installed models:
   ```powershell
   ollama list
   ```
3. If `qwen2.5:7b` is missing, download it:
   ```powershell
   ollama pull qwen2.5:7b
   ```
4. Verify port binding (`http://localhost:11434/api/tags`).

---

### 2. Local Model Weight Verification

#### Symptom:
`No valid local model checkpoint found at ./models/`.

#### Resolution:
1. Confirm directory structure:
   ```text
   models/
   └── saved_models/
       └── grammar_corrector_model/
           ├── config.json
           ├── model.safetensors
           ├── tokenizer_config.json
           └── vocab.json
   ```
2. Run validation check:
   ```powershell
   py verify_test_case.py
   ```

---

### 3. Missing Python Dependencies

#### Resolution:
Install dependencies via pip:
```powershell
pip install -r requirements.txt
```

---

### 4. Port Conflict on Port 8501

#### Symptom:
`[Errno 10048] error while attempting to bind on address ('127.0.0.1', 8501)`.

#### Resolution:
Identify and kill existing process or configure an alternative port:
```powershell
netstat -ano | findstr :8501
taskkill /PID <PID> /F
```
Or start Uvicorn directly on custom port:
```powershell
uvicorn backend.main:app --port 8502
```
