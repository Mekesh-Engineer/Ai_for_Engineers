# Experiment 8: Grammar Correction & Text Rewriting using Large Language Models (LLM)

A production-grade, dual-mode AI workspace and evaluation framework for **Automated Grammar Error Correction (GEC)**, multi-level prompt rewriting, and quantitative linguistic evaluation.

---

## 🚀 Key Features & Capabilities

### 1. Dual-Mode Local LLM Architecture
- **Mode 1 — Ollama Mode**: High-performance local REST API integration with `qwen2.5:7b` for conversational proofreading, style rewriting, and error diagnosis.
- **Mode 2 — Project Local Model Mode**: Standalone offline Transformer Seq2Seq checkpoint loaded from `./models/saved_models/grammar_corrector_model/` with process-level in-memory caching.

### 2. Multi-Level Prompt Engineering
- **Level 1 — Minimal Correction**: Focuses strictly on spelling mistakes, typos, and obvious verb errors while preserving author voice.
- **Level 2 — Standard GEC**: Comprehensive error correction across subject-verb agreement, verb tenses, prepositions, articles, and punctuation.
- **Level 3 — Comprehensive Rewriting**: Restructures sentences for maximum fluency, clarity, and readability.
- **Level 4 — Academic & Scientific Polish**: Elevates register to formal research publication standards with concise syntax.
- **Level 5 — Grammar Error Diagnostician**: Identifies, categorizes, and explains each error with before-and-after rationale.

### 3. Quantitative Evaluation Benchmarks
- **Exact Match %**: Ratio of generated corrections matching human gold standard.
- **Levenshtein Edit Distance**: Character-level minimum edit distance.
- **Token-Level Precision, Recall, and F1**: Overlap score between output and reference tokens.
- **Over-Correction & Under-Correction Detection**: Identifies unnecessary changes and missed errors.
- **Semantic Preservation**: Keyword retention score.

### 4. Local LLM Studio Web Interface
- **Dashboard & Metric Visualizations**: Real-time performance breakdown and distribution charts.
- **Interactive AI Chat**: Center conversation panel with fixed header, dedicated scrollbar on `#chat-messages-container`, prompt profiles, and file attachments.
- **Document Proofreader**: Multi-format document parser (.txt, .md, .py, .pdf, .docx, .csv, .json) with side-by-side diff preview.
- **Doc & Code Auditor**: Interactive repository tree, docstring grammar analysis, and error debugger.
- **Model Workbench**: Live diagnostics, latency benchmarks, and custom prompt creator.

---

## 📂 Project Structure

```text
llm_grammar_correction_project/
├── backend/
│   ├── api/                     # FastAPI route endpoints
│   │   ├── chat.py              # Multi-turn chat & SSE streaming
│   │   ├── files.py             # File upload & document proofreading
│   │   ├── models.py            # Mode switching & diagnostics
│   │   ├── project.py           # Workspace tree & doc auditor
│   │   └── settings.py          # Configuration & profiles
│   ├── config/                  # Settings and path resolution
│   ├── services/                # Backend business logic & LLM abstraction
│   │   ├── corrector.py         # Multi-chunk document proofreader
│   │   ├── file_service.py      # Multi-format document extractor
│   │   ├── llm_service.py       # Unified LLMProvider interface
│   │   ├── local_model_service.py # Transformers Seq2Seq runtime
│   │   ├── ollama_service.py    # Ollama REST API client
│   │   └── project_analyzer.py  # Repository documentation auditor
│   ├── utils/                   # Sanitizing logger & security validation
│   └── main.py                  # FastAPI application entrypoint
├── config/
│   ├── api_config.json          # Server & Ollama port configurations
│   ├── model_config.json        # Hyperparameters & generation defaults
│   └── prompts.json             # Role profiles & task prompt templates
├── data/
│   ├── raw/                     # Lang-8 & JFLEG raw corpora
│   │   ├── lang8_errors.csv
│   │   └── jfleg_dataset.csv
│   ├── processed/               # Cleaned & normalized sentence pairs
│   └── outputs/                 # Benchmark outputs and predictions
├── models/
│   ├── prompts/                 # Text prompt templates
│   └── saved_models/            # Local model weights (Safetensors / Configs)
├── notebooks/
│   ├── data_analysis.ipynb      # Corpus distribution & error type analytics
│   └── evaluation.ipynb         # Prompt comparison & metric visualizations
├── results/                     # Metric CSVs and publication-grade plots
│   ├── prompt_comparison.png
│   ├── error_type_distribution.png
│   ├── metric_distributions.png
│   ├── over_vs_under_correction.png
│   ├── accuracy_scores.csv
│   └── evaluation_report.txt
├── src/                         # Core Machine Learning Pipeline
│   ├── data_loader.py           # Corpus acquisition & splitting
│   ├── data_preprocessing.py    # Normalization & error tagging
│   ├── prompt_templates.py      # Prompt management
│   ├── corrector.py             # Inference pipeline & diff markup
│   ├── evaluation.py            # Linguistic metrics suite
│   ├── error_analyzer.py        # Error categorization diagnostics
│   └── visualization.py         # Matplotlib chart generator
├── tests/
│   └── test_studio_api.py       # 10 automated backend API tests
├── frontend/                    # Local LLM Studio Web UI
├── main.py                      # 7-stage ML pipeline orchestrator
├── verify_test_case.py          # 13 automated ML unit tests
├── start_studio.py              # Studio launcher script
├── requirements.txt             # Python dependencies
└── README.md
```

---

## 🛠️ Quick Start

### 1. Run Core ML Pipeline
Executes dataset loading, preprocessing, model inference, evaluation metrics, and generates visualization plots:
```powershell
py main.py
```

### 2. Run Automated Verification Suites
```powershell
py verify_test_case.py
py tests/test_studio_api.py
```

### 3. Launch Local LLM Studio Web Interface
```powershell
py start_studio.py
```
Open your browser at **`http://127.0.0.1:8501`**.
