# VOLTIX — Advanced Electrical & Electronics Engineering AI Platform

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-Flask%203.0%2B-green.svg)](https://flask.palletsprojects.com/)
[![Vector DB](https://img.shields.io/badge/Vector%20Store-FAISS-orange.svg)](https://github.com/facebookresearch/faiss)
[![LLM Inference](https://img.shields.io/badge/Local%20LLM-Ollama-purple.svg)](https://ollama.ai/)
[![Tests](https://img.shields.io/badge/Tests-36%20Passed-brightgreen.svg)](tests/)

**VOLTIX** is a production-grade, offline-first conversational AI assistant and RAG platform engineered specifically for **Electrical & Electronics Engineering (EEE)**. It integrates local LLMs (via Ollama), FAISS dense vector search, SymPy symbolic solvers, Pint unit converters, and MATLAB/Simulink script generators into a ChatGPT-style web interface.

---

## ⚡ Key Capabilities

### 1. Offline-First & Local LLM Inference
- **Local Ollama Daemon**: Powered by models such as `qwen2.5:7b`, `llama3.1:8b`, `deepseek-r1`, or `mistral`.
- **Dynamic Model Discovery & Health Checking**: Automatically detects installed models and server availability.
- **Server-Sent Events (SSE) Streaming**: Low-latency token-by-token streaming with real-time `AbortController` cancellation support.
- **Optional Cloud Fallbacks**: Google Gemini (`gemini-1.5-flash`, `gemini-2.0-flash`) and OpenAI (`gpt-4o`).

### 2. Multi-Disciplinary EEE Domain Agents
Includes 10 specialized agent personas with custom domain prompts:
1. **Electrical Machines**: Induction motors, synchronous machines, transformers, torque-speed curves.
2. **Power Systems**: Load flow, symmetrical/unsymmetrical fault analysis, per-unit systems, relay protection.
3. **Power Electronics**: Buck, Boost, Buck-Boost converters, inverters, PWM modulation, THD analysis.
4. **Control Systems**: Transfer functions, Laplace transforms, PID tuning, Bode/Nyquist plots, root locus.
5. **Circuit Theory**: KVL/KCL, Thevenin/Norton theorems, AC phasor analysis, RLC transients.
6. **Electric Vehicles (EV)**: Powertrain tractive effort, traction motor sizing, regenerative braking.
7. **Renewable Energy**: Solar PV arrays, MPPT (P&O / Incremental Conductance), wind turbines, microgrids.
8. **Battery & BMS**: Lithium-ion chemistries, SoC/SoH estimation, cell balancing, thermal management.
9. **Analog & Digital Electronics**: Op-Amps, active filters, small-signal models, logic design, ADC/DAC.
10. **MATLAB & Simulink**: Syntax-valid MATLAB scripts and Simulink model architecture blueprints.

### 3. Seven Academic & Engineering Modes
- 📖 **Learn**: Structured conceptual explanations with physical intuition and analogies.
- ✍️ **Practice**: Step-by-step problem solving with progressive hints.
- 📝 **Exam**: University/GATE style solution formatting with mark weightings and boxed answers.
- 🎯 **Viva**: Oral examination preparation with probing questions, common traps, and examiner follow-ups.
- 🔬 **Laboratory**: Full experiment manuals (Aim, Apparatus, Circuit Diagram, Tabular Column, Procedure, Precautions, Viva).
- 📊 **Research**: IEEE journal style with analytical formulations and state-of-the-art comparisons.
- 🛠️ **Project**: Engineering design with Bill of Materials (BOM) and hardware/firmware interfacing.

### 4. Deterministic Engineering Calculators & Solvers
- **Three-Phase & Single-Phase Power**: $P = \sqrt{3} V_L I_L \cos\phi$, line currents, reactive and apparent power.
- **Ohm's Law**: $V = IR, P = I^2 R = V^2 / R$.
- **Motor Slip & Torque**: $s = (N_s - N)/N_s$, $N_s = 120f/P$, $T = 60P / (2\pi N)$.
- **Power Factor Correction**: Capacitor bank reactive power ($Q_c$) and capacitance ($C$) sizing.
- **DC-DC Converters**: Duty cycle, inductor ripple $\Delta I_L$, capacitor ripple $\Delta V_c$, critical inductance $L_{min}$.
- **Battery & EV**: Pack kWh, discharge current at C-rate, driving range from Wh/km.
- **Solar PV**: Maximum Power Point ($P_{mp}$), Fill Factor ($FF$), array sizing.
- **SymPy Symbolic Math**: Algebraic equations, differentiation, integration, Laplace transforms, Inverse Laplace, transfer function poles/zeros, and RLC transient characteristics.
- **Pint Unit Converter**: Strict conversions across all EEE units ($kW \leftrightarrow W$, $rpm \leftrightarrow rad/s$, $kV \leftrightarrow V$, $\mu F$, $mH$).

### 5. Local RAG & Project Isolation
- **Supported Formats**: PDF, DOCX, TXT, MD, CSV.
- **Dense Vector Search**: FAISS index with `BAAI/bge-small-en-v1.5` or `all-MiniLM-L6-v2` embeddings.
- **Project Workspaces**: Independent knowledge bases where RAG search is strictly isolated to the active project.
- **Citation Badging**: Footnotes linking answers to specific source documents, page numbers, and similarity scores.
- **Prompt Injection Defense**: Untrusted retrieved content is isolated within strict security boundaries.

---

## 🏗️ Architecture & Codebase Structure

```text
domain_specific_qa_system_project/
├── app/
│   ├── __init__.py               # Flask Application Factory
│   ├── config.py                 # Configuration manager (YAML & .env)
│   ├── database.py               # Database initialization
│   ├── extensions.py             # SQLAlchemy & CORS extensions
│   │
│   ├── models/                   # SQLite Database Models
│   │   ├── project.py            # Project workspace entity
│   │   ├── conversation.py       # Conversation & Message entities
│   │   ├── document.py           # Document & Chunk metadata entities
│   │   ├── tool_log.py           # Engineering calculation audit logs
│   │   └── settings.py           # User preferences
│   │
│   ├── routes/                   # REST & SSE API Blueprints
│   │   ├── main.py               # Main UI rendering route
│   │   ├── chat.py               # /api/chat/stream SSE streaming
│   │   ├── documents.py          # /api/documents/ (Upload, List, Delete)
│   │   ├── conversations.py      # /api/conversations/ (CRUD, Search, Export)
│   │   ├── projects.py           # /api/projects/ (Project isolation CRUD)
│   │   ├── models.py             # /api/models/ (Discovery & Health check)
│   │   └── tools.py              # /api/tools/ (Interactive engineering solvers)
│   │
│   ├── services/                 # AI Orchestrator & Services
│   │   ├── router.py             # Intent & Agent routing engine
│   │   ├── prompt_builder.py     # 7 Academic modes & persona builder
│   │   ├── memory.py             # Multi-turn conversational memory
│   │   └── orchestrator.py       # Central pipeline coordinator
│   │
│   ├── agents/                   # 10 EEE Domain Specialist Agents
│   │   ├── base_agent.py         # Abstract base agent
│   │   ├── machines_agent.py
│   │   ├── power_systems_agent.py
│   │   ├── power_electronics_agent.py
│   │   ├── control_agent.py
│   │   ├── circuit_theory_agent.py
│   │   ├── ev_agent.py
│   │   ├── renewable_agent.py
│   │   ├── battery_agent.py
│   │   ├── electronics_agent.py
│   │   └── matlab_simulink_agent.py
│   │
│   ├── rag/                      # RAG & FAISS Vector Engine
│   │   ├── loader.py             # PDF, DOCX, TXT, MD, CSV loaders
│   │   ├── parser.py             # Noise cleaner & structure normalizer
│   │   ├── chunker.py            # Semantic overlapping chunker
│   │   ├── embedder.py           # SentenceTransformers encoder
│   │   ├── vector_store.py       # FAISS persistent index manager
│   │   ├── retriever.py          # Top-K similarity search with project filtering
│   │   └── citation.py           # Citation badging & security boundaries
│   │
│   ├── tools/                    # Deterministic Solvers
│   │   ├── ee_calculator.py      # Numerical EEE formulas
│   │   ├── sympy_solver.py       # Symbolic mathematics & Laplace
│   │   ├── unit_converter.py     # Pint-based engineering units
│   │   └── matlab_generator.py   # MATLAB code & Simulink blueprints
│   │
│   └── utils/                    # Utilities & Helpers
│       ├── logger.py             # Logging setup
│       ├── exceptions.py         # Custom application exceptions
│       └── validators.py         # File security & input validators
│
├── frontend/                     # Unified Modern ChatGPT-Style Frontend
│   ├── static/                   # Static CSS & JS Assets
│   │   ├── css/main.css          # Custom styling & scrollbar rules
│   │   └── js/
│   │       ├── api.js            # Frontend REST API client
│   │       ├── app.js            # Application entry point
│   │       ├── chat.js           # SSE stream controller & KaTeX auto-render
│   │       ├── code_highlighter.js # Highlight.js & one-click copy
│   │       ├── document_upload.js # Drag-and-drop file uploader
│   │       ├── math_renderer.js  # KaTeX math engine
│   │       └── ui.js             # Modals, tools tabs, project switcher
│   └── templates/                # Jinja2 HTML Templates
│       ├── base.html             # Master HTML layout
│       ├── chat.html             # ChatGPT-style interface
│       └── components/
│           ├── sidebar.html      # Sessions, model selector, modes, project picker
│           ├── calc_modal.html   # Interactive multi-tab engineering solvers
│           ├── project_modal.html # Project workspace manager
│           ├── settings_modal.html # LLM parameters & RAG configuration
│           └── source_modal.html # RAG document citation viewer
│
├── tests/                        # Pytest Automated Test Suite
│   ├── conftest.py               # Test client fixtures
│   ├── unit/                     # Unit tests
│   ├── integration/              # API and route integration tests
│   └── test_acceptance_scenarios.py # 11 Mandatory Acceptance Tests
│
├── config.yaml                   # System hyperparameters
├── requirements.txt              # Python dependencies
├── run.py                        # Application entry launcher
└── README.md                     # Project documentation
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites
- Python 3.10+ (Python 3.11 recommended)
- [Ollama](https://ollama.ai/) installed and running locally

### 2. Installation

```bash
# Clone or navigate to the repository
cd domain_specific_qa_system_project

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Pull Local LLM Model

```bash
# Pull recommended EEE model (qwen2.5:7b or llama3.1:8b)
ollama pull qwen2.5:7b
```

### 4. Run Automated Tests

```bash
pytest tests/ -v
```

### 5. Launch VOLTIX Web Application

```bash
python run.py
```

Open your browser at **`http://localhost:5000`** to access VOLTIX.

---

## 🧪 Acceptance Scenarios Verified

| # | Test Scenario | Verified Behavior |
|---|---|---|
| **1** | EEE Concept | Working principle of 3-phase induction motor correctly routed to `MachinesAgent` with KaTeX formatting. |
| **2** | Numerical Slip | $N_s = 1500$ RPM, $N = 1440$ RPM $\implies s = 4.0\%$ computed deterministically. |
| **3** | 3-Phase Current | $5$ kW, $415$ V, $0.8$ pf load $\implies I_L = 8.696$ A calculated accurately. |
| **4** | RAG Search | Ingests PDF/DOCX/TXT/MD/CSV and retrieves grounded context with cosine similarity scores. |
| **5** | Document Summary | Generates grounded summaries with citation footnotes. |
| **6** | MATLAB Generation | Synthesizes executable torque-slip and Bode plot scripts with parameter explanations. |
| **7** | Memory Persistence | Retains multi-turn conversation context across user turns in SQLite. |
| **8** | Model Selection | Queries `/api/models/` and `/api/models/health` for live Ollama daemon discovery. |
| **9** | Conversation History | Supports session loading, search, renaming, and Markdown/JSON export. |
| **10** | Document Lifecycle | Supports upload, indexing, querying, and full deletion (un-indexing from FAISS). |
| **11** | Offline Mode | Runs 100% locally with offline embeddings, FAISS, SymPy, and Ollama. |

---

## 📄 License & Attribution

Designed and developed for Electrical & Electronics Engineering research and education.
