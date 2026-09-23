import os
import sys
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse

from backend.config.settings import settings, PROJECT_ROOT
from backend.utils.logger import studio_logger
from backend.services.ollama_service import OllamaProvider
from backend.services.local_model_service import LocalModelProvider

from backend.api.models import router as models_router
from backend.api.chat import router as chat_router
from backend.api.files import router as files_router
from backend.api.project import router as project_router
from backend.api.settings import router as settings_router

FRONTEND_DIR = PROJECT_ROOT / "frontend"

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup sequence
    studio_logger.info("=" * 60)
    studio_logger.info("  LOCAL LLM STUDIO — Starting Backend Services")
    studio_logger.info(f"  Project Root: {PROJECT_ROOT}")
    studio_logger.info(f"  Active Mode:  {settings.active_mode}")
    studio_logger.info(f"  Ollama URL:   {settings.ollama_base_url} (Model: {settings.active_ollama_model})")
    studio_logger.info(f"  Local Models: {settings.local_model_dir}")
    studio_logger.info("=" * 60)
    
    # Pre-check Ollama status asynchronously
    try:
        ollama_ready = await OllamaProvider().is_available()
        studio_logger.info(f"Ollama Connectivity Status: {'CONNECTED' if ollama_ready else 'NOT RUNNING (Run `ollama serve`)'}")
    except Exception as e:
        studio_logger.warning(f"Ollama check warning: {e}")

    # Pre-check Local Model status
    try:
        local_ready = await LocalModelProvider().is_available()
        studio_logger.info(f"Local Model Status: {'AVAILABLE' if local_ready else 'NOT FOUND'}")
    except Exception as e:
        studio_logger.warning(f"Local model check warning: {e}")

    yield
    studio_logger.info("Local LLM Studio shutting down.")

app = FastAPI(
    title="Local LLM Studio",
    description="Dual-Mode Local AI Workspace (Ollama Mode + Project Local Model Mode)",
    version="1.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(models_router)
app.include_router(chat_router)
app.include_router(files_router)
app.include_router(project_router)
app.include_router(settings_router)

# Health Check Endpoint
@app.get("/api/health")
async def health_check():
    ollama_ready = await OllamaProvider().is_available()
    local_ready = await LocalModelProvider().is_available()
    return {
        "status": "healthy",
        "backend": True,
        "active_mode": settings.active_mode,
        "ollama_connected": ollama_ready,
        "local_model_available": local_ready,
        "ollama_model": settings.active_ollama_model,
        "version": "1.0.0"
    }

# Static file serving for Frontend
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
    if (FRONTEND_DIR / "css").exists():
        app.mount("/css", StaticFiles(directory=str(FRONTEND_DIR / "css")), name="css")
    if (FRONTEND_DIR / "js").exists():
        app.mount("/js", StaticFiles(directory=str(FRONTEND_DIR / "js")), name="js")

@app.get("/")
async def serve_index():
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return HTMLResponse("<h1>Local LLM Studio</h1><p>Frontend under construction.</p>")
