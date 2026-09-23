import time
import asyncio
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional

from backend.config.settings import settings
from backend.services.ollama_service import OllamaProvider
from backend.services.local_model_service import LocalModelProvider
from backend.services.llm_service import get_llm_provider
from backend.utils.logger import studio_logger

router = APIRouter(prefix="/api/models", tags=["Models"])

class SwitchModelRequest(BaseModel):
    mode: str = Field(..., description="'ollama' or 'local_model'")
    model_name_or_path: Optional[str] = Field(None, description="Model identifier for Ollama or directory path for Local Model")

class ModelComparisonRequest(BaseModel):
    prompt: str = Field(..., description="Sentence or text prompt to evaluate across models")
    system_prompt: Optional[str] = None

@router.get("")
@router.get("/overview")
@router.get("/health")
async def get_models_overview():
    """Retrieve full status of both Ollama and Local Model backends."""
    ollama_prov = OllamaProvider()
    local_prov = LocalModelProvider()

    ollama_info = await ollama_prov.get_model_info()
    local_info = await local_prov.get_model_info()

    return {
        "active_mode": settings.active_mode,
        "ollama": ollama_info,
        "local_model": local_info
    }

@router.get("/list")
async def list_available_models():
    """List all detected models across both Ollama and local storage."""
    ollama_prov = OllamaProvider()
    local_prov = LocalModelProvider()

    ollama_models = []
    if await ollama_prov.is_available():
        models_data = await ollama_prov.list_models()
        ollama_models = [m["name"] for m in models_data]

    local_models = local_prov.scan_models()

    return {
        "active_mode": settings.active_mode,
        "ollama_models": ollama_models,
        "local_models": local_models
    }

@router.post("/test-ollama")
async def test_ollama_connection():
    """Execute complete diagnostic test for Ollama backend."""
    ollama_prov = OllamaProvider()
    results = await ollama_prov.test_connection()
    return results

@router.post("/test-local")
async def test_local_connection():
    """Execute validation and diagnostic test for Local Model backend."""
    local_prov = LocalModelProvider()
    results = await local_prov.test_connection()
    return results

@router.post("/switch")
async def switch_model_mode(req: SwitchModelRequest):
    """Switch the active LLM mode and active model."""
    mode = req.mode.lower()
    if mode not in ["ollama", "local_model"]:
        raise HTTPException(status_code=400, detail=f"Invalid mode '{req.mode}'. Must be 'ollama' or 'local_model'.")

    settings.active_mode = mode
    if mode == "ollama" and req.model_name_or_path:
        settings.active_ollama_model = req.model_name_or_path
    elif mode == "local_model" and req.model_name_or_path:
        settings.active_local_model_path = req.model_name_or_path

    studio_logger.info(f"Switched model backend to mode='{settings.active_mode}', model='{req.model_name_or_path or 'default'}'")

    return {
        "status": "success",
        "active_mode": settings.active_mode,
        "active_ollama_model": settings.active_ollama_model,
        "active_local_model_path": settings.active_local_model_path
    }

@router.post("/compare")
async def compare_models_endpoint(req: ModelComparisonRequest):
    """Execute side-by-side inference across Ollama and Project Local Model."""
    ollama_prov = OllamaProvider()
    local_prov = LocalModelProvider()

    ollama_out = {"text": "Ollama service unavailable", "duration": 0.0}
    local_out = {"text": "Local model unavailable", "duration": 0.0}

    # Run Ollama inference
    t0 = time.time()
    try:
        res = await ollama_prov.generate(prompt=req.prompt, system_prompt=req.system_prompt)
        ollama_out = {
            "text": res.text,
            "duration": round(time.time() - t0, 2)
        }
    except Exception as e:
        ollama_out = {"text": f"Ollama error: {str(e)}", "duration": round(time.time() - t0, 2)}

    # Run Local Model inference
    t1 = time.time()
    try:
        res = await local_prov.generate(prompt=req.prompt, system_prompt=req.system_prompt)
        local_out = {
            "text": res.text,
            "duration": round(time.time() - t1, 2)
        }
    except Exception as e:
        local_out = {"text": f"Local model error: {str(e)}", "duration": round(time.time() - t1, 2)}

    return {
        "prompt": req.prompt,
        "ollama": ollama_out,
        "local_model": local_out
    }
