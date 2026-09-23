from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

from backend.config.settings import settings
from backend.utils.logger import studio_logger

router = APIRouter(prefix="/api/settings", tags=["Settings"])

class UpdateSettingsRequest(BaseModel):
    active_mode: Optional[str] = None
    active_ollama_model: Optional[str] = None
    ollama_base_url: Optional[str] = None
    ollama_url: Optional[str] = None
    ollama_model: Optional[str] = None
    active_local_model_path: Optional[str] = None
    active_system_profile: Optional[str] = None
    active_system_prompt: Optional[str] = None
    max_chunk_tokens: Optional[int] = None
    chunk_overlap_tokens: Optional[int] = None

class CustomProfileRequest(BaseModel):
    profile_id: str = Field(..., description="Unique profile identifier, e.g. 'custom_academic'")
    name: str = Field(..., description="Human readable name")
    description: str = Field(..., description="Short description")
    prompt: str = Field(..., description="System prompt text")

@router.get("")
async def get_settings():
    """Retrieve current system configuration and available profiles."""
    return {
        "settings": settings.to_dict(),
        "profiles": settings.get_all_profiles()
    }

@router.post("")
@router.put("")
async def update_settings(req: UpdateSettingsRequest):
    """Update active runtime studio settings."""
    if req.active_mode:
        settings.active_mode = req.active_mode.lower()
    if req.active_ollama_model or req.ollama_model:
        settings.active_ollama_model = req.active_ollama_model or req.ollama_model
    if req.ollama_base_url or req.ollama_url:
        settings.ollama_base_url = req.ollama_base_url or req.ollama_url
    if req.active_local_model_path:
        settings.active_local_model_path = req.active_local_model_path
    if req.active_system_profile:
        settings.active_system_profile = req.active_system_profile
        settings.active_system_prompt = settings.get_profile_prompt(req.active_system_profile)
    if req.active_system_prompt:
        settings.active_system_prompt = req.active_system_prompt
    if req.max_chunk_tokens:
        settings.max_chunk_tokens = req.max_chunk_tokens
    if req.chunk_overlap_tokens:
        settings.chunk_overlap_tokens = req.chunk_overlap_tokens

    studio_logger.info("Updated studio settings.")
    return {
        "status": "success",
        "settings": settings.to_dict()
    }

@router.post("/profile")
@router.post("/profiles")
async def save_profile(req: CustomProfileRequest):
    """Save or update a system prompt profile."""
    settings.save_custom_profile(
        profile_id=req.profile_id,
        name=req.name,
        description=req.description,
        prompt=req.prompt
    )
    return {
        "status": "success",
        "message": f"Profile '{req.name}' saved successfully.",
        "profiles": settings.get_all_profiles()
    }
