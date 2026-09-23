import json
import os
from pathlib import Path
from typing import Dict, Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"

class StudioSettings:
    """Manages system configuration, active LLM provider modes, and prompt templates for Grammar Studio."""

    def __init__(self):
        self.project_root = PROJECT_ROOT
        self.config_dir = CONFIG_DIR
        self.models_dir = MODELS_DIR
        self.model_config_path = CONFIG_DIR / "model_config.json"
        self.api_config_path = CONFIG_DIR / "api_config.json"
        self.prompts_path = CONFIG_DIR / "prompts.json"
        
        self.raw_model_config = self._load_json(self.model_config_path)
        self.raw_api_config = self._load_json(self.api_config_path)
        self.raw_prompts = self._load_json(self.prompts_path)

        server_cfg = self.raw_api_config.get("server", {})
        self.server_host = os.getenv("STUDIO_HOST", server_cfg.get("host", "127.0.0.1"))
        self.server_port = int(os.getenv("STUDIO_PORT", server_cfg.get("port", 8501)))

        studio_cfg = self.raw_model_config.get("studio", {})
        self.default_mode = studio_cfg.get("default_mode", "ollama")
        self.active_mode = self.default_mode
        
        ollama_cfg = studio_cfg.get("ollama", {})
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", ollama_cfg.get("base_url", "http://localhost:11434"))
        self.ollama_default_model = os.getenv("OLLAMA_MODEL", ollama_cfg.get("default_model", "qwen2.5:7b"))
        self.ollama_timeout = ollama_cfg.get("timeout_seconds", 120)
        self.active_ollama_model = self.ollama_default_model

        local_cfg = studio_cfg.get("local_model", {})
        self.local_model_dir = str((PROJECT_ROOT / local_cfg.get("directory", "./models")).resolve())
        self.preferred_local_model_path = str((PROJECT_ROOT / local_cfg.get("preferred_model_path", "./models/saved_models/grammar_corrector_model")).resolve())
        self.active_local_model_path = self.preferred_local_model_path

        chunking_cfg = studio_cfg.get("chunking", {})
        self.max_chunk_tokens = chunking_cfg.get("max_chunk_tokens", 1500)
        self.chunk_overlap_tokens = chunking_cfg.get("chunk_overlap_tokens", 150)
        self.hierarchical_threshold_tokens = chunking_cfg.get("hierarchical_threshold_tokens", 2500)

        self.default_system_profile = studio_cfg.get("default_system_profile", "standard_corrector")
        self.active_system_profile = self.default_system_profile
        self.active_system_prompt = self.get_profile_prompt(self.active_system_profile)

    def _load_json(self, path: Path) -> Dict[str, Any]:
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def get_profile_prompt(self, profile_key: str) -> str:
        profiles = self.raw_prompts.get("profiles", {})
        if profile_key in profiles:
            return profiles[profile_key].get("prompt", "")
        return profiles.get("standard_corrector", {}).get(
            "prompt", 
            "You are an expert Grammar Correction, Proofreading, and Academic Editing Assistant."
        )

    def get_all_profiles(self) -> Dict[str, Any]:
        return self.raw_prompts.get("profiles", {})

    def get_task_template(self, task_name: str) -> str:
        templates = self.raw_prompts.get("task_templates", {})
        return templates.get(task_name, "{text}")

    def save_custom_profile(self, profile_id: str, name: str, description: str, prompt: str):
        if "profiles" not in self.raw_prompts:
            self.raw_prompts["profiles"] = {}
        self.raw_prompts["profiles"][profile_id] = {
            "name": name,
            "description": description,
            "prompt": prompt
        }
        with open(self.prompts_path, "w", encoding="utf-8") as f:
            json.dump(self.raw_prompts, f, indent=2)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "active_mode": self.active_mode,
            "ollama": {
                "base_url": self.ollama_base_url,
                "active_model": self.active_ollama_model,
                "default_model": self.ollama_default_model,
                "timeout": self.ollama_timeout
            },
            "local_model": {
                "directory": self.local_model_dir,
                "active_model_path": self.active_local_model_path,
                "preferred_model_path": self.preferred_local_model_path
            },
            "system_profile": {
                "active_profile": self.active_system_profile,
                "active_prompt": self.active_system_prompt
            },
            "chunking": {
                "max_chunk_tokens": self.max_chunk_tokens,
                "chunk_overlap_tokens": self.chunk_overlap_tokens,
                "hierarchical_threshold": self.hierarchical_threshold_tokens
            }
        }

settings = StudioSettings()
