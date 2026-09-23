import time
import requests
from flask import Blueprint, jsonify, current_app
from app.llm import OllamaClient, OpenAIClient, GeminiClient
from app.llm.model_selector import ModelSelector

models_bp = Blueprint("models", __name__, url_prefix="/api/models")

@models_bp.route("/", methods=["GET"])
def get_available_models():
    """Discover available local Ollama models with tier capabilities and cloud fallback models."""
    ollama_client = OllamaClient(base_url=current_app.config["OLLAMA_BASE_URL"])
    local_models = ollama_client.list_models()

    cloud_models = []
    if current_app.config.get("OPENAI_API_KEY"):
        cloud_models.extend(OpenAIClient().list_models())
    if current_app.config.get("GEMINI_API_KEY"):
        cloud_models.extend(GeminiClient().list_models())

    return jsonify({
        "default_model": current_app.config.get("DEFAULT_MODEL", "qwen2.5:7b"),
        "local_models": local_models,
        "cloud_models": cloud_models,
        "model_matrix": ModelSelector.get_all_models_metadata(),
    })

@models_bp.route("/health", methods=["GET"])
def get_ollama_health():
    """Health check for local Ollama daemon status and latency."""
    base_url = current_app.config["OLLAMA_BASE_URL"]
    start_t = time.time()
    try:
        res = requests.get(f"{base_url}/api/tags", timeout=3)
        latency_ms = round((time.time() - start_t) * 1000, 2)
        if res.status_code == 200:
            data = res.json()
            models = [m["name"] for m in data.get("models", [])]
            return jsonify({
                "status": "online",
                "base_url": base_url,
                "latency_ms": latency_ms,
                "installed_models": models,
                "model_count": len(models)
            })
    except Exception as e:
        latency_ms = round((time.time() - start_t) * 1000, 2)
        return jsonify({
            "status": "offline",
            "base_url": base_url,
            "latency_ms": latency_ms,
            "error": str(e),
            "message": "Ollama local service is not responding. Ensure Ollama is running (`ollama serve`)."
        }), 503
