import json
import requests
from typing import Generator, List, Dict, Any, Optional
from app.llm.base import BaseLLMClient
from app.utils.logger import get_logger
from app.utils.exceptions import LLMConnectionError

logger = get_logger("voltix.ollama")

class OllamaClient(BaseLLMClient):
    """Local Ollama REST API client supporting real-time token streaming and native tool calling."""

    def __init__(self, base_url: str = "http://localhost:11434", default_model: str = "qwen2.5:7b"):
        self.base_url = base_url.rstrip("/")
        self.default_model = default_model

    def list_models(self) -> List[str]:
        """Fetch models installed in local Ollama daemon."""
        url = f"{self.base_url}/api/tags"
        try:
            res = requests.get(url, timeout=5)
            if res.status_code == 200:
                data = res.json()
                models = [m["name"] for m in data.get("models", [])]
                if models:
                    return models
        except Exception as e:
            logger.warning(f"Could not connect to Ollama daemon at {self.base_url}: {e}")
        return [
            "deepseek-v4-flash:cloud",
            "gemma4:cloud",
            "gpt-oss:120b-cloud",
            "qwen2.5:3b",
            "qwen2.5:7b",
            "gemma4:31b-cloud"
        ]

    def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        model_name: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Non-streaming chat completion with optional native tool calling."""
        model = model_name or self.default_model
        url = f"{self.base_url}/api/chat"

        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": options or {"temperature": 0.1, "top_p": 0.9}
        }

        if tools:
            payload["tools"] = tools

        try:
            response = requests.post(url, json=payload, timeout=90)
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Ollama chat error {response.status_code}: {response.text}")
                # Fallback to local 7B if a cloud model returns error
                if model != self.default_model:
                    logger.info(f"Retrying chat completion with fallback model: {self.default_model}")
                    payload["model"] = self.default_model
                    fb_res = requests.post(url, json=payload, timeout=90)
                    if fb_res.status_code == 200:
                        return fb_res.json()
                return {"message": {"role": "assistant", "content": f"Ollama Error: {response.text}"}}
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to connect to Ollama: {e}")
            raise LLMConnectionError(f"Ollama service unreachable at {self.base_url}.") from e

    def stream_generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        model_name: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Generator[str, None, None]:
        """Stream NDJSON tokens from Ollama /api/chat endpoint."""
        model = model_name or self.default_model
        url = f"{self.base_url}/api/chat"

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        if history:
            for item in history:
                messages.append({"role": item.get("sender", "user"), "content": item.get("content", "")})

        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "options": options or {"temperature": 0.2, "top_p": 0.9}
        }

        try:
            response = requests.post(url, json=payload, stream=True, timeout=90)
            if response.status_code != 200:
                # Try fallback to local default model
                if model != self.default_model:
                    logger.warning(f"Model {model} failed ({response.status_code}). Falling back to {self.default_model}")
                    payload["model"] = self.default_model
                    response = requests.post(url, json=payload, stream=True, timeout=90)

                if response.status_code != 200:
                    raise LLMConnectionError(f"Ollama returned status code {response.status_code}: {response.text}")

            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line.decode("utf-8"))
                    content = chunk.get("message", {}).get("content", "")
                    if content:
                        yield content
                    if chunk.get("done", False):
                        break

        except requests.exceptions.RequestException as req_err:
            logger.error(f"Failed to stream from Ollama: {req_err}")
            raise LLMConnectionError(f"Ollama local service unreachable at {self.base_url}. Ensure Ollama is running.") from req_err
