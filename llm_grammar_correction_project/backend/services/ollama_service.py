import json
import time
from typing import AsyncGenerator, Dict, Any, List, Optional
import httpx
from backend.services.llm_service import LLMProvider, LLMResponse, StreamChunk, ChatMessage
from backend.config.settings import settings
from backend.utils.logger import studio_logger

class OllamaProvider(LLMProvider):
    """Implementation of LLMProvider communicating with local Ollama HTTP service for Grammar Studio."""

    def __init__(self, base_url: Optional[str] = None, model: Optional[str] = None, timeout: float = 120.0):
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self.model = model or settings.active_ollama_model
        self.timeout = timeout

    @property
    def mode_name(self) -> str:
        return "ollama"

    async def is_available(self) -> bool:
        """Check if Ollama service is reachable."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                res = await client.get(f"{self.base_url}/api/tags")
                return res.status_code == 200
        except Exception:
            return False

    async def list_models(self) -> List[Dict[str, Any]]:
        """Retrieve list of available models from Ollama."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                res = await client.get(f"{self.base_url}/api/tags")
                if res.status_code == 200:
                    data = res.json()
                    models = data.get("models", [])
                    return [
                        {
                            "name": m.get("name"),
                            "size_bytes": m.get("size", 0),
                            "size_gb": round(m.get("size", 0) / (1024**3), 2),
                            "modified_at": m.get("modified_at"),
                            "format": m.get("details", {}).get("format", "unknown"),
                            "family": m.get("details", {}).get("family", "unknown"),
                            "parameter_size": m.get("details", {}).get("parameter_size", "unknown"),
                            "quantization_level": m.get("details", {}).get("quantization_level", "unknown")
                        }
                        for m in models
                    ]
        except Exception as e:
            studio_logger.error(f"Failed to list Ollama models: {e}")
        return []

    async def get_model_info(self) -> Dict[str, Any]:
        """Fetch status and metadata for current Ollama model."""
        available = await self.is_available()
        models = await self.list_models() if available else []
        model_names = [m["name"] for m in models]
        model_exists = self.model in model_names or any(self.model in m for m in model_names)

        return {
            "mode": "ollama",
            "base_url": self.base_url,
            "active_model": self.model,
            "is_reachable": available,
            "model_installed": model_exists,
            "available_models": models
        }

    async def test_connection(self) -> Dict[str, Any]:
        """Perform comprehensive 5-stage diagnostic test of Ollama backend."""
        results = {
            "backend_running": True,
            "ollama_reachable": False,
            "api_valid": False,
            "model_present": False,
            "test_generation": False,
            "latency_seconds": 0.0,
            "message": "",
            "details": {}
        }
        
        start_time = time.time()
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                # Stage 1: Reachability
                res = await client.get(f"{self.base_url}/api/tags")
                if res.status_code != 200:
                    results["message"] = f"Ollama returned HTTP status {res.status_code} on /api/tags"
                    return results
                
                results["ollama_reachable"] = True
                results["api_valid"] = True
                
                data = res.json()
                models = [m.get("name") for m in data.get("models", [])]
                results["details"]["installed_models"] = models

                # Stage 2: Model present
                if self.model in models or any(self.model in m for m in models):
                    results["model_present"] = True
                else:
                    results["message"] = f"Model '{self.model}' not found in installed models: {models}"
                    return results

                # Stage 3: Test Generation
                test_payload = {
                    "model": self.model,
                    "prompt": "Respond with the single word: OK",
                    "stream": False,
                    "options": {"temperature": 0.1, "num_predict": 10}
                }
                gen_res = await client.post(f"{self.base_url}/api/generate", json=test_payload, timeout=30.0)
                if gen_res.status_code == 200:
                    gen_data = gen_res.json()
                    results["test_generation"] = True
                    results["latency_seconds"] = round(time.time() - start_time, 2)
                    results["details"]["sample_output"] = gen_data.get("response", "").strip()
                    results["message"] = "All Ollama diagnostic checks passed successfully!"
                else:
                    results["message"] = f"Test generation failed with HTTP status {gen_res.status_code}"

        except httpx.ConnectError:
            results["message"] = f"Ollama is not running or cannot be reached at {self.base_url}. Please run 'ollama serve'."
        except httpx.TimeoutException:
            results["message"] = f"Request to Ollama at {self.base_url} timed out."
        except Exception as e:
            results["message"] = f"Ollama connection error: {str(e)}"

        return results

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        start_time = time.time()
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature
            }
        }
        if system_prompt:
            payload["system"] = system_prompt
        if max_tokens:
            payload["options"]["num_predict"] = max_tokens

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                res = await client.post(f"{self.base_url}/api/generate", json=payload)
                if res.status_code != 200:
                    error_msg = f"Ollama generate failed (HTTP {res.status_code}): {res.text}"
                    studio_logger.error(error_msg)
                    raise RuntimeError(error_msg)

                data = res.json()
                duration = time.time() - start_time
                resp = LLMResponse(
                    text=data.get("response", ""),
                    model=self.model,
                    mode="ollama",
                    duration_seconds=round(duration, 3),
                    prompt_tokens=data.get("prompt_eval_count"),
                    completion_tokens=data.get("eval_count"),
                    total_tokens=(data.get("prompt_eval_count", 0) or 0) + (data.get("eval_count", 0) or 0),
                    raw_response=data
                )
                studio_logger.log_inference("ollama", self.model, "generate", duration, {"eval_count": data.get("eval_count")})
                return resp
        except httpx.ConnectError:
            raise RuntimeError(f"Ollama is not running or unreachable at {self.base_url}. Run 'ollama serve' to start Ollama.")
        except Exception as e:
            studio_logger.error(f"Error in Ollama generate: {e}")
            raise

    async def chat(
        self,
        messages: List[ChatMessage],
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        start_time = time.time()
        payload = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": False,
            "options": {
                "temperature": temperature
            }
        }
        if max_tokens:
            payload["options"]["num_predict"] = max_tokens

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                res = await client.post(f"{self.base_url}/api/chat", json=payload)
                if res.status_code != 200:
                    error_msg = f"Ollama chat failed (HTTP {res.status_code}): {res.text}"
                    studio_logger.error(error_msg)
                    raise RuntimeError(error_msg)

                data = res.json()
                duration = time.time() - start_time
                msg_content = data.get("message", {}).get("content", "")
                resp = LLMResponse(
                    text=msg_content,
                    model=self.model,
                    mode="ollama",
                    duration_seconds=round(duration, 3),
                    prompt_tokens=data.get("prompt_eval_count"),
                    completion_tokens=data.get("eval_count"),
                    total_tokens=(data.get("prompt_eval_count", 0) or 0) + (data.get("eval_count", 0) or 0),
                    raw_response=data
                )
                studio_logger.log_inference("ollama", self.model, "chat", duration, {"eval_count": data.get("eval_count")})
                return resp
        except httpx.ConnectError:
            raise RuntimeError(f"Ollama is unreachable at {self.base_url}. Please check if Ollama is running.")
        except Exception as e:
            studio_logger.error(f"Error in Ollama chat: {e}")
            raise

    async def stream_chat(
        self,
        messages: List[ChatMessage],
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncGenerator[StreamChunk, None]:
        start_time = time.time()
        payload = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": True,
            "options": {
                "temperature": temperature
            }
        }
        if max_tokens:
            payload["options"]["num_predict"] = max_tokens

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream("POST", f"{self.base_url}/api/chat", json=payload) as response:
                    if response.status_code != 200:
                        error_body = await response.aread()
                        raise RuntimeError(f"Ollama stream failed ({response.status_code}): {error_body.decode('utf-8', errors='ignore')}")

                    async for line in response.aiter_lines():
                        if not line.strip():
                            continue
                        try:
                            chunk_data = json.loads(line)
                            content_piece = chunk_data.get("message", {}).get("content", "")
                            is_done = chunk_data.get("done", False)
                            
                            chunk = StreamChunk(
                                text=content_piece,
                                done=is_done,
                                model=self.model,
                                finish_reason="stop" if is_done else None,
                                total_duration_seconds=round(time.time() - start_time, 3) if is_done else None
                            )
                            yield chunk
                        except json.JSONDecodeError:
                            continue
        except httpx.ConnectError:
            yield StreamChunk(
                text=f"\n[Error: Ollama is unreachable at {self.base_url}. Please run 'ollama serve'.]",
                done=True,
                finish_reason="error"
            )
        except Exception as e:
            studio_logger.error(f"Error streaming from Ollama: {e}")
            yield StreamChunk(
                text=f"\n[Error streaming response: {str(e)}]",
                done=True,
                finish_reason="error"
            )
