from abc import ABC, abstractmethod
from typing import AsyncGenerator, Dict, Any, List, Optional
from pydantic import BaseModel, Field

class ChatMessage(BaseModel):
    role: str = Field(..., description="Role: 'system', 'user', or 'assistant'")
    content: str = Field(..., description="Message content")

class LLMResponse(BaseModel):
    text: str
    model: str
    mode: str
    duration_seconds: float = 0.0
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    finish_reason: Optional[str] = "stop"
    raw_response: Optional[Dict[str, Any]] = None

class StreamChunk(BaseModel):
    text: str
    done: bool = False
    model: Optional[str] = None
    finish_reason: Optional[str] = None
    total_duration_seconds: Optional[float] = None

class LLMProvider(ABC):
    """Common abstraction layer for all LLM backends (Ollama and Local Models)."""

    @property
    @abstractmethod
    def mode_name(self) -> str:
        """Returns the mode name ('ollama' or 'local_model')."""
        pass

    @abstractmethod
    async def is_available(self) -> bool:
        """Checks if the backend is currently reachable and ready."""
        pass

    @abstractmethod
    async def get_model_info(self) -> Dict[str, Any]:
        """Returns metadata about the active model and backend status."""
        pass

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """Single-shot text generation."""
        pass

    @abstractmethod
    async def chat(
        self,
        messages: List[ChatMessage],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """Multi-turn conversation generation."""
        pass

    @abstractmethod
    async def stream_chat(
        self,
        messages: List[ChatMessage],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncGenerator[StreamChunk, None]:
        """Streams conversation responses token-by-token or chunk-by-chunk."""
        pass

def get_llm_provider(mode: Optional[str] = None, model_name_or_path: Optional[str] = None) -> LLMProvider:
    """Factory function returning the configured LLMProvider instance."""
    from backend.config.settings import settings
    from backend.services.ollama_service import OllamaProvider
    from backend.services.local_model_service import LocalModelProvider

    selected_mode = (mode or settings.active_mode).lower()

    if selected_mode == "ollama":
        model = model_name_or_path or settings.active_ollama_model
        return OllamaProvider(base_url=settings.ollama_base_url, model=model, timeout=settings.ollama_timeout)
    elif selected_mode in ["local_model", "local", "transformers"]:
        model_path = model_name_or_path or settings.active_local_model_path
        return LocalModelProvider(model_path=model_path)
    else:
        raise ValueError(f"Unknown LLM mode '{selected_mode}'. Supported modes: 'ollama', 'local_model'.")
