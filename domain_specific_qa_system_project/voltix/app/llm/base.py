from abc import ABC, abstractmethod
from typing import Generator, List, Dict, Any, Optional

class BaseLLMClient(ABC):
    """Abstract interface for streaming LLM clients."""

    @abstractmethod
    def stream_generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        model_name: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Generator[str, None, None]:
        """Yield tokens synchronously or asynchronously from the model provider."""
        pass

    @abstractmethod
    def list_models(self) -> List[str]:
        """Return list of available models from provider."""
        pass
