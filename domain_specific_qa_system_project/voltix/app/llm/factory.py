from typing import Optional
from app.llm.base import BaseLLMClient
from app.llm.ollama_client import OllamaClient
from app.llm.openai_client import OpenAIClient
from app.llm.gemini_client import GeminiClient
from app.utils.logger import get_logger

logger = get_logger("voltix.llm_factory")

class LLMFactory:
    """Factory to instantiate LLM providers dynamically based on model name or user choice."""

    @staticmethod
    def get_client(model_name: Optional[str] = None, base_url: str = "http://localhost:11434") -> BaseLLMClient:
        model = (model_name or "").lower()

        if "gpt" in model:
            logger.info(f"Instantiating OpenAI client for model '{model_name}'")
            return OpenAIClient()
        elif "gemini" in model:
            logger.info(f"Instantiating Gemini client for model '{model_name}'")
            return GeminiClient()
        else:
            # Default to local Ollama
            logger.info(f"Instantiating Ollama client for model '{model_name or 'qwen2.5:7b'}'")
            return OllamaClient(base_url=base_url, default_model=model_name or "qwen2.5:7b")
