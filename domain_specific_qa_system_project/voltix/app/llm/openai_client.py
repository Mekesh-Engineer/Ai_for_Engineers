import os
from typing import Generator, List, Dict, Any, Optional
from app.llm.base import BaseLLMClient
from app.utils.logger import get_logger

logger = get_logger("voltix.openai")

class OpenAIClient(BaseLLMClient):
    """Optional OpenAI API streaming client."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")

    def list_models(self) -> List[str]:
        return ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"]

    def stream_generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        model_name: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Generator[str, None, None]:
        if not self.api_key:
            yield "[OpenAI API key not configured. Please set OPENAI_API_KEY in .env file.]"
            return

        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key)
            model = model_name or "gpt-4o-mini"

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            if history:
                for item in history:
                    messages.append({"role": item.get("sender", "user"), "content": item.get("content", "")})
            messages.append({"role": "user", "content": prompt})

            response = client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True,
                temperature=options.get("temperature", 0.2) if options else 0.2,
            )

            for chunk in response:
                delta = chunk.choices[0].delta.content if chunk.choices else ""
                if delta:
                    yield delta

        except Exception as e:
            logger.error(f"OpenAI streaming error: {e}")
            yield f"[OpenAI API Error: {str(e)}]"
