import os
from typing import Generator, List, Dict, Any, Optional
from app.llm.base import BaseLLMClient
from app.utils.logger import get_logger

logger = get_logger("voltix.gemini")

class GeminiClient(BaseLLMClient):
    """Optional Google Gemini API streaming client."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")

    def list_models(self) -> List[str]:
        return ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-2.0-flash"]

    def stream_generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        model_name: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Generator[str, None, None]:
        if not self.api_key:
            yield "[Gemini API key not configured. Please set GEMINI_API_KEY in .env file.]"
            return

        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            model_id = model_name or "gemini-1.5-flash"
            model = genai.GenerativeModel(model_id, system_instruction=system_prompt if system_prompt else None)

            full_prompt = ""
            if history:
                for h in history:
                    full_prompt += f"{h.get('sender', 'User')}: {h.get('content', '')}\n"
            full_prompt += f"User: {prompt}\nAssistant:"

            response = model.generate_content(full_prompt, stream=True)
            for chunk in response:
                if chunk.text:
                    yield chunk.text

        except Exception as e:
            logger.error(f"Gemini API streaming error: {e}")
            yield f"[Gemini API Error: {str(e)}]"
