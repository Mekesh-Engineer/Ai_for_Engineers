import os
import time
import json
from pathlib import Path
from typing import AsyncGenerator, Dict, Any, List, Optional
import torch

from backend.services.llm_service import LLMProvider, LLMResponse, StreamChunk, ChatMessage
from backend.config.settings import settings
from backend.utils.logger import studio_logger

# Global in-memory cache to prevent redundant model weight reloads
_LOADED_MODELS_CACHE: Dict[str, Dict[str, Any]] = {}

class LocalModelProvider(LLMProvider):
    """Implementation of LLMProvider for locally hosted model checkpoints in ./models/."""

    def __init__(self, model_path: Optional[str] = None):
        self.models_dir = Path(settings.local_model_dir)
        self.model_path = Path(model_path or settings.active_local_model_path)
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def mode_name(self) -> str:
        return "local_model"

    def scan_models(self) -> List[Dict[str, Any]]:
        """Discover and validate all compatible model directories/files inside ./models/."""
        found_models = []
        if not self.models_dir.exists():
            return found_models

        search_dirs = [self.models_dir]
        for root, dirs, files in os.walk(self.models_dir):
            rel_path = Path(root).relative_to(self.models_dir)
            if len(rel_path.parts) <= 3:
                search_dirs.append(Path(root))

        seen_paths = set()
        for directory in search_dirs:
            if not directory.is_dir():
                continue
            
            config_file = directory / "config.json"
            safetensors_files = list(directory.glob("*.safetensors"))
            bin_files = list(directory.glob("*.bin"))
            gguf_files = list(directory.glob("*.gguf"))

            if config_file.exists() and (safetensors_files or bin_files):
                abs_str = str(directory.resolve())
                if abs_str not in seen_paths:
                    seen_paths.add(abs_str)
                    model_size = sum(f.stat().st_size for f in directory.iterdir() if f.is_file())
                    try:
                        with open(config_file, "r", encoding="utf-8") as f:
                            cfg_data = json.load(f)
                            arch = cfg_data.get("architectures", ["Unknown"])[0]
                            model_type = cfg_data.get("model_type", "transformers")
                    except Exception:
                        arch = "Transformers"
                        model_type = "transformers"

                    found_models.append({
                        "name": directory.name,
                        "path": abs_str,
                        "relative_path": str(directory.relative_to(settings.project_root)),
                        "format": "safetensors" if safetensors_files else "pytorch_bin",
                        "architecture": arch,
                        "model_type": model_type,
                        "size_bytes": model_size,
                        "size_mb": round(model_size / (1024 * 1024), 2),
                        "is_valid": True
                    })

            for gguf in gguf_files:
                abs_str = str(gguf.resolve())
                if abs_str not in seen_paths:
                    seen_paths.add(abs_str)
                    found_models.append({
                        "name": gguf.name,
                        "path": abs_str,
                        "relative_path": str(gguf.relative_to(settings.project_root)),
                        "format": "GGUF",
                        "architecture": "llama.cpp GGUF",
                        "model_type": "gguf",
                        "size_bytes": gguf.stat().st_size,
                        "size_mb": round(gguf.stat().st_size / (1024 * 1024), 2),
                        "is_valid": True
                    })

        return found_models

    async def is_available(self) -> bool:
        """Check if active local model directory is valid and exists."""
        if self.model_path.exists() and self.model_path.is_dir():
            config_file = self.model_path / "config.json"
            weights = list(self.model_path.glob("*.safetensors")) + list(self.model_path.glob("*.bin"))
            return config_file.exists() and len(weights) > 0
        return False

    async def get_model_info(self) -> Dict[str, Any]:
        """Get information about available local models and current selection."""
        models = self.scan_models()
        is_ready = await self.is_available()
        
        current_info = {}
        if is_ready:
            cfg_path = self.model_path / "config.json"
            if cfg_path.exists():
                try:
                    with open(cfg_path, "r", encoding="utf-8") as f:
                        current_info = json.load(f)
                except Exception:
                    pass

        return {
            "mode": "local_model",
            "models_directory": str(self.models_dir),
            "active_model_path": str(self.model_path),
            "is_available": is_ready,
            "architecture": current_info.get("architectures", ["Unknown"])[0] if current_info else "None",
            "device": self._device,
            "discovered_models": models
        }

    def _ensure_loaded(self):
        """Lazy loader with global caching for Transformers local pipeline."""
        model_key = str(self.model_path.resolve())
        if model_key in _LOADED_MODELS_CACHE:
            cached = _LOADED_MODELS_CACHE[model_key]
            self._tokenizer = cached["tokenizer"]
            self._model = cached["model"]
            self._is_seq2seq = cached["is_seq2seq"]
            return

        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoConfig

        if not self.model_path.exists():
            raise FileNotFoundError(f"Local model path not found: {self.model_path}")

        studio_logger.info(f"Loading local model from {self.model_path} onto {self._device}...")
        config = AutoConfig.from_pretrained(str(self.model_path))
        tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))

        # Determine architecture class
        arch_str = str(getattr(config, "architectures", [""]))
        if any(term in arch_str for term in ["ConditionalGeneration", "Seq2SeqLM", "Bart", "T5", "Pegasus", "Marian"]):
            is_seq2seq = True
            model = AutoModelForSeq2SeqLM.from_pretrained(
                str(self.model_path),
                torch_dtype=torch.float32
            ).to(self._device)
        else:
            is_seq2seq = False
            model = AutoModelForCausalLM.from_pretrained(
                str(self.model_path),
                torch_dtype=torch.float32
            ).to(self._device)

        model.eval()
        _LOADED_MODELS_CACHE[model_key] = {
            "tokenizer": tokenizer,
            "model": model,
            "is_seq2seq": is_seq2seq
        }
        self._tokenizer = tokenizer
        self._model = model
        self._is_seq2seq = is_seq2seq

        studio_logger.info(f"Successfully loaded and cached local model ({'Seq2Seq' if is_seq2seq else 'CausalLM'}) onto {self._device}.")

    async def test_connection(self) -> Dict[str, Any]:
        """Test local model detection, loading, and small inference."""
        results = {
            "backend_running": True,
            "model_detected": False,
            "model_valid": False,
            "model_loaded": False,
            "test_generation": False,
            "latency_seconds": 0.0,
            "message": "",
            "details": {}
        }
        
        start_time = time.time()
        try:
            models = self.scan_models()
            results["details"]["found_models"] = [m["name"] for m in models]
            
            if not await self.is_available():
                results["message"] = f"No valid local model checkpoint found at: {self.model_path}. Please check ./models/."
                return results

            results["model_detected"] = True
            results["model_valid"] = True

            self._ensure_loaded()
            results["model_loaded"] = True

            test_prompt = "He go to laboratory yesterday for doing the experiment."
            gen_res = await self.generate(test_prompt, max_tokens=64)
            
            results["test_generation"] = True
            results["latency_seconds"] = round(time.time() - start_time, 2)
            results["details"]["sample_output"] = gen_res.text
            results["message"] = "Local model successfully validated and generated test inference!"

        except Exception as e:
            studio_logger.error(f"Local model test failed: {e}")
            results["message"] = f"Local model error: {str(e)}"

        return results

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: Optional[int] = 128,
        **kwargs
    ) -> LLMResponse:
        start_time = time.time()
        self._ensure_loaded()

        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        inputs = self._tokenizer(full_prompt, return_tensors="pt", max_length=1024, truncation=True).to(self._device)
        input_token_count = inputs["input_ids"].shape[1]

        max_len = max_tokens or 128
        with torch.no_grad():
            if self._is_seq2seq:
                outputs = self._model.generate(
                    **inputs,
                    max_length=max_len,
                    min_length=5,
                    num_beams=2,
                    early_stopping=True,
                    no_repeat_ngram_size=3
                )
            else:
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=max_len,
                    do_sample=temperature > 0.1,
                    temperature=max(temperature, 0.1),
                    top_p=0.9,
                    pad_token_id=self._tokenizer.eos_token_id
                )

        output_tokens = outputs.shape[1]
        decoded_text = self._tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        
        if not self._is_seq2seq and decoded_text.startswith(full_prompt):
            decoded_text = decoded_text[len(full_prompt):].strip()

        duration = time.time() - start_time
        studio_logger.log_inference("local_model", self.model_path.name, "generate", duration, {"tokens": output_tokens})

        return LLMResponse(
            text=decoded_text,
            model=self.model_path.name,
            mode="local_model",
            duration_seconds=round(duration, 3),
            prompt_tokens=input_token_count,
            completion_tokens=output_tokens,
            total_tokens=input_token_count + output_tokens
        )

    async def chat(
        self,
        messages: List[ChatMessage],
        temperature: float = 0.2,
        max_tokens: Optional[int] = 128,
        **kwargs
    ) -> LLMResponse:
        prompt_parts = []
        for m in messages:
            if m.role == "system":
                prompt_parts.append(f"System: {m.content}")
            elif m.role == "user":
                prompt_parts.append(f"User: {m.content}")
            elif m.role == "assistant":
                prompt_parts.append(f"Assistant: {m.content}")
        
        prompt_parts.append("Assistant:")
        formatted_prompt = "\n".join(prompt_parts)
        return await self.generate(formatted_prompt, temperature=temperature, max_tokens=max_tokens, **kwargs)

    async def stream_chat(
        self,
        messages: List[ChatMessage],
        temperature: float = 0.2,
        max_tokens: Optional[int] = 128,
        **kwargs
    ) -> AsyncGenerator[StreamChunk, None]:
        start_time = time.time()
        try:
            res = await self.chat(messages, temperature=temperature, max_tokens=max_tokens, **kwargs)
            words = res.text.split(" ")
            for i, word in enumerate(words):
                piece = word + (" " if i < len(words) - 1 else "")
                is_last = (i == len(words) - 1)
                yield StreamChunk(
                    text=piece,
                    done=is_last,
                    model=self.model_path.name,
                    finish_reason="stop" if is_last else None,
                    total_duration_seconds=round(time.time() - start_time, 3) if is_last else None
                )
        except Exception as e:
            studio_logger.error(f"Local model stream error: {e}")
            yield StreamChunk(
                text=f"\n[Error generating from local model: {str(e)}]",
                done=True,
                finish_reason="error"
            )

LocalModelService = LocalModelProvider
