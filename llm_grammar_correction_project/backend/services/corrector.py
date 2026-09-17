import time
import re
from typing import Dict, Any, List, Optional
from backend.services.llm_service import LLMProvider, get_llm_provider
from backend.services.file_service import file_service
from backend.config.settings import settings
from backend.utils.logger import studio_logger
from src.evaluation import EvaluationModule
from src.corrector import GrammarCorrector

class DocumentCorrector:
    """Multi-stage document grammar correction, text rewriting, and proofreading engine."""

    def __init__(self, provider: Optional[LLMProvider] = None):
        self._provider = provider
        self.evaluator = EvaluationModule()

    def get_provider(self, mode: Optional[str] = None) -> LLMProvider:
        if self._provider:
            return self._provider
        return get_llm_provider(mode=mode)

    async def correct_document(
        self,
        text: str,
        filename: str = "document",
        mode: Optional[str] = None,
        correction_level: str = "standard",
        system_prompt: Optional[str] = None,
        custom_instructions: Optional[str] = None
    ) -> Dict[str, Any]:
        start_time = time.time()
        provider = self.get_provider(mode)
        
        words = text.split()
        total_words = len(words)
        est_tokens = file_service.estimate_tokens(text)

        chunks = file_service.chunk_text(
            text,
            max_chunk_tokens=settings.max_chunk_tokens,
            overlap_tokens=settings.chunk_overlap_tokens
        )

        is_hierarchical = len(chunks) > 1

        studio_logger.info(
            f"Proofreading '{filename}' ({total_words} words, ~{est_tokens} tokens) using mode '{provider.mode_name}' with level '{correction_level}'. Chunks: {len(chunks)}"
        )

        chunk_corrections = []
        final_corrected_text = ""

        # Map correction_level to prompt template
        task_map = {
            "minimal": "minimal_correction",
            "standard": "standard_correction",
            "rewrite": "comprehensive_rewriting",
            "comprehensive": "comprehensive_rewriting",
            "academic": "academic_polish",
            "technical": "technical_editor",
            "diagnose": "error_detection"
        }
        template_key = task_map.get(correction_level.lower(), "standard_correction")

        if not is_hierarchical:
            prompt_template = settings.get_task_template(template_key)
            prompt = prompt_template.format(text=text)
            if custom_instructions:
                prompt += f"\n\nAdditional user instructions: {custom_instructions}"

            res = await provider.generate(
                prompt=prompt,
                system_prompt=system_prompt or settings.active_system_prompt,
                temperature=0.2
            )
            final_corrected_text = res.text
        else:
            # Multi-chunk processing
            for i, chunk in enumerate(chunks):
                prompt_template = settings.get_task_template(template_key)
                chunk_prompt = prompt_template.format(text=chunk["text"])
                if custom_instructions:
                    chunk_prompt += f"\n\nAdditional user instructions: {custom_instructions}"

                chunk_res = await provider.generate(
                    prompt=chunk_prompt,
                    system_prompt=system_prompt or settings.active_system_prompt,
                    temperature=0.2
                )
                chunk_corrections.append({
                    "chunk_index": i,
                    "original_words": chunk["word_count"],
                    "corrected_words": len(chunk_res.text.split()),
                    "corrected_text": chunk_res.text
                })

            final_corrected_text = "\n\n".join([c["corrected_text"] for c in chunk_corrections])

        duration = round(time.time() - start_time, 2)
        corrected_words = len(final_corrected_text.split())
        
        # Calculate quantitative evaluation metrics
        edit_dist = self.evaluator.calculate_levenshtein_distance(text, final_corrected_text)
        token_metrics = self.evaluator.calculate_token_f1(final_corrected_text, text)
        diff_markup = GrammarCorrector.generate_diff_markup(text, final_corrected_text)
        is_exact = text.strip() == final_corrected_text.strip()

        return {
            "filename": filename,
            "mode_used": provider.mode_name,
            "correction_level": correction_level,
            "is_hierarchical": is_hierarchical,
            "total_chunks": len(chunks),
            "original_words": total_words,
            "corrected_words": corrected_words,
            "levenshtein_distance": edit_dist,
            "token_precision": token_metrics["precision"],
            "token_recall": token_metrics["recall"],
            "token_f1": token_metrics["f1"],
            "is_unchanged": is_exact,
            "duration_seconds": duration,
            "diff_markup": diff_markup,
            "chunk_corrections": chunk_corrections,
            "corrected_text": final_corrected_text
        }

document_corrector = DocumentCorrector()
