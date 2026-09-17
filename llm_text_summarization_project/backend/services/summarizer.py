import time
from typing import Dict, Any, List, Optional
from backend.services.llm_service import LLMProvider, get_llm_provider
from backend.services.file_service import file_service
from backend.config.settings import settings
from backend.utils.logger import studio_logger

class HierarchicalSummarizer:
    """Multi-stage document summarization engine supporting both single-pass and hierarchical chunk aggregation."""

    def __init__(self, provider: Optional[LLMProvider] = None):
        self._provider = provider

    def get_provider(self, mode: Optional[str] = None) -> LLMProvider:
        if self._provider:
            return self._provider
        return get_llm_provider(mode=mode)

    async def summarize_document(
        self,
        text: str,
        filename: str = "document",
        mode: Optional[str] = None,
        task: str = "summarize",
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

        studio_logger.info(f"Summarizing '{filename}' ({total_words} words, ~{est_tokens} tokens) using mode '{provider.mode_name}'. Hierarchical: {is_hierarchical} ({len(chunks)} chunks)")

        chunk_summaries = []
        final_summary_text = ""

        if not is_hierarchical:
            # Single pass summarization
            prompt_template = settings.get_task_template(task)
            prompt = prompt_template.format(text=text)
            if custom_instructions:
                prompt += f"\n\nAdditional user instructions: {custom_instructions}"

            res = await provider.generate(
                prompt=prompt,
                system_prompt=system_prompt or settings.active_system_prompt,
                temperature=0.3
            )
            final_summary_text = res.text
        else:
            # Hierarchical pipeline
            # Phase 1: Summarize each chunk
            for i, chunk in enumerate(chunks):
                chunk_prompt = f"Summarize section {i+1} of {len(chunks)} from the document '{filename}'. Capture all critical facts, technical details, and data points:\n\n{chunk['text']}"
                chunk_res = await provider.generate(
                    prompt=chunk_prompt,
                    system_prompt="You are a precise technical summarizer. Extract essential information concisely without losing critical facts.",
                    temperature=0.3
                )
                chunk_summaries.append({
                    "chunk_index": i,
                    "chunk_words": chunk["word_count"],
                    "summary": chunk_res.text
                })

            # Phase 2: Combine and synthesize final summary
            combined_chunk_text = "\n\n".join([f"--- Section {s['chunk_index']+1} Summary ---\n{s['summary']}" for s in chunk_summaries])
            synthesis_prompt = f"Synthesize the following section summaries from '{filename}' into a cohesive, structured, and comprehensive executive summary:\n\n{combined_chunk_text}"
            if custom_instructions:
                synthesis_prompt += f"\n\nAdditional user instructions: {custom_instructions}"

            final_res = await provider.generate(
                prompt=synthesis_prompt,
                system_prompt=system_prompt or settings.active_system_prompt,
                temperature=0.3
            )
            final_summary_text = final_res.text

        duration = round(time.time() - start_time, 2)
        summary_words = len(final_summary_text.split())
        compression_ratio = round(summary_words / max(1, total_words), 3)

        return {
            "filename": filename,
            "mode_used": provider.mode_name,
            "is_hierarchical": is_hierarchical,
            "total_chunks": len(chunks),
            "original_words": total_words,
            "summary_words": summary_words,
            "compression_ratio": compression_ratio,
            "duration_seconds": duration,
            "chunk_summaries": chunk_summaries,
            "final_summary": final_summary_text
        }

document_summarizer = HierarchicalSummarizer()
