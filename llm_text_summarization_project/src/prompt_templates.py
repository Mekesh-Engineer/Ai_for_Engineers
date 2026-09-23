# -*- coding: utf-8 -*-
"""
prompt_templates.py
-------------------
Prompt engineering templates, prompt version managers, and LangChain integration
for Experiment 7: LLM Text Summarization.
"""

import os
from typing import Dict, List, Optional, Any

# Try importing langchain PromptTemplate if available, fallback gracefully
try:
    from langchain_core.prompts import PromptTemplate
    _LANGCHAIN_AVAILABLE = True
except ImportError:
    try:
        from langchain.prompts import PromptTemplate
        _LANGCHAIN_AVAILABLE = True
    except ImportError:
        _LANGCHAIN_AVAILABLE = False


# Default built-in prompt templates
BUILTIN_PROMPTS = {
    "v1": {
        "name": "Direct Concise Prompt (v1)",
        "description": "Standard direct task instruction asking for a 2-3 sentence overview.",
        "template": (
            "You are an expert summarization assistant.\n"
            "Please provide a concise, coherent summary of the following document in 2 to 3 sentences.\n"
            "Capture the main objective and primary outcome without adding any outside knowledge.\n\n"
            "Document:\n{document}\n\n"
            "Summary:"
        ),
        "target_style": "paragraph",
        "length_constraint": "2-3 sentences"
    },
    "few_shot": {
        "name": "Few-Shot Demonstration Prompt (few_shot)",
        "description": "In-context demonstration with paired technical document and executive reference summary.",
        "template": (
            "You are an expert technical summarizer. Study the demonstration example below, then summarize the target document using the same style, conciseness, and rigor.\n\n"
            "### Demonstration Example:\n"
            "Document: Deep convolutional neural networks were applied to automated optical inspection of semiconductor silicon wafers. The model identified micro-defects with 99.4% accuracy, outperforming human manual inspection by 3x in throughput.\n"
            "Summary: Convolutional neural networks achieved 99.4% accuracy in semiconductor wafer defect detection, tripling inspection throughput compared to manual methods.\n\n"
            "### Target Document:\n"
            "{document}\n\n"
            "Summary:"
        ),
        "target_style": "few_shot_paragraph",
        "length_constraint": "2-3 sentences"
    },
    "v2": {
        "name": "Structured Technical Summary (v2)",
        "description": "Formal engineering summary constrained to under 100 words focusing on methodology & outcomes.",
        "template": (
            "You are a professional technical editor.\n"
            "Summarize the following document into a single coherent paragraph.\n"
            "Requirements:\n"
            "1. Target length: Keep the summary strictly under 100 words.\n"
            "2. Content: Focus on technical methodology, key quantitative findings, and engineering implications.\n"
            "3. Tone: Maintain an objective, academic register.\n"
            "4. Constraint: Output only the summary paragraph without preamble or concluding remarks.\n\n"
            "Document:\n{document}\n\n"
            "Summary:"
        ),
        "target_style": "technical_paragraph",
        "length_constraint": "under 100 words"
    },
    "best": {
        "name": "Executive & Bullet Insights (best_prompt)",
        "description": "Executive summary with core takeaway sentence and 2-3 structured insight bullets.",
        "template": (
            "You are an advanced AI research assistant specialized in executive technical summarization.\n"
            "Generate a high-impact, abstractive summary of the provided text according to these strict rules:\n\n"
            "1. CORE TAKEAWAY: Synthesize the central thesis and primary impact in 1 clear opening sentence.\n"
            "2. KEY INSIGHTS: Provide 2 to 3 concise bullet points highlighting key mechanisms, innovations, or experimental results.\n"
            "3. CONSTRAINTS:\n"
            "   - Stay strictly faithful to the source document; do NOT hallucinate or extrapolate facts.\n"
            "   - Total summary length must be between 50 and 130 words.\n"
            "   - Output ONLY the formatted summary (no conversational prefixes like 'Here is your summary').\n\n"
            "Document:\n{document}\n\n"
            "Summary:"
        ),
        "target_style": "executive_bullets",
        "length_constraint": "50-130 words"
    }
}


class PromptManager:
    """Manages prompt templates, custom prompt files, and formatted prompt construction."""

    def __init__(self, prompt_dir: Optional[str] = None):
        self.prompts: Dict[str, Dict[str, Any]] = dict(BUILTIN_PROMPTS)
        if prompt_dir and os.path.isdir(prompt_dir):
            self._load_prompts_from_dir(prompt_dir)

    def _load_prompts_from_dir(self, prompt_dir: str) -> None:
        """Load text prompt templates from directory if present."""
        mapping = {
            "prompt_v1.txt": "v1",
            "prompt_few_shot.txt": "few_shot",
            "prompt_v2.txt": "v2",
            "best_prompt.txt": "best"
        }
        for filename, key in mapping.items():
            filepath = os.path.join(prompt_dir, filename)
            if os.path.exists(filepath):
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                if key in self.prompts:
                    self.prompts[key]["template"] = content
                else:
                    self.prompts[key] = {
                        "name": f"Custom {key}",
                        "description": f"Loaded from {filename}",
                        "template": content,
                        "target_style": "custom",
                        "length_constraint": "unspecified"
                    }

    def get_prompt_keys(self) -> List[str]:
        """Return list of available prompt template identifiers."""
        return list(self.prompts.keys())

    def get_prompt_info(self, key: str) -> Dict[str, Any]:
        """Return metadata and template for a prompt key."""
        if key not in self.prompts:
            raise KeyError(f"Prompt key '{key}' not found. Available: {self.get_prompt_keys()}")
        return self.prompts[key]

    def format_prompt(self, key: str, document_text: str) -> str:
        """Construct full prompt string for a given document."""
        info = self.get_prompt_info(key)
        template_str = info["template"]
        return template_str.format(document=document_text)

    def get_langchain_template(self, key: str) -> Any:
        """Return LangChain PromptTemplate object if langchain is installed."""
        if not _LANGCHAIN_AVAILABLE:
            raise ImportError("LangChain is not installed in the current environment.")
        info = self.get_prompt_info(key)
        return PromptTemplate(
            input_variables=["document"],
            template=info["template"]
        )


def create_summarization_prompt(
    document_text: str,
    prompt_version: str = "best",
    prompt_dir: Optional[str] = None
) -> str:
    """Convenience function to generate formatted prompt string for a document."""
    manager = PromptManager(prompt_dir=prompt_dir)
    return manager.format_prompt(prompt_version, document_text)
