# -*- coding: utf-8 -*-
"""
prompt_templates.py
-------------------
Prompt templates and PromptManager for Experiment 8: Automated Grammar Error Correction & Text Rewriting.
Controls prompt engineering variations across correction intensities (Minimal/Zero-Shot, Few-Shot,
Standard GEC, Constrained, Academic Polish, Best Structured, Error Detection).
"""

import os
import json
from typing import Dict, List, Optional, Any


BUILTIN_PROMPTS: Dict[str, str] = {
    "v1": (
        "Fix only obvious spelling mistakes and basic agreement errors without altering sentence structure:\n"
        "\"{text}\"\n"
        "Corrected:"
    ),
    "minimal": (
        "Fix only obvious spelling mistakes and basic agreement errors without altering sentence structure:\n"
        "\"{text}\"\n"
        "Corrected:"
    ),
    "few_shot": (
        "Here is a demonstration of how to correct grammatical and syntactic errors:\n\n"
        "Demonstration Example:\n"
        "Input: \"The researcher discuss about the convergence rate of gradient descent.\"\n"
        "Correction: \"The researcher discusses the convergence rate of gradient descent.\"\n\n"
        "Input: \"Each of the component have been tested under high pressure.\"\n"
        "Correction: \"Each of the components has been tested under high pressure.\"\n\n"
        "Target Sentence to Correct:\n"
        "Input: \"{text}\"\n"
        "Correction:"
    ),
    "v2": (
        "Correct all grammatical, spelling, punctuation, verb tense, and preposition errors in this sentence. "
        "Keep the output under 100 words in an objective, academic and technical tone:\n"
        "\"{text}\"\n"
        "Corrected:"
    ),
    "standard": (
        "Correct all grammar, spelling, punctuation, verb tense, and preposition errors in this sentence cleanly:\n"
        "\"{text}\"\n"
        "Corrected:"
    ),
    "rewrite": (
        "Rewrite the following sentence for superior clarity, professional flow, and natural academic style while strictly preserving its original meaning:\n"
        "\"{text}\"\n"
        "Rewritten:"
    ),
    "academic": (
        "Rewrite this sentence in formal academic English suitable for a scientific engineering publication:\n"
        "\"{text}\"\n"
        "Polished:"
    ),
    "best": (
        "CORE CORRECTION & POLISH TASK:\n"
        "Analyze the following technical sentence, resolve all grammatical, agreement, tense, and orthographic defects, "
        "and produce a flawless, professionally polished sentence preserving all domain terminology:\n\n"
        "SOURCE TEXT:\n"
        "\"{text}\"\n\n"
        "CORRECTED OUTPUT:"
    ),
    "best_prompt": (
        "CORE CORRECTION & POLISH TASK:\n"
        "Analyze the following technical sentence, resolve all grammatical, agreement, tense, and orthographic defects, "
        "and produce a flawless, professionally polished sentence preserving all domain terminology:\n\n"
        "SOURCE TEXT:\n"
        "\"{text}\"\n\n"
        "CORRECTED OUTPUT:"
    ),
    "detection": (
        "Identify and explain all grammatical, punctuation, and spelling errors in this sentence:\n"
        "\"{text}\"\n"
        "Analysis:"
    )
}


class PromptTemplates:
    """Static catalog of prompt templates for GEC."""
    TEMPLATES = BUILTIN_PROMPTS

    @classmethod
    def get_template(cls, version: str = "standard") -> str:
        key = version.lower().strip()
        return cls.TEMPLATES.get(key, cls.TEMPLATES.get("standard", "{text}"))

    @classmethod
    def format_prompt(cls, version: str, text: str) -> str:
        tpl = cls.get_template(version)
        return tpl.format(text=text.strip())


class PromptManager:
    """Manages built-in and user-customized prompt engineering templates."""

    def __init__(self, prompt_dir: Optional[str] = None, default_version: str = "standard"):
        self.prompt_dir = prompt_dir
        self.active_version = default_version
        self.custom_templates: Dict[str, str] = {}
        self.templates: Dict[str, str] = dict(BUILTIN_PROMPTS)

        if prompt_dir and os.path.exists(prompt_dir):
            self.load_from_directory(prompt_dir)

    def load_from_directory(self, directory_path: str) -> None:
        """Loads prompt templates from JSON files in the given directory."""
        if not os.path.exists(directory_path):
            return
        for fname in os.listdir(directory_path):
            if fname.endswith(".json"):
                fpath = os.path.join(directory_path, fname)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if isinstance(data, dict):
                            self.custom_templates.update(data)
                            self.templates.update(data)
                except Exception:
                    pass

    def get_prompt_keys(self) -> List[str]:
        """Returns all available prompt template identifier keys."""
        return list(self.templates.keys())

    def list_available_versions(self) -> List[str]:
        return self.get_prompt_keys()

    def set_active_version(self, version: str) -> None:
        v = version.lower().strip()
        if v in self.templates:
            self.active_version = v
        else:
            raise ValueError(f"Unknown prompt version: {version}")

    def register_template(self, name: str, template: str) -> None:
        if "{text}" not in template:
            raise ValueError("Template must contain '{text}' placeholder.")
        key = name.lower().strip()
        self.custom_templates[key] = template
        self.templates[key] = template

    def format_prompt(self, version_key: str, text: str) -> str:
        key = version_key.lower().strip()
        tpl = self.templates.get(key, self.templates.get("standard", "{text}"))
        return tpl.format(text=text.strip())

    def get_prompt(self, text: str, version: Optional[str] = None) -> str:
        v = (version or self.active_version).lower().strip()
        return self.format_prompt(v, text)


def create_correction_prompt(text: str, version: str = "standard") -> str:
    """Helper function to format text with the specified prompt version."""
    pm = PromptManager()
    return pm.format_prompt(version, text)
