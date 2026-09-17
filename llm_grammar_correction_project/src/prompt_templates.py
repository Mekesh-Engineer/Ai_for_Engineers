"""
Prompt Templates and Prompt Manager for Experiment 8
Controls prompt variations across correction intensities (Minimal, Standard, Rewriting, Error Detection).
"""

from typing import Dict, List, Optional

class PromptTemplates:
    TEMPLATES = {
        "minimal": "Fix only obvious spelling mistakes and basic agreement errors without altering sentence structure:\n\"{text}\"\nCorrected:",
        "standard": "Correct all grammar, spelling, punctuation, verb tense, and preposition errors in this sentence cleanly:\n\"{text}\"\nCorrected:",
        "rewrite": "Rewrite the following sentence for superior clarity, professional flow, and natural academic style while strictly preserving its original meaning:\n\"{text}\"\nRewritten:",
        "detection": "Identify and explain all grammatical, punctuation, and spelling errors in this sentence:\n\"{text}\"\nAnalysis:",
        "academic": "Rewrite this sentence in formal academic English suitable for a scientific engineering publication:\n\"{text}\"\nPolished:"
    }

    @classmethod
    def get_template(cls, version: str = "standard") -> str:
        """Retrieves a prompt template by key."""
        return cls.TEMPLATES.get(version.lower(), cls.TEMPLATES["standard"])

    @classmethod
    def format_prompt(cls, version: str, text: str) -> str:
        """Formats the input sentence into the requested template."""
        tpl = cls.get_template(version)
        return tpl.format(text=text.strip())


class PromptManager:
    def __init__(self, default_version: str = "standard"):
        self.active_version = default_version
        self.custom_templates: Dict[str, str] = {}

    def set_active_version(self, version: str) -> None:
        """Sets the active prompt version."""
        if version in PromptTemplates.TEMPLATES or version in self.custom_templates:
            self.active_version = version
        else:
            raise ValueError(f"Unknown prompt version: {version}")

    def register_template(self, name: str, template: str) -> None:
        """Registers a user-defined prompt template."""
        if "{text}" not in template:
            raise ValueError("Template must contain '{text}' placeholder.")
        self.custom_templates[name] = template

    def get_prompt(self, text: str, version: Optional[str] = None) -> str:
        """Formats the input sentence using the active or specified prompt version."""
        v = version or self.active_version
        if v in self.custom_templates:
            return self.custom_templates[v].format(text=text.strip())
        return PromptTemplates.format_prompt(v, text)

    def list_available_versions(self) -> List[str]:
        """Returns all available prompt template names."""
        return list(PromptTemplates.TEMPLATES.keys()) + list(self.custom_templates.keys())
