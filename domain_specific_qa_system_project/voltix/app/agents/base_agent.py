from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional

class BaseAgent(ABC):
    """Abstract base class for domain-specific EEE sub-agents."""

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def domain_description(self) -> str:
        pass

    @abstractmethod
    def get_system_prompt(self, context_str: str = "") -> str:
        pass

    def prepare_prompt(self, query: str, context_str: str = "") -> str:
        if context_str:
            return f"{context_str}\n\nUser Question: {query}"
        return query
