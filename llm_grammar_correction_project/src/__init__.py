"""
Automated Grammar Correction and Text Rewriting using Large Language Models (LLM)
Experiment 8 Package Initialization
"""

from .data_loader import DataLoaderModule
from .data_preprocessing import PreprocessingModule
from .prompt_templates import PromptTemplates, PromptManager
from .corrector import GrammarCorrector
from .evaluation import EvaluationModule
from .error_analyzer import ErrorAnalyzerModule
from .visualization import VisualizationModule

__version__ = "1.0.0"
__author__ = "Mekesh Kumar M"
__experiment__ = "Experiment 8: Grammar Correction and Text Rewriting using LLMs"

__all__ = [
    "DataLoaderModule",
    "PreprocessingModule",
    "PromptTemplates",
    "PromptManager",
    "GrammarCorrector",
    "EvaluationModule",
    "ErrorAnalyzerModule",
    "VisualizationModule"
]
