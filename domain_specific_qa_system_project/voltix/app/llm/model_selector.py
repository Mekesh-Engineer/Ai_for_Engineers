from typing import Dict, Any, Optional
from app.utils.logger import get_logger

logger = get_logger("voltix.model_selector")

class ModelSelector:
    """
    Tier-based Dynamic Model Selector for multi-model Ollama orchestrations.
    
    Model Matrix:
    - Tier 0: qwen2.5:3b (Fast Router, Intent Classifier, Memory Summarizer - Offline)
    - Tier 1: qwen2.5:7b (Local Tool Executor, ReAct Engine, Offline Fallback)
    - Tier 2: deepseek-v4-flash:cloud (High-Speed Reasoning, Code Synthesis, Real-time Stream)
    - Tier 3: gemma4:cloud (Structured Task Planner, JSON Schema Validator)
    - Tier 4: gemma4:31b-cloud (Deep Domain Specialist, Literature Synthesis, Complex Systems)
    - Tier 5: gpt-oss:120b-cloud (Frontier Proof Engine, Complex Math Proofs, Architectural Reasoning)
    """

    DEFAULT_TIERS = {
        "router": "qwen2.5:3b",
        "tool_executor": "qwen2.5:7b",
        "fast_reasoning": "deepseek-v4-flash:cloud",
        "planner": "gemma4:cloud",
        "deep_specialist": "gemma4:31b-cloud",
        "frontier_reasoning": "gpt-oss:120b-cloud",
    }

    CAPABILITIES = {
        "qwen2.5:3b": {
            "tier": 0,
            "label": "Qwen 2.5 3B (Fast Edge)",
            "type": "local",
            "best_for": "Routing, Summarization, Zero-latency classification"
        },
        "qwen2.5:7b": {
            "tier": 1,
            "label": "Qwen 2.5 7B (Offline Specialist)",
            "type": "local",
            "best_for": "Local Tool Calling, ReAct Execution, Offline Math"
        },
        "deepseek-v4-flash:cloud": {
            "tier": 2,
            "label": "DeepSeek V4 Flash (Speed & Code)",
            "type": "cloud",
            "best_for": "Fast Code Generation, Iterative Solving, Low Latency"
        },
        "gemma4:cloud": {
            "tier": 3,
            "label": "Gemma 4 (Task Planner)",
            "type": "cloud",
            "best_for": "Task Decomposition, JSON Schema, Plan Validation"
        },
        "gemma4:31b-cloud": {
            "tier": 4,
            "label": "Gemma 4 31B (Deep Domain)",
            "type": "cloud",
            "best_for": "Complex EEE Systems, Literature Synthesis, Deep Analysis"
        },
        "gpt-oss:120b-cloud": {
            "tier": 5,
            "label": "GPT-OSS 120B (Frontier Proof Engine)",
            "type": "cloud",
            "best_for": "Frontier Reasoning, Mathematical Proofs, Multi-Step Architecture"
        },
    }

    @classmethod
    def select_model(cls, intent: str, complexity: str = "medium", requested_model: Optional[str] = None) -> str:
        """Selects the most suitable model based on user intent and complexity."""
        if requested_model and requested_model not in ("auto", "default", "", None):
            return requested_model

        intent_upper = (intent or "").upper()
        
        # 1. Complex proofs / Frontier reasoning
        if "PROOF" in intent_upper or "THEORY_DEEP" in intent_upper or complexity == "frontier":
            return cls.DEFAULT_TIERS["frontier_reasoning"]
            
        # 2. Deep domain EEE or academic research
        if "RESEARCH" in intent_upper or "COMPLEX_DESIGN" in intent_upper or complexity == "high":
            return cls.DEFAULT_TIERS["deep_specialist"]

        # 3. MATLAB / Code generation / Fast iterative solving
        if "MATLAB" in intent_upper or "CODE" in intent_upper or "SIMULINK" in intent_upper:
            return cls.DEFAULT_TIERS["fast_reasoning"]

        # 4. Multi-step planning
        if "PLAN" in intent_upper or "MULTI_STEP" in intent_upper:
            return cls.DEFAULT_TIERS["planner"]

        # 5. Deterministic numerical calculations & tools (Local 7B)
        if "NUMERICAL" in intent_upper or "TOOL" in intent_upper or "CALCULATOR" in intent_upper:
            return cls.DEFAULT_TIERS["tool_executor"]

        # Default fallback
        return cls.DEFAULT_TIERS["tool_executor"]

    @classmethod
    def get_all_models_metadata(cls) -> Dict[str, Any]:
        """Returns structured metadata for all registered models."""
        return cls.CAPABILITIES
