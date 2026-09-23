import json
from typing import Dict, Any, List, Optional
from app.llm.ollama_client import OllamaClient
from app.utils.logger import get_logger

logger = get_logger("voltix.agents.planner")

class TaskPlanner:
    """
    Decomposes complex, multi-faceted engineering user queries into an ordered task plan.
    Utilizes Gemma 4 or GPT-OSS 120B to structure reasoning steps and identify needed tools.
    """

    @staticmethod
    def generate_plan(
        user_query: str,
        ollama_client: Optional[OllamaClient] = None,
        model_name: str = "gemma4:cloud"
    ) -> Dict[str, Any]:
        """
        Analyzes the user's objective and generates a step-by-step execution plan.
        """
        client = ollama_client or OllamaClient()

        system_prompt = (
            "You are an expert AI Task Planner for an Electrical & Electronics Engineering platform. "
            "Analyze the user request and determine if it requires multiple steps or specialized tools.\n"
            "Available tool hints: [three_phase_power, single_phase_power, ohms_law, induction_motor_slip, "
            "power_factor_correction, buck_converter_sizing, boost_converter_sizing, battery_ev_calculations, "
            "solar_pv_sizing, sympy_solve_equation, sympy_differentiate, sympy_integrate, sympy_laplace_transform, "
            "sympy_transfer_function, sympy_rlc_transient, unit_converter, fetch_local_web_page, rag_search, none].\n\n"
            "Respond ONLY with a valid JSON object matching this exact schema:\n"
            "{\n"
            '  "requires_multi_step": true,\n'
            '  "objective": "Concise summary of user goal",\n'
            '  "steps": [\n'
            '    {"step": 1, "description": "...", "tool_hint": "..."},\n'
            '    {"step": 2, "description": "...", "tool_hint": "..."}\n'
            '  ]\n'
            "}"
        )

        try:
            res = client.chat_completion(
                model_name=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_query}
                ],
                options={"temperature": 0.0}
            )

            raw_text = res.get("message", {}).get("content", "{}")
            start = raw_text.find("{")
            end = raw_text.rfind("}") + 1
            if start != -1 and end > start:
                json_data = json.loads(raw_text[start:end])
                if "steps" in json_data and isinstance(json_data["steps"], list):
                    return json_data

        except Exception as e:
            logger.warning(f"TaskPlanner could not parse plan with {model_name}: {e}")

        # Default fallback single-step plan
        return {
            "requires_multi_step": False,
            "objective": user_query,
            "steps": [{"step": 1, "description": user_query, "tool_hint": "none"}]
        }
