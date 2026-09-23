from app.agents.base_agent import BaseAgent

class PowerSystemsAgent(BaseAgent):
    """Specialist for Power Systems Analysis, Load Flow, Protection, and Transmission."""

    @property
    def name(self) -> str:
        return "Power Systems Specialist"

    @property
    def domain_description(self) -> str:
        return "Power system analysis, load flow (Gauss-Seidel, Newton-Raphson), fault analysis, per-unit systems, transmission line parameters, relay protection."

    def get_system_prompt(self, context_str: str = "") -> str:
        prompt = (
            "You are VOLTIX Power Systems Specialist Agent. "
            "Examine grid stability, symmetrical/unsymmetrical fault analysis, per-unit impedance calculations, "
            "and transmission line dynamics. Include formulas for complex power S = P + jQ and admittance matrices Y_bus."
        )
        if context_str:
            prompt += f"\n\n{context_str}"
        return prompt
