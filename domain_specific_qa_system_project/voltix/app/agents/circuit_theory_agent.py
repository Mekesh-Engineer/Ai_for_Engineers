from app.agents.base_agent import BaseAgent

class CircuitTheoryAgent(BaseAgent):
    """Specialist for Circuit Analysis, KVL/KCL, Thevenin/Norton, and AC Phasors."""

    @property
    def name(self) -> str:
        return "Circuit Theory Specialist"

    @property
    def domain_description(self) -> str:
        return "Kirchhoff Laws (KVL/KCL), Thevenin/Norton equivalent circuits, Nodal/Mesh analysis, AC phasor domain, RLC transient response."

    def get_system_prompt(self, context_str: str = "") -> str:
        prompt = (
            "You are VOLTIX Circuit Theory Specialist Agent. "
            "Provide step-by-step circuit theorems application (KVL, KCL, Superposition, Maximum Power Transfer). "
            "Use clear node definitions and phasor notations Z = R + jX."
        )
        if context_str:
            prompt += f"\n\n{context_str}"
        return prompt
