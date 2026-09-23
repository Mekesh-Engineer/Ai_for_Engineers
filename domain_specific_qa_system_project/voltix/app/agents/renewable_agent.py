from app.agents.base_agent import BaseAgent

class RenewableAgent(BaseAgent):
    """Specialist for Renewable Energy Systems (Solar PV, Wind, MPPT, Microgrids)."""

    @property
    def name(self) -> str:
        return "Renewable Energy & Microgrids Specialist"

    @property
    def domain_description(self) -> str:
        return "Solar Photovoltaic (PV) IV/PV curves, Maximum Power Point Tracking (Perturb & Observe, Incremental Conductance), Wind turbine generators (DFIG, PMSG), microgrid islanding."

    def get_system_prompt(self, context_str: str = "") -> str:
        prompt = (
            "You are VOLTIX Renewable Energy & Microgrids Specialist Agent. "
            "Analyze PV cell modeling (single diode model), temperature coefficients, MPPT tracking algorithms, "
            "grid-tied inverters, and microgrid synchronization."
        )
        if context_str:
            prompt += f"\n\n{context_str}"
        return prompt
