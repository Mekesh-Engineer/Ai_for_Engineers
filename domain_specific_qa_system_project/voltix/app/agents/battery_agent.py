from app.agents.base_agent import BaseAgent

class BatteryAgent(BaseAgent):
    """Specialist for Battery Management Systems (BMS), Cell Balancing, and Energy Storage."""

    @property
    def name(self) -> str:
        return "Battery & Energy Storage Specialist"

    @property
    def domain_description(self) -> str:
        return "Lithium-ion chemistries (NMC, LFP, LTO), State of Charge (SoC) estimation (Coulomb counting, Kalman Filter), State of Health (SoH), active/passive cell balancing, thermal runaway prevention."

    def get_system_prompt(self, context_str: str = "") -> str:
        prompt = (
            "You are VOLTIX Battery & Energy Storage Specialist Agent. "
            "Analyze battery chemistries, equivalent circuit models (Thevenin/RC models), "
            "BMS architecture, cell balancing circuits, C-rate limits, and thermal management."
        )
        if context_str:
            prompt += f"\n\n{context_str}"
        return prompt
