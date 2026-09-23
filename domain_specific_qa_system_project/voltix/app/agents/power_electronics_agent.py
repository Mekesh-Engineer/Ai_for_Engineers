from app.agents.base_agent import BaseAgent

class PowerElectronicsAgent(BaseAgent):
    """Specialist for Power Electronics, Converters, Inverters, and PWM."""

    @property
    def name(self) -> str:
        return "Power Electronics Specialist"

    @property
    def domain_description(self) -> str:
        return "Buck/Boost/Buck-Boost DC-DC converters, single & 3-phase inverters, rectifiers, PWM switching schemes, ripple & THD."

    def get_system_prompt(self, context_str: str = "") -> str:
        prompt = (
            "You are VOLTIX Power Electronics Specialist Agent. "
            "Analyze switching converters, semiconductor devices (IGBT, MOSFET), duty cycles (D), "
            "inductor ripple current ΔIL, capacitor ripple voltage ΔVc, and harmonic distortion."
        )
        if context_str:
            prompt += f"\n\n{context_str}"
        return prompt
