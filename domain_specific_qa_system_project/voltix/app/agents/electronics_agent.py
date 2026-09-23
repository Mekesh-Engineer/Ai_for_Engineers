from app.agents.base_agent import BaseAgent

class ElectronicsAgent(BaseAgent):
    """Specialist for Analog and Digital Electronics, Op-Amps, Filters, and Semiconductor Devices."""

    @property
    def name(self) -> str:
        return "Analog & Digital Electronics Specialist"

    @property
    def domain_description(self) -> str:
        return "Operational amplifiers (inverting, non-inverting, differential, instrumentation), active filters (Butterworth, Chebyshev), BJT/MOSFET small-signal analysis, logic families, ADC/DAC converters."

    def get_system_prompt(self, context_str: str = "") -> str:
        prompt = (
            "You are VOLTIX Analog & Digital Electronics Specialist Agent. "
            "Analyze amplifier configurations, frequency response, cut-off frequencies, small-signal hybrid-pi models, "
            "and digital logic timing diagrams. Provide exact derivations and component values."
        )
        if context_str:
            prompt += f"\n\n{context_str}"
        return prompt
