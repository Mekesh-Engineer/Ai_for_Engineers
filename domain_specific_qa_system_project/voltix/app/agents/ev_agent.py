from app.agents.base_agent import BaseAgent

class EVAgent(BaseAgent):
    """Specialist for Electric Vehicles, EV Powertrains, Motor Sizing, and Inverters."""

    @property
    def name(self) -> str:
        return "Electric Vehicle & Powertrain Specialist"

    @property
    def domain_description(self) -> str:
        return "EV powertrains, traction motors (PMSM, BLDC, Induction), battery pack integration, regenerative braking, onboard chargers, DC fast charging."

    def get_system_prompt(self, context_str: str = "") -> str:
        prompt = (
            "You are VOLTIX Electric Vehicles & Powertrain Specialist Agent. "
            "Analyze vehicle dynamics, tractive effort equations (aerodynamic drag, rolling resistance, grade resistance), "
            "traction motor torque/power sizing, inverter modulation, and EV range calculations. "
            "Format math using KaTeX LaTeX notation."
        )
        if context_str:
            prompt += f"\n\n{context_str}"
        return prompt
