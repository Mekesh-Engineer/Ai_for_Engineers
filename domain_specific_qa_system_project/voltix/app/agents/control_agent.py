from app.agents.base_agent import BaseAgent

class ControlSystemsAgent(BaseAgent):
    """Specialist for Control Systems, Feedback, Transfer Functions, and PID Tuning."""

    @property
    def name(self) -> str:
        return "Control Systems Specialist"

    @property
    def domain_description(self) -> str:
        return "Transfer functions, Laplace transforms, state-space representations, Bode/Nyquist stability plots, root locus, PID controller design."

    def get_system_prompt(self, context_str: str = "") -> str:
        prompt = (
            "You are VOLTIX Control Systems Specialist Agent. "
            "Analyze closed-loop control systems, damping ratio ζ, natural frequency ωn, settling time, "
            "phase and gain margins, and state transition matrices."
        )
        if context_str:
            prompt += f"\n\n{context_str}"
        return prompt
