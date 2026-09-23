from app.agents.base_agent import BaseAgent

class MatlabSimulinkAgent(BaseAgent):
    """Specialist for MATLAB Code Generation and Simulink Modeling Guidance."""

    @property
    def name(self) -> str:
        return "MATLAB & Simulink Specialist"

    @property
    def domain_description(self) -> str:
        return "Executable MATLAB script generation, Control System Toolbox (`tf`, `bode`, `step`), Simscape Electrical, Simulink block parameter setup."

    def get_system_prompt(self, context_str: str = "") -> str:
        prompt = (
            "You are VOLTIX MATLAB & Simulink Specialist Agent. "
            "Synthesize clean, self-contained, syntax-valid MATLAB code blocks. "
            "Provide explanation of parameters, variable definitions, and step-by-step Simulink block setup guidelines."
        )
        if context_str:
            prompt += f"\n\n{context_str}"
        return prompt
