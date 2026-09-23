from app.agents.base_agent import BaseAgent

class MachinesAgent(BaseAgent):
    """Specialist for Electrical Machines (Transformers, Induction Motors, Synchronous Machines, DC Motors)."""

    @property
    def name(self) -> str:
        return "Electrical Machines Specialist"

    @property
    def domain_description(self) -> str:
        return "Transformers, induction motors, synchronous generators, DC machines, equivalent circuits, torque-speed characteristics."

    def get_system_prompt(self, context_str: str = "") -> str:
        prompt = (
            "You are VOLTIX Electrical Machines Specialist Agent. "
            "Provide rigorous, step-by-step engineering analysis for electrical machinery problems. "
            "Always state Given parameters, standard equations (e.g. Ns = 120f/P, s = (Ns-N)/Ns, V1/V2 = N1/N2), "
            "perform exact substitutions, and clearly state final units (RPM, kW, N.m, %, A, V). "
            "Format math with KaTeX LaTeX notation."
        )
        if context_str:
            prompt += f"\n\n{context_str}"
        return prompt
