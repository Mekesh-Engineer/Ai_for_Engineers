import re
from typing import Dict, Any
from app.utils.logger import get_logger

logger = get_logger("voltix.router")

class QueryRouter:
    """Classifies user intent across all EEE disciplines, tool requirements, and problem types."""

    @staticmethod
    def classify_intent(query: str) -> Dict[str, Any]:
        text = query.lower()

        # 1. Summary / Document Analysis Intents
        if any(w in text for w in ["summarize this document", "summary of document", "summarize uploaded", "document summary"]):
            return {"intent": "DOCUMENT_SUMMARY", "agent": "CircuitTheoryAgent", "use_tool": False}
        if any(w in text for w in ["analyze this pdf", "document analysis", "table in document", "according to document"]):
            return {"intent": "DOCUMENT_ANALYSIS", "agent": "CircuitTheoryAgent", "use_tool": False}

        # 2. Simulink Modeling
        if "simulink" in text or "simscape" in text or "block diagram in simulink" in text:
            return {"intent": "SIMULINK", "agent": "MatlabSimulinkAgent", "use_tool": False}

        # 3. MATLAB Code Generation / Debugging
        if any(w in text for w in ["matlab", "bode plot", "step response", "nyquist", "root locus", "plot script"]):
            return {"intent": "MATLAB", "agent": "MatlabSimulinkAgent", "use_tool": False}

        # 4. Numerical / Deterministic Calculations
        calc_patterns = [
            r"\bcalculate\b", r"\bsolve\b", r"\bcompute\b", r"\bfind the\b",
            r"\bcurrent drawn\b", r"\bslip\b", r"\bpower factor\b", r"\bturns ratio\b",
            r"\bduty cycle\b", r"\befficiency\b", r"\bvoltage regulation\b", r"\btorque\b",
            r"\blaplace\b", r"\bintegrate\b", r"\bdifferentiate\b", r"\bresonant freq\b"
        ]
        for pat in calc_patterns:
            if re.search(pat, text):
                # Identify which specialized agent should accompany the calculation
                agent = "CircuitTheoryAgent"
                if any(w in text for w in ["motor", "transformer", "induction", "synchronous", "generator", "rotor"]):
                    agent = "MachinesAgent"
                elif any(w in text for w in ["buck", "boost", "converter", "inverter", "rectifier", "pwm", "duty"]):
                    agent = "PowerElectronicsAgent"
                elif any(w in text for w in ["grid", "fault", "transmission", "busbar", "load flow", "3-phase", "three-phase", "three phase"]):
                    agent = "PowerSystemsAgent"
                elif any(w in text for w in ["ev", "electric vehicle", "battery", "soc", "bms", "c-rate"]):
                    agent = "EVAgent"
                elif any(w in text for w in ["pv", "solar", "mppt", "wind", "irradiance"]):
                    agent = "RenewableAgent"
                elif any(w in text for w in ["transfer function", "damping", "zeta", "omega", "pid"]):
                    agent = "ControlSystemsAgent"

                return {"intent": "NUMERICAL", "agent": agent, "use_tool": True}

        # 5. Electric Vehicles & Powertrains
        if any(w in text for w in ["ev ", "electric vehicle", "powertrain", "tractive effort", "regenerative braking", "onboard charger"]):
            return {"intent": "EV", "agent": "EVAgent", "use_tool": False}

        # 6. Battery & BMS
        if any(w in text for w in ["battery", "bms", "cell balancing", "state of charge", "soc", "soh", "lithium-ion", "c-rate"]):
            return {"intent": "BATTERY", "agent": "BatteryAgent", "use_tool": False}

        # 7. Renewable Energy & Microgrids
        if any(w in text for w in ["solar", "photovoltaic", "pv array", "mppt", "wind turbine", "dfig", "microgrid", "island"]):
            return {"intent": "RENEWABLE", "agent": "RenewableAgent", "use_tool": False}

        # 8. Electrical Machines
        if any(w in text for w in ["motor", "transformer", "induction", "synchronous", "stator", "rotor", "armature", "dc motor", "stepper", "bldc"]):
            return {"intent": "MACHINE_ANALYSIS", "agent": "MachinesAgent", "use_tool": False}

        # 9. Power Systems & Protection
        if any(w in text for w in ["power system", "grid", "transmission line", "fault analysis", "load flow", "busbar", "relay", "circuit breaker", "per unit"]):
            return {"intent": "POWER_SYSTEM", "agent": "PowerSystemsAgent", "use_tool": False}

        # 10. Power Electronics & Drives
        if any(w in text for w in ["power electronics", "converter", "inverter", "rectifier", "pwm", "buck", "boost", "igbt", "mosfet", "thd", "cycloconverter"]):
            return {"intent": "POWER_ELECTRONICS", "agent": "PowerElectronicsAgent", "use_tool": False}

        # 11. Control Systems
        if any(w in text for w in ["control system", "transfer function", "feedback", "pid", "bode", "nyquist", "root locus", "state space", "gain margin", "phase margin"]):
            return {"intent": "CONTROL_SYSTEM", "agent": "ControlSystemsAgent", "use_tool": False}

        # 12. Analog & Digital Electronics
        if any(w in text for w in ["op-amp", "operational amplifier", "filter", "butterworth", "transistor", "bjt", "mosfet", "logic gate", "adc", "dac"]):
            return {"intent": "CIRCUIT_ANALYSIS", "agent": "ElectronicsAgent", "use_tool": False}

        # 13. Casual Conversational
        if any(w == text.strip() for w in ["hi", "hello", "hey", "who are you", "what can you do", "help", "thanks", "thank you"]):
            return {"intent": "CASUAL", "agent": "CircuitTheoryAgent", "use_tool": False}

        # Default EEE Conceptual Query
        return {"intent": "EEE_CONCEPT", "agent": "CircuitTheoryAgent", "use_tool": False}
