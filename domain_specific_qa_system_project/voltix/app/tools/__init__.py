from app.tools.registry import ToolRegistry
from app.tools.ee_calculator import EECalculator
from app.tools.sympy_solver import SymPySolver
from app.tools.unit_converter import UnitConverter
from app.tools.matlab_generator import MatlabScriptGenerator
from app.tools.web_inspector import fetch_local_web_page

# ---------------------------------------------------------
# Tool Registration for Agentic Function Calling
# ---------------------------------------------------------

@ToolRegistry.register(
    name="three_phase_power",
    description="Calculate active/apparent power or line current for balanced 3-phase electrical systems."
)
def tool_three_phase_power(power_val: float = 0.0, voltage: float = 415.0, power_factor: float = 0.85, line_current: float = 0.0) -> dict:
    p = power_val if power_val > 0 else None
    if p is not None and p < 500: # if given in kW (e.g., 15 kW), convert to Watts
        p = p * 1000.0
    il = line_current if line_current > 0 else None
    return EECalculator.three_phase_power(p=p, vl=voltage, il=il, pf=power_factor)

@ToolRegistry.register(
    name="single_phase_power",
    description="Calculate active power, reactive power, apparent power, and current for a 1-phase AC circuit."
)
def tool_single_phase_power(voltage: float, current: float, power_factor: float = 0.85) -> dict:
    return EECalculator.single_phase_power(voltage, current, power_factor)

@ToolRegistry.register(
    name="ohms_law",
    description="Calculates missing Ohm's Law quantity (Voltage, Current, Resistance, or Power)."
)
def tool_ohms_law(voltage: float = 0.0, current: float = 0.0, resistance: float = 0.0) -> dict:
    return EECalculator.ohms_law(voltage or None, current or None, resistance or None)

@ToolRegistry.register(
    name="induction_motor_slip",
    description="Calculates induction motor slip, synchronous speed, and slip percentage."
)
def tool_motor_slip(sync_speed: float = 0.0, rotor_speed: float = 0.0, poles: int = 4, frequency: float = 50.0) -> dict:
    return EECalculator.induction_motor_slip(sync_speed or None, rotor_speed or None, poles, frequency)

@ToolRegistry.register(
    name="power_factor_correction",
    description="Sizes capacitor bank kVAR and capacitance (microfarads) to improve electrical power factor."
)
def tool_pf_correction(active_power_kw: float, current_pf: float, target_pf: float = 0.95, voltage_v: float = 415.0, frequency_hz: float = 50.0) -> dict:
    return EECalculator.power_factor_correction(active_power_kw, current_pf, target_pf, voltage_v, frequency_hz)

@ToolRegistry.register(
    name="buck_converter_sizing",
    description="Designs a DC-DC Buck converter: duty cycle, minimum inductance L, and output capacitance C."
)
def tool_buck_converter(vin: float, vout: float, iout: float, fsw_khz: float = 50.0, delta_il_pct: float = 30.0, delta_vo_pct: float = 1.0) -> dict:
    return EECalculator.buck_converter(vin, vout, iout, fsw_khz, delta_il_pct, delta_vo_pct)

@ToolRegistry.register(
    name="boost_converter_sizing",
    description="Designs a DC-DC Boost converter: duty cycle, inductor L, output capacitor C, and diode stress."
)
def tool_boost_converter(vin: float, vout: float, iout: float, fsw_khz: float = 50.0, delta_il_pct: float = 30.0, delta_vo_pct: float = 1.0) -> dict:
    return EECalculator.boost_converter(vin, vout, iout, fsw_khz, delta_il_pct, delta_vo_pct)

@ToolRegistry.register(
    name="battery_ev_calculations",
    description="Calculates EV driving range, charging time (AC/DC), and energy consumption per 100km."
)
def tool_battery_ev(battery_kwh: float, consumption_wh_km: float = 150.0, charging_power_kw: float = 22.0) -> dict:
    return EECalculator.battery_ev_calculations(battery_kwh, consumption_wh_km, charging_power_kw)

@ToolRegistry.register(
    name="solar_pv_sizing",
    description="Calculates solar PV panel Fill Factor (FF), maximum power output, efficiency, and array sizing."
)
def tool_solar_pv(voc: float, isc: float, vmp: float, imp: float, irrad_w_m2: float = 1000.0, temp_c: float = 25.0) -> dict:
    return EECalculator.solar_pv_sizing(voc, isc, vmp, imp, irrad_w_m2, temp_c)

@ToolRegistry.register(
    name="sympy_solve_equation",
    description="Analytically solve an algebraic or polynomial equation using SymPy."
)
def tool_sympy_solve(equation_str: str, variable: str = "x") -> dict:
    return SymPySolver.solve_equation(equation_str, variable)

@ToolRegistry.register(
    name="sympy_differentiate",
    description="Compute symbolic derivative of a mathematical expression with steps."
)
def tool_sympy_diff(expression_str: str, variable: str = "x", order: int = 1) -> dict:
    return SymPySolver.differentiate(expression_str, variable, order)

@ToolRegistry.register(
    name="sympy_integrate",
    description="Compute symbolic definite or indefinite integral of a mathematical expression."
)
def tool_sympy_integrate(expression_str: str, variable: str = "x", lower_limit: str = None, upper_limit: str = None) -> dict:
    return SymPySolver.integrate(expression_str, variable, lower_limit, upper_limit)

@ToolRegistry.register(
    name="sympy_laplace_transform",
    description="Compute symbolic unilateral Laplace transform L{f(t)} = F(s) with ROC convergence."
)
def tool_sympy_laplace(expression_str: str, t_var: str = "t", s_var: str = "s") -> dict:
    return SymPySolver.laplace_transform(expression_str, t_var, s_var)

@ToolRegistry.register(
    name="sympy_inverse_laplace",
    description="Compute symbolic Inverse Laplace transform L^-1{F(s)} = f(t)."
)
def tool_sympy_inv_laplace(expression_str: str, s_var: str = "s", t_var: str = "t") -> dict:
    return SymPySolver.inverse_laplace(expression_str, s_var, t_var)

@ToolRegistry.register(
    name="sympy_transfer_function",
    description="Analyze a control system transfer function H(s): poles, zeros, stability, and characteristic polynomial."
)
def tool_sympy_tf(num_poly: str, den_poly: str, s_var: str = "s") -> dict:
    return SymPySolver.transfer_function_analysis(num_poly, den_poly, s_var)

@ToolRegistry.register(
    name="sympy_rlc_transient",
    description="Analytically evaluate series RLC transient response: damping factor, natural frequency, damping state."
)
def tool_sympy_rlc(r_val: float, l_val: float, c_val: float) -> dict:
    return SymPySolver.rlc_transient_response(r_val, l_val, c_val)

@ToolRegistry.register(
    name="unit_converter",
    description="Convert between engineering units (e.g. kW to W, rpm to rad/s, kV to V, uF to F)."
)
def tool_unit_converter(value: float, from_unit: str, to_unit: str) -> dict:
    return UnitConverter.convert(value, from_unit, to_unit)

__all__ = [
    "ToolRegistry",
    "EECalculator",
    "SymPySolver",
    "UnitConverter",
    "MatlabScriptGenerator",
    "fetch_local_web_page",
]
