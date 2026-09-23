import sympy as sp
from typing import Dict, Any, Optional, List
from app.utils.logger import get_logger

logger = get_logger("voltix.sympy_solver")

class SymPySolver:
    """Advanced Symbolic computation engine for Electrical Engineering mathematical analysis."""

    @staticmethod
    def solve_equation(expr_str: str, variable_str: str = "x") -> Dict[str, Any]:
        """Solves an algebraic equation (e.g., '2*x + 5 = 15' or 's**2 + 3*s + 2 = 0')."""
        try:
            var = sp.Symbol(variable_str)
            if "=" in expr_str:
                lhs, rhs = expr_str.split("=", 1)
                eq = sp.Eq(sp.sympify(lhs.strip()), sp.sympify(rhs.strip()))
            else:
                eq = sp.sympify(expr_str)

            solutions = sp.solve(eq, var)
            return {
                "success": True,
                "operation": "Solve Algebraic Equation",
                "expression": expr_str,
                "variable": variable_str,
                "solutions": [str(s) for s in solutions],
                "latex": sp.latex(solutions),
                "formatted": f"Solutions for {variable_str}: {', '.join([str(s) for s in solutions])}"
            }
        except Exception as e:
            logger.error(f"SymPy solve error: {e}")
            return {"success": False, "error": str(e)}

    @staticmethod
    def differentiate(expr_str: str, variable_str: str = "t", order: int = 1) -> Dict[str, Any]:
        """Performs symbolic differentiation."""
        try:
            var = sp.Symbol(variable_str)
            expr = sp.sympify(expr_str)
            derivative = sp.diff(expr, var, order)
            return {
                "success": True,
                "operation": f"Derivative (Order {order})",
                "expression": expr_str,
                "derivative": str(derivative),
                "latex": sp.latex(derivative),
                "formatted": f"d^{order}/d{variable_str}^{order} [{expr_str}] = {derivative}"
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    @staticmethod
    def integrate(expr_str: str, variable_str: str = "t", lower: Optional[str] = None, upper: Optional[str] = None) -> Dict[str, Any]:
        """Performs symbolic definite or indefinite integration."""
        try:
            var = sp.Symbol(variable_str)
            expr = sp.sympify(expr_str)
            if lower is not None and upper is not None:
                res = sp.integrate(expr, (var, sp.sympify(lower), sp.sympify(upper)))
                formatted = f"∫_{lower}^{upper} ({expr_str}) d{variable_str} = {res}"
            else:
                res = sp.integrate(expr, var)
                formatted = f"∫ ({expr_str}) d{variable_str} = {res} + C"

            return {
                "success": True,
                "operation": "Integration",
                "expression": expr_str,
                "result": str(res),
                "latex": sp.latex(res),
                "formatted": formatted
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    @staticmethod
    def laplace_transform(expr_str: str, t_var: str = "t", s_var: str = "s") -> Dict[str, Any]:
        """Computes unilateral Laplace transform L{f(t)} = F(s)."""
        try:
            t = sp.Symbol(t_var, positive=True)
            s = sp.Symbol(s_var)
            expr = sp.sympify(expr_str, locals={t_var: t, s_var: s, "exp": sp.exp, "sin": sp.sin, "cos": sp.cos})
            res = sp.laplace_transform(expr, t, s, noconds=True)
            return {
                "success": True,
                "operation": "Laplace Transform",
                "time_domain": expr_str,
                "frequency_domain": str(res),
                "latex": sp.latex(res),
                "formatted": f"L{{{expr_str}}} = {res}"
            }
        except Exception as e:
            return {"success": False, "error": f"Laplace transform calculation failed: {str(e)}"}

    @staticmethod
    def inverse_laplace_transform(expr_str: str, s_var: str = "s", t_var: str = "t") -> Dict[str, Any]:
        """Computes inverse Laplace transform L^-1{F(s)} = f(t)."""
        try:
            s = sp.Symbol(s_var)
            t = sp.Symbol(t_var, positive=True)
            expr = sp.sympify(expr_str, locals={s_var: s, t_var: t, "exp": sp.exp, "sin": sp.sin, "cos": sp.cos})
            res = sp.inverse_laplace_transform(expr, s, t)
            return {
                "success": True,
                "operation": "Inverse Laplace Transform",
                "frequency_domain": expr_str,
                "time_domain": str(res),
                "latex": sp.latex(res),
                "formatted": f"L^-1{{{expr_str}}} = {res}"
            }
        except Exception as e:
            return {"success": False, "error": f"Inverse Laplace calculation failed: {str(e)}"}

    @staticmethod
    def transfer_function_analysis(num_str: str, den_str: str) -> Dict[str, Any]:
        """Analyzes poles, zeros, and characteristic stability for G(s) = Num(s) / Den(s)."""
        try:
            s = sp.Symbol("s")
            num = sp.sympify(num_str)
            den = sp.sympify(den_str)

            zeros = sp.solve(num, s)
            poles = sp.solve(den, s)

            # Stability check: all poles must have negative real parts
            is_stable = True
            pole_details = []
            for p in poles:
                val = complex(p.evalf())
                pole_details.append(f"{round(val.real, 4)} + {round(val.imag, 4)}j" if val.imag != 0 else f"{round(val.real, 4)}")
                if val.real >= 0:
                    is_stable = False

            return {
                "success": True,
                "operation": "Transfer Function Pole-Zero Analysis",
                "transfer_function": f"({num_str}) / ({den_str})",
                "zeros": [str(z) for z in zeros],
                "poles": [str(p) for p in poles],
                "pole_values": pole_details,
                "is_stable": is_stable,
                "stability_summary": "BIBO Stable (All poles in Open Left Half-Plane)" if is_stable else "Unstable or Marginally Stable (Poles in RHP or on imaginary axis)"
            }
        except Exception as e:
            return {"success": False, "error": f"Transfer function analysis failed: {str(e)}"}

    @staticmethod
    def rlc_transient_response(r: float, l: float, c: float) -> Dict[str, Any]:
        """Analyzes series RLC transient damping and natural resonance characteristics."""
        if l <= 0 or c <= 0:
            return {"success": False, "error": "L and C must be positive numbers"}

        alpha = r / (2.0 * l)
        w0 = 1.0 / sp.sqrt(l * c)
        w0_val = float(w0.evalf())

        if alpha > w0_val:
            response_type = "Overdamped (Two distinct real negative roots)"
            s1 = -alpha + sp.sqrt(alpha**2 - w0_val**2).evalf()
            s2 = -alpha - sp.sqrt(alpha**2 - w0_val**2).evalf()
            roots_str = f"s1 = {round(float(s1), 4)}, s2 = {round(float(s2), 4)}"
        elif abs(alpha - w0_val) < 1e-6:
            response_type = "Critically Damped (Repeated real root, fastest response without overshoot)"
            roots_str = f"s1,2 = -{round(alpha, 4)}"
        else:
            response_type = "Underdamped (Complex conjugate roots, oscillatory response)"
            wd = sp.sqrt(w0_val**2 - alpha**2).evalf()
            roots_str = f"s1,2 = -{round(alpha, 4)} ± j{round(float(wd), 4)}"

        return {
            "success": True,
            "operation": "Series RLC Transient Characteristic Analysis",
            "given": {"Resistance (R)": f"{r} Ω", "Inductance (L)": f"{l} H", "Capacitance (C)": f"{c} F"},
            "alpha_rad_s": round(alpha, 4),
            "w0_rad_s": round(w0_val, 4),
            "response_type": response_type,
            "characteristic_roots": roots_str,
            "formula": "α = R / (2L), ω0 = 1 / √(LC), s², s1,2 = -α ± √(α² - ω0²)"
        }
