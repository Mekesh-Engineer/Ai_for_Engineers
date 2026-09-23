import math
from typing import Dict, Any, Optional
from app.utils.logger import get_logger

logger = get_logger("voltix.ee_calculator")

class EECalculator:
    """Comprehensive, deterministic engineering solvers for core Electrical & Electronics Engineering formulas."""

    @staticmethod
    def three_phase_power(
        p: Optional[float] = None,
        vl: Optional[float] = None,
        il: Optional[float] = None,
        pf: Optional[float] = None
    ) -> Dict[str, Any]:
        """Calculates 3-phase real power P = sqrt(3) * V_L * I_L * pf or solves for line current I_L."""
        if il is None and p is not None and vl is not None and pf is not None:
            il_calc = p / (math.sqrt(3) * vl * pf)
            s_calc = math.sqrt(3) * vl * il_calc
            q_calc = math.sqrt(3) * vl * il_calc * math.sin(math.acos(min(max(pf, -1.0), 1.0)))
            return {
                "success": True,
                "tool_name": "Three-Phase Power & Current Solver",
                "given": {"Power (P)": f"{p} W ({round(p/1000, 3)} kW)", "Line Voltage (V_L)": f"{vl} V", "Power Factor (cos φ)": pf},
                "required": "Line Current (I_L), Apparent Power (S), Reactive Power (Q)",
                "formula": "P = √3 · V_L · I_L · cos(φ) => I_L = P / (√3 · V_L · cos φ)",
                "substitution": f"I_L = {p} / (1.73205 · {vl} · {pf})",
                "calculation": f"I_L = {round(il_calc, 4)} A",
                "result": {
                    "line_current_a": round(il_calc, 4),
                    "apparent_power_va": round(s_calc, 2),
                    "reactive_power_var": round(q_calc, 2),
                    "power_factor": pf
                },
                "units": "A",
                "assumptions": ["Balanced 3-phase system", "Sinusoidal voltages and currents"]
            }
        elif p is None and vl is not None and il is not None and pf is not None:
            p_calc = math.sqrt(3) * vl * il * pf
            s_calc = math.sqrt(3) * vl * il
            q_calc = math.sqrt(3) * vl * il * math.sin(math.acos(min(max(pf, -1.0), 1.0)))
            return {
                "success": True,
                "tool_name": "Three-Phase Power Solver",
                "given": {"Line Voltage (V_L)": f"{vl} V", "Line Current (I_L)": f"{il} A", "Power Factor (cos φ)": pf},
                "required": "Active Power (P), Apparent Power (S), Reactive Power (Q)",
                "formula": "P = √3 · V_L · I_L · cos(φ)",
                "substitution": f"P = 1.73205 · {vl} · {il} · {pf}",
                "calculation": f"P = {round(p_calc, 2)} W ({round(p_calc/1000, 3)} kW)",
                "result": {
                    "active_power_w": round(p_calc, 2),
                    "active_power_kw": round(p_calc/1000, 3),
                    "apparent_power_va": round(s_calc, 2),
                    "reactive_power_var": round(q_calc, 2)
                },
                "units": "W",
                "assumptions": ["Balanced 3-phase system"]
            }
        return {"success": False, "error": "Provide either (P, V_L, pf) to find I_L, or (V_L, I_L, pf) to find P."}

    @staticmethod
    def single_phase_power(v: float, i: float, pf: float) -> Dict[str, Any]:
        """Single-phase power calculations: P = V * I * pf."""
        p = v * i * pf
        s = v * i
        theta = math.acos(min(max(pf, -1.0), 1.0))
        q = v * i * math.sin(theta)
        return {
            "success": True,
            "tool_name": "Single-Phase AC Power Solver",
            "given": {"Voltage (V)": f"{v} V", "Current (I)": f"{i} A", "Power Factor": pf},
            "required": "Active Power (P), Reactive Power (Q), Apparent Power (S)",
            "formula": "P = V · I · cos(φ), Q = V · I · sin(φ), S = V · I",
            "substitution": f"P = {v} · {i} · {pf}",
            "calculation": f"P = {round(p, 2)} W, Q = {round(q, 2)} VAR, S = {round(s, 2)} VA",
            "result": {"p_w": round(p, 2), "q_var": round(q, 2), "s_va": round(s, 2)},
            "units": "W",
            "assumptions": ["Sinusoidal steady-state AC circuit"]
        }

    @staticmethod
    def ohms_law(v: Optional[float] = None, i: Optional[float] = None, r: Optional[float] = None) -> Dict[str, Any]:
        """Calculates missing parameter in Ohm's Law and associated power dissipation."""
        if v is None and i is not None and r is not None:
            v_calc = i * r
            p_calc = (i ** 2) * r
            return {
                "success": True,
                "tool_name": "Ohm's Law Solver",
                "given": {"Current (I)": f"{i} A", "Resistance (R)": f"{r} Ω"},
                "required": "Voltage (V), Power (P)",
                "formula": "V = I · R, P = I² · R",
                "substitution": f"V = {i} · {r}, P = ({i})² · {r}",
                "calculation": f"V = {round(v_calc, 4)} V, P = {round(p_calc, 4)} W",
                "result": {"voltage_v": round(v_calc, 4), "power_w": round(p_calc, 4), "current_a": i, "resistance_ohm": r},
                "units": "V",
                "assumptions": ["Linear, time-invariant resistance"]
            }
        elif i is None and v is not None and r is not None:
            if r == 0:
                return {"success": False, "error": "Resistance cannot be zero (short circuit)"}
            i_calc = v / r
            p_calc = (v ** 2) / r
            return {
                "success": True,
                "tool_name": "Ohm's Law Solver",
                "given": {"Voltage (V)": f"{v} V", "Resistance (R)": f"{r} Ω"},
                "required": "Current (I), Power (P)",
                "formula": "I = V / R, P = V² / R",
                "substitution": f"I = {v} / {r}, P = ({v})² / {r}",
                "calculation": f"I = {round(i_calc, 4)} A, P = {round(p_calc, 4)} W",
                "result": {"current_a": round(i_calc, 4), "power_w": round(p_calc, 4), "voltage_v": v, "resistance_ohm": r},
                "units": "A",
                "assumptions": ["Linear, time-invariant resistance"]
            }
        elif r is None and v is not None and i is not None:
            if i == 0:
                return {"success": False, "error": "Current cannot be zero (open circuit)"}
            r_calc = v / i
            p_calc = v * i
            return {
                "success": True,
                "tool_name": "Ohm's Law Solver",
                "given": {"Voltage (V)": f"{v} V", "Current (I)": f"{i} A"},
                "required": "Resistance (R), Power (P)",
                "formula": "R = V / I, P = V · I",
                "substitution": f"R = {v} / {i}, P = {v} · {i}",
                "calculation": f"R = {round(r_calc, 4)} Ω, P = {round(p_calc, 4)} W",
                "result": {"resistance_ohm": round(r_calc, 4), "power_w": round(p_calc, 4), "voltage_v": v, "current_a": i},
                "units": "Ω",
                "assumptions": ["Linear, time-invariant resistance"]
            }
        return {"success": False, "error": "Provide exactly two parameters among (v, i, r)."}

    @staticmethod
    def induction_motor_slip(ns: float, n: float) -> Dict[str, Any]:
        """Calculates induction motor slip: s = (Ns - N) / Ns."""
        if ns <= 0:
            return {"success": False, "error": "Synchronous speed Ns must be greater than 0"}
        slip = (ns - n) / ns
        slip_pct = slip * 100.0
        return {
            "success": True,
            "tool_name": "Induction Motor Slip Calculator",
            "given": {"Synchronous Speed (Ns)": f"{ns} RPM", "Rotor Speed (N)": f"{n} RPM"},
            "required": "Slip (s in per-unit and percentage)",
            "formula": "s = (Ns - N) / Ns",
            "substitution": f"s = ({ns} - {n}) / {ns}",
            "calculation": f"s = {round(slip, 4)} pu ({round(slip_pct, 2)}%)",
            "result": {
                "slip_pu": round(slip, 4),
                "slip_percent": round(slip_pct, 2),
                "synchronous_speed_rpm": ns,
                "rotor_speed_rpm": n
            },
            "units": "%",
            "assumptions": ["Standard induction machine operation in motoring mode"]
        }

    @staticmethod
    def synchronous_speed(f: float, p: int) -> Dict[str, Any]:
        """Calculates synchronous speed: Ns = 120 * f / P."""
        if p <= 0 or p % 2 != 0:
            return {"success": False, "error": "Poles P must be a positive even integer (2, 4, 6, 8, ...)"}
        ns = (120.0 * f) / p
        return {
            "success": True,
            "tool_name": "Synchronous Speed Calculator",
            "given": {"Supply Frequency (f)": f"{f} Hz", "Number of Poles (P)": p},
            "required": "Synchronous Speed (Ns)",
            "formula": "Ns = 120 · f / P",
            "substitution": f"Ns = 120 · {f} / {p}",
            "calculation": f"Ns = {round(ns, 2)} RPM",
            "result": {"synchronous_speed_rpm": round(ns, 2), "frequency_hz": f, "poles": p},
            "units": "RPM",
            "assumptions": ["Standard AC stator winding supply"]
        }

    @staticmethod
    def motor_torque(power_w: float, speed_rpm: float) -> Dict[str, Any]:
        """Calculates shaft torque: T = P / omega = (60 * P) / (2 * pi * N)."""
        if speed_rpm <= 0:
            return {"success": False, "error": "Rotor speed must be greater than 0"}
        omega = (2.0 * math.pi * speed_rpm) / 60.0
        torque = power_w / omega
        return {
            "success": True,
            "tool_name": "Shaft Torque Calculator",
            "given": {"Mechanical Power (P)": f"{power_w} W ({round(power_w/1000, 3)} kW)", "Speed (N)": f"{speed_rpm} RPM"},
            "required": "Mechanical Torque (T), Angular Velocity (ω)",
            "formula": "ω = 2π·N / 60, T = P / ω = 60·P / (2π·N)",
            "substitution": f"T = 60 · {power_w} / (2 · 3.14159 · {speed_rpm})",
            "calculation": f"ω = {round(omega, 3)} rad/s, T = {round(torque, 3)} N·m",
            "result": {"torque_nm": round(torque, 3), "angular_velocity_rad_s": round(omega, 3)},
            "units": "N·m",
            "assumptions": ["Neglecting stray mechanical friction unless specified"]
        }

    @staticmethod
    def transformer_efficiency_regulation(
        rating_kva: float,
        pf: float,
        p_iron_w: float,
        p_cu_fl_w: float,
        fraction_load: float = 1.0,
        v_nl: Optional[float] = None,
        v_fl: Optional[float] = None
    ) -> Dict[str, Any]:
        """Calculates transformer efficiency at fractional load and voltage regulation."""
        p_out = fraction_load * (rating_kva * 1000.0) * pf
        p_cu_actual = (fraction_load ** 2) * p_cu_fl_w
        total_losses = p_iron_w + p_cu_actual
        p_in = p_out + total_losses
        eff = (p_out / p_in) * 100.0 if p_in > 0 else 0.0

        vr_pct = None
        if v_nl is not None and v_fl is not None and v_fl > 0:
            vr_pct = ((v_nl - v_fl) / v_fl) * 100.0

        return {
            "success": True,
            "tool_name": "Transformer Efficiency & Regulation Calculator",
            "given": {
                "Rating": f"{rating_kva} kVA",
                "Load Fraction (x)": fraction_load,
                "Power Factor": pf,
                "Core Loss (Pi)": f"{p_iron_w} W",
                "Full-load Copper Loss (Pcu)": f"{p_cu_fl_w} W"
            },
            "required": "Efficiency (η) and Voltage Regulation (VR)",
            "formula": "η = (x·S·cos φ) / (x·S·cos φ + Pi + x²·Pcu) · 100%, VR = (V_nl - V_fl) / V_fl · 100%",
            "substitution": f"P_out = {fraction_load}·{rating_kva*1000}·{pf} = {p_out} W, Total Losses = {p_iron_w} + ({fraction_load}²·{p_cu_fl_w}) = {round(total_losses, 2)} W",
            "calculation": f"Efficiency = {round(eff, 2)}%" + (f", Voltage Regulation = {round(vr_pct, 2)}%" if vr_pct is not None else ""),
            "result": {
                "efficiency_percent": round(eff, 2),
                "total_losses_w": round(total_losses, 2),
                "copper_loss_w": round(p_cu_actual, 2),
                "iron_loss_w": p_iron_w,
                "voltage_regulation_percent": round(vr_pct, 2) if vr_pct is not None else None
            },
            "units": "%",
            "assumptions": ["Constant core losses independent of load"]
        }

    @staticmethod
    def power_factor_correction(
        active_power_kw: float,
        initial_pf: float,
        target_pf: float,
        voltage_v: float,
        freq_hz: float = 50.0
    ) -> Dict[str, Any]:
        """Calculates capacitor VARs (Qc) and capacitance (C) required to improve power factor."""
        p_w = active_power_kw * 1000.0
        theta1 = math.acos(min(max(initial_pf, 0.01), 1.0))
        theta2 = math.acos(min(max(target_pf, 0.01), 1.0))

        qc = p_w * (math.tan(theta1) - math.tan(theta2))
        c_farad = qc / (2.0 * math.pi * freq_hz * (voltage_v ** 2)) if voltage_v > 0 else 0.0
        c_uf = c_farad * 1e6

        return {
            "success": True,
            "tool_name": "Power Factor Correction Calculator",
            "given": {
                "Active Power (P)": f"{active_power_kw} kW",
                "Initial PF (cos φ1)": initial_pf,
                "Target PF (cos φ2)": target_pf,
                "Voltage (V)": f"{voltage_v} V",
                "Frequency (f)": f"{freq_hz} Hz"
            },
            "required": "Required Capacitor Bank Reactive Power (Qc), Required Capacitance (C)",
            "formula": "Qc = P · (tan φ1 - tan φ2), C = Qc / (2π · f · V²)",
            "substitution": f"Qc = {p_w} · (tan({round(math.degrees(theta1), 2)}°) - tan({round(math.degrees(theta2), 2)}°))",
            "calculation": f"Qc = {round(qc, 2)} VAR ({round(qc/1000, 3)} kVAR), C = {round(c_uf, 2)} μF",
            "result": {
                "qc_kvar": round(qc / 1000.0, 3),
                "qc_var": round(qc, 2),
                "capacitance_uf": round(c_uf, 2),
                "capacitance_farad": c_farad
            },
            "units": "kVAR / μF",
            "assumptions": ["Pure shunt capacitor compensation at fundamental frequency"]
        }

    @staticmethod
    def dc_dc_converter_buck_boost(
        topology: str,
        vin: float,
        vout: float,
        iout: float,
        freq_hz: float,
        delta_il_ratio: float = 0.2,
        delta_vc_ratio: float = 0.01
    ) -> Dict[str, Any]:
        """Calculates Duty cycle D, Inductor ripple ΔIL, critical inductance L_min, and filter capacitance C."""
        topology_lower = topology.lower()
        if "buck" in topology_lower and "boost" not in topology_lower:
            # Buck: Vo = D * Vin => D = Vo / Vin
            d = vout / vin if vin > 0 else 0.0
            r_load = vout / iout if iout > 0 else 1.0
            delta_il = delta_il_ratio * iout
            l_val = ((vin - vout) * d) / (freq_hz * delta_il) if (freq_hz * delta_il) > 0 else 0.0
            delta_vc = delta_vc_ratio * vout
            c_val = delta_il / (8.0 * freq_hz * delta_vc) if (8.0 * freq_hz * delta_vc) > 0 else 0.0
            l_crit = ((1.0 - d) * r_load) / (2.0 * freq_hz)

            return {
                "success": True,
                "tool_name": "Buck (Step-Down) Converter Design Calculator",
                "given": {"Input Voltage (Vin)": f"{vin} V", "Output Voltage (Vo)": f"{vout} V", "Load Current (Io)": f"{iout} A", "Switching Frequency (fs)": f"{freq_hz} Hz"},
                "required": "Duty Cycle (D), Filter Inductance (L), Filter Capacitance (C), Critical Inductance (L_crit)",
                "formula": "D = Vo / Vin, L = (Vin - Vo)·D / (fs · ΔIL), C = ΔIL / (8·fs · ΔVc), L_crit = (1 - D)·R / (2·fs)",
                "substitution": f"D = {vout} / {vin} = {round(d, 4)}",
                "calculation": f"Duty Cycle = {round(d*100, 2)}%, L = {round(l_val*1e6, 2)} μH, C = {round(c_val*1e6, 2)} μF, L_crit = {round(l_crit*1e6, 2)} μH",
                "result": {
                    "topology": "Buck",
                    "duty_cycle": round(d, 4),
                    "duty_cycle_pct": round(d * 100, 2),
                    "inductance_uh": round(l_val * 1e6, 2),
                    "capacitance_uf": round(c_val * 1e6, 2),
                    "critical_inductance_uh": round(l_crit * 1e6, 2)
                },
                "units": "Duty % / μH / μF",
                "assumptions": ["Continuous Conduction Mode (CCM)", "Ideal switches with zero on-state drop"]
            }
        elif "boost" in topology_lower:
            # Boost: Vo = Vin / (1 - D) => D = 1 - Vin / Vo
            d = 1.0 - (vin / vout) if vout > 0 else 0.0
            iin = (vout * iout) / vin if vin > 0 else iout
            r_load = vout / iout if iout > 0 else 1.0
            delta_il = delta_il_ratio * iin
            l_val = (vin * d) / (freq_hz * delta_il) if (freq_hz * delta_il) > 0 else 0.0
            delta_vc = delta_vc_ratio * vout
            c_val = (iout * d) / (freq_hz * delta_vc) if (freq_hz * delta_vc) > 0 else 0.0
            l_crit = (d * ((1.0 - d) ** 2) * r_load) / (2.0 * freq_hz)

            return {
                "success": True,
                "tool_name": "Boost (Step-Up) Converter Design Calculator",
                "given": {"Input Voltage (Vin)": f"{vin} V", "Output Voltage (Vo)": f"{vout} V", "Load Current (Io)": f"{iout} A", "Switching Frequency (fs)": f"{freq_hz} Hz"},
                "required": "Duty Cycle (D), Filter Inductance (L), Filter Capacitance (C), Critical Inductance (L_crit)",
                "formula": "D = 1 - (Vin / Vo), L = (Vin · D) / (fs · ΔIL), C = (Io · D) / (fs · ΔVc), L_crit = D·(1-D)²·R / (2·fs)",
                "substitution": f"D = 1 - ({vin} / {vout}) = {round(d, 4)}",
                "calculation": f"Duty Cycle = {round(d*100, 2)}%, L = {round(l_val*1e6, 2)} μH, C = {round(c_val*1e6, 2)} μF, L_crit = {round(l_crit*1e6, 2)} μH",
                "result": {
                    "topology": "Boost",
                    "duty_cycle": round(d, 4),
                    "duty_cycle_pct": round(d * 100, 2),
                    "inductance_uh": round(l_val * 1e6, 2),
                    "capacitance_uf": round(c_val * 1e6, 2),
                    "critical_inductance_uh": round(l_crit * 1e6, 2)
                },
                "units": "Duty % / μH / μF",
                "assumptions": ["Continuous Conduction Mode (CCM)", "Ideal switches"]
            }
        return {"success": False, "error": "Supported topologies are 'Buck' or 'Boost'"}

    @staticmethod
    def battery_ev_calculations(
        voltage_v: float,
        capacity_ah: float,
        c_rate: float = 1.0,
        wh_per_km: Optional[float] = None
    ) -> Dict[str, Any]:
        """Calculates Battery Energy (kWh), Maximum discharge current (A), Discharge time, and EV Driving Range."""
        energy_wh = voltage_v * capacity_ah
        energy_kwh = energy_wh / 1000.0
        discharge_current_a = capacity_ah * c_rate
        discharge_time_hours = 1.0 / c_rate if c_rate > 0 else 0.0

        est_range_km = None
        if wh_per_km and wh_per_km > 0:
            est_range_km = energy_wh / wh_per_km

        return {
            "success": True,
            "tool_name": "Battery & EV Sizing Calculator",
            "given": {
                "Pack Nominal Voltage": f"{voltage_v} V",
                "Nominal Capacity": f"{capacity_ah} Ah",
                "C-Rate": f"{c_rate}C",
                "Vehicle Consumption": f"{wh_per_km} Wh/km" if wh_per_km else "N/A"
            },
            "required": "Total Energy Capacity (kWh), Discharge Current (A), Discharge Duration, Range (km)",
            "formula": "Energy = V · Ah / 1000, I_dis = C_rate · Ah, t = 1 / C_rate, Range = Energy_Wh / (Wh/km)",
            "substitution": f"Energy = {voltage_v} · {capacity_ah} / 1000 = {round(energy_kwh, 3)} kWh",
            "calculation": f"Energy = {round(energy_kwh, 3)} kWh, Discharge Current = {round(discharge_current_a, 2)} A at {c_rate}C duration {round(discharge_time_hours*60, 1)} min" + (f", Driving Range = {round(est_range_km, 1)} km" if est_range_km else ""),
            "result": {
                "energy_kwh": round(energy_kwh, 3),
                "energy_wh": round(energy_wh, 1),
                "discharge_current_a": round(discharge_current_a, 2),
                "discharge_time_minutes": round(discharge_time_hours * 60.0, 1),
                "estimated_range_km": round(est_range_km, 1) if est_range_km else None
            },
            "units": "kWh / A / km",
            "assumptions": ["100% usable DoD (Depth of Discharge) unless battery reserve factored"]
        }

    @staticmethod
    def solar_pv_calculations(
        voc: float,
        isc: float,
        vmp: float,
        imp: float,
        target_kw: Optional[float] = None
    ) -> Dict[str, Any]:
        """Calculates Solar PV Maximum Power (Pmp), Fill Factor (FF), and Array Sizing."""
        pmp = vmp * imp
        ptheoretical = voc * isc
        fill_factor = pmp / ptheoretical if ptheoretical > 0 else 0.0

        num_panels = None
        if target_kw and target_kw > 0 and pmp > 0:
            num_panels = math.ceil((target_kw * 1000.0) / pmp)

        return {
            "success": True,
            "tool_name": "Solar PV Panel & Array Calculator",
            "given": {
                "Open Circuit Voltage (Voc)": f"{voc} V",
                "Short Circuit Current (Isc)": f"{isc} A",
                "Voltage at MPP (Vmp)": f"{vmp} V",
                "Current at MPP (Imp)": f"{imp} A",
                "Target System Power": f"{target_kw} kW" if target_kw else "N/A"
            },
            "required": "Peak Power (Pmp), Fill Factor (FF), Number of Panels",
            "formula": "Pmp = Vmp · Imp, FF = Pmp / (Voc · Isc), N_panels = ceil(Target_W / Pmp)",
            "substitution": f"Pmp = {vmp} · {imp} = {round(pmp, 2)} W, FF = {round(pmp, 2)} / ({voc} · {isc})",
            "calculation": f"Pmp = {round(pmp, 2)} Wp, Fill Factor = {round(fill_factor * 100, 2)}%" + (f", Total Panels = {num_panels} panels for {target_kw} kW" if num_panels else ""),
            "result": {
                "peak_power_w": round(pmp, 2),
                "fill_factor_pct": round(fill_factor * 100, 2),
                "theoretical_max_w": round(ptheoretical, 2),
                "required_panels_count": num_panels
            },
            "units": "Wp / %",
            "assumptions": ["Standard Test Conditions (STC): Irradiance 1000 W/m², AM 1.5, Cell Temp 25°C"]
        }
