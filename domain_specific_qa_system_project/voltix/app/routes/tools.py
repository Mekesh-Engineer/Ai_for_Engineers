from flask import Blueprint, request, jsonify
from app.tools import EECalculator, SymPySolver, UnitConverter, MatlabScriptGenerator

tools_bp = Blueprint("tools", __name__, url_prefix="/api/tools")

@tools_bp.route("/three-phase-power", methods=["POST"])
def three_phase_power():
    data = request.get_json() or {}
    p = float(data["p"]) if "p" in data and data["p"] is not None else None
    vl = float(data["vl"]) if "vl" in data and data["vl"] is not None else None
    il = float(data["il"]) if "il" in data and data["il"] is not None else None
    pf = float(data["pf"]) if "pf" in data and data["pf"] is not None else None
    res = EECalculator.three_phase_power(p=p, vl=vl, il=il, pf=pf)
    return jsonify(res)

@tools_bp.route("/single-phase-power", methods=["POST"])
def single_phase_power():
    data = request.get_json() or {}
    v = float(data.get("v", 230))
    i = float(data.get("i", 10))
    pf = float(data.get("pf", 0.8))
    res = EECalculator.single_phase_power(v=v, i=i, pf=pf)
    return jsonify(res)

@tools_bp.route("/ohms-law", methods=["POST"])
def ohms_law():
    data = request.get_json() or {}
    v = float(data["v"]) if "v" in data and data["v"] is not None else None
    i = float(data["i"]) if "i" in data and data["i"] is not None else None
    r = float(data["r"]) if "r" in data and data["r"] is not None else None
    res = EECalculator.ohms_law(v=v, i=i, r=r)
    return jsonify(res)

@tools_bp.route("/slip", methods=["POST"])
def slip():
    data = request.get_json() or {}
    ns = data.get("ns")
    n = data.get("n")
    if ns is None or n is None:
        return jsonify({"success": False, "error": "Parameters ns and n are required"}), 400
    res = EECalculator.induction_motor_slip(ns=float(ns), n=float(n))
    return jsonify(res)

@tools_bp.route("/synchronous-speed", methods=["POST"])
def synchronous_speed():
    data = request.get_json() or {}
    f = float(data.get("f", 50))
    p = int(data.get("p", 4))
    res = EECalculator.synchronous_speed(f=f, p=p)
    return jsonify(res)

@tools_bp.route("/motor-torque", methods=["POST"])
def motor_torque():
    data = request.get_json() or {}
    power = float(data.get("power_w", 1000))
    speed = float(data.get("speed_rpm", 1500))
    res = EECalculator.motor_torque(power_w=power, speed_rpm=speed)
    return jsonify(res)

@tools_bp.route("/transformer", methods=["POST"])
def transformer():
    data = request.get_json() or {}
    rating = float(data.get("rating_kva", 10))
    pf = float(data.get("pf", 0.8))
    pi = float(data.get("p_iron_w", 100))
    pcu = float(data.get("p_cu_fl_w", 200))
    x = float(data.get("fraction_load", 1.0))
    v_nl = float(data["v_nl"]) if "v_nl" in data and data["v_nl"] is not None else None
    v_fl = float(data["v_fl"]) if "v_fl" in data and data["v_fl"] is not None else None
    res = EECalculator.transformer_efficiency_regulation(
        rating_kva=rating, pf=pf, p_iron_w=pi, p_cu_fl_w=pcu, fraction_load=x, v_nl=v_nl, v_fl=v_fl
    )
    return jsonify(res)

@tools_bp.route("/power-factor-correction", methods=["POST"])
def power_factor_correction():
    data = request.get_json() or {}
    p_kw = float(data.get("active_power_kw", 50))
    pf1 = float(data.get("initial_pf", 0.7))
    pf2 = float(data.get("target_pf", 0.95))
    v = float(data.get("voltage_v", 415))
    f = float(data.get("freq_hz", 50))
    res = EECalculator.power_factor_correction(active_power_kw=p_kw, initial_pf=pf1, target_pf=pf2, voltage_v=v, freq_hz=f)
    return jsonify(res)

@tools_bp.route("/dc-dc-converter", methods=["POST"])
def dc_dc_converter():
    data = request.get_json() or {}
    top = data.get("topology", "Buck")
    vin = float(data.get("vin", 24))
    vout = float(data.get("vout", 5))
    iout = float(data.get("iout", 2))
    freq = float(data.get("freq_hz", 50000))
    res = EECalculator.dc_dc_converter_buck_boost(topology=top, vin=vin, vout=vout, iout=iout, freq_hz=freq)
    return jsonify(res)

@tools_bp.route("/battery-ev", methods=["POST"])
def battery_ev():
    data = request.get_json() or {}
    v = float(data.get("voltage_v", 400))
    ah = float(data.get("capacity_ah", 100))
    c_rate = float(data.get("c_rate", 1.0))
    wh_per_km = float(data["wh_per_km"]) if "wh_per_km" in data and data["wh_per_km"] is not None else None
    res = EECalculator.battery_ev_calculations(voltage_v=v, capacity_ah=ah, c_rate=c_rate, wh_per_km=wh_per_km)
    return jsonify(res)

@tools_bp.route("/solar-pv", methods=["POST"])
def solar_pv():
    data = request.get_json() or {}
    voc = float(data.get("voc", 45))
    isc = float(data.get("isc", 10))
    vmp = float(data.get("vmp", 36))
    imp = float(data.get("imp", 9.2))
    target_kw = float(data["target_kw"]) if "target_kw" in data and data["target_kw"] is not None else None
    res = EECalculator.solar_pv_calculations(voc=voc, isc=isc, vmp=vmp, imp=imp, target_kw=target_kw)
    return jsonify(res)

# --- SymPy Symbolic Endpoints ---

@tools_bp.route("/sympy/solve", methods=["POST"])
def sympy_solve():
    data = request.get_json() or {}
    expr = data.get("expression", "")
    var = data.get("variable", "x")
    res = SymPySolver.solve_equation(expr, var)
    return jsonify(res)

@tools_bp.route("/sympy/diff", methods=["POST"])
def sympy_diff():
    data = request.get_json() or {}
    expr = data.get("expression", "")
    var = data.get("variable", "t")
    order = int(data.get("order", 1))
    res = SymPySolver.differentiate(expr, var, order)
    return jsonify(res)

@tools_bp.route("/sympy/integrate", methods=["POST"])
def sympy_integrate():
    data = request.get_json() or {}
    expr = data.get("expression", "")
    var = data.get("variable", "t")
    lower = data.get("lower")
    upper = data.get("upper")
    res = SymPySolver.integrate(expr, var, lower, upper)
    return jsonify(res)

@tools_bp.route("/sympy/laplace", methods=["POST"])
def sympy_laplace():
    data = request.get_json() or {}
    expr = data.get("expression", "")
    res = SymPySolver.laplace_transform(expr)
    return jsonify(res)

@tools_bp.route("/sympy/inverse-laplace", methods=["POST"])
def sympy_inverse_laplace():
    data = request.get_json() or {}
    expr = data.get("expression", "")
    res = SymPySolver.inverse_laplace_transform(expr)
    return jsonify(res)

@tools_bp.route("/sympy/transfer-function", methods=["POST"])
def sympy_transfer_function():
    data = request.get_json() or {}
    num = data.get("num", "1")
    den = data.get("den", "s**2 + 2*s + 1")
    res = SymPySolver.transfer_function_analysis(num, den)
    return jsonify(res)

@tools_bp.route("/sympy/rlc", methods=["POST"])
def sympy_rlc():
    data = request.get_json() or {}
    r = float(data.get("r", 10))
    l = float(data.get("l", 0.1))
    c = float(data.get("c", 0.001))
    res = SymPySolver.rlc_transient_response(r, l, c)
    return jsonify(res)

# --- Unit Converter & MATLAB Endpoints ---

@tools_bp.route("/unit-convert", methods=["POST"])
def convert_unit():
    data = request.get_json() or {}
    val = data.get("value")
    from_u = data.get("from_unit")
    to_u = data.get("to_unit")
    if val is None or not from_u or not to_u:
        return jsonify({"success": False, "error": "Parameters value, from_unit, and to_unit are required"}), 400
    res = UnitConverter.convert(float(val), from_u, to_u)
    return jsonify(res)

@tools_bp.route("/matlab/bode", methods=["POST"])
def generate_bode():
    data = request.get_json() or {}
    num = data.get("num", [1])
    den = data.get("den", [1, 1])
    title = data.get("title", "Bode Plot Analysis")
    script = MatlabScriptGenerator.generate_bode_plot_script(num, den, title)
    return jsonify({"success": True, "script": script})

@tools_bp.route("/matlab/torque-speed", methods=["POST"])
def generate_torque_speed():
    data = request.get_json() or {}
    v = float(data.get("v_phase", 230))
    f = float(data.get("f", 50))
    p = int(data.get("poles", 4))
    script = MatlabScriptGenerator.generate_induction_motor_torque_speed_script(v_phase=v, f=f, poles=p)
    return jsonify({"success": True, "script": script})

@tools_bp.route("/matlab/simulink-blueprint", methods=["POST"])
def generate_simulink():
    data = request.get_json() or {}
    topic = data.get("topic", "Buck Converter DC-DC Simulation")
    blueprint = MatlabScriptGenerator.generate_simulink_model_blueprint(topic)
    return jsonify({"success": True, "blueprint": blueprint})
