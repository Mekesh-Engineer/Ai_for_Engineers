import pytest
from app.tools.ee_calculator import EECalculator

def test_ohms_law():
    res1 = EECalculator.ohms_law(v=None, i=2.0, r=10.0)
    assert res1["success"] is True
    assert res1["result"]["voltage_v"] == 20.0
    assert res1["result"]["power_w"] == 40.0

    res2 = EECalculator.ohms_law(v=230.0, i=None, r=23.0)
    assert res2["success"] is True
    assert res2["result"]["current_a"] == 10.0

def test_three_phase_power_and_current():
    # 5 kW, 415 V, 3-phase, 0.8 pf => I_L = 5000 / (sqrt(3) * 415 * 0.8) ~= 8.696 A
    res = EECalculator.three_phase_power(p=5000, vl=415, il=None, pf=0.8)
    assert res["success"] is True
    assert abs(res["result"]["line_current_a"] - 8.696) < 0.01

def test_induction_motor_slip():
    # Ns = 1500, N = 1440 => s = 60 / 1500 = 0.04 (4%)
    res = EECalculator.induction_motor_slip(ns=1500, n=1440)
    assert res["success"] is True
    assert res["result"]["slip_pu"] == 0.04
    assert res["result"]["slip_percent"] == 4.0

def test_synchronous_speed():
    res = EECalculator.synchronous_speed(f=50, p=4)
    assert res["success"] is True
    assert res["result"]["synchronous_speed_rpm"] == 1500.0

def test_motor_torque():
    # P = 7500 W, N = 1440 RPM => T = (60 * 7500) / (2 * pi * 1440) ~= 49.736 N*m
    res = EECalculator.motor_torque(power_w=7500, speed_rpm=1440)
    assert res["success"] is True
    assert abs(res["result"]["torque_nm"] - 49.736) < 0.05

def test_buck_converter():
    res = EECalculator.dc_dc_converter_buck_boost(topology="Buck", vin=24, vout=5, iout=2, freq_hz=50000)
    assert res["success"] is True
    assert abs(res["result"]["duty_cycle"] - (5/24)) < 0.001

def test_battery_ev():
    res = EECalculator.battery_ev_calculations(voltage_v=400, capacity_ah=100, c_rate=1.5, wh_per_km=150)
    assert res["success"] is True
    assert res["result"]["energy_kwh"] == 40.0
    assert res["result"]["discharge_current_a"] == 150.0
    assert abs(res["result"]["estimated_range_km"] - (40000/150)) < 0.1
