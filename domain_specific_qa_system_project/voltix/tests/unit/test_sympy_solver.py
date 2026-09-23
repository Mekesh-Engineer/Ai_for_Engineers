import pytest
from app.tools.sympy_solver import SymPySolver

def test_solve_equation():
    res = SymPySolver.solve_equation("s**2 + 3*s + 2 = 0", "s")
    assert res["success"] is True
    assert "-1" in res["solutions"]
    assert "-2" in res["solutions"]

def test_differentiate():
    res = SymPySolver.differentiate("t**3 + 2*t", "t", order=1)
    assert res["success"] is True
    assert "3*t**2 + 2" in res["derivative"]

def test_integrate():
    res = SymPySolver.integrate("3*t**2", "t")
    assert res["success"] is True
    assert "t**3" in res["result"]

def test_laplace_transform():
    # L{exp(-2*t)} = 1 / (s + 2)
    res = SymPySolver.laplace_transform("exp(-2*t)")
    assert res["success"] is True
    assert "1/(s + 2)" in res["frequency_domain"]

def test_transfer_function():
    # G(s) = 1 / (s^2 + 3s + 2) => Poles at -1, -2 (BIBO Stable)
    res = SymPySolver.transfer_function_analysis("1", "s**2 + 3*s + 2")
    assert res["success"] is True
    assert res["is_stable"] is True

def test_rlc_transient():
    # R = 100, L = 0.1, C = 0.0001 => alpha = 500, w0 = 316.22 => Overdamped
    res = SymPySolver.rlc_transient_response(r=100, l=0.1, c=0.0001)
    assert res["success"] is True
    assert "Overdamped" in res["response_type"]
