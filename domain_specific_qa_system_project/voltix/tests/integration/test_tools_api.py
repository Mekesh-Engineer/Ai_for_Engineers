import pytest

def test_tools_api_3phase(client):
    res = client.post("/api/tools/three-phase-power", json={"p": 5000, "vl": 415, "pf": 0.8})
    assert res.status_code == 200
    data = res.get_json()
    assert data["success"] is True
    assert "line_current_a" in data["result"]
    assert abs(data["result"]["line_current_a"] - 8.696) < 0.01

def test_tools_api_slip(client):
    res = client.post("/api/tools/slip", json={"ns": 1500, "n": 1440})
    assert res.status_code == 200
    data = res.get_json()
    assert data["success"] is True
    assert data["result"]["slip_percent"] == 4.0

def test_tools_api_sympy(client):
    res = client.post("/api/tools/sympy/solve", json={"expression": "s**2 + 5*s + 6 = 0", "variable": "s"})
    assert res.status_code == 200
    data = res.get_json()
    assert data["success"] is True
    assert "-2" in data["solutions"]
    assert "-3" in data["solutions"]

def test_tools_api_unit(client):
    res = client.post("/api/tools/unit-convert", json={"value": 10, "from_unit": "kW", "to_unit": "W"})
    assert res.status_code == 200
    data = res.get_json()
    assert data["success"] is True
    assert data["result_value"] == 10000.0
