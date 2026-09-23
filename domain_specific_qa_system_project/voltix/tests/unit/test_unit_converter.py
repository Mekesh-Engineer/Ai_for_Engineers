import pytest
from app.tools.unit_converter import UnitConverter

def test_unit_conversion_power():
    res = UnitConverter.convert(5, "kW", "W")
    assert res["success"] is True
    assert res["result_value"] == 5000.0

def test_unit_conversion_frequency():
    res = UnitConverter.convert(1500, "rpm", "rad/s")
    assert res["success"] is True
    assert abs(res["result_value"] - 157.079633) < 0.01

def test_unit_conversion_current():
    res = UnitConverter.convert(2.5, "A", "mA")
    assert res["success"] is True
    assert res["result_value"] == 2500.0

def test_incompatible_units():
    res = UnitConverter.convert(10, "kW", "A")
    assert res["success"] is False
    assert "Incompatible" in res["error"]
