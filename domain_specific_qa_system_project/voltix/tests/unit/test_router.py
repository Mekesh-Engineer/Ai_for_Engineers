import pytest
from app.services.router import QueryRouter

def test_router_intent_classification():
    res1 = QueryRouter.classify_intent("Calculate slip of 3-phase induction motor running at 1440 RPM")
    assert res1["intent"] == "NUMERICAL"
    assert res1["agent"] == "MachinesAgent"
    assert res1["use_tool"] is True

    res2 = QueryRouter.classify_intent("Generate MATLAB code to plot Bode diagram")
    assert res2["intent"] == "MATLAB"
    assert res2["agent"] == "MatlabSimulinkAgent"

    res3 = QueryRouter.classify_intent("Explain how a Boost converter works in continuous conduction mode")
    assert res3["intent"] == "POWER_ELECTRONICS"
    assert res3["agent"] == "PowerElectronicsAgent"

    res4 = QueryRouter.classify_intent("Summarize this document")
    assert res4["intent"] == "DOCUMENT_SUMMARY"
