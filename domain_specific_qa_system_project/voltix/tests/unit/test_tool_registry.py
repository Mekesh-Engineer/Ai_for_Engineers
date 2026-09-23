from app.tools.registry import ToolRegistry
import app.tools # ensures tools are registered

def test_tool_registry_registration_and_execution():
    @ToolRegistry.register(
        name="test_adder_tool",
        description="Adds two floating point numbers."
    )
    def adder(a: float, b: float) -> dict:
        return {"sum": a + b}

    result = ToolRegistry.execute("test_adder_tool", {"a": 5.5, "b": 4.5})
    assert result == {"sum": 10.0}

def test_tool_registry_three_phase():
    result = ToolRegistry.execute("three_phase_power", {
        "power_val": 15.0,
        "voltage": 415.0,
        "power_factor": 0.85
    })
    assert result.get("success") is True
    assert "result" in result
    assert result["result"]["line_current_a"] == 24.5507

def test_tool_registry_schemas():
    schemas = ToolRegistry.get_ollama_tools()
    assert isinstance(schemas, list)
    assert len(schemas) > 5
    tool_names = [s["function"]["name"] for s in schemas]
    assert "three_phase_power" in tool_names
    assert "sympy_solve_equation" in tool_names
    assert "fetch_local_web_page" in tool_names
