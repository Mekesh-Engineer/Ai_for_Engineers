from unittest.mock import MagicMock
from app.agents.planner import TaskPlanner

def test_task_planner_json_generation():
    mock_client = MagicMock()
    mock_client.chat_completion.return_value = {
        "message": {
            "content": '{"requires_multi_step": true, "objective": "Calculate motor current and torque", "steps": [{"step": 1, "description": "Calc current", "tool_hint": "three_phase_power"}]}'
        }
    }

    plan = TaskPlanner.generate_plan(
        user_query="Calculate current for 15kW motor and find torque",
        ollama_client=mock_client
    )

    assert plan["requires_multi_step"] is True
    assert len(plan["steps"]) == 1
    assert plan["steps"][0]["tool_hint"] == "three_phase_power"

def test_task_planner_fallback_on_invalid_output():
    mock_client = MagicMock()
    mock_client.chat_completion.return_value = {
        "message": {"content": "I cannot formulate this in JSON format."}
    }

    plan = TaskPlanner.generate_plan(
        user_query="Explain Thevenin theorem",
        ollama_client=mock_client
    )

    assert plan["requires_multi_step"] is False
    assert plan["objective"] == "Explain Thevenin theorem"
