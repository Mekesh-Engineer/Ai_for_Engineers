from flask import Blueprint, request, jsonify, current_app
from app.agents.planner import TaskPlanner
from app.tools.registry import ToolRegistry
from app.llm.ollama_client import OllamaClient
from app.llm.model_selector import ModelSelector

agent_bp = Blueprint("agent", __name__, url_prefix="/api/agent")

@agent_bp.route("/plan", methods=["POST"])
def generate_task_plan():
    """Generates an agentic execution plan for a complex user query."""
    data = request.get_json() or {}
    query = data.get("query", "").strip()
    model = data.get("model", "gemma4:cloud")

    if not query:
        return jsonify({"error": "Query parameter is required"}), 400

    client = OllamaClient(base_url=current_app.config["OLLAMA_BASE_URL"])
    plan = TaskPlanner.generate_plan(user_query=query, ollama_client=client, model_name=model)
    return jsonify(plan)

@agent_bp.route("/tools", methods=["GET"])
def list_agent_tools():
    """Returns all registered agent tools with their OpenAPI/JSON schemas."""
    return jsonify({
        "tools": ToolRegistry.get_all_tools_metadata(),
        "ollama_schemas": ToolRegistry.get_ollama_tools()
    })

@agent_bp.route("/model-matrix", methods=["GET"])
def get_model_matrix():
    """Returns the agentic multi-model tiering matrix."""
    return jsonify(ModelSelector.get_all_models_metadata())
