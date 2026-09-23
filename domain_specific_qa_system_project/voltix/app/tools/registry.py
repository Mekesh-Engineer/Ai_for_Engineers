import inspect
from typing import Callable, Dict, Any, List, Optional
from app.utils.logger import get_logger

logger = get_logger("voltix.tools.registry")

class ToolRegistry:
    """Central registry for Agentic Tool Calling with automatic JSON schema generation."""
    _tools: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def register(cls, name: str, description: str):
        """Decorator to register a function as an agent-callable tool."""
        def decorator(func: Callable):
            sig = inspect.signature(func)
            parameters: Dict[str, Any] = {
                "type": "object",
                "properties": {},
                "required": []
            }
            
            for param_name, param in sig.parameters.items():
                if param_name in ("self", "cls"):
                    continue
                param_type = "string"
                if param.annotation in (int, float):
                    param_type = "number"
                elif param.annotation == bool:
                    param_type = "boolean"
                elif param.annotation == dict:
                    param_type = "object"
                elif param.annotation == list:
                    param_type = "array"

                parameters["properties"][param_name] = {
                    "type": param_type,
                    "description": f"Parameter {param_name}"
                }
                if param.default == inspect.Parameter.empty:
                    parameters["required"].append(param_name)

            cls._tools[name] = {
                "name": name,
                "description": description,
                "func": func,
                "parameters": parameters,
                "schema": {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": description,
                        "parameters": parameters
                    }
                }
            }
            return func
        return decorator

    @classmethod
    def get_ollama_tools(cls) -> List[Dict[str, Any]]:
        """Returns the list of tool schemas for Ollama's `tools` parameter."""
        return [tool["schema"] for tool in cls._tools.values()]

    @classmethod
    def get_all_tools_metadata(cls) -> Dict[str, Any]:
        """Returns descriptive dictionary of all registered tools."""
        return {
            name: {
                "name": info["name"],
                "description": info["description"],
                "parameters": info["parameters"]
            }
            for name, info in cls._tools.items()
        }

    @classmethod
    def execute(cls, name: str, kwargs: Optional[Dict[str, Any]] = None) -> Any:
        """Executes a registered tool safely with kwargs."""
        if name not in cls._tools:
            return {"error": f"Tool '{name}' is not registered."}
        try:
            kwargs = kwargs or {}
            logger.info(f"Executing tool '{name}' with args: {kwargs}")
            return cls._tools[name]["func"](**kwargs)
        except Exception as e:
            logger.error(f"Execution error in tool '{name}': {e}", exc_info=True)
            return {"error": f"Execution error in tool '{name}': {str(e)}"}
