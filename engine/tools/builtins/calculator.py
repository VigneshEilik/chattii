from typing import Any

from engine.tools.base import BaseTool


class CalculatorTool(BaseTool):
    name = "calculator"
    description = "Evaluates mathematical expressions safely."
    parameters_schema = {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "Math expression e.g. 48392 * 928",
            }
        },
        "required": ["expression"],
    }

    def execute(self, expression: str, **kwargs: Any) -> Any:
        try:
            # Safe restricted eval
            allowed = set("0123456789+-*/(). ")
            if not all(c in allowed for c in expression):
                return "Error: Invalid characters in expression."
            return str(eval(expression, {"__builtins__": {}}, {}))
        except Exception as e:
            return f"Error evaluating expression: {e}"
