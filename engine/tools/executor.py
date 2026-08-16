from typing import Any, Dict, List, Optional

from engine.tools.base import BaseTool
from engine.tools.builtins.calculator import CalculatorTool
from engine.tools.builtins.datetime_tool import DateTimeTool


class ToolExecutor:
    """Registry and execution engine for function calling tools."""

    def __init__(self) -> None:
        self._tools: Dict[str, BaseTool] = {}
        # Register default tools
        self.register_tool(CalculatorTool())
        self.register_tool(DateTimeTool())

    def register_tool(self, tool: BaseTool) -> None:
        self._tools[tool.name] = tool

    def get_tool(self, name: str) -> Optional[BaseTool]:
        return self._tools.get(name)

    def list_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters_schema,
            }
            for t in self._tools.values()
        ]

    def execute_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        tool = self.get_tool(name)
        if not tool:
            raise ValueError(f"Tool '{name}' is not registered.")
        return tool.execute(**arguments)


_tool_executor_instance: ToolExecutor = ToolExecutor()


def get_tool_executor() -> ToolExecutor:
    return _tool_executor_instance
