import pytest

from engine.tools.builtins.calculator import CalculatorTool
from engine.tools.builtins.datetime_tool import DateTimeTool
from engine.tools.executor import ToolExecutor


def test_calculator_valid():
    calc = CalculatorTool()
    assert calc.execute(expression="48392 * 928") == "44907776"


def test_calculator_invalid_character():
    calc = CalculatorTool()
    res = calc.execute(expression="import os; os.system('echo hi')")
    assert "Error: Invalid characters" in res


def test_calculator_syntax_error():
    calc = CalculatorTool()
    res = calc.execute(expression="2 + * 3")
    assert "Error evaluating" in res


def test_datetime_tool():
    dt = DateTimeTool()
    res = dt.execute()
    assert "utc_now" in res
    assert "date" in res
    assert "time" in res


def test_tool_executor_builtin_registration():
    executor = ToolExecutor()
    tools = executor.list_tools()
    tool_names = {t["name"] for t in tools}
    assert "calculator" in tool_names
    assert "datetime" in tool_names


def test_tool_executor_execute_calculator():
    executor = ToolExecutor()
    res = executor.execute_tool("calculator", {"expression": "10 + 20"})
    assert res == "30"


def test_tool_executor_execute_datetime():
    executor = ToolExecutor()
    res = executor.execute_tool("datetime", {})
    assert "date" in res


def test_tool_executor_unregistered_tool():
    executor = ToolExecutor()
    with pytest.raises(ValueError):
        executor.execute_tool("nonexistent_tool", {})


def test_calculator_tool_schema():
    calc = CalculatorTool()
    assert calc.name == "calculator"
    assert "expression" in calc.parameters_schema["properties"]


def test_datetime_tool_schema():
    dt = DateTimeTool()
    assert dt.name == "datetime"
    assert dt.parameters_schema["type"] == "object"
