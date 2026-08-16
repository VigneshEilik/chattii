from engine.tools.executor import get_tool_executor


def test_calculator_multiplication():
    executor = get_tool_executor()
    res = executor.execute_tool("calculator", {"expression": "125 * 48"})
    assert res == "6000"


def test_datetime_tool_execution():
    executor = get_tool_executor()
    res = executor.execute_tool("datetime", {})
    assert "utc_now" in res
    assert "date" in res


def test_tool_registration():
    executor = get_tool_executor()
    tools = executor.list_tools()
    names = {t["name"] for t in tools}
    assert "calculator" in names
    assert "datetime" in names
