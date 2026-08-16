# Chatii Function Calling & Tools Engine

Chatii supports extensible function calling tool execution.

## Builtin Tools
- `calculator`: Evaluates mathematical expressions safely.
- `datetime`: Provides current UTC date, time, and timezone metadata.

## Tool Interface
Inherit from `BaseTool` in `engine/tools/base.py` and register with `ToolExecutor`.
