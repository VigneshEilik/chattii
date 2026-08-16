from datetime import datetime, timezone
from typing import Any

from engine.tools.base import BaseTool


class DateTimeTool(BaseTool):
    name = "datetime"
    description = "Returns current date, time, and timezone information."
    parameters_schema = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    def execute(self, **kwargs: Any) -> Any:
        now = datetime.now(timezone.utc)
        return {
            "utc_now": now.isoformat(),
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H:%M:%S UTC"),
            "day": now.strftime("%A"),
        }
