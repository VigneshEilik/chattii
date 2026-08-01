from typing import Any, Dict


class CommunicationProfiler:
    """Tracks latency overheads of distributed all-reduce operations."""

    def __init__(self) -> None:
        self.total_comm_time = 0.0

    def record_operation(self, duration: float) -> None:
        self.total_comm_time += duration

    def get_summary(self) -> Dict[str, Any]:
        return {
            "total_communication_seconds": self.total_comm_time,
        }
