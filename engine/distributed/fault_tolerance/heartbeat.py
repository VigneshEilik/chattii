import time
from typing import Dict


class DistributedHeartbeat:
    """Manages active process heartbeats to detect disconnected training nodes."""

    def __init__(self) -> None:
        self.last_heartbeats: Dict[int, float] = {}

    def record_heartbeat(self, rank: int) -> None:
        self.last_heartbeats[rank] = time.time()

    def check_health(self, rank: int, timeout_seconds: float = 30.0) -> bool:
        if rank not in self.last_heartbeats:
            return False
        return (time.time() - self.last_heartbeats[rank]) < timeout_seconds
