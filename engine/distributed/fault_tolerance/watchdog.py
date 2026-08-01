from typing import Callable, Optional

from engine.distributed.fault_tolerance.heartbeat import DistributedHeartbeat


class DistributedWatchdog:
    """Monitors worker status and triggers actions on disconnected or hung training nodes."""

    def __init__(
        self, world_size: int, on_failure: Optional[Callable[[int], None]] = None
    ) -> None:
        self.world_size = world_size
        self.heartbeat = DistributedHeartbeat()
        self.on_failure = on_failure

    def step(self, rank: int) -> None:
        self.heartbeat.record_heartbeat(rank)

    def monitor(self, timeout_seconds: float = 30.0) -> None:
        for r in range(self.world_size):
            if not self.heartbeat.check_health(r, timeout_seconds):
                print(f"Watchdog Alert: Node rank {r} disconnected or failed.")
                if self.on_failure:
                    self.on_failure(r)
