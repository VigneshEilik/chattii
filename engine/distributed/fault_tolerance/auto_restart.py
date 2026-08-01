import time
from typing import Any, Callable


class DistributedAutoRestart:
    """Manages automatic job respawning and elastic state recoveries on node failures."""

    def __init__(self, max_restarts: int = 3) -> None:
        self.max_restarts = max_restarts
        self.restart_count = 0

    def trigger_restart(self, restart_fn: Callable[[], Any]) -> bool:
        if self.restart_count < self.max_restarts:
            self.restart_count += 1
            print(
                f"Auto-Restart Triggered ({self.restart_count}/{self.max_restarts}). Recovery in progress..."
            )
            time.sleep(2)  # Cooldown before restarting
            restart_fn()
            return True
        print("Max process restarts exceeded. Halting training.")
        return False
