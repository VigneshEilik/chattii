import os
from typing import Optional


class DistributedRecoveryManager:
    """Orchestrates reloading model checkpoints and matching target states upon recovery."""

    @staticmethod
    def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
        if not os.path.exists(checkpoint_dir):
            return None
        checkpoints = [
            os.path.join(checkpoint_dir, d)
            for d in os.listdir(checkpoint_dir)
            if os.path.isdir(os.path.join(checkpoint_dir, d))
        ]
        if not checkpoints:
            return None
        # Return most recently modified directory
        return max(checkpoints, key=os.path.getmtime)
