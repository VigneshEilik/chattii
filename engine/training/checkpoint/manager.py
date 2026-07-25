import os
import shutil


class CheckpointManager:
    """Manages saving rotations, deletions, and cleanups of training checkpoints."""

    def __init__(self, checkpoint_dir: str, keep_top_n: int = 3) -> None:
        self.checkpoint_dir = checkpoint_dir
        self.keep_top_n = keep_top_n
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def rotate_checkpoints(self) -> None:
        """
        Scans checkpoint directory and removes the oldest 'step_' subdirectories
        to keep only the top N step checkpoints.
        """
        if self.keep_top_n <= 0:
            return

        # Find all valid directories matching step_N format
        step_folders = []
        for name in os.listdir(self.checkpoint_dir):
            path = os.path.join(self.checkpoint_dir, name)
            if os.path.isdir(path) and name.startswith("step_"):
                try:
                    step_val = int(name.split("_")[1])
                    step_folders.append((step_val, path))
                except ValueError:
                    pass

        # If we have more checkpoints than keep_top_n threshold
        if len(step_folders) <= self.keep_top_n:
            return

        # Sort folders by step values ascending (oldest first)
        step_folders.sort(key=lambda x: x[0])

        # Delete oldest folders
        num_deletes = len(step_folders) - self.keep_top_n
        for i in range(num_deletes):
            shutil.rmtree(step_folders[i][1], ignore_errors=True)
