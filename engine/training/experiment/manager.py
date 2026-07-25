import os
import re


class ExperimentManager:
    """Generates sequential run IDs and manages experiment directories."""

    def __init__(self, root: str = "experiments") -> None:
        self.root = root
        os.makedirs(self.root, exist_ok=True)

    def generate_run_id(self, prefix: str = "experiment") -> str:
        """
        Scans experiment directory and returns a sequential ID (e.g. experiment_001).
        """
        existing = os.listdir(self.root)
        numbers = []
        pattern = re.compile(rf"^{prefix}_(\d+)")

        for name in existing:
            match = pattern.match(name)
            if match:
                numbers.append(int(match.group(1)))

        next_num = max(numbers) + 1 if numbers else 1
        return f"{prefix}_{next_num:03d}"
