import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


class BPECheckpointer:
    """
    Saves periodic checkpoints during BPE training iterations.
    """

    @staticmethod
    def save_checkpoint(
        output_dir: str,
        merge_count: int,
        vocab_map: Dict[str, int],
        merges: List[Tuple[str, str]],
        statistics: Dict[str, Any],
    ) -> None:
        """
        Writes current state variables to a checkpoint file.
        """
        path = Path(output_dir) / "checkpoints"
        path.mkdir(parents=True, exist_ok=True)

        checkpoint_data = {
            "merge_count": merge_count,
            "vocab_size": len(vocab_map),
            "vocab": vocab_map,
            "merges": [[p[0], p[1]] for p in merges],
            "statistics": statistics,
        }

        checkpoint_file = path / f"checkpoint_{merge_count}.json"
        try:
            with open(checkpoint_file, "w", encoding="utf-8") as f:
                json.dump(checkpoint_data, f, indent=2)
        except OSError:
            pass
