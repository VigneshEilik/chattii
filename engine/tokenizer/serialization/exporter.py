from pathlib import Path
from typing import Dict


class VocabularyExporter:
    """
    Exports vocabulary structures into alternative model configurations.
    """

    @staticmethod
    def export_to_txt(vocab_map: Dict[str, int], output_file: str) -> None:
        """
        Exports a flat line-by-line token list ordered by token ID.
        """
        path = Path(output_file)
        path.parent.mkdir(parents=True, exist_ok=True)

        sorted_tokens = sorted(vocab_map.keys(), key=lambda tok: vocab_map[tok])

        with open(path, "w", encoding="utf-8") as f:
            for token in sorted_tokens:
                f.write(f"{token}\n")
