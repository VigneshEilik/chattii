import os
from typing import Dict

from engine.tokenizer.vocabulary.exceptions import SerializationError


class VocabularyImporter:
    """
    Imports vocabulary maps from alternative standard formats.
    """

    @staticmethod
    def import_from_txt(txt_file: str) -> Dict[str, int]:
        """
        Imports a plain line-by-line token list (where line number matches ID).
        """
        if not os.path.exists(txt_file):
            raise FileNotFoundError(f"Importer source file not found: {txt_file}")

        vocab_map: Dict[str, int] = {}
        with open(txt_file, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                token = line.rstrip("\n")
                if token in vocab_map:
                    raise SerializationError(
                        f"Duplicate token '{token}' detected during "
                        f"text import at line {idx + 1}."
                    )
                vocab_map[token] = idx
        return vocab_map
