import json
from pathlib import Path
from typing import Dict

from engine.tokenizer.algorithms.bpe.visualizer import BPEVisualizer
from engine.tokenizer.serialization.io import VocabularySerializer


class TokenizerVisualizer:
    """
    Renders ASCII charts of token properties (e.g. length distributions, merge ranks).
    """

    @staticmethod
    def render_length_distribution(vocab_dir: str) -> str:
        """
        Generates an ASCII bar chart depicting token lengths in characters.
        """
        data = VocabularySerializer.load(vocab_dir)
        vocab = data.get("vocab", {})

        lengths: Dict[int, int] = {}
        for token in vocab.keys():
            length = len(token)
            lengths[length] = lengths.get(length, 0) + 1

        if not lengths:
            return "No tokens found to compute distribution."

        lines = ["\n--- Token Length Distribution ---"]
        max_count = max(lengths.values())
        max_chars = 30  # maximum visual bar width

        for length in sorted(lengths.keys()):
            count = lengths[length]
            bar_size = int(count * max_chars / max_count) if max_count > 0 else 0
            bar = "*" * bar_size
            lines.append(f"  {length:2d} chars | {bar:<{max_chars}} ({count})")

        return "\n".join(lines)

    @staticmethod
    def render_merge_history(vocab_dir: str) -> str:
        """
        Loads BPE merge records and prints a progress line chart.
        """
        history_path = Path(vocab_dir) / "merge_history.json"
        if not history_path.exists():
            return "No BPE merge history found."

        try:
            with open(history_path, "r", encoding="utf-8") as f:
                history = json.load(f)
        except Exception as e:
            return f"Failed to load merge history: {e}"

        if not history:
            return "Merge history is empty."

        ranks = [item["rank"] for item in history]
        frequencies = [float(item["frequency"]) for item in history]

        return BPEVisualizer.render_chart(
            "BPE Merge Frequencies over Iterations",
            ranks,
            frequencies,
            height=8,
            width=40,
        )
