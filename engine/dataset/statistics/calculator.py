from collections import Counter
from typing import Any, Dict, List

from engine.dataset.base.dataset import TextSample, TokenizedSample


class DatasetStatisticsCalculator:
    """Aggregates and computes statistical metrics over text and tokenized samples."""

    def __init__(self) -> None:
        self.sample_count = 0
        self.token_count = 0
        self.char_lengths: List[int] = []
        self.token_lengths: List[int] = []
        self.languages: Counter = Counter()
        self.char_freq: Counter = Counter()
        self.token_freq: Counter = Counter()
        self.eos_count = 0
        self.pad_count = 0
        self.special_token_count = 0

    def add_sample(
        self,
        text_sample: TextSample,
        tokenized_sample: TokenizedSample,
        pad_token_id: int = 0,
        eos_token_id: int = 2,
    ) -> None:
        self.sample_count += 1
        self.char_lengths.append(len(text_sample.text))
        self.token_lengths.append(len(tokenized_sample.input_ids))

        for char in text_sample.text:
            self.char_freq[char] += 1

        for token in tokenized_sample.input_ids:
            self.token_freq[token] += 1
            if token == eos_token_id:
                self.eos_count += 1
            elif token == pad_token_id:
                self.pad_count += 1

        self.token_count += len(tokenized_sample.input_ids)
        lang = text_sample.metadata.get("language") or "en"
        self.languages[lang] += 1

    def calculate(self, vocab_size: int = 32000) -> Dict[str, Any]:
        if self.sample_count == 0:
            return {}

        avg_char = sum(self.char_lengths) / self.sample_count
        avg_token = sum(self.token_lengths) / self.sample_count

        unique_tokens = len(self.token_freq)
        coverage = (unique_tokens / max(1, vocab_size)) * 100
        oov_tokens = self.token_freq.get(
            0, 0
        )  # Standard BPE unknown fallback token is 0 or unk
        oov_ratio = (oov_tokens / max(1, self.token_count)) * 100
        padding_pct = (self.pad_count / max(1, self.token_count)) * 100

        return {
            "sample_count": self.sample_count,
            "token_count": self.token_count,
            "char_stats": {
                "average": avg_char,
                "max": max(self.char_lengths),
                "min": min(self.char_lengths),
            },
            "token_stats": {
                "average": avg_token,
                "max": max(self.token_lengths),
                "min": min(self.token_lengths),
            },
            "vocabulary": {
                "unique_tokens_seen": unique_tokens,
                "coverage_pct": coverage,
                "oov_ratio_pct": oov_ratio,
            },
            "special_tokens": {
                "eos_frequency": self.eos_count,
                "pad_frequency": self.pad_count,
            },
            "padding_pct": padding_pct,
            "languages": dict(self.languages),
            "top_characters": dict(self.char_freq.most_common(10)),
            "top_tokens": dict(self.token_freq.most_common(10)),
        }
