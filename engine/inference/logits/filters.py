from typing import Dict, List

import torch

from engine.inference.logits.processor import LogitsProcessor


class BadWordProcessor(LogitsProcessor):
    """Sets bad word token logits to -inf to prevent them from being generated."""

    def __init__(self, bad_word_ids: List[int]) -> None:
        self.bad_word_ids = bad_word_ids

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        for idx in self.bad_word_ids:
            if 0 <= idx < scores.size(-1):
                scores[:, idx] = float("-inf")
        return scores


class SuppressTokenProcessor(LogitsProcessor):
    """Suppresses specific tokens by setting their logits to -inf."""

    def __init__(self, suppress_ids: List[int]) -> None:
        self.suppress_ids = suppress_ids

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        for idx in self.suppress_ids:
            if 0 <= idx < scores.size(-1):
                scores[:, idx] = float("-inf")
        return scores


class AllowedTokenProcessor(LogitsProcessor):
    """Filters logits to allow only a set of specified tokens."""

    def __init__(self, allowed_ids: List[int]) -> None:
        self.allowed_ids = set(allowed_ids)

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        mask = torch.ones_like(scores, dtype=torch.bool)
        for idx in self.allowed_ids:
            if 0 <= idx < scores.size(-1):
                mask[:, idx] = False
        scores[mask] = float("-inf")
        return scores


class ForcedTokenProcessor(LogitsProcessor):
    """Forces the generation of a specific token at a specific step."""

    def __init__(self, forced_token_map: Dict[int, int]) -> None:
        self.forced_token_map = forced_token_map

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        step = input_ids.size(1)
        if step in self.forced_token_map:
            forced_id = self.forced_token_map[step]
            scores.fill_(float("-inf"))
            scores[:, forced_id] = 0.0
        return scores
