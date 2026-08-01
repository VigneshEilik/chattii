from typing import List

import torch


class LogitsProcessor:
    """Base interface for modifying model output token logit distributions."""

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class LogitsProcessorList(List[LogitsProcessor]):
    """Orchestrates sequential execution of registered logit processors."""

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        for processor in self:
            scores = processor(input_ids, scores)
        return scores
