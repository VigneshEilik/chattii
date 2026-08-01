from typing import Any, Optional

import torch
import torch.nn as nn


class SingleTokenGenerator:
    """Manages model evaluation passes and extracts logit distributions for the final sequence token."""

    @staticmethod
    def step(
        model: nn.Module,
        input_ids: torch.Tensor,
        cache: Optional[Any] = None,
        use_cache: bool = True,
    ) -> torch.Tensor:
        # Run forward pass through transformer stack
        logits = model(input_ids, cache=cache, use_cache=use_cache)
        # Extract last token logits: shape (batch_size, vocab_size)
        return logits[:, -1, :]
