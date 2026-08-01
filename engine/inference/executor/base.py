from typing import Any

import torch


class BaseExecutor:
    """Base interface for model evaluation executors."""

    def __init__(self, model: Any) -> None:
        self.model = model

    def execute(self, input_ids: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        raise NotImplementedError
