from typing import Any, Callable

import torch
import torch.utils.checkpoint


class GradientCheckpointing:
    """Applies activation checkpointing to optimize memory consumption."""

    @staticmethod
    def checkpoint(
        func: Callable[..., Any], *args: Any, use_reentrant: bool = False, **kwargs: Any
    ) -> Any:
        """
        Executes functions with checkpointing to save GPU activation memory.
        """
        return torch.utils.checkpoint.checkpoint(
            func, *args, use_reentrant=use_reentrant, **kwargs
        )
