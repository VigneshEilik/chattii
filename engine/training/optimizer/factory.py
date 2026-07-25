from typing import Any

import torch.nn as nn
from torch.optim import Optimizer

from engine.training.optimizer.filter import WeightDecayFilter
from engine.training.optimizer.registry import OptimizerRegistry


class OptimizerFactory:
    """Factory for dynamically creating and configuring optimizer instances."""

    @staticmethod
    def create(
        name: str, model: nn.Module, lr: float, weight_decay: float = 0.0, **kwargs: Any
    ) -> Optimizer:
        """
        Resolves optimizer type, splits parameter groups to filter weight decay,
        and returns the initialized Optimizer instance.
        """
        cls_type = OptimizerRegistry.get(name)

        # Separate parameters into decay/no-decay groups
        param_groups = WeightDecayFilter.get_parameter_groups(model, weight_decay)

        # Instantiate optimizer with parameter groups and custom arguments
        return cls_type(param_groups, lr=lr, **kwargs)
