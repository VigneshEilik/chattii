from typing import Any, List, Optional

import torch.nn as nn

from engine.tokenizer.service import TokenizerService
from engine.training.config.schema import ModelTrainingConfig
from engine.training.dataset.dataset import ChatiiDataset
from engine.training.registry import TrainerRegistry
from engine.training.trainer import BaseTrainer


class TrainerFactory:
    """Factory for dynamically creating and configuring Trainer instances."""

    @staticmethod
    def create(
        name: str,
        model: nn.Module,
        tokenizer: TokenizerService,
        config: ModelTrainingConfig,
        train_dataset: ChatiiDataset,
        val_dataset: Optional[ChatiiDataset] = None,
        callbacks: Optional[List[Any]] = None,
    ) -> BaseTrainer:
        """
        Resolves trainer type and returns the initialized trainer instance.
        """
        cls_type = TrainerRegistry.get(name)
        return cls_type(
            model=model,
            tokenizer=tokenizer,
            config=config,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            callbacks=callbacks,
        )
