# ruff: noqa: E501
from pathlib import Path
from typing import Any, List, Optional, Union

import torch.nn as nn

from engine.tokenizer.service import TokenizerService
from engine.training.config.loader import TrainingConfigLoader
from engine.training.config.validator import TrainingConfigValidator
from engine.training.dataset.dataset import ChatiiDataset
from engine.training.factory import TrainerFactory
from engine.training.trainer import BaseTrainer


class TrainingService:
    """Orchestrates configuration validation, data setup, and model training execution."""

    def __init__(self, config_dir: Union[str, Path] = "configs") -> None:
        self.config = TrainingConfigLoader.load_from_yaml(config_dir)
        TrainingConfigValidator.validate(self.config)

    def run_training(
        self,
        model: nn.Module,
        tokenizer: TokenizerService,
        train_file: str,
        val_file: Optional[str] = None,
        callbacks: Optional[List[Any]] = None,
        resume_dir: Optional[str] = None,
    ) -> BaseTrainer:
        """
        Loads training/validation datasets, instantiates SFT trainer,
        restores from checkpoints if requested, and launches the training loop.
        """
        # Load and preprocess training dataset
        train_dataset = ChatiiDataset(
            train_file,
            tokenizer,
            sequence_length=self.config.sequence_length,
            clean=self.config.preprocessing.clean,
            deduplicate=self.config.preprocessing.deduplicate,
            language_filter=self.config.preprocessing.language_filter,
            quality_filter=self.config.preprocessing.quality_filter,
        )

        # Load validation dataset if specified
        val_dataset = None
        if val_file:
            val_dataset = ChatiiDataset(
                val_file,
                tokenizer,
                sequence_length=self.config.sequence_length,
                clean=self.config.preprocessing.clean,
                deduplicate=self.config.preprocessing.deduplicate,
                language_filter=self.config.preprocessing.language_filter,
                quality_filter=self.config.preprocessing.quality_filter,
            )

        # Instantiate trainer using factory
        trainer = TrainerFactory.create(
            "sft",
            model=model,
            tokenizer=tokenizer,
            config=self.config,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            callbacks=callbacks,
        )

        # Auto-resume from checkpoint directory on-demand
        if resume_dir and Path(resume_dir).exists():
            trainer.resume_from_checkpoint(resume_dir)

        trainer.train()
        return trainer
