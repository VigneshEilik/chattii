from typing import Any, Dict, Optional

import torch

from engine.transformer.config import ModelMathConfig
from engine.transformer.factory import ActivationFactory, NormalizationFactory
from engine.transformer.model import ChatiiModel
from engine.transformer.position.factory import PositionFactory


class TransformerMathService:
    """
    Service layer orchestrating mathematical components in Milestone 2.
    """

    def __init__(self, config: ModelMathConfig) -> None:
        self.config = config
        self.activation = ActivationFactory.create(config.activation)
        self.normalization = NormalizationFactory.create(
            config.normalization, dim=config.d_model
        )
        self.position_embedding = PositionFactory.create(
            config.position_embedding,
            dim=config.d_model,
            d_model=config.d_model,
            max_seq_len=config.max_seq_len,
        )


class TransformerModelService:
    """
    Service orchestration layer serving as the primary API for ChatiiModel actions.
    """

    def __init__(self, config: Optional[ModelMathConfig] = None) -> None:
        self.config = config or ModelMathConfig.load_from_yaml()
        self.model = ChatiiModel(self.config)

    def forward(
        self,
        input_ids: torch.Tensor,
        use_cache: bool = False,
        cache: Optional[Any] = None,
    ) -> torch.Tensor:
        """
        Runs model forward pass.
        """
        return self.model(input_ids, cache=cache, use_cache=use_cache)

    def save(self, save_dir: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Saves model weights and parameters.
        """
        self.model.save(save_dir, metadata=metadata)

    def load(self, load_dir: str) -> None:
        """
        Loads model weights from designated folder.
        """
        self.model.load(load_dir)
