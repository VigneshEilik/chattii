from typing import Any, Dict, List, Optional

import torch.nn as nn


class ParameterGroups:
    """Configures layer-specific learning rates for fine-tuning embeddings/heads."""

    @staticmethod
    def get_groups(
        model: nn.Module,
        base_lr: float,
        embedding_lr: Optional[float] = None,
        lm_head_lr: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Groups model parameters by layers (Embedding, Transformer, LMHead)
        and configures separate learning rates.
        """
        groups = []
        embedding_params = []
        lm_head_params = []
        transformer_params = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            if "token_embedding" in name:
                embedding_params.append(param)
            elif "lm_head" in name:
                lm_head_params.append(param)
            else:
                transformer_params.append(param)

        # 1. Base Transformer Stack Parameters
        groups.append({"params": transformer_params, "lr": base_lr})

        # 2. Embedding Layer Parameters
        if embedding_params:
            groups.append(
                {
                    "params": embedding_params,
                    "lr": embedding_lr if embedding_lr is not None else base_lr,
                }
            )

        # 3. LM Head Parameters
        if lm_head_params:
            groups.append(
                {
                    "params": lm_head_params,
                    "lr": lm_head_lr if lm_head_lr is not None else base_lr,
                }
            )

        return groups
