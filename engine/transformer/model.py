from typing import Any, Dict, Optional

import torch

from engine.transformer.blocks.transformer import TransformerStack
from engine.transformer.cache.attention_cache import AttentionCache
from engine.transformer.checkpoint.load import CheckpointLoader
from engine.transformer.checkpoint.save import CheckpointSaver
from engine.transformer.config import ModelMathConfig
from engine.transformer.embedding.lm_head import LMHead
from engine.transformer.embedding.token import TokenEmbedding
from engine.transformer.nn.module import Module
from engine.transformer.nn.normalization import RMSNorm


class ChatiiModel(Module):
    """
    Complete Decoder-only Transformer model architecture.
    """

    def __init__(
        self, config: Optional[ModelMathConfig] = None, weight_tying: bool = True
    ) -> None:
        super().__init__()
        self.config = config or ModelMathConfig.load_from_yaml()
        self.weight_tying = weight_tying

        # 1. Token Embeddings
        self.token_embedding = TokenEmbedding(
            vocab_size=self.config.vocab_size,
            hidden_size=self.config.d_model,
            device=self.config.device,
            dtype=self.config.dtype,
        )

        # 2. Sequential Blocks Stack
        self.stack = TransformerStack(
            num_layers=self.config.num_layers,
            d_model=self.config.d_model,
            num_heads=self.config.n_heads,
            d_ff=self.config.d_ff,
            dropout=self.config.dropout,
            rope_theta=self.config.rope_theta,
            max_seq_len=self.config.max_seq_len,
            activation=self.config.activation,
            normalization=self.config.normalization,
            device=self.config.device,
            dtype=self.config.dtype,
        )

        # 3. Final Norm layer
        self.norm = RMSNorm(
            dim=self.config.d_model, device=self.config.device, dtype=self.config.dtype
        )

        # 4. Language Model prediction Head
        tied_weight = self.token_embedding.weight if weight_tying else None
        self.lm_head = LMHead(
            vocab_size=self.config.vocab_size,
            hidden_size=self.config.d_model,
            bias=False,
            tied_weight=tied_weight,
            device=self.config.device,
            dtype=self.config.dtype,
        )

        self.register_module("token_embedding", self.token_embedding)
        self.register_module("stack", self.stack)
        self.register_module("norm", self.norm)
        self.register_module("lm_head", self.lm_head)

    def forward(
        self,
        input_ids: torch.Tensor,
        cache: Optional[AttentionCache] = None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        """
        Executes end-to-end forward pass.
        """
        seq_len = input_ids.shape[1]
        start_pos = 0
        if use_cache and cache is not None:
            layer_cache = cache.get_cache(0)
            if layer_cache.k is not None:
                start_pos = layer_cache.k.shape[2]
        total_seq_len = start_pos + seq_len

        # Construct causal additive mask where row index <= col index
        mask = torch.zeros((seq_len, total_seq_len), device=input_ids.device)
        col_indices = torch.arange(total_seq_len, device=input_ids.device)[None, :]
        row_indices = torch.arange(start_pos, total_seq_len, device=input_ids.device)[
            :, None
        ]
        mask.masked_fill_(col_indices > row_indices, float("-inf"))
        mask = mask.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, seq_len, total_seq_len)

        h = self.token_embedding(input_ids)
        h = self.stack(h, mask=mask, cache=cache, use_cache=use_cache)
        h = self.norm(h)
        logits = self.lm_head(h)
        return logits

    def save(self, save_dir: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Serializes model checkpoints including config.json, metadata.json, and model.pt.
        """
        CheckpointSaver.save(save_dir, self.state_dict(), self.config, metadata)

    def load(self, load_dir: str) -> None:
        """
        Loads parameters and configurations from a checkpoint directory.
        """
        state_dict, config, metadata = CheckpointLoader.load(load_dir)
        self.config = config
        self.load_state_dict(state_dict)

    def num_parameters(self, only_trainable: bool = False) -> int:
        """
        Returns total parameters count.
        """
        params = self.parameters()
        if only_trainable:
            return sum(p.data.numel() for p in params if p.requires_grad)
        return sum(p.data.numel() for p in params)

    def summary(self) -> str:
        """
        Summarizes model structure details.
        """
        lines = [
            "--- ChatiiModel Architecture Summary ---",
            f"Model Vocab Size : {self.config.vocab_size}",
            f"Model Dimension  : {self.config.d_model}",
            f"Number of Layers : {self.config.num_layers}",
            f"Attention Heads  : {self.config.n_heads}",
            f"FFN Dimension    : {self.config.d_ff}",
            f"Activation FFN   : {self.config.activation}",
            f"Weight Tying     : {self.weight_tying}",
            f"Total Parameters : {self.num_parameters():,}",
        ]
        return "\n".join(lines)

    def compile(self) -> "ChatiiModel":
        """
        Stub method wrapping PyTorch 2.0+ torch.compile if supported.
        """
        if hasattr(torch, "compile"):
            try:
                self.compiled_forward = torch.compile(self.forward)
            except Exception:
                pass
        return self

    def freeze(self) -> None:
        """
        Freezes all parameters for execution during inference.
        """
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze(self) -> None:
        """
        Enables gradients back for training steps.
        """
        for p in self.parameters():
            p.requires_grad = True
