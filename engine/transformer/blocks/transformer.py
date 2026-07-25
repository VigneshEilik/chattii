from typing import Optional, Union

import torch

from engine.transformer.blocks.decoder import DecoderBlock
from engine.transformer.cache.attention_cache import AttentionCache
from engine.transformer.nn.module import Module


class TransformerStack(Module):
    """
    Sequential stack of N Decoder Block layers.
    """

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.0,
        rope_theta: float = 10000.0,
        max_seq_len: int = 2048,
        activation: str = "swiglu",
        normalization: str = "rmsnorm",
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[Union[str, torch.dtype]] = None,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.layers_list = []

        for i in range(num_layers):
            layer = DecoderBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout=dropout,
                rope_theta=rope_theta,
                max_seq_len=max_seq_len,
                activation=activation,
                normalization=normalization,
                device=device,
                dtype=dtype,
            )
            self.register_module(f"layer_{i}", layer)
            self.layers_list.append(layer)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cache: Optional[AttentionCache] = None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        """
        Processes inputs sequentially through each decoder block in the stack.
        """
        for i, layer in enumerate(self.layers_list):
            layer_cache = (
                cache.get_cache(i) if (use_cache and cache is not None) else None
            )
            x = layer(x, mask=mask, cache=layer_cache, use_cache=use_cache)
        return x
