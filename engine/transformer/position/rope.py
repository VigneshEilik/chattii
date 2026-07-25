from typing import Optional, Tuple, Union

import torch

from engine.transformer.math.device import DeviceManager
from engine.transformer.nn.module import Module


def precompute_freqs_cis(
    dim: int,
    end: int,
    theta: float = 10000.0,
    device: Optional[Union[str, torch.device]] = None,
) -> torch.Tensor:
    """
    Precomputes complex frequencies e^(i * m * theta) for Rotary Position Embeddings.
    """
    target_device = DeviceManager.get_device(device)
    dim_range = torch.arange(0, dim, 2, device=target_device)[: (dim // 2)].float()
    freqs = 1.0 / (theta ** (dim_range / dim))
    t = torch.arange(end, device=target_device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64 tensor
    return freqs_cis


def apply_rotary_emb(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies rotary embeddings to query and key tensors.
    """
    xq_complex = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_complex = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    # Reshape freqs_cis for broadcasting: (1, seq_len, 1, head_dim // 2)
    shape = [1, xq.size(1), 1, xq.size(-1) // 2]
    freqs_cis_b = freqs_cis.view(*shape)

    xq_out = torch.view_as_real(xq_complex * freqs_cis_b).flatten(-2)
    xk_out = torch.view_as_real(xk_complex * freqs_cis_b).flatten(-2)

    return xq_out.type_as(xq), xk_out.type_as(xk)


class RotaryPositionalEmbedding(Module):
    """
    Rotary Position Embedding (RoPE) module.
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 2048,
        theta: float = 10000.0,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.freqs_cis = precompute_freqs_cis(dim, max_seq_len, theta, device=device)

    def forward(
        self, xq: torch.Tensor, xk: torch.Tensor, start_pos: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = xq.size(1)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seq_len].to(xq.device)
        return apply_rotary_emb(xq, xk, freqs_cis)
