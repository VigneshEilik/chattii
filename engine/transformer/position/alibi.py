import math
from typing import List, Optional, Union

import torch

from engine.transformer.math.device import DeviceManager
from engine.transformer.math.precision import PrecisionManager
from engine.transformer.nn.module import Module


def get_alibi_slopes(n_heads: int) -> List[float]:
    """
    Computes ALiBi slope constants for a given number of attention heads.
    """

    def get_slopes_power_of_2(n: int) -> List[float]:
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * (ratio**i) for i in range(n)]

    if math.log2(n_heads).is_integer():
        return get_slopes_power_of_2(n_heads)
    else:
        closest_power_of_2 = 2 ** math.floor(math.log2(n_heads))
        extra_slopes = get_alibi_slopes(2 * closest_power_of_2)[0::2]
        return (
            get_slopes_power_of_2(closest_power_of_2)
            + extra_slopes[: n_heads - closest_power_of_2]
        )


class ALiBiPositionalBias(Module):
    """
    Attention with Linear Biases (ALiBi) additive position bias.
    """

    def __init__(
        self,
        num_heads: int,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[Union[str, torch.dtype]] = None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        target_device = DeviceManager.get_device(device)
        target_dtype = PrecisionManager.get_dtype(dtype) if dtype else torch.float32

        slopes = torch.tensor(
            get_alibi_slopes(num_heads), device=target_device, dtype=target_dtype
        )
        self.slopes = slopes.view(1, num_heads, 1, 1)

    def forward(
        self, seq_len: int, device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Generates additive ALiBi bias matrix of shape (1, num_heads, seq_len, seq_len).
        """
        target_device = device or self.slopes.device
        # Causal distance matrix: positions i, j
        context_position = torch.arange(seq_len, device=target_device)[:, None]
        memory_position = torch.arange(seq_len, device=target_device)[None, :]
        relative_position = memory_position - context_position
        relative_position = torch.abs(relative_position).unsqueeze(0).unsqueeze(0)

        # Multiply by slopes (-m * distance)
        scaled_pos = relative_position.to(dtype=self.slopes.dtype)
        alibi = -1.0 * self.slopes.to(target_device) * scaled_pos
        return alibi
