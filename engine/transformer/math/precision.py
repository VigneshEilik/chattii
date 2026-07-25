from typing import Dict, Union

import torch

from engine.transformer.math.exceptions import PrecisionError


class PrecisionManager:
    """
    Manages data precision types for Chatii (float32, float16, bfloat16, int64).
    """

    _DTYPE_MAP: Dict[str, torch.dtype] = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "int64": torch.int64,
        "int32": torch.int32,
    }

    @classmethod
    def get_dtype(cls, name_or_dtype: Union[str, torch.dtype]) -> torch.dtype:
        """
        Resolves string label or torch.dtype to canonical torch.dtype.
        """
        if isinstance(name_or_dtype, torch.dtype):
            return name_or_dtype
        key = str(name_or_dtype).lower().strip()
        if key not in cls._DTYPE_MAP:
            raise PrecisionError(
                f"Unsupported precision '{name_or_dtype}'. "
                f"Supported values: {list(cls._DTYPE_MAP.keys())}"
            )
        return cls._DTYPE_MAP[key]

    @classmethod
    def cast(cls, tensor: torch.Tensor, dtype: Union[str, torch.dtype]) -> torch.Tensor:
        """
        Casts a target PyTorch tensor to the requested precision.
        """
        target_dtype = cls.get_dtype(dtype)
        return tensor.to(dtype=target_dtype)
