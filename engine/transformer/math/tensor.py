from typing import Any, Optional, Tuple, Union

import torch

from engine.transformer.math.device import DeviceManager
from engine.transformer.math.precision import PrecisionManager


class TensorUtils:
    """
    Utility methods for creating, initializing, and casting PyTorch tensors.
    """

    @staticmethod
    def create(
        data: Any,
        dtype: Optional[Union[str, torch.dtype]] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> torch.Tensor:
        """
        Creates a tensor from raw data with device and precision placement.
        """
        target_dtype = PrecisionManager.get_dtype(dtype) if dtype else None
        target_device = DeviceManager.get_device(device)
        return torch.tensor(data, dtype=target_dtype, device=target_device)

    @staticmethod
    def zeros(
        shape: Tuple[int, ...],
        dtype: Optional[Union[str, torch.dtype]] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> torch.Tensor:
        """
        Allocates a tensor filled with zeros.
        """
        target_dtype = PrecisionManager.get_dtype(dtype) if dtype else torch.float32
        target_device = DeviceManager.get_device(device)
        return torch.zeros(shape, dtype=target_dtype, device=target_device)

    @staticmethod
    def ones(
        shape: Tuple[int, ...],
        dtype: Optional[Union[str, torch.dtype]] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> torch.Tensor:
        """
        Allocates a tensor filled with ones.
        """
        target_dtype = PrecisionManager.get_dtype(dtype) if dtype else torch.float32
        target_device = DeviceManager.get_device(device)
        return torch.ones(shape, dtype=target_dtype, device=target_device)

    @staticmethod
    def randn(
        shape: Tuple[int, ...],
        dtype: Optional[Union[str, torch.dtype]] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> torch.Tensor:
        """
        Allocates a tensor filled with random normal values (mean 0, std 1).
        """
        target_dtype = PrecisionManager.get_dtype(dtype) if dtype else torch.float32
        target_device = DeviceManager.get_device(device)
        return torch.randn(shape, dtype=target_dtype, device=target_device)
