from typing import Optional, Union

import torch

from engine.transformer.math.exceptions import DeviceError


class DeviceManager:
    """
    Compute device manager providing CPU, CUDA, and Apple MPS abstraction.
    """

    @classmethod
    def get_device(
        cls, device_input: Optional[Union[str, torch.device]] = None
    ) -> torch.device:
        """
        Validates and returns a PyTorch device object.
        """
        if device_input is None:
            return torch.device("cpu")
        if isinstance(device_input, torch.device):
            return device_input

        dev_str = str(device_input).lower().strip()

        if dev_str.startswith("cuda"):
            if not torch.cuda.is_available():
                raise DeviceError("CUDA compute device requested but is not available.")
            return torch.device(dev_str)

        if dev_str.startswith("mps"):
            has_mps = (
                hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            )
            if not has_mps:
                raise DeviceError(
                    "Apple MPS compute device requested but is not available."
                )
            return torch.device(dev_str)

        if dev_str == "cpu":
            return torch.device("cpu")

        raise DeviceError(f"Unsupported device identifier: '{device_input}'")

    @classmethod
    def move_to_device(
        cls,
        tensor: torch.Tensor,
        device_input: Optional[Union[str, torch.device]] = None,
    ) -> torch.Tensor:
        """
        Transfers a tensor onto the requested compute hardware device.
        """
        target_dev = cls.get_device(device_input)
        return tensor.to(device=target_dev)
