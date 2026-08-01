from typing import Any

import torch


class AMPAutocastContext:
    """Manages Automatic Mixed Precision autocast context scopes."""

    @staticmethod
    def get_autocast(
        device_type: str, enabled: bool = True, dtype: torch.dtype = torch.float16
    ) -> Any:
        return torch.amp.autocast(device_type=device_type, enabled=enabled, dtype=dtype)
