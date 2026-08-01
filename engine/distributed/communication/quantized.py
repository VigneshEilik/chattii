import torch


class QuantizedCommunicator:
    """Performs 8-bit quantization on gradients/tensors to optimize bandwidth."""

    @staticmethod
    def compress(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.numel() == 0:
            return tensor
        min_val = tensor.min()
        max_val = tensor.max()
        scale = (max_val - min_val) / 255.0
        scale = torch.clamp(scale, min=1e-8)
        quantized = torch.clamp(((tensor - min_val) / scale).round(), 0, 255).to(
            torch.uint8
        )
        return quantized

    @staticmethod
    def decompress(
        quantized_tensor: torch.Tensor,
        min_val: float,
        scale: float,
        original_dtype: torch.dtype,
    ) -> torch.Tensor:
        return (quantized_tensor.to(original_dtype) * scale) + min_val
