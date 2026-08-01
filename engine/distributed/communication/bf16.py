import torch


class BF16Communicator:
    """Compresses gradients/tensors to BF16 to reduce communication bandwidth."""

    @staticmethod
    def compress(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(torch.bfloat16)

    @staticmethod
    def decompress(tensor: torch.Tensor, original_dtype: torch.dtype) -> torch.Tensor:
        return tensor.to(original_dtype)
