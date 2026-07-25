from typing import List, Union

import torch

from engine.transformer.math.exceptions import TensorShapeError
from engine.transformer.math.validation import TensorValidator


class MatrixOps:
    """
    Core matrix and batched matrix operations for Chatii Transformer components.
    """

    @staticmethod
    def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication (2D or batched N-D).
        """
        TensorValidator.check_non_empty(a, "matrix_a")
        TensorValidator.check_non_empty(b, "matrix_b")
        try:
            return torch.matmul(a, b)
        except RuntimeError as e:
            raise TensorShapeError(
                f"Incompatible matrix shapes for multiplication: "
                f"A={tuple(a.shape)}, B={tuple(b.shape)}: {e}"
            ) from e

    @staticmethod
    def batch_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Explicit batched matrix multiplication (expects >= 3D inputs).
        """
        if a.ndim < 3 or b.ndim < 3:
            raise TensorShapeError(
                f"Batch matrix multiplication expects >=3D tensors, "
                f"got A={a.ndim}D and B={b.ndim}D"
            )
        return MatrixOps.matmul(a, b)

    @staticmethod
    def transpose(tensor: torch.Tensor, dim0: int = -2, dim1: int = -1) -> torch.Tensor:
        """
        Transposes two target dimensions of a tensor.
        """
        return torch.transpose(tensor, dim0, dim1)

    @staticmethod
    def reshape(tensor: torch.Tensor, *shape: int) -> torch.Tensor:
        """
        Reshapes a tensor to target shape dimensions.
        """
        try:
            return tensor.reshape(*shape)
        except RuntimeError as e:
            raise TensorShapeError(
                f"Cannot reshape tensor {tuple(tensor.shape)} to {shape}: {e}"
            ) from e

    @staticmethod
    def view(tensor: torch.Tensor, *shape: int) -> torch.Tensor:
        """
        Returns a new tensor with specified shape layout.
        """
        try:
            return tensor.contiguous().view(*shape)
        except RuntimeError as e:
            raise TensorShapeError(
                f"Cannot view tensor {tuple(tensor.shape)} as {shape}: {e}"
            ) from e

    @staticmethod
    def flatten(
        tensor: torch.Tensor, start_dim: int = 1, end_dim: int = -1
    ) -> torch.Tensor:
        """
        Flattens contiguous tensor dimensions.
        """
        return torch.flatten(tensor, start_dim=start_dim, end_dim=end_dim)

    @staticmethod
    def concatenate(tensors: List[torch.Tensor], dim: int = 0) -> torch.Tensor:
        """
        Concatenates a list of tensors along a specified dimension.
        """
        if not tensors:
            raise TensorShapeError("Cannot concatenate an empty list of tensors.")
        try:
            return torch.cat(tensors, dim=dim)
        except RuntimeError as e:
            shapes = [tuple(t.shape) for t in tensors]
            raise TensorShapeError(
                f"Concatenation error along dimension {dim} for shapes {shapes}: {e}"
            ) from e

    @staticmethod
    def split(
        tensor: torch.Tensor,
        split_size_or_sections: Union[int, List[int]],
        dim: int = 0,
    ) -> List[torch.Tensor]:
        """
        Splits a tensor into multiple chunks along a specified dimension.
        """
        return list(torch.split(tensor, split_size_or_sections, dim=dim))
