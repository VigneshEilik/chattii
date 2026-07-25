from typing import List, Tuple, Union

import torch

from engine.transformer.math.exceptions import TensorShapeError


class ShapeUtils:
    """
    Utilities for tensor shape inspection, reshaping, transpositions, and splitting.
    """

    @staticmethod
    def flatten(
        tensor: torch.Tensor, start_dim: int = 1, end_dim: int = -1
    ) -> torch.Tensor:
        """
        Flattens contiguous dimensions of a tensor into a single dimension.
        """
        return torch.flatten(tensor, start_dim=start_dim, end_dim=end_dim)

    @staticmethod
    def reshape(tensor: torch.Tensor, shape: Tuple[int, ...]) -> torch.Tensor:
        """
        Reshapes a tensor to the target shape dimensions.
        """
        try:
            return tensor.reshape(shape)
        except RuntimeError as e:
            raise TensorShapeError(
                f"Cannot reshape tensor of shape {tuple(tensor.shape)} to {shape}: {e}"
            ) from e

    @staticmethod
    def transpose(tensor: torch.Tensor, dim0: int, dim1: int) -> torch.Tensor:
        """
        Swaps two dimensions of a tensor.
        """
        return torch.transpose(tensor, dim0, dim1)

    @staticmethod
    def permute(tensor: torch.Tensor, dims: Tuple[int, ...]) -> torch.Tensor:
        """
        Permutes the dimensions of a tensor according to the specified axis order.
        """
        return tensor.permute(dims)

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
