import torch

from engine.transformer.math.exceptions import (
    NumericalInstabilityError,
    TensorShapeError,
)


class TensorValidator:
    """
    Sanity and numerical stability checks for PyTorch tensors.
    """

    @staticmethod
    def check_nan(tensor: torch.Tensor, name: str = "tensor") -> None:
        """
        Raises NumericalInstabilityError if NaN values exist in the tensor.
        """
        if torch.isnan(tensor).any():
            raise NumericalInstabilityError(f"NaN values detected in '{name}'.")

    @staticmethod
    def check_inf(tensor: torch.Tensor, name: str = "tensor") -> None:
        """
        Raises NumericalInstabilityError if Inf values exist in the tensor.
        """
        if torch.isinf(tensor).any():
            raise NumericalInstabilityError(f"Inf values detected in '{name}'.")

    @staticmethod
    def check_numerical_stability(tensor: torch.Tensor, name: str = "tensor") -> None:
        """
        Assures neither NaN nor Inf values contaminate tensor computations.
        """
        TensorValidator.check_nan(tensor, name)
        TensorValidator.check_inf(tensor, name)

    @staticmethod
    def check_non_empty(tensor: torch.Tensor, name: str = "tensor") -> None:
        """
        Raises TensorShapeError if the tensor contains zero elements.
        """
        if tensor.numel() == 0:
            raise TensorShapeError(f"Tensor '{name}' is empty (numel = 0).")
