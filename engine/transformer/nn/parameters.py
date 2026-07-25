import torch


class Parameter:
    """
    Wrapper around a PyTorch Tensor representing a learnable neural network parameter.
    """

    def __init__(self, data: torch.Tensor, requires_grad: bool = True) -> None:
        if not isinstance(data, torch.Tensor):
            raise TypeError(f"Parameter data must be a torch.Tensor, got {type(data)}")
        self.data = data
        self.data.requires_grad_(requires_grad)

    @property
    def requires_grad(self) -> bool:
        """Returns True if gradients are enabled for this parameter."""
        return self.data.requires_grad

    @requires_grad.setter
    def requires_grad(self, value: bool) -> None:
        self.data.requires_grad_(value)

    @property
    def shape(self) -> torch.Size:
        """Returns the shape dimensions of the parameter tensor."""
        return self.data.shape

    @property
    def dtype(self) -> torch.dtype:
        """Returns the precision data type of the parameter tensor."""
        return self.data.dtype

    @property
    def device(self) -> torch.device:
        """Returns the compute device hosting the parameter tensor."""
        return self.data.device

    def __repr__(self) -> str:
        return (
            f"Parameter(shape={tuple(self.data.shape)}, "
            f"dtype={self.data.dtype}, device={self.data.device})"
        )
