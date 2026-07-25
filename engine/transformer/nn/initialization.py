import torch


class Initializer:
    """
    Provides weight initialization strategies for neural network parameters.
    """

    @staticmethod
    def zeros_(tensor: torch.Tensor) -> torch.Tensor:
        """Fills target tensor with zeros in-place."""
        return torch.nn.init.zeros_(tensor)

    @staticmethod
    def ones_(tensor: torch.Tensor) -> torch.Tensor:
        """Fills target tensor with ones in-place."""
        return torch.nn.init.ones_(tensor)

    @staticmethod
    def normal_(
        tensor: torch.Tensor, mean: float = 0.0, std: float = 1.0
    ) -> torch.Tensor:
        """Fills target tensor with values drawn from normal distribution."""
        return torch.nn.init.normal_(tensor, mean=mean, std=std)

    @staticmethod
    def uniform_(tensor: torch.Tensor, a: float = -1.0, b: float = 1.0) -> torch.Tensor:
        """Fills target tensor with values drawn from uniform distribution U(a, b)."""
        return torch.nn.init.uniform_(tensor, a=a, b=b)

    @staticmethod
    def xavier_uniform_(tensor: torch.Tensor, gain: float = 1.0) -> torch.Tensor:
        """Fills target tensor using Xavier (Glorot) uniform initialization."""
        return torch.nn.init.xavier_uniform_(tensor, gain=gain)

    @staticmethod
    def xavier_normal_(tensor: torch.Tensor, gain: float = 1.0) -> torch.Tensor:
        """Fills target tensor using Xavier (Glorot) normal initialization."""
        return torch.nn.init.xavier_normal_(tensor, gain=gain)

    @staticmethod
    def kaiming_uniform_(
        tensor: torch.Tensor,
        a: float = 0,
        mode: str = "fan_in",
        nonlinearity: str = "leaky_relu",
    ) -> torch.Tensor:
        """Fills target tensor using Kaiming (He) uniform initialization."""
        return torch.nn.init.kaiming_uniform_(
            tensor, a=a, mode=mode, nonlinearity=nonlinearity
        )

    @staticmethod
    def kaiming_normal_(
        tensor: torch.Tensor,
        a: float = 0,
        mode: str = "fan_in",
        nonlinearity: str = "leaky_relu",
    ) -> torch.Tensor:
        """Fills target tensor using Kaiming (He) normal initialization."""
        return torch.nn.init.kaiming_normal_(
            tensor, a=a, mode=mode, nonlinearity=nonlinearity
        )
