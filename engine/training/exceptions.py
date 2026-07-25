# ruff: noqa: E501, N818
class TrainingError(Exception):
    """Base exception class for all training errors in Chatii."""

    pass


class ConfigurationError(TrainingError):
    """Raised when training configuration validation fails."""

    pass


class CheckpointError(TrainingError):
    """Raised when checkpoint serialization or deserialization fails signature validation."""

    pass


class EarlyStoppingTrigger(TrainingError):
    """Signal raised to gracefully halt training loop when loss ceilings are reached."""

    pass


class NaNLossError(TrainingError):
    """Raised when loss evaluates to NaN or Inf during training iterations."""

    pass
