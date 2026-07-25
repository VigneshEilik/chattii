class MathError(Exception):
    """Base exception for all mathematical and tensor operations in Chatii."""

    pass


class TensorShapeError(MathError):
    """Raised when tensor shapes are incompatible or invalid for an operation."""

    pass


class PrecisionError(MathError):
    """Raised when an unsupported precision or type mismatch is encountered."""

    pass


class DeviceError(MathError):
    """Raised when a specified compute device is unavailable or invalid."""

    pass


class NumericalInstabilityError(MathError):
    """Raised when NaN or Inf values are detected in tensor computations."""

    pass
