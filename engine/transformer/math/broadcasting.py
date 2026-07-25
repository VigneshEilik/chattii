from typing import Tuple

from engine.transformer.math.exceptions import TensorShapeError


class BroadcastingValidator:
    """
    Validates array broadcasting rules between tensor shapes.
    """

    @staticmethod
    def is_broadcastable(shape1: Tuple[int, ...], shape2: Tuple[int, ...]) -> bool:
        """
        Returns True if two shapes are compatible under PyTorch broadcasting rules.
        """
        for d1, d2 in zip(reversed(shape1), reversed(shape2), strict=False):
            if d1 != 1 and d2 != 1 and d1 != d2:
                return False
        return True

    @staticmethod
    def validate(shape1: Tuple[int, ...], shape2: Tuple[int, ...]) -> None:
        """
        Raises TensorShapeError if shapes cannot be broadcasted together.
        """
        if not BroadcastingValidator.is_broadcastable(shape1, shape2):
            raise TensorShapeError(
                f"Incompatible dimensions for broadcasting: {shape1} and {shape2}"
            )
