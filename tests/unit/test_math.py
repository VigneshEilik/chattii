import pytest
import torch

from engine.transformer.math.broadcasting import BroadcastingValidator
from engine.transformer.math.device import DeviceManager
from engine.transformer.math.exceptions import (
    NumericalInstabilityError,
    TensorShapeError,
)
from engine.transformer.math.matrix import MatrixOps
from engine.transformer.math.precision import PrecisionManager
from engine.transformer.math.shape import ShapeUtils
from engine.transformer.math.validation import TensorValidator


def test_precision_and_device_managers() -> None:
    assert PrecisionManager.get_dtype("float32") == torch.float32
    assert PrecisionManager.get_dtype("bfloat16") == torch.bfloat16

    t = torch.randn(2, 2)
    t_fp16 = PrecisionManager.cast(t, "float16")
    assert t_fp16.dtype == torch.float16

    dev = DeviceManager.get_device("cpu")
    assert dev == torch.device("cpu")


def test_broadcasting_and_shape_utils() -> None:
    assert BroadcastingValidator.is_broadcastable((1, 512), (8, 512))
    assert not BroadcastingValidator.is_broadcastable((4, 512), (8, 512))

    with pytest.raises(TensorShapeError):
        BroadcastingValidator.validate((4, 512), (8, 512))

    t = torch.randn(2, 3, 4)
    flattened = ShapeUtils.flatten(t, 1, 2)
    assert flattened.shape == (2, 12)


def test_tensor_validator() -> None:
    t_nan = torch.tensor([1.0, float("nan")])
    with pytest.raises(NumericalInstabilityError):
        TensorValidator.check_nan(t_nan)

    t_inf = torch.tensor([1.0, float("inf")])
    with pytest.raises(NumericalInstabilityError):
        TensorValidator.check_inf(t_inf)

    empty_t = torch.empty(0)
    with pytest.raises(TensorShapeError):
        TensorValidator.check_non_empty(empty_t)


def test_matrix_operations() -> None:
    a = torch.randn(4, 8)
    b = torch.randn(8, 16)
    c = MatrixOps.matmul(a, b)
    assert c.shape == (4, 16)

    # Batched matmul
    b_a = torch.randn(2, 4, 8)
    b_b = torch.randn(2, 8, 16)
    b_c = MatrixOps.batch_matmul(b_a, b_b)
    assert b_c.shape == (2, 4, 16)

    # Transpose & reshape
    t_a = MatrixOps.transpose(a, 0, 1)
    assert t_a.shape == (8, 4)

    reshaped = MatrixOps.reshape(a, 32)
    assert reshaped.shape == (32,)
