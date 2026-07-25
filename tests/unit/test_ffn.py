import torch

from engine.transformer.feedforward.factory import FeedForwardFactory
from engine.transformer.feedforward.mlp import MLP
from engine.transformer.feedforward.swiglu import SwiGLU
from engine.transformer.feedforward.validator import FeedForwardValidator


def test_mlp_and_swiglu() -> None:
    d_model = 64
    d_ff = 256

    mlp = FeedForwardFactory.create("mlp", d_model=d_model, d_ff=d_ff)
    swiglu = FeedForwardFactory.create("swiglu", d_model=d_model, d_ff=d_ff)

    assert isinstance(mlp, MLP)
    assert isinstance(swiglu, SwiGLU)

    x = torch.randn(2, 10, d_model)
    out_mlp = mlp(x)
    out_swiglu = swiglu(x)

    assert out_mlp.shape == (2, 10, d_model)
    assert out_swiglu.shape == (2, 10, d_model)

    # Validate numeric properties
    ok_mlp, errs_mlp = FeedForwardValidator.validate_feedforward(mlp, x, out_mlp)
    ok_swiglu, errs_swiglu = FeedForwardValidator.validate_feedforward(
        swiglu, x, out_swiglu
    )

    assert ok_mlp
    assert ok_swiglu
    assert len(errs_mlp) == 0
    assert len(errs_swiglu) == 0
