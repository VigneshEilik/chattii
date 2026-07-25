import torch

from engine.transformer.blocks.transformer import TransformerStack


def test_transformer_stack() -> None:
    d_model = 64
    num_heads = 4
    d_ff = 256
    num_layers = 3

    stack = TransformerStack(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        activation="swiglu",
        normalization="rmsnorm",
    )

    x = torch.randn(2, 10, d_model)
    out = stack(x)
    assert out.shape == (2, 10, d_model)
