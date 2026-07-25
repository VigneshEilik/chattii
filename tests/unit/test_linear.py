import torch

from engine.transformer.nn.linear import Linear


def test_linear_layer_forward() -> None:
    layer = Linear(in_features=16, out_features=32, bias=True)
    assert layer.weight.shape == (32, 16)
    assert layer.bias is not None and layer.bias.shape == (32,)

    x = torch.randn(4, 16)
    out = layer(x)
    assert out.shape == (4, 32)


def test_linear_no_bias() -> None:
    layer = Linear(in_features=16, out_features=32, bias=False)
    assert layer.bias is None

    x = torch.randn(2, 512, 16)
    out = layer(x)
    assert out.shape == (2, 512, 32)
