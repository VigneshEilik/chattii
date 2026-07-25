import torch

from engine.transformer.nn.normalization import LayerNorm, RMSNorm


def test_layernorm() -> None:
    ln = LayerNorm(normalized_shape=16)
    x = torch.randn(4, 16) * 10.0 + 5.0
    out = ln(x)
    assert out.shape == (4, 16)
    # Mean across feature dim should be ~0, var ~1
    assert torch.allclose(out.mean(dim=-1), torch.zeros(4), atol=1e-3)


def test_rmsnorm() -> None:
    rms = RMSNorm(dim=16)
    x = torch.randn(4, 16) * 5.0
    out = rms(x)
    assert out.shape == (4, 16)
    # Root mean square across feature dim should be ~1
    rms_val = torch.sqrt(out.pow(2).mean(dim=-1))
    assert torch.allclose(rms_val, torch.ones(4), atol=1e-3)
