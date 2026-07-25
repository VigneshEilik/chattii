import torch

from engine.transformer.position.alibi import ALiBiPositionalBias
from engine.transformer.position.rope import RotaryPositionalEmbedding
from engine.transformer.position.sinusoidal import SinusoidalPositionalEncoding


def test_sinusoidal_positional_encoding() -> None:
    pe = SinusoidalPositionalEncoding(d_model=32, max_len=100)
    x = torch.zeros(2, 10, 32)
    out = pe(x)
    assert out.shape == (2, 10, 32)
    assert not torch.equal(out[0, 0], out[0, 1])


def test_rope_positional_embedding() -> None:
    rope = RotaryPositionalEmbedding(dim=16, max_seq_len=64)
    xq = torch.randn(2, 10, 4, 16)
    xk = torch.randn(2, 10, 4, 16)
    out_q, out_k = rope(xq, xk)
    assert out_q.shape == (2, 10, 4, 16)
    assert out_k.shape == (2, 10, 4, 16)


def test_alibi_positional_bias() -> None:
    alibi = ALiBiPositionalBias(num_heads=8)
    bias = alibi(seq_len=16)
    assert bias.shape == (1, 8, 16, 16)
    assert (bias <= 0.0).all()
