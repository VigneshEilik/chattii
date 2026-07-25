import torch

from engine.transformer.attention.multi_head import MultiHeadSelfAttention
from engine.transformer.attention.validator import AttentionValidator


def test_multi_head_attention_block() -> None:
    d_model = 64
    num_heads = 4
    layer = MultiHeadSelfAttention(d_model=d_model, num_heads=num_heads)

    x = torch.randn(2, 10, d_model)
    out = layer(x)
    assert out.shape == (2, 10, d_model)

    # Validate output stability
    ok, errors = AttentionValidator.validate_attention(layer, x, out)
    assert ok
    assert len(errors) == 0
