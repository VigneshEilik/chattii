import torch

from engine.transformer.blocks.decoder import DecoderBlock


def test_decoder_block() -> None:
    d_model = 64
    num_heads = 4
    d_ff = 256
    layer = DecoderBlock(
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        activation="swiglu",
        normalization="rmsnorm",
    )

    x = torch.randn(2, 10, d_model)
    out = layer(x)
    assert out.shape == (2, 10, d_model)
