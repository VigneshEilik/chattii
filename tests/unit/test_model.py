import torch

from engine.transformer.config import ModelMathConfig
from engine.transformer.model import ChatiiModel
from engine.transformer.validator import TransformerValidator


def test_chatii_model_forward() -> None:
    config = ModelMathConfig(
        activation="swiglu",
        normalization="rmsnorm",
        d_model=64,
        n_heads=4,
        d_ff=256,
        num_layers=2,
        vocab_size=1000,
    )

    model = ChatiiModel(config)

    # Validate parameters sum
    p_count = model.num_parameters()
    assert p_count > 0

    # Validate summary info
    summ = model.summary()
    assert "ChatiiModel Architecture Summary" in summ

    # Forward check
    x = torch.randint(0, 1000, (2, 10))
    logits = model(x)
    assert logits.shape == (2, 10, 1000)

    # Validate model
    ok, errors = TransformerValidator.validate_model(model, x, logits)
    assert ok
    assert len(errors) == 0


def test_compile_freeze_unfreeze() -> None:
    config = ModelMathConfig(
        activation="swiglu",
        normalization="rmsnorm",
        d_model=64,
        n_heads=4,
        d_ff=256,
        num_layers=2,
        vocab_size=1000,
    )
    model = ChatiiModel(config)

    model.compile()

    # Freeze params
    model.freeze()
    for p in model.parameters():
        assert not p.requires_grad

    # Unfreeze params
    model.unfreeze()
    for p in model.parameters():
        assert p.requires_grad
