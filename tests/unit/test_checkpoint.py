import tempfile

import torch

from engine.transformer.checkpoint.converter import CheckpointConverter
from engine.transformer.config import ModelMathConfig
from engine.transformer.health import TransformerHealthChecker
from engine.transformer.model import ChatiiModel


def test_checkpoint_saving_and_loading() -> None:
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
    x = torch.randint(0, 1000, (1, 10))
    original_logits = model(x)

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Save checkpoint
        model.save(tmp_dir)

        # Audit directory health
        ok, errors = TransformerHealthChecker.check_health(tmp_dir)
        assert ok
        assert len(errors) == 0

        # Load into new model
        new_model = ChatiiModel(config)
        new_model.load(tmp_dir)

        # Validate equivalence
        loaded_logits = new_model(x)
        assert torch.allclose(original_logits, loaded_logits, atol=1e-5)


def test_checkpoint_converter() -> None:
    state_dict = {"layers.0.attn.q_proj.weight": torch.ones(10)}
    flat = CheckpointConverter.to_flat(state_dict)
    assert "layers__0__attn__q_proj__weight" in flat

    restored = CheckpointConverter.from_flat(flat)
    assert "layers.0.attn.q_proj.weight" in restored
