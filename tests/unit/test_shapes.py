import pytest
import torch

from engine.transformer.config import ModelMathConfig
from engine.transformer.model import ChatiiModel


@pytest.mark.parametrize("batch_size", [1, 2, 4])
@pytest.mark.parametrize("seq_len", [8, 32, 128])
def test_various_shapes(batch_size: int, seq_len: int) -> None:
    config = ModelMathConfig(vocab_size=1000, d_model=64, n_heads=4, num_layers=2)
    model = ChatiiModel(config)
    x = torch.randint(0, 1000, (batch_size, seq_len))
    logits = model(x)
    assert logits.shape == (batch_size, seq_len, 1000)
