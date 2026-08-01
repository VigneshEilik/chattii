import torch
import torch.nn as nn

from engine.inference.runtime.token_generator import SingleTokenGenerator


class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.param = nn.Parameter(torch.zeros(1))

    def forward(self, input_ids, cache=None, use_cache=True):
        # Return logits of shape (batch, seq, vocab=10)
        b, t = input_ids.shape
        return torch.randn(b, t, 10, device=input_ids.device)


def test_single_token_generator():
    model = MockModel()
    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    logits = SingleTokenGenerator.step(model, input_ids)
    assert logits.shape == (1, 10)
