import torch
import torch.nn as nn

from engine.inference.config import GenerationConfig
from engine.inference.runtime.generation_loop import AutoregressiveGenerationLoop


class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.param = nn.Parameter(torch.zeros(1))

    def forward(self, input_ids, cache=None, use_cache=True):
        b, t = input_ids.shape
        return torch.randn(b, t, 10, device=input_ids.device)


def test_autoregressive_generation_loop():
    model = MockModel()
    config = GenerationConfig(config_dict={"max_new_tokens": 4, "do_sample": False})
    tokens = list(AutoregressiveGenerationLoop.generate(model, [1, 2], config))
    assert len(tokens) == 4
