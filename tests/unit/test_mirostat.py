import torch

from engine.inference.sampling.mirostat import MirostatFilter


def test_mirostat_filter():
    filt = MirostatFilter(target_entropy=2.0)
    logits = torch.tensor([1.0, 2.0, 3.0, 4.0])
    filtered = filt(logits)
    assert filtered.shape == (4,)
