import torch

from engine.inference.sampling.typical import TypicalFilter


def test_typical_filter():
    filt = TypicalFilter(mass=0.9)
    logits = torch.tensor([1.0, 2.0, 3.0, 4.0])
    filtered = filt(logits)
    assert filtered.shape == (4,)
