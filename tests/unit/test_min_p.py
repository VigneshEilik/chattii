import torch

from engine.inference.sampling.min_p import MinPFilter


def test_min_p_filter():
    filt = MinPFilter(min_p=0.05)
    logits = torch.tensor([1.0, 2.0, 3.0, 4.0])
    filtered = filt(logits)
    assert filtered.shape == (4,)
