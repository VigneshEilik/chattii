import torch

from engine.inference.sampling.topp import TopPFilter


def test_topp_filter():
    filt = TopPFilter(p=0.9)
    # Give clear high values for top 2 values
    logits = torch.tensor([10.0, 10.0, 1.0, 0.1])
    filtered = filt(logits)
    assert filtered[0] == 10.0
    assert filtered[1] == 10.0
    assert filtered[2] == float("-inf")
    assert filtered[3] == float("-inf")
