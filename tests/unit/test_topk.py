import torch

from engine.inference.sampling.topk import TopKFilter


def test_topk_filter():
    filt = TopKFilter(k=2)
    logits = torch.tensor([1.0, 2.0, 3.0, 4.0])
    filtered = filt(logits)
    assert filtered[0] == float("-inf")
    assert filtered[1] == float("-inf")
    assert filtered[2] == 3.0
    assert filtered[3] == 4.0
