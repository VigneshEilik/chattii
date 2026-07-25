import torch

from engine.transformer.mask.causal import CausalMask
from engine.transformer.mask.combined import CombinedMask
from engine.transformer.mask.padding import PaddingMask


def test_causal_mask() -> None:
    mask = CausalMask.create(seq_len=4, additive=True)
    assert mask.shape == (4, 4)
    assert mask[0, 0] == 0.0
    assert mask[0, 1] == float("-inf")


def test_padding_mask() -> None:
    ids = torch.tensor([[1, 2, 0, 0], [5, 0, 0, 0]])
    mask = PaddingMask.create(ids, pad_id=0, additive=True)
    assert mask.shape == (2, 1, 1, 4)
    assert mask[0, 0, 0, 0] == 0.0
    assert mask[0, 0, 0, 2] == float("-inf")


def test_combined_mask() -> None:
    ids = torch.tensor([[1, 2, 0], [5, 0, 0]])
    mask = CombinedMask.create(ids, pad_id=0)
    assert mask.shape == (2, 1, 3, 3)
    assert mask[0, 0, 0, 0] == 0.0
    assert mask[0, 0, 0, 1] == float("-inf")
