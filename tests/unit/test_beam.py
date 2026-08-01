import torch

from engine.inference.decoding.beam import (
    BeamSearchDecoder,
    GroupBeamSearchDecoder,
)


def test_beam_search_decoder():
    dec = BeamSearchDecoder(beam_width=2)
    logits = torch.tensor([[1.0, 3.0, 2.0, 4.0]])
    res = dec.decode(logits)
    assert res.numel() == 1


def test_group_beam_search_decoder():
    dec = GroupBeamSearchDecoder(num_groups=2)
    logits = torch.tensor([[1.0, 3.0, 2.0, 4.0]])
    res = dec.decode(logits)
    assert res.item() == 3
