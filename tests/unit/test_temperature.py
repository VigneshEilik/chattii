import torch

from engine.inference.sampling.temperature import TemperatureFilter


def test_temperature_filter():
    filt = TemperatureFilter(temperature=0.5)
    logits = torch.tensor([1.0, 2.0, 3.0])
    filtered = filt(logits)
    assert torch.allclose(filtered, torch.tensor([2.0, 4.0, 6.0]))
