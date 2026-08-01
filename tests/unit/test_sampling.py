from engine.inference.sampling.factory import SamplingFactory
from engine.inference.sampling.registry import SamplingRegistry
from engine.inference.sampling.temperature import TemperatureFilter


def test_sampling_registry():
    # Verify temperature is registered
    cls = SamplingRegistry.get("temperature")
    assert cls == TemperatureFilter


def test_sampling_factory():
    filt = SamplingFactory.create("temperature", temperature=0.5)
    assert isinstance(filt, TemperatureFilter)
    assert filt.temperature == 0.5
