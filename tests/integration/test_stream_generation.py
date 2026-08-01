from engine.inference.config import GenerationConfig
from engine.inference.service import InferenceService


def test_stream_generation_integration():
    service = InferenceService()
    service.load_model()

    cfg = GenerationConfig(config_dict={"max_new_tokens": 4})
    tokens = list(service.stream("hello world", config=cfg))
    assert len(tokens) > 0
    assert all(isinstance(t, str) for t in tokens)
