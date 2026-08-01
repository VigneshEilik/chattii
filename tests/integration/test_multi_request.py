from engine.inference.config import GenerationConfig
from engine.inference.service import InferenceService


def test_multi_request_integration():
    service = InferenceService()
    service.load_model()
    cfg = GenerationConfig(config_dict={"max_new_tokens": 4})

    # Sequential requests
    r1 = service.generate("hello", config=cfg)
    r2 = service.generate("world", config=cfg)
    assert r1.generated_tokens > 0
    assert r2.generated_tokens > 0
