from engine.inference.config import GenerationConfig
from engine.inference.service import InferenceService


def test_long_context_integration():
    service = InferenceService()
    service.load_model()
    # Prompt is longer than the standard test max context size of 2048, or we test truncation
    long_prompt = "a" * 3000
    cfg = GenerationConfig(config_dict={"max_new_tokens": 4})
    resp = service.generate(long_prompt, config=cfg)
    assert resp.generated_tokens > 0
