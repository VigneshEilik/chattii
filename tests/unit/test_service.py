from engine.inference.config import GenerationConfig
from engine.inference.service import InferenceService


def test_inference_service_initialization():
    service = InferenceService()
    service.load_model()
    assert service.model is not None
    assert service.tokenizer is not None
    assert service.engine is not None

    # Test basic generation
    config = GenerationConfig(config_dict={"max_new_tokens": 5})
    resp = service.generate("hello", config=config)
    assert resp.generated_tokens > 0
    assert isinstance(resp.text, str)
