from engine.inference.config import GenerationConfig
from engine.inference.service import InferenceService


def test_batch_generation_integration():
    service = InferenceService()
    service.load_model()
    cfg = GenerationConfig(config_dict={"max_new_tokens": 4})
    prompts = ["explain rope", "what is self attention"]
    responses = service.batch_generate(prompts, config=cfg)
    assert len(responses) == 2
    assert responses[0].generated_tokens > 0
    assert responses[1].generated_tokens > 0
