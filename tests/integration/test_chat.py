from engine.inference.config import GenerationConfig
from engine.inference.service import InferenceService


def test_chat_integration():
    service = InferenceService()
    service.load_model()
    session = service.new_session(is_chat=True)
    session.add_message("user", "Hello there")

    cfg = GenerationConfig(config_dict={"max_new_tokens": 4})
    resp = service.chat(session, config=cfg)
    assert len(session.messages) == 2
    assert resp.generated_tokens > 0
