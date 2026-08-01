from engine.inference.config import GenerationConfig
from engine.inference.service import InferenceService


def test_multi_chat_sessions_integration():
    service = InferenceService()
    service.load_model()
    cfg = GenerationConfig(config_dict={"max_new_tokens": 4})

    s1 = service.new_session(session_id="chat_1", is_chat=True)
    s2 = service.new_session(session_id="chat_2", is_chat=True)

    s1.add_message("user", "Session 1 user prompt")
    s2.add_message("user", "Session 2 user prompt")

    r1 = service.chat(s1, config=cfg)
    r2 = service.chat(s2, config=cfg)

    assert len(s1.messages) == 2
    assert len(s2.messages) == 2
    assert r1.generated_tokens > 0
    assert r2.generated_tokens > 0
