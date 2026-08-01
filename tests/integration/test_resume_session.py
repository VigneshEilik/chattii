from engine.inference.config import GenerationConfig
from engine.inference.service import InferenceService


def test_resume_session_integration():
    service = InferenceService()
    service.load_model()
    cfg = GenerationConfig(config_dict={"max_new_tokens": 4})
    session = service.new_session(session_id="resume_test", is_chat=True)

    # First turn
    session.add_message("user", "Hello")
    service.chat(session, config=cfg)
    assert len(session.messages) == 2

    # Second turn
    session.add_message("user", "How are you?")
    service.chat(session, config=cfg)
    assert len(session.messages) == 4
