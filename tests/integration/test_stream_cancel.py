from engine.inference.config import GenerationConfig
from engine.inference.service import InferenceService


def test_stream_cancel_integration():
    service = InferenceService()
    service.load_model()
    cfg = GenerationConfig(config_dict={"max_new_tokens": 10})
    session = service.new_session(session_id="cancel_session", is_chat=False)

    from engine.inference.request import GenerationRequest

    req = GenerationRequest(
        prompt="hello world", config=cfg, session_id="cancel_session"
    )
    tokens = []
    for token in service.stream(req):
        tokens.append(token)
        # Cancel after first token
        session.cancel()
    # Verify that stream was successfully interrupted
    assert len(tokens) <= 2
