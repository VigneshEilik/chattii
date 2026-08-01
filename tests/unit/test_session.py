from engine.inference.session import ChatSession, GenerationSession


def test_generation_session():
    session = GenerationSession()
    assert session.session_id is not None
    assert session.cancelled is False
    session.cancel()
    assert session.cancelled is True


def test_chat_session():
    session = ChatSession()
    session.add_message("user", "hello")
    session.add_message("assistant", "hi")
    assert len(session.messages) == 2
    session.reset()
    assert len(session.messages) == 0
