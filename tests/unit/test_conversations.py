import pytest

from engine.platform.conversations.models import MessageRole
from engine.platform.conversations.repository import ConversationRepository


def test_conversation_creation():
    repo = ConversationRepository()
    conv = repo.create_conversation("usr-1", "Test Title", "chatii-v1")
    assert conv.id.startswith("conv-")
    assert conv.title == "Test Title"
    assert conv.user_id == "usr-1"


def test_get_conversation():
    repo = ConversationRepository()
    conv = repo.create_conversation("usr-2", "Chat 2")
    fetched = repo.get_conversation(conv.id)
    assert fetched == conv


def test_list_user_conversations():
    repo = ConversationRepository()
    conv1 = repo.create_conversation("usr-3", "Chat A")
    conv2 = repo.create_conversation("usr-3", "Chat B")
    repo.create_conversation("usr-4", "Other Chat")

    user_convs = repo.list_user_conversations("usr-3")
    assert len(user_convs) == 2
    assert {c.id for c in user_convs} == {conv1.id, conv2.id}


def test_rename_conversation():
    repo = ConversationRepository()
    conv = repo.create_conversation("usr-5", "Old Title")
    renamed = repo.rename_conversation(conv.id, "New Title")
    assert renamed is not None
    assert renamed.title == "New Title"


def test_delete_conversation():
    repo = ConversationRepository()
    conv = repo.create_conversation("usr-6", "To Delete")
    assert repo.delete_conversation(conv.id) is True
    assert repo.get_conversation(conv.id) is None


def test_add_message_success():
    repo = ConversationRepository()
    conv = repo.create_conversation("usr-7", "Message Chat")
    msg = repo.add_message(conv.id, MessageRole.USER, "Hello model")
    assert msg.id.startswith("msg-")
    assert msg.role == MessageRole.USER
    assert msg.content == "Hello model"
    assert conv.message_count == 1


def test_add_message_nonexistent_conversation():
    repo = ConversationRepository()
    with pytest.raises(ValueError):
        repo.add_message("conv-invalid", MessageRole.USER, "Hello")


def test_get_messages():
    repo = ConversationRepository()
    conv = repo.create_conversation("usr-8", "Chat History")
    repo.add_message(conv.id, MessageRole.USER, "Hi")
    repo.add_message(conv.id, MessageRole.ASSISTANT, "Hello user")

    msgs = repo.get_messages(conv.id)
    assert len(msgs) == 2
    assert msgs[0].role == MessageRole.USER
    assert msgs[1].role == MessageRole.ASSISTANT


def test_message_roles():
    repo = ConversationRepository()
    conv = repo.create_conversation("usr-9", "Role Chat")
    m_sys = repo.add_message(conv.id, MessageRole.SYSTEM, "You are helpful.")
    m_tool = repo.add_message(conv.id, MessageRole.TOOL, "Result: 42")
    assert m_sys.role == MessageRole.SYSTEM
    assert m_tool.role == MessageRole.TOOL


def test_conversation_updated_at_on_message():
    repo = ConversationRepository()
    conv = repo.create_conversation("usr-10", "Timing Chat")
    t0 = conv.updated_at
    repo.add_message(conv.id, MessageRole.USER, "Ping")
    assert conv.updated_at >= t0


def test_get_messages_empty():
    repo = ConversationRepository()
    assert repo.get_messages("nonexistent") == []


def test_delete_conversation_clears_messages():
    repo = ConversationRepository()
    conv = repo.create_conversation("usr-11", "Clear Chat")
    repo.add_message(conv.id, MessageRole.USER, "Test")
    repo.delete_conversation(conv.id)
    assert repo.get_messages(conv.id) == []
