import pytest

from app.services.conversation_service import ConversationService


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    from app.config.settings import settings
    from app.db.database import init_db

    monkeypatch.setattr(
        settings, "DATABASE_PATH", str(tmp_path / "test.db")
    )
    init_db()


class TestCreateAndExists:

    def test_create_returns_a_usable_id(self):
        conversation_id = ConversationService.create_conversation()

        assert conversation_id
        assert ConversationService.conversation_exists(conversation_id)

    def test_unknown_id_does_not_exist(self):
        assert not ConversationService.conversation_exists("nope")

    def test_each_create_call_is_distinct(self):
        a = ConversationService.create_conversation()
        b = ConversationService.create_conversation()

        assert a != b


class TestGetOrCreate:

    def test_none_creates_a_new_conversation(self):
        conversation_id = ConversationService.get_or_create(None)

        assert ConversationService.conversation_exists(conversation_id)

    def test_existing_id_is_reused(self):
        original = ConversationService.create_conversation()

        result = ConversationService.get_or_create(original)

        assert result == original

    def test_unknown_id_falls_back_to_a_new_conversation(self):
        result = ConversationService.get_or_create("does-not-exist")

        assert result != "does-not-exist"
        assert ConversationService.conversation_exists(result)


class TestMessages:

    def test_add_and_retrieve_a_message(self):
        conversation_id = ConversationService.create_conversation()

        ConversationService.add_message(
            conversation_id, role="user", content="hello"
        )

        history = ConversationService.get_full_history(conversation_id)
        assert len(history) == 1
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "hello"

    def test_messages_ordered_oldest_first(self):
        conversation_id = ConversationService.create_conversation()

        ConversationService.add_message(
            conversation_id, "user", "first"
        )
        ConversationService.add_message(
            conversation_id, "assistant", "second"
        )
        ConversationService.add_message(
            conversation_id, "user", "third"
        )

        history = ConversationService.get_full_history(conversation_id)
        assert [m["content"] for m in history] == [
            "first",
            "second",
            "third",
        ]

    def test_sources_round_trip_through_json(self):
        conversation_id = ConversationService.create_conversation()
        sources = [
            {"index": 1, "document": "doc.pdf", "heading": "H",
             "page": None, "cited": True}
        ]

        ConversationService.add_message(
            conversation_id, "assistant", "answer", sources=sources
        )

        history = ConversationService.get_full_history(conversation_id)
        assert history[0]["sources"] == sources

    def test_message_with_no_sources_stores_none(self):
        conversation_id = ConversationService.create_conversation()

        ConversationService.add_message(
            conversation_id, "user", "question"
        )

        history = ConversationService.get_full_history(conversation_id)
        assert history[0]["sources"] is None

    def test_messages_scoped_to_their_own_conversation(self):
        conv_a = ConversationService.create_conversation()
        conv_b = ConversationService.create_conversation()

        ConversationService.add_message(conv_a, "user", "in A")
        ConversationService.add_message(conv_b, "user", "in B")

        history_a = ConversationService.get_full_history(conv_a)
        assert [m["content"] for m in history_a] == ["in A"]


class TestRecentMessagesForPrompting:

    def test_returns_role_and_content_only(self):
        conversation_id = ConversationService.create_conversation()
        ConversationService.add_message(
            conversation_id, "user", "hello"
        )

        recent = ConversationService.get_recent_messages(conversation_id)

        assert recent == [{"role": "user", "content": "hello"}]

    def test_respects_the_limit_keeping_most_recent(self):
        conversation_id = ConversationService.create_conversation()
        for i in range(10):
            ConversationService.add_message(
                conversation_id, "user", f"message {i}"
            )

        recent = ConversationService.get_recent_messages(
            conversation_id, limit=4
        )

        assert [m["content"] for m in recent] == [
            "message 6",
            "message 7",
            "message 8",
            "message 9",
        ]

    def test_still_oldest_first_within_the_limited_window(self):
        conversation_id = ConversationService.create_conversation()
        ConversationService.add_message(conversation_id, "user", "a")
        ConversationService.add_message(conversation_id, "assistant", "b")
        ConversationService.add_message(conversation_id, "user", "c")

        recent = ConversationService.get_recent_messages(
            conversation_id, limit=2
        )

        assert [m["content"] for m in recent] == ["b", "c"]

    def test_empty_conversation_returns_empty_list(self):
        conversation_id = ConversationService.create_conversation()

        assert ConversationService.get_recent_messages(conversation_id) == []
