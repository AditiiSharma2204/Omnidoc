import pytest

from app.schemas.retrieval import RetrievedChunk
from app.services.chat_service import ChatService


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    from app.config.settings import settings
    from app.db.database import init_db

    monkeypatch.setattr(
        settings, "DATABASE_PATH", str(tmp_path / "test.db")
    )
    init_db()


def _chunk(chunk_id, text="text", heading="H", title="doc.pdf", page=None):
    return RetrievedChunk(
        score=0.9,
        document_id="doc-1",
        chunk_id=chunk_id,
        heading=heading,
        title=title,
        section=None,
        page=page,
        text=text,
    )


class TestChatSourceCitations:
    """
    ChatService.chat() ties the response's `sources` indices to the
    same "Context N" numbering PromptBuilder put in the prompt, and
    flags which ones the model actually cited -- fully mocked, no
    real retrieval or LLM call.
    """

    def _wire(self, monkeypatch, chunks, answer):
        import app.services.chat_service as cs

        monkeypatch.setattr(
            cs.RetrievalService, "search", lambda **kwargs: chunks
        )
        monkeypatch.setattr(
            cs.LLMService,
            "generate",
            lambda system, user, history=None: answer,
        )

    def test_sources_indices_match_context_numbering(self, monkeypatch):
        chunks = [_chunk("c1"), _chunk("c2"), _chunk("c3")]
        self._wire(monkeypatch, chunks, "The answer is here [2].")

        result = ChatService.chat("question")

        assert [s["index"] for s in result["sources"]] == [1, 2, 3]

    def test_cited_flag_reflects_actual_citation_markers(
        self, monkeypatch
    ):
        chunks = [_chunk("c1"), _chunk("c2"), _chunk("c3")]
        self._wire(monkeypatch, chunks, "Supported by [1] and [3].")

        result = ChatService.chat("question")

        cited = {s["index"]: s["cited"] for s in result["sources"]}
        assert cited == {1: True, 2: False, 3: True}

    def test_all_retrieved_contexts_included_even_if_uncited(
        self, monkeypatch
    ):
        """
        Every retrieved context is listed (with cited=False if
        unused), not just the ones the model happened to cite -- so
        the UI can show "available but not used", and no source is
        silently dropped.
        """
        chunks = [_chunk("c1"), _chunk("c2")]
        self._wire(monkeypatch, chunks, "No citations here at all.")

        result = ChatService.chat("question")

        assert len(result["sources"]) == 2
        assert all(not s["cited"] for s in result["sources"])

    def test_no_context_produces_no_sources(self, monkeypatch):
        self._wire(
            monkeypatch, [], "I couldn't find that information in the uploaded documents."
        )

        result = ChatService.chat("question")

        assert result["sources"] == []

    def test_source_fields_carry_document_metadata(self, monkeypatch):
        chunks = [
            _chunk("c1", heading="Skills", title="resume.pdf", page=2)
        ]
        self._wire(monkeypatch, chunks, "Answer [1].")

        result = ChatService.chat("question")

        source = result["sources"][0]
        assert source["document"] == "resume.pdf"
        assert source["heading"] == "Skills"
        assert source["page"] == 2

    def test_out_of_range_citation_does_not_crash(self, monkeypatch):
        """
        If the model hallucinates a citation number that doesn't
        correspond to any retrieved context (e.g. [8] when only 2
        contexts exist), that must not match any real source -- a
        hallucinated citation is visible by simply not appearing in
        `sources`, not a crash.
        """
        chunks = [_chunk("c1"), _chunk("c2")]
        self._wire(monkeypatch, chunks, "Supported by [8].")

        result = ChatService.chat("question")

        assert len(result["sources"]) == 2
        assert all(not s["cited"] for s in result["sources"])


class TestConversationMemory:

    def _wire(self, monkeypatch, chunks, answer, captured_history=None):
        import app.services.chat_service as cs

        monkeypatch.setattr(
            cs.RetrievalService, "search", lambda **kwargs: chunks
        )

        def fake_generate(system, user, history=None):
            if captured_history is not None:
                captured_history.append(history)
            return answer

        monkeypatch.setattr(cs.LLMService, "generate", fake_generate)

    def test_first_turn_returns_a_new_conversation_id(self, monkeypatch):
        self._wire(monkeypatch, [], "answer")

        result = ChatService.chat("question")

        assert result["conversation_id"]

    def test_passing_conversation_id_continues_it(self, monkeypatch):
        self._wire(monkeypatch, [], "answer")

        first = ChatService.chat("question one")
        second = ChatService.chat(
            "question two", conversation_id=first["conversation_id"]
        )

        assert second["conversation_id"] == first["conversation_id"]

    def test_unknown_conversation_id_starts_a_new_one(self, monkeypatch):
        self._wire(monkeypatch, [], "answer")

        result = ChatService.chat(
            "question", conversation_id="does-not-exist"
        )

        assert result["conversation_id"] != "does-not-exist"

    def test_second_turn_sees_first_turn_as_history(self, monkeypatch):
        captured = []
        self._wire(
            monkeypatch,
            [],
            "The person's name is Aditii Sharma.",
            captured_history=captured,
        )

        first = ChatService.chat("What is this person's name?")
        ChatService.chat(
            "What did she study?",
            conversation_id=first["conversation_id"],
        )

        # First call: no prior turns yet.
        assert captured[0] == []
        # Second call: sees the first Q&A as history.
        roles_and_content = [(m["role"], m["content"]) for m in captured[1]]
        assert (
            "user",
            "What is this person's name?",
        ) in roles_and_content
        assert (
            "assistant",
            "The person's name is Aditii Sharma.",
        ) in roles_and_content

    def test_separate_conversations_do_not_share_history(
        self, monkeypatch
    ):
        captured = []
        self._wire(monkeypatch, [], "answer", captured_history=captured)

        conv_a = ChatService.chat("question in conversation A")
        ChatService.chat("question in conversation B")  # fresh conversation
        ChatService.chat(
            "second question in A",
            conversation_id=conv_a["conversation_id"],
        )

        # Third call (second turn of conversation A) must not see
        # conversation B's turn in its history.
        assert not any(
            "conversation B" in m["content"] for m in captured[2]
        )

    def test_conversation_endpoint_returns_full_history(self, monkeypatch):
        self._wire(monkeypatch, [], "answer")

        result = ChatService.chat("question")

        from app.services.conversation_service import ConversationService

        history = ConversationService.get_full_history(
            result["conversation_id"]
        )
        assert [m["role"] for m in history] == ["user", "assistant"]
        assert history[0]["content"] == "question"
        assert history[1]["content"] == "answer"
