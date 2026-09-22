from app.prompts.prompt_builder import PromptBuilder
from app.schemas.retrieval import RetrievedChunk


def _chunk(text, score=0.9, title="Doc.pdf", heading="Heading", **kw):
    return RetrievedChunk(
        score=score,
        document_id=kw.get("document_id", "doc-1"),
        chunk_id=kw.get("chunk_id", "c1"),
        heading=heading,
        title=title,
        section=kw.get("section"),
        page=kw.get("page"),
        text=text,
    )


class TestNoContext:

    def test_no_chunks_asks_model_to_say_it_cant_find_it(self):
        system, user = PromptBuilder.build(
            question="What is the capital of France?",
            retrieved_chunks=[],
        )

        assert "OmniDoc AI" in system
        assert PromptBuilder.NO_CONTEXT_MESSAGE in user
        assert "capital of France" in user


class TestSystemUserSplit:

    def test_returns_separate_system_and_user_messages(self):
        chunks = [_chunk("Some retrieved content.")]

        system, user = PromptBuilder.build(
            question="What does it say?",
            retrieved_chunks=chunks,
        )

        # The system prompt must be the instructions, not the
        # document content or question.
        assert "Rules" in system
        assert "Some retrieved content." not in system

        # The user message must carry the actual context/question,
        # not the instruction rules baked in.
        assert "Some retrieved content." in user
        assert "What does it say?" in user

    def test_no_leading_whitespace_indentation_bug(self):
        """
        Regression test: the original prompt builder baked 4-space
        indentation into every line via an indented f-string.
        """
        chunks = [_chunk("content")]

        _, user = PromptBuilder.build("question", chunks)

        for line in user.splitlines():
            assert not line.startswith("    "), (
                f"line has leftover indentation: {line!r}"
            )


class TestContextBudget:

    def test_fits_under_budget_keeps_all_chunks(self):
        chunks = [_chunk("short " * 5) for _ in range(3)]

        _, user = PromptBuilder.build("q", chunks)

        assert user.count("Context ") == 3

    def test_oversized_context_is_trimmed_not_sent_raw(
        self, monkeypatch
    ):
        import app.prompts.prompt_builder as pb

        # Force a tiny budget so we can assert trimming happens
        # deterministically without depending on real settings.
        monkeypatch.setattr(pb.settings, "LLM_NUM_CTX", 300)
        monkeypatch.setattr(
            pb, "RESERVED_TOKENS_FOR_SYSTEM_AND_ANSWER", 0
        )

        big_chunks = [_chunk("x" * 500) for _ in range(5)]

        _, user = PromptBuilder.build("q", big_chunks)

        # Must keep at least one chunk and must not include all five
        # full-size blobs verbatim.
        assert user.count("Context ") < 5
        assert user.count("Context ") >= 1

    def test_always_keeps_at_least_one_chunk_even_if_oversized(
        self, monkeypatch
    ):
        import app.prompts.prompt_builder as pb

        monkeypatch.setattr(pb.settings, "LLM_NUM_CTX", 10)
        monkeypatch.setattr(
            pb, "RESERVED_TOKENS_FOR_SYSTEM_AND_ANSWER", 0
        )

        chunks = [_chunk("x" * 10000)]

        _, user = PromptBuilder.build("q", chunks)

        assert user.count("Context ") == 1
