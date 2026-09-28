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
        system, user, contexts = PromptBuilder.build(
            question="What is the capital of France?",
            retrieved_chunks=[],
        )

        assert "OmniDoc AI" in system
        assert PromptBuilder.NO_CONTEXT_MESSAGE in user
        assert "capital of France" in user
        assert contexts == []


class TestSystemUserSplit:

    def test_returns_separate_system_and_user_messages(self):
        chunks = [_chunk("Some retrieved content.")]

        system, user, contexts = PromptBuilder.build(
            question="What does it say?",
            retrieved_chunks=chunks,
        )

        # The system prompt must be the instructions, not the
        # document content or question.
        assert "Other rules" in system
        assert "Some retrieved content." not in system

        # The user message must carry the actual context/question,
        # not the instruction rules baked in.
        assert "Some retrieved content." in user
        assert "What does it say?" in user
        assert contexts == chunks

    def test_no_leading_whitespace_indentation_bug(self):
        """
        Regression test: the original prompt builder baked 4-space
        indentation into every line via an indented f-string.
        """
        chunks = [_chunk("content")]

        _, user, _contexts = PromptBuilder.build("question", chunks)

        for line in user.splitlines():
            assert not line.startswith("    "), (
                f"line has leftover indentation: {line!r}"
            )


class TestCitationInstruction:

    def test_system_prompt_instructs_bracket_citations(self):
        system, _, _ = PromptBuilder.build("q", [_chunk("content")])

        assert "[1]" in system

    def test_user_message_ends_with_citation_reminder(self):
        """
        Small models follow instructions placed close to where they
        start generating better than ones buried earlier -- this
        reinforcement is deliberate, not redundant with the system
        prompt's citation rule.
        """
        _, user, _ = PromptBuilder.build("q", [_chunk("content")])

        assert user.rstrip().endswith(PromptBuilder.CITATION_REMINDER)

    def test_context_numbering_matches_returned_contexts_order(self):
        """
        A caller builds a citation-aligned sources list by enumerating
        `contexts` 1-indexed -- this must exactly match the "Context
        N" numbers the model actually sees in the prompt.
        """
        chunks = [
            _chunk("first", chunk_id="c1"),
            _chunk("second", chunk_id="c2"),
            _chunk("third", chunk_id="c3"),
        ]

        _, user, contexts = PromptBuilder.build("q", chunks)

        assert [c.chunk_id for c in contexts] == ["c1", "c2", "c3"]
        assert "Context 1" in user
        assert "Context 2" in user
        assert "Context 3" in user


class TestContextBudget:

    def test_fits_under_budget_keeps_all_chunks(self):
        chunks = [_chunk("short " * 5) for _ in range(3)]

        _, user, contexts = PromptBuilder.build("q", chunks)

        assert user.count("Context ") == 3
        assert len(contexts) == 3

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

        _, user, contexts = PromptBuilder.build("q", big_chunks)

        # Must keep at least one chunk and must not include all five
        # full-size blobs verbatim.
        assert user.count("Context ") < 5
        assert user.count("Context ") >= 1
        # contexts must reflect exactly what's in the prompt, not the
        # original untrimmed list.
        assert len(contexts) == user.count("Context ")

    def test_always_keeps_at_least_one_chunk_even_if_oversized(
        self, monkeypatch
    ):
        import app.prompts.prompt_builder as pb

        monkeypatch.setattr(pb.settings, "LLM_NUM_CTX", 10)
        monkeypatch.setattr(
            pb, "RESERVED_TOKENS_FOR_SYSTEM_AND_ANSWER", 0
        )

        chunks = [_chunk("x" * 10000)]

        _, user, contexts = PromptBuilder.build("q", chunks)

        assert user.count("Context ") == 1
        assert len(contexts) == 1
