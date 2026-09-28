import pytest

from app.schemas.retrieval import RetrievedChunk
from app.services.reranker_service import RerankerService


class _StubCrossEncoder:
    """
    A fake CrossEncoder that scores each (query, text) pair by how
    many words they share -- deterministic, no real model needed.
    """

    def predict(self, pairs):
        scores = []
        for query, text in pairs:
            query_words = set(query.lower().split())
            text_words = set(text.lower().split())
            scores.append(float(len(query_words & text_words)))
        return scores


def _chunk(chunk_id, text, score=0.0):
    return RetrievedChunk(
        score=score,
        document_id="doc-a",
        chunk_id=chunk_id,
        heading="H",
        title="doc-a.pdf",
        section=None,
        page=None,
        text=text,
    )


class TestRerankerService:

    def test_empty_candidates_returns_empty(self, monkeypatch):
        import app.services.reranker_service as rs

        monkeypatch.setattr(
            rs.RerankerModel, "get_model", lambda: _StubCrossEncoder()
        )

        assert RerankerService.rerank("query", [], top_k=5) == []

    def test_reorders_by_cross_encoder_score(self, monkeypatch):
        import app.services.reranker_service as rs

        monkeypatch.setattr(
            rs.RerankerModel, "get_model", lambda: _StubCrossEncoder()
        )

        # "python sql" shares 2 words with c2, 0 with c1 -- a
        # cross-encoder should rank c2 first even though c1 has a
        # higher pre-rerank score (simulating dense/bm25 disagreeing
        # with what's actually relevant).
        candidates = [
            _chunk("c1", "completely unrelated text here", score=0.99),
            _chunk("c2", "skills include python and sql", score=0.1),
        ]

        results = RerankerService.rerank(
            "python sql skills", candidates, top_k=5
        )

        assert results[0].chunk_id == "c2"

    def test_respects_top_k(self, monkeypatch):
        import app.services.reranker_service as rs

        monkeypatch.setattr(
            rs.RerankerModel, "get_model", lambda: _StubCrossEncoder()
        )

        candidates = [
            _chunk("c1", "python"),
            _chunk("c2", "python sql"),
            _chunk("c3", "python sql c++"),
        ]

        results = RerankerService.rerank(
            "python sql c++", candidates, top_k=2
        )

        assert len(results) == 2
        assert results[0].chunk_id == "c3"  # most word overlap

    def test_scores_are_overwritten_with_reranker_scores(
        self, monkeypatch
    ):
        import app.services.reranker_service as rs

        monkeypatch.setattr(
            rs.RerankerModel, "get_model", lambda: _StubCrossEncoder()
        )

        candidates = [_chunk("c1", "python sql", score=0.5)]

        results = RerankerService.rerank("python sql", candidates, top_k=5)

        # Stub scores by word overlap: "python sql" vs "python sql" -> 2
        assert results[0].score == 2.0


class TestSearchRerankWiring:
    """
    Confirms RetrievalService.search() actually calls into reranking
    when enabled, without loading a real model.
    """

    def test_rerank_true_calls_reranker_and_overrides_order(
        self, monkeypatch
    ):
        from app.schemas.retrieval import RetrievedChunk as RC
        from app.services.retrieval_service import RetrievalService

        def fake_retrieve(cls, query, top_k, document_ids, mode, embedding_query=None):
            return [
                RC(
                    score=0.9,
                    document_id="doc-a",
                    chunk_id="c1",
                    heading="H",
                    title="doc-a.pdf",
                    section=None,
                    page=None,
                    text="irrelevant",
                ),
                RC(
                    score=0.1,
                    document_id="doc-a",
                    chunk_id="c2",
                    heading="H",
                    title="doc-a.pdf",
                    section=None,
                    page=None,
                    text="python sql",
                ),
            ]

        monkeypatch.setattr(
            RetrievalService, "_retrieve", classmethod(fake_retrieve)
        )

        import app.services.reranker_service as rrs

        monkeypatch.setattr(
            rrs.RerankerModel, "get_model", lambda: _StubCrossEncoder()
        )

        results = RetrievalService.search(
            "python sql", top_k=5, mode="hybrid", rerank=True
        )

        # Pre-rerank order was c1, c2; the stub cross-encoder should
        # flip it because c2 actually overlaps with the query.
        assert results[0].chunk_id == "c2"

    def test_rerank_false_skips_reranker_entirely(self, monkeypatch):
        from app.schemas.retrieval import RetrievedChunk as RC
        from app.services.retrieval_service import RetrievalService

        def fake_retrieve(cls, query, top_k, document_ids, mode, embedding_query=None):
            return [
                RC(
                    score=0.9,
                    document_id="doc-a",
                    chunk_id="c1",
                    heading="H",
                    title="doc-a.pdf",
                    section=None,
                    page=None,
                    text="irrelevant",
                ),
            ]

        monkeypatch.setattr(
            RetrievalService, "_retrieve", classmethod(fake_retrieve)
        )

        # No RerankerModel stub -- if search() tried to rerank, this
        # would attempt to load the real model and fail/hang in a
        # unit test, so this also proves rerank=False truly skips it.
        results = RetrievalService.search(
            "python sql", top_k=5, mode="hybrid", rerank=False
        )

        assert results[0].chunk_id == "c1"
        assert results[0].score == 0.9


def _chunk_scored(chunk_id, score):
    return RetrievedChunk(
        score=score,
        document_id="doc-a",
        chunk_id=chunk_id,
        heading="H",
        title="doc-a.pdf",
        section=None,
        page=None,
        text="text",
    )


class TestScoreThreshold:
    """
    RetrievalService._apply_score_threshold -- the refusal-threshold
    mechanism. Pure logic, no model involved.
    """

    def test_none_threshold_returns_everything_unchanged(self, monkeypatch):
        """
        RERANK_SCORE_THRESHOLD is no longer None by default (calibrated
        from real data on 2026-09-28, see settings.py's comment) -- this
        covers the disabled state explicitly rather than relying on it
        being the default.
        """
        from app.config.settings import settings
        from app.services.retrieval_service import RetrievalService

        monkeypatch.setattr(settings, "RERANK_SCORE_THRESHOLD", None)

        results = [_chunk_scored("c1", -10.0), _chunk_scored("c2", -50.0)]

        filtered = RetrievalService._apply_score_threshold(results)

        assert filtered == results

    def test_drops_results_below_threshold(self, monkeypatch):
        from app.config.settings import settings
        from app.services.retrieval_service import RetrievalService

        monkeypatch.setattr(settings, "RERANK_SCORE_THRESHOLD", -20.0)

        results = [
            _chunk_scored("above", -10.0),
            _chunk_scored("at", -20.0),
            _chunk_scored("below", -30.0),
        ]

        filtered = RetrievalService._apply_score_threshold(results)

        assert [r.chunk_id for r in filtered] == ["above", "at"]

    def test_all_below_threshold_returns_empty(self, monkeypatch):
        from app.config.settings import settings
        from app.services.retrieval_service import RetrievalService

        monkeypatch.setattr(settings, "RERANK_SCORE_THRESHOLD", 0.0)

        results = [_chunk_scored("c1", -5.0), _chunk_scored("c2", -1.0)]

        filtered = RetrievalService._apply_score_threshold(results)

        assert filtered == []

    def test_threshold_only_applies_when_reranking(self, monkeypatch):
        """
        The threshold is calibrated for reranker (cross-encoder)
        scores specifically -- it must not silently filter dense/
        BM25/hybrid-without-rerank results on an incompatible scale.
        """
        from app.config.settings import settings
        from app.schemas.retrieval import RetrievedChunk as RC
        from app.services.retrieval_service import RetrievalService

        # A threshold that would drop everything if it were (wrongly)
        # applied to raw dense/hybrid scores, which are small
        # positive floats, not cross-encoder logits.
        monkeypatch.setattr(settings, "RERANK_SCORE_THRESHOLD", -100.0)

        def fake_retrieve(cls, query, top_k, document_ids, mode, embedding_query=None):
            return [RC(
                score=0.02,  # a plausible RRF-fused hybrid score
                document_id="doc-a",
                chunk_id="c1",
                heading="H",
                title="doc-a.pdf",
                section=None,
                page=None,
                text="text",
            )]

        monkeypatch.setattr(
            RetrievalService, "_retrieve", classmethod(fake_retrieve)
        )

        results = RetrievalService.search(
            "q", top_k=5, mode="hybrid", rerank=False
        )

        # threshold=-100 with score=0.02 would pass anyway, so use a
        # threshold that WOULD filter to prove it's not even consulted
        # on the no-rerank path: score 0.02 < a threshold of 1.0.
        monkeypatch.setattr(settings, "RERANK_SCORE_THRESHOLD", 1.0)

        results = RetrievalService.search(
            "q", top_k=5, mode="hybrid", rerank=False
        )

        assert len(results) == 1  # not filtered -- rerank=False path

    def test_empty_after_filtering_triggers_prompt_builder_refusal(
        self, monkeypatch
    ):
        """
        End-to-end check that an all-filtered retrieval flows into
        the existing "no context" refusal path, not a crash or an
        empty-context prompt sent to the LLM.
        """
        from app.prompts.prompt_builder import PromptBuilder

        system, user, contexts = PromptBuilder.build(
            question="What is this person's name?",
            retrieved_chunks=[],
        )

        assert PromptBuilder.NO_CONTEXT_MESSAGE in user
        assert contexts == []
