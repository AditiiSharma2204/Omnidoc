import numpy as np
import pytest

from app.services.retrieval_service import RetrievalService


@pytest.fixture
def stub_dense(monkeypatch):
    """
    Stubs the dense path to return a fixed, ordered list of
    (document_id, chunk_id) results with fixed scores, without
    touching FAISS/the embedding model.
    """

    def make(results: dict):
        # results: {(doc_id, chunk_id): score}
        def fake_dense_search(cls, query, top_k, document_ids, embedding_query=None):
            from app.schemas.retrieval import RetrievedChunk

            return {
                key: RetrievedChunk(
                    score=score,
                    document_id=key[0],
                    chunk_id=key[1],
                    heading="H",
                    title=f"{key[0]}.pdf",
                    section=None,
                    page=None,
                    text=f"dense text for {key}",
                )
                for key, score in results.items()
            }

        monkeypatch.setattr(
            RetrievalService,
            "_dense_search",
            classmethod(fake_dense_search),
        )

    return make


@pytest.fixture
def stub_bm25(monkeypatch):
    def make(results: dict):
        def fake_bm25_search(cls, query, top_k, document_ids, embedding_query=None):
            from app.schemas.retrieval import RetrievedChunk

            return {
                key: RetrievedChunk(
                    score=score,
                    document_id=key[0],
                    chunk_id=key[1],
                    heading="H",
                    title=f"{key[0]}.pdf",
                    section=None,
                    page=None,
                    text=f"bm25 text for {key}",
                )
                for key, score in results.items()
            }

        monkeypatch.setattr(
            RetrievalService,
            "_bm25_search",
            classmethod(fake_bm25_search),
        )

    return make


class TestHybridFusion:

    def test_doc_ranked_first_in_both_lists_wins(
        self, stub_dense, stub_bm25
    ):
        stub_dense(
            {("doc-a", "c1"): 0.9, ("doc-a", "c2"): 0.5}
        )
        stub_bm25(
            {("doc-a", "c1"): 5.0, ("doc-a", "c2"): 1.0}
        )

        results = RetrievalService.search("q", top_k=5, mode="hybrid", rerank=False)

        assert results[0].chunk_id == "c1"

    def test_doc_missing_from_dense_can_still_surface_via_bm25(
        self, stub_dense, stub_bm25
    ):
        """
        This is the whole point of hybrid retrieval: a chunk dense
        search misses entirely (e.g. a short exact-match query) can
        still be found via lexical overlap.
        """
        stub_dense({("doc-a", "c1"): 0.9})  # c2 not found by dense at all
        stub_bm25(
            {("doc-a", "c2"): 8.0, ("doc-a", "c1"): 1.0}
        )

        results = RetrievalService.search("q", top_k=5, mode="hybrid", rerank=False)

        chunk_ids = [r.chunk_id for r in results]
        assert "c2" in chunk_ids

    def test_hybrid_result_count_is_union_of_both_lists(
        self, stub_dense, stub_bm25
    ):
        stub_dense({("doc-a", "c1"): 0.9})
        stub_bm25({("doc-a", "c2"): 5.0})

        results = RetrievalService.search("q", top_k=5, mode="hybrid", rerank=False)

        assert {r.chunk_id for r in results} == {"c1", "c2"}

    def test_dense_mode_ignores_bm25_results(
        self, stub_dense, stub_bm25
    ):
        stub_dense({("doc-a", "c1"): 0.9})
        stub_bm25({("doc-a", "c2"): 5.0})

        results = RetrievalService.search("q", top_k=5, mode="dense", rerank=False)

        assert [r.chunk_id for r in results] == ["c1"]

    def test_bm25_mode_ignores_dense_results(
        self, stub_dense, stub_bm25
    ):
        stub_dense({("doc-a", "c1"): 0.9})
        stub_bm25({("doc-a", "c2"): 5.0})

        results = RetrievalService.search("q", top_k=5, mode="bm25", rerank=False)

        assert [r.chunk_id for r in results] == ["c2"]

    def test_unknown_mode_raises(self, stub_dense, stub_bm25):
        stub_dense({})
        stub_bm25({})

        with pytest.raises(ValueError):
            RetrievalService.search("q", top_k=5, mode="nonsense", rerank=False)

    def test_top_k_is_respected_after_fusion(
        self, stub_dense, stub_bm25
    ):
        stub_dense(
            {
                ("doc-a", "c1"): 0.9,
                ("doc-a", "c2"): 0.8,
                ("doc-a", "c3"): 0.7,
            }
        )
        stub_bm25({})

        results = RetrievalService.search("q", top_k=2, mode="hybrid", rerank=False)

        assert len(results) == 2


class TestReciprocalRankFusion:

    def test_pure_function_matches_hand_computed_scores(self):
        fused = RetrievalService._reciprocal_rank_fusion(
            [
                [("a", "1"), ("a", "2")],  # a/1 rank 1, a/2 rank 2
                [("a", "2"), ("a", "1")],  # a/2 rank 1, a/1 rank 2
            ],
            k=60,
        )

        expected_a1 = 1 / 61 + 1 / 62
        expected_a2 = 1 / 62 + 1 / 61

        assert fused[("a", "1")] == pytest.approx(expected_a1)
        assert fused[("a", "2")] == pytest.approx(expected_a2)
        assert fused[("a", "1")] == pytest.approx(fused[("a", "2")])


class TestQueryRewriteWiring:
    """
    Confirms query_rewrite generates a hypothetical answer and feeds
    it to the DENSE leg's embedding only -- BM25 must always see the
    literal query, never the fabricated hypothetical text.
    """

    @pytest.fixture
    def capture_dense_embedding_query(self, monkeypatch):
        captured = {}

        def fake_dense_search(
            cls, query, top_k, document_ids, embedding_query=None
        ):
            captured["query"] = query
            captured["embedding_query"] = embedding_query
            return {}

        monkeypatch.setattr(
            RetrievalService, "_dense_search", classmethod(fake_dense_search)
        )
        return captured

    @pytest.fixture
    def capture_bm25_query(self, monkeypatch):
        captured = {}

        def fake_bm25_search(cls, query, top_k, document_ids):
            captured["query"] = query
            return {}

        monkeypatch.setattr(
            RetrievalService, "_bm25_search", classmethod(fake_bm25_search)
        )
        return captured

    def test_rewrite_true_passes_hypothetical_to_dense_only(
        self,
        capture_dense_embedding_query,
        capture_bm25_query,
        monkeypatch,
    ):
        import app.services.query_rewrite_service as qrs

        monkeypatch.setattr(
            qrs.QueryRewriteService,
            "generate_hypothetical_answer",
            lambda question: "The person's name is Aditii Sharma.",
        )

        RetrievalService.search(
            "What is this person's name?",
            top_k=5,
            mode="hybrid",
            rerank=False,
            query_rewrite=True,
        )

        assert (
            capture_dense_embedding_query["embedding_query"]
            == "The person's name is Aditii Sharma."
        )
        assert (
            capture_dense_embedding_query["query"]
            == "What is this person's name?"
        )
        assert (
            capture_bm25_query["query"] == "What is this person's name?"
        )

    def test_rewrite_false_does_not_call_llm_or_override_embedding(
        self, capture_dense_embedding_query, capture_bm25_query
    ):
        # No QueryRewriteService stub at all -- if search() tried to
        # call the real thing, this would attempt a real LLM call and
        # fail/hang, so this also proves query_rewrite=False (the
        # default) truly skips it.
        RetrievalService.search(
            "What is this person's name?",
            top_k=5,
            mode="hybrid",
            rerank=False,
            query_rewrite=False,
        )

        assert capture_dense_embedding_query["embedding_query"] is None

    def test_rewrite_defaults_to_settings_flag(
        self, capture_dense_embedding_query, capture_bm25_query, monkeypatch
    ):
        from app.config.settings import settings

        monkeypatch.setattr(settings, "QUERY_REWRITE_ENABLED", False)

        RetrievalService.search(
            "q", top_k=5, mode="hybrid", rerank=False
        )

        assert capture_dense_embedding_query["embedding_query"] is None
