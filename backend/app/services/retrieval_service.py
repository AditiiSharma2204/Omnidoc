import json
from pathlib import Path

import numpy as np

from app.config.settings import settings
from app.embeddings.model import EmbeddingModel
from app.schemas.retrieval import RetrievedChunk
from app.vectorstore.bm25_service import BM25Service
from app.vectorstore.faiss_service import FAISSService


class RetrievalService:
    """
    Retrieval over the indexed corpus.

    Supports three modes (settings.RETRIEVAL_MODE): "dense" (FAISS/
    BGE-M3 only, the original behavior), "bm25" (lexical-only, for
    comparison), and "hybrid" (both, fused with Reciprocal Rank
    Fusion -- the default). Measured on the seed eval set: hybrid
    matches or beats dense-only at every k, with no per-question
    regressions (see backend/eval/README.md for the full ablation).
    It is NOT a universal fix, though -- a query that shares no
    vocabulary with its target chunk (e.g. "what is this person's
    name?" when the document never uses the word "name") defeats
    BM25 exactly as it defeats dense embedding similarity; that
    specific case is still a miss in every mode as of this writing.
    """

    # ==========================================================
    # Embedding / chunk loading
    # ==========================================================

    @staticmethod
    def embed_query(text: str) -> np.ndarray:
        model = EmbeddingModel.get_model()
        embedding = model.encode(
            [text],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embedding.astype(np.float32)

    @staticmethod
    def load_chunk(document_id: str, chunk_index: int) -> dict:
        """
        Load a single chunk from a processed document.
        """
        chunks_path = (
            Path(settings.DOCUMENTS_DIR) / document_id / "chunks.json"
        )

        if not chunks_path.exists():
            raise FileNotFoundError(
                f"Chunks not found for document '{document_id}'."
            )

        with chunks_path.open("r", encoding="utf-8") as f:
            chunks = json.load(f)

        if chunk_index >= len(chunks):
            raise IndexError(f"Chunk index {chunk_index} out of range.")

        return chunks[chunk_index]

    @classmethod
    def _to_retrieved_chunk(
        cls, metadata: dict, score: float
    ) -> RetrievedChunk:
        chunk = cls.load_chunk(
            document_id=metadata["document_id"],
            chunk_index=metadata["chunk_index"],
        )
        return RetrievedChunk(
            score=score,
            document_id=metadata["document_id"],
            chunk_id=metadata["chunk_id"],
            section=chunk["metadata"].get("section"),
            heading=metadata.get("heading"),
            title=metadata.get("title"),
            page=metadata.get("page"),
            text=chunk["text"],
        )

    @staticmethod
    def _fetch_depth(
        top_k: int, document_ids: list[str] | None
    ) -> int:
        """
        How many candidates to pull from an index before filtering/
        fusing. Filtering to a specific document can throw away most
        candidates, so search deeper in that case; capped so a large
        corpus doesn't blow up latency on every filtered query.
        """
        fetch_k = top_k * 3
        if document_ids:
            fetch_k = min(max(fetch_k, top_k * 10), 200)
        return fetch_k

    # ==========================================================
    # Dense (FAISS)
    # ==========================================================

    @classmethod
    def _dense_search(
        cls,
        query: str,
        top_k: int,
        document_ids: list[str] | None,
        embedding_query: str | None = None,
    ) -> dict[tuple[str, str], RetrievedChunk]:
        """
        Returns {(document_id, chunk_id): RetrievedChunk}, deduped
        (near-duplicate FAISS hits keep the highest score).

        `embedding_query`, when given, is embedded INSTEAD of `query`
        -- used for HyDE-style query rewriting, where a hypothetical
        answer passage (not the literal question) produces a better
        embedding for similarity search. `query` itself is unused in
        that case but kept as a parameter for a consistent dense/bm25
        call signature.
        """
        query_embedding = cls.embed_query(embedding_query or query)
        fetch_k = cls._fetch_depth(top_k, document_ids)

        scores, vector_ids = FAISSService.search(
            query_embedding=query_embedding, top_k=fetch_k
        )

        allowed = set(document_ids) if document_ids else None
        results: dict[tuple[str, str], RetrievedChunk] = {}

        for score, vector_id in zip(scores, vector_ids):
            if vector_id == -1:
                continue

            metadata = FAISSService.get_vector_metadata(int(vector_id))
            if metadata is None:
                continue

            if allowed is not None and metadata["document_id"] not in allowed:
                continue

            key = (metadata["document_id"], metadata["chunk_id"])
            if key in results and results[key].score >= score:
                continue

            results[key] = cls._to_retrieved_chunk(metadata, float(score))

        return results

    # ==========================================================
    # Lexical (BM25)
    # ==========================================================

    @classmethod
    def _bm25_search(
        cls,
        query: str,
        top_k: int,
        document_ids: list[str] | None,
    ) -> dict[tuple[str, str], RetrievedChunk]:
        """
        Returns {(document_id, chunk_id): RetrievedChunk}.
        """
        fetch_k = cls._fetch_depth(top_k, document_ids)
        allowed = set(document_ids) if document_ids else None

        results: dict[tuple[str, str], RetrievedChunk] = {}

        for score, metadata in BM25Service.search(query, fetch_k):
            if allowed is not None and metadata["document_id"] not in allowed:
                continue

            key = (metadata["document_id"], metadata["chunk_id"])
            if key in results and results[key].score >= score:
                continue

            results[key] = cls._to_retrieved_chunk(metadata, float(score))

        return results

    # ==========================================================
    # Fusion
    # ==========================================================

    @staticmethod
    def _reciprocal_rank_fusion(
        ranked_lists: list[list[tuple[str, str]]],
        k: int,
    ) -> dict[tuple[str, str], float]:
        """
        Standard RRF: score(d) = sum over lists containing d of
        1 / (k + rank_in_that_list), rank 1-indexed. A document
        doesn't need to appear in every list -- being highly ranked
        in even one list is enough to score well, which is exactly
        the point (a doc dense retrieval misses entirely can still
        surface via BM25, and vice versa).
        """
        fused: dict[tuple[str, str], float] = {}
        for ranked in ranked_lists:
            for rank, key in enumerate(ranked, start=1):
                fused[key] = fused.get(key, 0.0) + 1.0 / (k + rank)
        return fused

    # ==========================================================
    # Public entry point
    # ==========================================================

    @classmethod
    def search(
        cls,
        query: str,
        top_k: int = 5,
        document_ids: list[str] | None = None,
        mode: str | None = None,
        rerank: bool | None = None,
        query_rewrite: bool | None = None,
    ) -> list[RetrievedChunk]:
        """
        Search the indexed corpus and return deduplicated chunks,
        best first.

        `document_ids`, when given, restricts results to those
        documents. `mode` overrides settings.RETRIEVAL_MODE, `rerank`
        overrides settings.RERANK_ENABLED, and `query_rewrite`
        overrides settings.QUERY_REWRITE_ENABLED for this call (used
        by the eval harness to A/B without touching global config).
        """
        should_rerank = (
            settings.RERANK_ENABLED if rerank is None else rerank
        )
        should_rewrite = (
            settings.QUERY_REWRITE_ENABLED
            if query_rewrite is None
            else query_rewrite
        )

        embedding_query = None
        if should_rewrite:
            # Local import: keeps the extra LLM-call dependency out
            # of the path for callers that never rewrite (e.g. the
            # eval harness's --compare-modes dense/bm25/hybrid runs).
            from app.services.query_rewrite_service import (
                QueryRewriteService,
            )

            embedding_query = (
                QueryRewriteService.generate_hypothetical_answer(query)
            )

        fetch_k = (
            top_k * settings.RERANK_CANDIDATE_MULTIPLIER
            if should_rerank
            else top_k
        )

        candidates = cls._retrieve(
            query, fetch_k, document_ids, mode, embedding_query
        )

        if not should_rerank:
            return candidates[:top_k]

        # Local import avoids a hard dependency for callers that
        # never rerank (e.g. it keeps the reranker model out of the
        # import chain for a pure dense/bm25 comparison in the eval
        # harness's --compare-modes).
        from app.services.reranker_service import RerankerService

        reranked = RerankerService.rerank(query, candidates, top_k)

        return cls._apply_score_threshold(reranked)

    @staticmethod
    def _apply_score_threshold(
        results: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        """
        Drops results scoring below settings.RERANK_SCORE_THRESHOLD.
        A no-op when the threshold is unset (the default -- see the
        settings comment for why this isn't calibrated yet). If this
        empties the list, the caller (PromptBuilder) already treats
        an empty retrieval as "no context found" and refuses rather
        than answering from weak matches.
        """
        threshold = settings.RERANK_SCORE_THRESHOLD

        if threshold is None:
            return results

        return [r for r in results if r.score >= threshold]

    @classmethod
    def _retrieve(
        cls,
        query: str,
        top_k: int,
        document_ids: list[str] | None,
        mode: str | None,
        embedding_query: str | None = None,
    ) -> list[RetrievedChunk]:
        """
        Runs the selected retrieval mode (dense/bm25/hybrid) and
        returns up to top_k candidates, best first. Split out from
        `search()` so reranking can request a wider candidate pool
        than the final top_k without duplicating the dense/bm25/
        hybrid dispatch logic.

        `embedding_query`, when given, is used for the dense leg's
        embedding instead of `query` (HyDE-style rewriting); BM25
        always uses the literal `query`, since a fabricated
        hypothetical answer helps semantic similarity but would just
        inject noise into lexical matching.
        """
        mode = mode or settings.RETRIEVAL_MODE

        if mode == "dense":
            dense = cls._dense_search(
                query, top_k, document_ids, embedding_query
            )
            ranked = sorted(
                dense.values(), key=lambda r: r.score, reverse=True
            )
            return ranked[:top_k]

        if mode == "bm25":
            bm25 = cls._bm25_search(query, top_k, document_ids)
            ranked = sorted(
                bm25.values(), key=lambda r: r.score, reverse=True
            )
            return ranked[:top_k]

        if mode != "hybrid":
            raise ValueError(f"Unknown RETRIEVAL_MODE: {mode!r}")

        dense = cls._dense_search(
            query, top_k, document_ids, embedding_query
        )
        bm25 = cls._bm25_search(query, top_k, document_ids)

        dense_ranked_keys = [
            k
            for k, _ in sorted(
                dense.items(), key=lambda kv: kv[1].score, reverse=True
            )
        ]
        bm25_ranked_keys = [
            k
            for k, _ in sorted(
                bm25.items(), key=lambda kv: kv[1].score, reverse=True
            )
        ]

        fused_scores = cls._reciprocal_rank_fusion(
            [dense_ranked_keys, bm25_ranked_keys], k=settings.RRF_K
        )

        chunks_by_key = {**dense, **bm25}  # either has the chunk data we need

        fused_results = [
            RetrievedChunk(
                **{
                    **chunks_by_key[key].model_dump(),
                    "score": fused_score,
                }
            )
            for key, fused_score in fused_scores.items()
        ]

        fused_results.sort(key=lambda r: r.score, reverse=True)

        return fused_results[:top_k]
