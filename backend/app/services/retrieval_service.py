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
    def embed_query(query: str) -> np.ndarray:
        model = EmbeddingModel.get_model()
        embedding = model.encode(
            [query],
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
    ) -> dict[tuple[str, str], RetrievedChunk]:
        """
        Returns {(document_id, chunk_id): RetrievedChunk}, deduped
        (near-duplicate FAISS hits keep the highest score).
        """
        query_embedding = cls.embed_query(query)
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
    ) -> list[RetrievedChunk]:
        """
        Search the indexed corpus and return deduplicated chunks,
        best first.

        `document_ids`, when given, restricts results to those
        documents. `mode` overrides settings.RETRIEVAL_MODE for this
        call (used by the eval harness to A/B dense vs. hybrid
        without touching global config).
        """
        mode = mode or settings.RETRIEVAL_MODE

        if mode == "dense":
            dense = cls._dense_search(query, top_k, document_ids)
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

        dense = cls._dense_search(query, top_k, document_ids)
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
