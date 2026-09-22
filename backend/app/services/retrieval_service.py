import json
from pathlib import Path

import numpy as np

from app.config.settings import settings
from app.embeddings.model import EmbeddingModel
from app.schemas.retrieval import RetrievedChunk
from app.vectorstore.faiss_service import FAISSService


class RetrievalService:
    """
    Semantic retrieval over the global FAISS index.
    """

    @staticmethod
    def embed_query(query: str) -> np.ndarray:
        """
        Generate an embedding for the user's query.
        """

        model = EmbeddingModel.get_model()

        embedding = model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        return embedding.astype(np.float32)

    @staticmethod
    def load_chunk(
        document_id: str,
        chunk_index: int,
    ) -> dict:
        """
        Load a single chunk from a processed document.
        """

        chunks_path = (
            Path(settings.DOCUMENTS_DIR)
            / document_id
            / "chunks.json"
        )

        if not chunks_path.exists():
            raise FileNotFoundError(
                f"Chunks not found for document '{document_id}'."
            )

        with chunks_path.open(
            "r",
            encoding="utf-8",
        ) as f:
            chunks = json.load(f)

        if chunk_index >= len(chunks):
            raise IndexError(
                f"Chunk index {chunk_index} out of range."
            )

        return chunks[chunk_index]

    @classmethod
    def search(
        cls,
        query: str,
        top_k: int = 5,
        document_ids: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        """
        Search the vector database and return deduplicated semantic
        chunks.

        `document_ids`, when given, restricts results to those
        documents. FAISS's IndexFlatIP has no native metadata
        filter, so we over-fetch and filter in Python; the fetch
        depth scales up when filtering to a small subset so we
        don't starve the result set.
        """

        query_embedding = cls.embed_query(query)

        fetch_k = top_k * 3

        if document_ids:
            # Filtering can throw away most candidates, so search
            # deeper. Capped so a huge corpus doesn't blow up
            # latency on every filtered query.
            fetch_k = min(max(fetch_k, top_k * 10), 200)

        scores, vector_ids = FAISSService.search(
            query_embedding=query_embedding,
            top_k=fetch_k,
        )

        allowed_documents = (
            set(document_ids) if document_ids else None
        )

        unique_results = {}

        for score, vector_id in zip(scores, vector_ids):

            if vector_id == -1:
                continue

            metadata = FAISSService.get_vector_metadata(
                int(vector_id)
            )

            if metadata is None:
                continue

            if (
                allowed_documents is not None
                and metadata["document_id"] not in allowed_documents
            ):
                continue

            chunk = cls.load_chunk(
                document_id=metadata["document_id"],
                chunk_index=metadata["chunk_index"],
            )

            # -------------------------------------------------
            # Use document + chunk id as unique key
            # -------------------------------------------------

            unique_key = (
                metadata["document_id"],
                metadata["chunk_id"],
            )

            retrieved = RetrievedChunk(
                score=float(score),
                document_id=metadata["document_id"],
                chunk_id=metadata["chunk_id"],
                section=chunk["metadata"].get("section"),
                heading=metadata.get("heading"),
                title=metadata.get("title"),
                page=metadata.get("page"),
                text=chunk["text"],
            )

            # Keep only the highest scoring duplicate
            if (
                unique_key not in unique_results
                or score > unique_results[unique_key].score
            ):
                unique_results[unique_key] = retrieved

        results = list(unique_results.values())

        results.sort(
            key=lambda x: x.score,
            reverse=True,
        )

        return results[:top_k]