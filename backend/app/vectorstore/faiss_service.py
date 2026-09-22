import json
from pathlib import Path

import faiss
import numpy as np

from app.config.settings import settings


class FAISSService:
    """
    Global FAISS vector store for OmniDoc.
    """

    INDEX_FILE = "index.faiss"
    METADATA_FILE = "metadata.json"

    # ==========================================================
    # Paths
    # ==========================================================

    @classmethod
    def get_index_path(cls) -> Path:
        return Path(settings.VECTORSTORE_DIR) / cls.INDEX_FILE

    @classmethod
    def get_metadata_path(cls) -> Path:
        return Path(settings.VECTORSTORE_DIR) / cls.METADATA_FILE

    # ==========================================================
    # Index Management
    # ==========================================================

    @classmethod
    def create_index(cls, dimension: int):
        """
        Creates a cosine-similarity FAISS index.
        """
        return faiss.IndexFlatIP(dimension)

    @classmethod
    def load_index(cls):
        """
        Loads the FAISS index if it exists.
        """
        path = cls.get_index_path()

        if not path.exists():
            return None

        return faiss.read_index(str(path))

    @classmethod
    def save_index(cls, index) -> None:
        """
        Saves the FAISS index.
        """
        faiss.write_index(
            index,
            str(cls.get_index_path()),
        )

    # ==========================================================
    # Metadata Management
    # ==========================================================

    @classmethod
    def load_metadata(cls) -> list:
        """
        Loads vector metadata.
        """
        path = cls.get_metadata_path()

        if not path.exists():
            return []

        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    @classmethod
    def save_metadata(cls, metadata: list) -> None:
        """
        Saves vector metadata.
        """
        with cls.get_metadata_path().open(
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                metadata,
                f,
                indent=4,
                ensure_ascii=False,
            )

    # ==========================================================
    # Document Indexing
    # ==========================================================

    @classmethod
    def add_document(cls, document_folder: Path) -> None:
        """
        Adds one processed document into the global FAISS index.
        """

        embeddings_path = document_folder / "embeddings.npy"
        chunks_path = document_folder / "chunks.json"

        if not embeddings_path.exists():
            raise FileNotFoundError(
                "embeddings.npy not found"
            )

        if not chunks_path.exists():
            raise FileNotFoundError(
                "chunks.json not found"
            )

        embeddings = np.load(
            embeddings_path
        ).astype(np.float32)

        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)

        with chunks_path.open(
            "r",
            encoding="utf-8",
        ) as f:

            chunks = json.load(f)

        document_id = document_folder.name

        metadata = cls.load_metadata()

        # ------------------------------------------
        # Prevent duplicate indexing
        # ------------------------------------------

        if any(
            item["document_id"] == document_id
            for item in metadata
        ):
            raise ValueError(
                f"Document '{document_id}' is already indexed."
            )

        # ------------------------------------------
        # Load/Create FAISS index
        # ------------------------------------------

        dimension = embeddings.shape[1]

        index = cls.load_index()

        if index is None:

            index = cls.create_index(dimension)

        else:

            if index.d != dimension:
                raise ValueError(
                    f"Embedding dimension mismatch "
                    f"(Index={index.d}, "
                    f"Embeddings={dimension})"
                )

        # ------------------------------------------
        # Add vectors
        # ------------------------------------------

        start_vector = index.ntotal

        index.add(embeddings)

        # ------------------------------------------
        # Save metadata
        # ------------------------------------------

        for i, chunk in enumerate(chunks):

            metadata.append(
                {
                    "vector_id": start_vector + i,
                    "document_id": document_id,
                    "chunk_id": chunk["chunk_id"],
                    "chunk_index": i,
                    "heading": chunk["metadata"].get("heading"),
                    "page": chunk.get("page"),
                    "title": chunk["metadata"].get("title"),
                }
            )

        cls.save_index(index)
        cls.save_metadata(metadata)

        print(
            f"Indexed {len(chunks)} chunks "
            f"({index.ntotal} total vectors)"
        )

    # ==========================================================
    # Rebuild (used by document delete)
    # ==========================================================

    @classmethod
    def rebuild_index(cls, documents_dir: Path) -> None:
        """
        Rebuilds the global index from every remaining document's
        saved embeddings + chunks on disk.

        IndexFlatIP has no per-vector delete, and vector ids are
        plain list positions into metadata.json, so removing one
        document safely means rebuilding from source rather than
        trying to patch the index in place.
        """

        index = None
        metadata: list[dict] = []

        if not documents_dir.exists():
            folders = []
        else:
            folders = sorted(
                p for p in documents_dir.iterdir() if p.is_dir()
            )

        for document_folder in folders:

            embeddings_path = document_folder / "embeddings.npy"
            chunks_path = document_folder / "chunks.json"

            if not embeddings_path.exists() or not chunks_path.exists():
                continue

            embeddings = np.load(embeddings_path).astype(np.float32)
            faiss.normalize_L2(embeddings)

            with chunks_path.open("r", encoding="utf-8") as f:
                chunks = json.load(f)

            if index is None:
                index = cls.create_index(embeddings.shape[1])

            start_vector = index.ntotal
            index.add(embeddings)

            document_id = document_folder.name

            for i, chunk in enumerate(chunks):
                metadata.append(
                    {
                        "vector_id": start_vector + i,
                        "document_id": document_id,
                        "chunk_id": chunk["chunk_id"],
                        "chunk_index": i,
                        "heading": chunk["metadata"].get("heading"),
                        "page": chunk.get("page"),
                        "title": chunk["metadata"].get("title"),
                    }
                )

        if index is None:
            # No documents left: drop the index/metadata files
            # entirely instead of leaving a stale empty one.
            cls.get_index_path().unlink(missing_ok=True)
            cls.get_metadata_path().unlink(missing_ok=True)
            return

        cls.save_index(index)
        cls.save_metadata(metadata)

    # ==========================================================
    # Search
    # ==========================================================

    @classmethod
    def search(
        cls,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ):
        """
        Searches the global FAISS index.
        """

        index = cls.load_index()

        if index is None:
            raise RuntimeError(
                "Vector index not found."
            )

        query_embedding = query_embedding.astype(
            np.float32
        )

        faiss.normalize_L2(query_embedding)

        scores, indices = index.search(
            query_embedding,
            top_k,
        )

        return scores[0], indices[0]

    # ==========================================================
    # Metadata Lookup
    # ==========================================================

    @classmethod
    def get_vector_metadata(
        cls,
        vector_id: int,
    ):
        """
        Returns metadata for one vector.
        """

        metadata = cls.load_metadata()

        if (
            vector_id < 0
            or vector_id >= len(metadata)
        ):
            return None

        return metadata[vector_id]