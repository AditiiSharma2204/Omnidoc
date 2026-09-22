import json

import numpy as np
import pytest

from app.services.retrieval_service import RetrievalService


def _write_chunks(documents_dir, document_id, chunks):
    """
    chunks: list of (chunk_id, heading, text) tuples.
    """
    folder = documents_dir / document_id
    folder.mkdir(parents=True, exist_ok=True)

    payload = [
        {
            "chunk_id": chunk_id,
            "text": text,
            "page": None,
            "metadata": {
                "heading": heading,
                "section": None,
                "title": f"{document_id}.pdf",
            },
        }
        for chunk_id, heading, text in chunks
    ]

    (folder / "chunks.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


@pytest.fixture
def documents_dir(tmp_path, monkeypatch):
    from app.config.settings import settings

    monkeypatch.setattr(settings, "DOCUMENTS_DIR", str(tmp_path))
    return tmp_path


def _fake_metadata_lookup(entries):
    """
    Builds a fake FAISSService.get_vector_metadata that indexes into
    a fixed list of metadata dicts, mirroring the real
    list-position-as-vector-id scheme.
    """

    def lookup(vector_id):
        if 0 <= vector_id < len(entries):
            return entries[vector_id]
        return None

    return lookup


class TestDeduplication:

    def test_near_duplicate_hits_keep_highest_score(
        self, documents_dir, monkeypatch
    ):
        _write_chunks(
            documents_dir,
            "doc-a",
            [("c1", "Skills", "Python, SQL, C++")],
        )

        entries = [
            {
                "vector_id": 0,
                "document_id": "doc-a",
                "chunk_id": "c1",
                "chunk_index": 0,
                "heading": "Skills",
                "page": None,
                "title": "doc-a.pdf",
            }
        ]

        monkeypatch.setattr(
            RetrievalService, "embed_query", lambda q: np.zeros((1, 4))
        )

        import app.services.retrieval_service as rs

        monkeypatch.setattr(
            rs.FAISSService,
            "search",
            lambda query_embedding, top_k: (
                np.array([0.5, 0.9]),
                np.array([0, 0]),  # same vector hit twice
            ),
        )
        monkeypatch.setattr(
            rs.FAISSService,
            "get_vector_metadata",
            _fake_metadata_lookup(entries),
        )

        results = RetrievalService.search("skills?", top_k=5)

        assert len(results) == 1
        assert results[0].score == 0.9

    def test_minus_one_vector_ids_are_skipped(
        self, documents_dir, monkeypatch
    ):
        """
        FAISS returns -1 for unfilled slots when the index has fewer
        vectors than top_k; these must not crash or become results.
        """
        _write_chunks(
            documents_dir, "doc-a", [("c1", "Skills", "Python")]
        )

        entries = [
            {
                "vector_id": 0,
                "document_id": "doc-a",
                "chunk_id": "c1",
                "chunk_index": 0,
                "heading": "Skills",
                "page": None,
                "title": "doc-a.pdf",
            }
        ]

        monkeypatch.setattr(
            RetrievalService, "embed_query", lambda q: np.zeros((1, 4))
        )

        import app.services.retrieval_service as rs

        monkeypatch.setattr(
            rs.FAISSService,
            "search",
            lambda query_embedding, top_k: (
                np.array([0.9, 0.0, 0.0]),
                np.array([0, -1, -1]),
            ),
        )
        monkeypatch.setattr(
            rs.FAISSService,
            "get_vector_metadata",
            _fake_metadata_lookup(entries),
        )

        results = RetrievalService.search("skills?", top_k=5)

        assert len(results) == 1


class TestDocumentFiltering:

    def test_document_ids_filters_out_other_documents(
        self, documents_dir, monkeypatch
    ):
        _write_chunks(
            documents_dir, "doc-a", [("c1", "Skills", "Python skills")]
        )
        _write_chunks(
            documents_dir, "doc-b", [("c1", "Skills", "Java skills")]
        )

        entries = [
            {
                "vector_id": 0,
                "document_id": "doc-a",
                "chunk_id": "c1",
                "chunk_index": 0,
                "heading": "Skills",
                "page": None,
                "title": "doc-a.pdf",
            },
            {
                "vector_id": 1,
                "document_id": "doc-b",
                "chunk_id": "c1",
                "chunk_index": 0,
                "heading": "Skills",
                "page": None,
                "title": "doc-b.pdf",
            },
        ]

        monkeypatch.setattr(
            RetrievalService, "embed_query", lambda q: np.zeros((1, 4))
        )

        import app.services.retrieval_service as rs

        monkeypatch.setattr(
            rs.FAISSService,
            "search",
            lambda query_embedding, top_k: (
                np.array([0.9, 0.8]),
                np.array([0, 1]),
            ),
        )
        monkeypatch.setattr(
            rs.FAISSService,
            "get_vector_metadata",
            _fake_metadata_lookup(entries),
        )

        results = RetrievalService.search(
            "skills?", top_k=5, document_ids=["doc-b"]
        )

        assert len(results) == 1
        assert results[0].document_id == "doc-b"

    def test_no_filter_returns_all_documents(
        self, documents_dir, monkeypatch
    ):
        _write_chunks(
            documents_dir, "doc-a", [("c1", "Skills", "Python skills")]
        )
        _write_chunks(
            documents_dir, "doc-b", [("c1", "Skills", "Java skills")]
        )

        entries = [
            {
                "vector_id": 0,
                "document_id": "doc-a",
                "chunk_id": "c1",
                "chunk_index": 0,
                "heading": "Skills",
                "page": None,
                "title": "doc-a.pdf",
            },
            {
                "vector_id": 1,
                "document_id": "doc-b",
                "chunk_id": "c1",
                "chunk_index": 0,
                "heading": "Skills",
                "page": None,
                "title": "doc-b.pdf",
            },
        ]

        monkeypatch.setattr(
            RetrievalService, "embed_query", lambda q: np.zeros((1, 4))
        )

        import app.services.retrieval_service as rs

        monkeypatch.setattr(
            rs.FAISSService,
            "search",
            lambda query_embedding, top_k: (
                np.array([0.9, 0.8]),
                np.array([0, 1]),
            ),
        )
        monkeypatch.setattr(
            rs.FAISSService,
            "get_vector_metadata",
            _fake_metadata_lookup(entries),
        )

        results = RetrievalService.search("skills?", top_k=5)

        assert {r.document_id for r in results} == {"doc-a", "doc-b"}
