import json

import numpy as np
import pytest

from app.vectorstore.faiss_service import FAISSService


def _write_document(documents_dir, document_id, vectors, headings):
    folder = documents_dir / document_id
    folder.mkdir(parents=True, exist_ok=True)

    np.save(
        folder / "embeddings.npy",
        np.array(vectors, dtype=np.float32),
    )

    chunks = [
        {
            "chunk_id": f"{document_id}-{i}",
            "text": f"chunk {i}",
            "page": None,
            "metadata": {"heading": heading, "title": f"{document_id}.pdf"},
        }
        for i, heading in enumerate(headings)
    ]
    (folder / "chunks.json").write_text(
        json.dumps(chunks), encoding="utf-8"
    )


@pytest.fixture
def vectorstore(tmp_path, monkeypatch):
    from app.config.settings import settings

    store_dir = tmp_path / "vectorstore"
    store_dir.mkdir()
    monkeypatch.setattr(settings, "VECTORSTORE_DIR", str(store_dir))
    return store_dir


class TestAddDocument:

    def test_add_document_builds_index_and_metadata(
        self, tmp_path, vectorstore
    ):
        doc_folder = tmp_path / "documents" / "doc-a"
        _write_document(
            tmp_path / "documents",
            "doc-a",
            [[1.0, 0.0], [0.0, 1.0]],
            ["Heading1", "Heading2"],
        )

        FAISSService.add_document(doc_folder)

        index = FAISSService.load_index()
        assert index.ntotal == 2

        metadata = FAISSService.load_metadata()
        assert len(metadata) == 2
        assert metadata[0]["document_id"] == "doc-a"

    def test_add_document_rejects_duplicate_document_id(
        self, tmp_path, vectorstore
    ):
        doc_folder = tmp_path / "documents" / "doc-a"
        _write_document(
            tmp_path / "documents",
            "doc-a",
            [[1.0, 0.0]],
            ["H"],
        )

        FAISSService.add_document(doc_folder)

        with pytest.raises(ValueError):
            FAISSService.add_document(doc_folder)


class TestRebuildIndex:

    def test_rebuild_excludes_deleted_document(
        self, tmp_path, vectorstore
    ):
        documents_dir = tmp_path / "documents"
        _write_document(
            documents_dir, "doc-a", [[1.0, 0.0]], ["H-a"]
        )
        _write_document(
            documents_dir, "doc-b", [[0.0, 1.0]], ["H-b"]
        )

        FAISSService.add_document(documents_dir / "doc-a")
        FAISSService.add_document(documents_dir / "doc-b")

        assert FAISSService.load_index().ntotal == 2

        # Simulate deleting doc-a: remove its folder, then rebuild.
        import shutil

        shutil.rmtree(documents_dir / "doc-a")
        FAISSService.rebuild_index(documents_dir)

        index = FAISSService.load_index()
        assert index.ntotal == 1

        metadata = FAISSService.load_metadata()
        assert len(metadata) == 1
        assert metadata[0]["document_id"] == "doc-b"

    def test_rebuild_with_no_documents_removes_index_files(
        self, tmp_path, vectorstore
    ):
        documents_dir = tmp_path / "documents"
        _write_document(documents_dir, "doc-a", [[1.0, 0.0]], ["H"])
        FAISSService.add_document(documents_dir / "doc-a")

        import shutil

        shutil.rmtree(documents_dir / "doc-a")
        FAISSService.rebuild_index(documents_dir)

        assert not FAISSService.get_index_path().exists()
        assert not FAISSService.get_metadata_path().exists()
        assert FAISSService.load_index() is None
        assert FAISSService.load_metadata() == []

    def test_vector_ids_are_reassigned_contiguously_after_rebuild(
        self, tmp_path, vectorstore
    ):
        documents_dir = tmp_path / "documents"
        _write_document(
            documents_dir,
            "doc-a",
            [[1.0, 0.0], [0.9, 0.1]],
            ["H1", "H2"],
        )
        _write_document(
            documents_dir, "doc-b", [[0.0, 1.0]], ["H3"]
        )

        FAISSService.add_document(documents_dir / "doc-a")
        FAISSService.add_document(documents_dir / "doc-b")

        import shutil

        shutil.rmtree(documents_dir / "doc-a")
        FAISSService.rebuild_index(documents_dir)

        metadata = FAISSService.load_metadata()
        assert [m["vector_id"] for m in metadata] == list(
            range(len(metadata))
        )
