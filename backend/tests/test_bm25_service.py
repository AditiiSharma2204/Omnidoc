import json

import pytest

from app.vectorstore.bm25_service import BM25Service, tokenize


def _write_document(documents_dir, document_id, chunk_texts):
    folder = documents_dir / document_id
    folder.mkdir(parents=True, exist_ok=True)

    chunks = [
        {
            "chunk_id": f"{document_id}-{i}",
            "text": text,
            "page": None,
            "metadata": {"heading": f"H{i}", "title": f"{document_id}.pdf"},
        }
        for i, text in enumerate(chunk_texts)
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


class TestTokenize:

    def test_lowercases_and_splits_on_punctuation(self):
        assert tokenize("Aditii Sharma, AI/ML Engineer!") == [
            "aditii",
            "sharma",
            "ai",
            "ml",
            "engineer",
        ]

    def test_empty_string(self):
        assert tokenize("") == []


class TestRebuildIndex:

    def test_search_finds_exact_term_match(self, tmp_path, vectorstore):
        # A term appearing in exactly 1 of N chunks has near-zero idf
        # when N is tiny (BM25 genuinely treats "in half the corpus"
        # terms as uninformative) -- use enough chunks that a
        # single-occurrence term has clearly positive idf, matching
        # realistic corpus sizes.
        documents_dir = tmp_path / "documents"
        _write_document(
            documents_dir,
            "doc-a",
            [
                "Aditii Sharma is a Computer Science student.",
                "Skills include Python and SQL.",
                "Certifications include NPTEL courses.",
                "Achievements include a scholarship award.",
                "Projects include a portfolio recommendation system.",
            ],
        )

        BM25Service.rebuild_index(documents_dir)

        results = BM25Service.search("Aditii Sharma", top_k=5)

        assert len(results) >= 1
        assert results[0][1]["document_id"] == "doc-a"
        assert results[0][1]["chunk_index"] == 0

    def test_no_matches_returns_empty(self, tmp_path, vectorstore):
        documents_dir = tmp_path / "documents"
        _write_document(documents_dir, "doc-a", ["Python and SQL skills."])

        BM25Service.rebuild_index(documents_dir)

        results = BM25Service.search("zzz nonexistent term qqq", top_k=5)

        assert results == []

    def test_empty_corpus_removes_index_files(
        self, tmp_path, vectorstore
    ):
        documents_dir = tmp_path / "documents"
        documents_dir.mkdir()

        BM25Service.rebuild_index(documents_dir)

        assert not BM25Service.get_index_path().exists()
        assert BM25Service.search("anything", top_k=5) == []

    def test_rebuild_excludes_deleted_document(
        self, tmp_path, vectorstore
    ):
        documents_dir = tmp_path / "documents"
        _write_document(
            documents_dir,
            "doc-a",
            ["alpha term here", "filler chunk one", "filler chunk two"],
        )
        _write_document(
            documents_dir,
            "doc-b",
            ["beta term here", "filler chunk three", "filler chunk four"],
        )

        BM25Service.rebuild_index(documents_dir)
        assert len(BM25Service.search("alpha", top_k=5)) >= 1
        assert len(BM25Service.search("beta", top_k=5)) >= 1

        import shutil

        shutil.rmtree(documents_dir / "doc-a")
        BM25Service.rebuild_index(documents_dir)

        assert BM25Service.search("alpha", top_k=5) == []
        results = BM25Service.search("beta", top_k=5)
        assert all(r[1]["document_id"] == "doc-b" for r in results)

    def test_search_before_any_index_built_returns_empty(
        self, vectorstore
    ):
        assert BM25Service.search("anything", top_k=5) == []
