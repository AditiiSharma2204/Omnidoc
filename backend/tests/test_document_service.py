import io

import pytest
from fastapi import UploadFile

from app.services.document_service import DocumentService


class _FakeParser:
    def __init__(self, result=None, error=None):
        self._result = result
        self._error = error

    def parse(self, file_path):
        if self._error:
            raise self._error
        return self._result


def _make_upload_file(filename="doc.pdf", content=b"%PDF-1.4 fake"):
    from starlette.datastructures import Headers

    return UploadFile(
        filename=filename,
        file=io.BytesIO(content),
        headers=Headers({"content-type": "application/pdf"}),
    )


@pytest.fixture
def wired_document_service(tmp_path, monkeypatch):
    """
    Wires DocumentService to a tmp storage dir and stubs out every
    heavy stage (validation is real; parsing/chunking/embedding/
    indexing are stubbed) so these tests exercise only the
    orchestration + status bookkeeping in save_document, not the ML
    stack.
    """
    import app.services.document_service as ds
    from app.config.settings import settings

    monkeypatch.setattr(settings, "DOCUMENTS_DIR", str(tmp_path))

    async def fake_validate(file):
        content = await file.read()
        file.file.seek(0)
        return len(content)

    monkeypatch.setattr(
        ds.ValidationService, "validate", staticmethod(fake_validate)
    )

    return ds


class TestStatusTransitions:
    """
    Regression tests for the original bug: success got stuck at
    "embedded" (never reaching "indexed"), and the except branch
    mislabeled failures as "indexed" instead of "failed".
    """

    @pytest.mark.asyncio
    async def test_successful_pipeline_reaches_indexed(
        self, wired_document_service, monkeypatch
    ):
        ds = wired_document_service

        monkeypatch.setattr(
            ds.ParserFactory,
            "get_parser",
            staticmethod(
                lambda path: _FakeParser(
                    result={
                        "document_id": "x",
                        "title": "original.pdf",
                        "pages": 1,
                        "text": "## H\n\nbody",
                        "tables": [],
                        "figures": [],
                        "metadata": {},
                    }
                )
            ),
        )
        monkeypatch.setattr(
            ds.ChunkingService, "chunk_document", lambda folder: []
        )
        monkeypatch.setattr(
            ds.EmbeddingService, "generate", lambda folder: None
        )
        monkeypatch.setattr(
            ds.FAISSService, "add_document", lambda folder: None
        )

        service = DocumentService()
        result = await service.save_document(
            _make_upload_file(filename="resume.pdf")
        )

        assert result["status"] == "indexed"
        assert "successfully" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_failed_parse_sets_failed_not_indexed(
        self, wired_document_service, monkeypatch
    ):
        ds = wired_document_service

        monkeypatch.setattr(
            ds.ParserFactory,
            "get_parser",
            staticmethod(
                lambda path: _FakeParser(
                    error=ValueError("Unsupported file type: .xyz")
                )
            ),
        )

        service = DocumentService()
        result = await service.save_document(
            _make_upload_file(filename="weird.pdf")
        )

        assert result["status"] == "failed"
        assert "Unsupported file type" in result["message"]

    @pytest.mark.asyncio
    async def test_failed_embedding_sets_failed_not_indexed(
        self, wired_document_service, monkeypatch
    ):
        ds = wired_document_service

        monkeypatch.setattr(
            ds.ParserFactory,
            "get_parser",
            staticmethod(
                lambda path: _FakeParser(
                    result={
                        "document_id": "x",
                        "title": "original.pdf",
                        "pages": 1,
                        "text": "## H\n\nbody",
                        "tables": [],
                        "figures": [],
                        "metadata": {},
                    }
                )
            ),
        )
        monkeypatch.setattr(
            ds.ChunkingService, "chunk_document", lambda folder: []
        )

        def boom(folder):
            raise RuntimeError("embedding blew up")

        monkeypatch.setattr(ds.EmbeddingService, "generate", boom)

        service = DocumentService()
        result = await service.save_document(_make_upload_file())

        assert result["status"] == "failed"
        assert "embedding blew up" in result["message"]

    @pytest.mark.asyncio
    async def test_original_filename_overrides_parser_title(
        self, wired_document_service, monkeypatch, tmp_path
    ):
        """
        Regression test: the parser only ever sees the on-disk path
        ("original.pdf"), so it can't know the real upload filename.
        save_document must stamp the real filename onto the parsed
        result before it's persisted.
        """
        ds = wired_document_service

        captured = {}

        def fake_save(folder, parsed_document):
            captured["title"] = parsed_document["title"]

        monkeypatch.setattr(ds.ParsedDocumentService, "save", fake_save)
        monkeypatch.setattr(
            ds.ParserFactory,
            "get_parser",
            staticmethod(
                lambda path: _FakeParser(
                    result={
                        "document_id": "x",
                        "title": "original.pdf",
                        "pages": 1,
                        "text": "## H\n\nbody",
                        "tables": [],
                        "figures": [],
                        "metadata": {},
                    }
                )
            ),
        )
        monkeypatch.setattr(
            ds.ChunkingService, "chunk_document", lambda folder: []
        )
        monkeypatch.setattr(
            ds.EmbeddingService, "generate", lambda folder: None
        )
        monkeypatch.setattr(
            ds.FAISSService, "add_document", lambda folder: None
        )

        service = DocumentService()
        await service.save_document(
            _make_upload_file(filename="Aditii_Resume.pdf")
        )

        assert captured["title"] == "Aditii_Resume.pdf"
