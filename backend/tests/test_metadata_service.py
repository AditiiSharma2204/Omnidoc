from datetime import datetime, timezone

import pytest

from app.schemas.document_metadata import DocumentMetadata
from app.services.metadata_service import MetadataService


@pytest.fixture
def isolated_db(tmp_path, monkeypatch):
    from app.config.settings import settings
    from app.db.database import init_db

    monkeypatch.setattr(
        settings, "DATABASE_PATH", str(tmp_path / "test.db")
    )
    init_db()


def _metadata(document_id="doc-1", status="uploaded", **overrides):
    defaults = dict(
        document_id=document_id,
        original_filename="resume.pdf",
        stored_filename="original.pdf",
        mime_type="application/pdf",
        size=1024,
        upload_time=datetime.now(timezone.utc),
        status=status,
        parse_error=None,
    )
    defaults.update(overrides)
    return DocumentMetadata(**defaults)


class TestSaveAndLoad:

    def test_round_trips_all_fields(self, isolated_db):
        original = _metadata(
            document_id="doc-1",
            original_filename="My Report.docx",
            status="indexed",
        )

        MetadataService.save(original)
        loaded = MetadataService.load("doc-1")

        assert loaded.document_id == "doc-1"
        assert loaded.original_filename == "My Report.docx"
        assert loaded.status == "indexed"
        assert loaded.parse_error is None

    def test_missing_document_returns_none(self, isolated_db):
        assert MetadataService.load("nonexistent") is None

    def test_save_is_upsert_not_insert_only(self, isolated_db):
        """
        Regression-shaped test: the real pipeline calls save()
        repeatedly on the same document_id as status progresses
        (uploaded -> parsed -> chunked -> ... -> indexed/failed).
        This must update in place, not fail on a duplicate primary
        key or create a second row.
        """
        MetadataService.save(_metadata(document_id="doc-1", status="uploaded"))
        MetadataService.save(_metadata(document_id="doc-1", status="parsed"))
        MetadataService.save(_metadata(document_id="doc-1", status="indexed"))

        loaded = MetadataService.load("doc-1")
        assert loaded.status == "indexed"
        assert len(MetadataService.list_all()) == 1

    def test_parse_error_persists(self, isolated_db):
        MetadataService.save(
            _metadata(
                document_id="doc-1",
                status="failed",
                parse_error="Unsupported file type: .xyz",
            )
        )

        loaded = MetadataService.load("doc-1")
        assert loaded.status == "failed"
        assert loaded.parse_error == "Unsupported file type: .xyz"


class TestListAll:

    def test_empty_database_returns_empty_list(self, isolated_db):
        assert MetadataService.list_all() == []

    def test_returns_newest_first(self, isolated_db):
        MetadataService.save(
            _metadata(
                document_id="old",
                upload_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
            )
        )
        MetadataService.save(
            _metadata(
                document_id="new",
                upload_time=datetime(2026, 6, 1, tzinfo=timezone.utc),
            )
        )

        results = MetadataService.list_all()

        assert [r.document_id for r in results] == ["new", "old"]


class TestDelete:

    def test_delete_removes_the_row(self, isolated_db):
        MetadataService.save(_metadata(document_id="doc-1"))
        MetadataService.delete("doc-1")

        assert MetadataService.load("doc-1") is None
        assert MetadataService.list_all() == []

    def test_delete_nonexistent_id_does_not_raise(self, isolated_db):
        MetadataService.delete("never-existed")
