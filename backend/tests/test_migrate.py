import json
from datetime import datetime, timezone

import pytest

from app.db.migrate import migrate_json_metadata_to_sqlite
from app.services.metadata_service import MetadataService


@pytest.fixture
def isolated_env(tmp_path, monkeypatch):
    from app.config.settings import settings
    from app.db.database import init_db

    documents_dir = tmp_path / "documents"
    documents_dir.mkdir()

    monkeypatch.setattr(settings, "DOCUMENTS_DIR", str(documents_dir))
    monkeypatch.setattr(
        settings, "DATABASE_PATH", str(tmp_path / "test.db")
    )
    init_db()

    return documents_dir


def _write_legacy_metadata_json(documents_dir, document_id, **overrides):
    folder = documents_dir / document_id
    folder.mkdir(parents=True, exist_ok=True)

    data = {
        "document_id": document_id,
        "original_filename": "resume.pdf",
        "stored_filename": "original.pdf",
        "mime_type": "application/pdf",
        "size": 1024,
        "upload_time": datetime.now(timezone.utc).isoformat(),
        "status": "indexed",
        "parse_error": None,
    }
    data.update(overrides)

    (folder / "metadata.json").write_text(
        json.dumps(data), encoding="utf-8"
    )


class TestMigration:

    def test_migrates_legacy_json_documents(self, isolated_env):
        _write_legacy_metadata_json(
            isolated_env, "doc-1", original_filename="report.pdf"
        )
        _write_legacy_metadata_json(
            isolated_env, "doc-2", original_filename="slides.pptx"
        )

        migrated = migrate_json_metadata_to_sqlite()

        assert migrated == 2
        ids = {m.document_id for m in MetadataService.list_all()}
        assert ids == {"doc-1", "doc-2"}

    def test_is_idempotent_does_not_duplicate_on_rerun(
        self, isolated_env
    ):
        _write_legacy_metadata_json(isolated_env, "doc-1")

        first_run = migrate_json_metadata_to_sqlite()
        second_run = migrate_json_metadata_to_sqlite()

        assert first_run == 1
        assert second_run == 0
        assert len(MetadataService.list_all()) == 1

    def test_skips_documents_already_in_sqlite(self, isolated_env):
        """
        A document saved directly to SQLite (post-migration, the
        normal path) must not be re-imported or overwritten by a
        stale metadata.json that might still be sitting on disk.
        """
        _write_legacy_metadata_json(
            isolated_env, "doc-1", original_filename="OLD_NAME.pdf"
        )
        migrate_json_metadata_to_sqlite()

        from app.schemas.document_metadata import DocumentMetadata

        MetadataService.save(
            DocumentMetadata(
                document_id="doc-1",
                original_filename="NEW_NAME.pdf",
                stored_filename="original.pdf",
                mime_type="application/pdf",
                size=999,
                upload_time=datetime.now(timezone.utc),
                status="indexed",
            )
        )

        migrated_again = migrate_json_metadata_to_sqlite()

        assert migrated_again == 0
        loaded = MetadataService.load("doc-1")
        assert loaded.original_filename == "NEW_NAME.pdf"

    def test_ignores_folders_without_metadata_json(self, isolated_env):
        (isolated_env / "no-metadata-here").mkdir()

        migrated = migrate_json_metadata_to_sqlite()

        assert migrated == 0
        assert MetadataService.list_all() == []

    def test_no_documents_dir_returns_zero(self, tmp_path, monkeypatch):
        from app.config.settings import settings
        from app.db.database import init_db

        monkeypatch.setattr(
            settings, "DOCUMENTS_DIR", str(tmp_path / "does-not-exist")
        )
        monkeypatch.setattr(
            settings, "DATABASE_PATH", str(tmp_path / "test.db")
        )
        init_db()

        assert migrate_json_metadata_to_sqlite() == 0
