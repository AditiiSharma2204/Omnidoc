import json
from pathlib import Path

from app.config.settings import settings
from app.db.database import get_connection
from app.schemas.document_metadata import DocumentMetadata
from app.services.metadata_service import MetadataService


def migrate_json_metadata_to_sqlite() -> int:
    """
    One-time, idempotent migration from the old per-document
    metadata.json layout to SQLite. Skips any document_id already in
    the documents table, so it's safe to call on every startup --
    documents saved after the migration go straight to SQLite and
    are never touched by this function again.

    Returns the number of documents actually migrated.
    """
    documents_dir = Path(settings.DOCUMENTS_DIR)

    if not documents_dir.exists():
        return 0

    with get_connection() as conn:
        existing_ids = {
            row["document_id"]
            for row in conn.execute(
                "SELECT document_id FROM documents"
            ).fetchall()
        }

    migrated = 0

    for folder in documents_dir.iterdir():

        if not folder.is_dir():
            continue

        if folder.name in existing_ids:
            continue

        metadata_path = folder / "metadata.json"

        if not metadata_path.exists():
            continue

        with metadata_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        MetadataService.save(DocumentMetadata(**data))
        migrated += 1

    return migrated
