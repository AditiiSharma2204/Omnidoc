from app.db.database import get_connection
from app.schemas.document_metadata import DocumentMetadata


class MetadataService:
    """
    Document metadata persistence, backed by SQLite (was per-document
    JSON files under storage/documents/<id>/metadata.json). Moved for
    the reasons any "flat files as a database" setup eventually hits:
    no atomic multi-field updates, no efficient listing without
    scanning every folder, and no real query capability (e.g.
    "documents with status=failed") without reading everything into
    memory first.
    """

    @staticmethod
    def save(metadata: DocumentMetadata) -> None:
        with get_connection() as conn:
            conn.execute(
                """
                INSERT INTO documents (
                    document_id, original_filename, stored_filename,
                    mime_type, size, upload_time, status, parse_error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(document_id) DO UPDATE SET
                    original_filename = excluded.original_filename,
                    stored_filename = excluded.stored_filename,
                    mime_type = excluded.mime_type,
                    size = excluded.size,
                    upload_time = excluded.upload_time,
                    status = excluded.status,
                    parse_error = excluded.parse_error
                """,
                (
                    metadata.document_id,
                    metadata.original_filename,
                    metadata.stored_filename,
                    metadata.mime_type,
                    metadata.size,
                    metadata.upload_time.isoformat(),
                    metadata.status,
                    metadata.parse_error,
                ),
            )

    @staticmethod
    def load(document_id: str) -> DocumentMetadata | None:
        with get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE document_id = ?",
                (document_id,),
            ).fetchone()

        if row is None:
            return None

        return DocumentMetadata(**dict(row))

    @staticmethod
    def list_all() -> list[DocumentMetadata]:
        """
        Newest first.
        """
        with get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM documents ORDER BY upload_time DESC"
            ).fetchall()

        return [DocumentMetadata(**dict(row)) for row in rows]

    @staticmethod
    def delete(document_id: str) -> None:
        with get_connection() as conn:
            conn.execute(
                "DELETE FROM documents WHERE document_id = ?",
                (document_id,),
            )
