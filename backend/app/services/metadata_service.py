import json
from pathlib import Path

from app.schemas.document_metadata import DocumentMetadata


class MetadataService:

    @staticmethod
    def save(document_folder: Path, metadata: DocumentMetadata):

        metadata_path = document_folder / "metadata.json"

        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(
                metadata.model_dump(mode="json"),
                f,
                indent=4,
            )

    @staticmethod
    def load(document_folder: Path) -> DocumentMetadata | None:

        metadata_path = document_folder / "metadata.json"

        if not metadata_path.exists():
            return None

        with metadata_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        return DocumentMetadata(**data)