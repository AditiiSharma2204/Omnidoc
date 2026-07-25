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