import json
from pathlib import Path


class ParsedDocumentService:

    @staticmethod
    def save(
        document_folder: Path,
        parsed_document: dict,
    ):

        parsed_path = document_folder / "parsed.json"

        with parsed_path.open(
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                parsed_document,
                f,
                indent=4,
                ensure_ascii=False,
            )