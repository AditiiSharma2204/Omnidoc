from pathlib import Path

from app.parser.base_parser import BaseParser
from app.schemas.parsed_document import ParsedDocument
from app.services.docling_service import DoclingService


class PDFParser(BaseParser):

    def __init__(self):
        self.docling = DoclingService()

    def parse(self, file_path: str):

        document = self.docling.convert(file_path)

        text = document.export_to_markdown()

        pages = len(getattr(document, "pages", []))

        parsed = ParsedDocument(
            document_id=Path(file_path).parent.name,
            title=Path(file_path).name,
            pages=pages,
            text=text,
            tables=[],
            figures=[],
            metadata={},
        )

        return parsed.model_dump()