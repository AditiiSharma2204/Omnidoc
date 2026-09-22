from pathlib import Path

from app.core.constants import DOCLING_EXTENSIONS
from app.parser.docling_parser import DoclingParser


class ParserFactory:

    @staticmethod
    def get_parser(file_path: str):

        extension = Path(file_path).suffix.lower()

        if extension in DOCLING_EXTENSIONS:
            return DoclingParser()

        raise ValueError(f"Unsupported file type: {extension}")
