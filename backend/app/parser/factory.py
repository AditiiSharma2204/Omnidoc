from pathlib import Path

from app.parser.pdf_parser import PDFParser


class ParserFactory:

    @staticmethod
    def get_parser(file_path: str):

        extension = Path(file_path).suffix.lower()

        if extension == ".pdf":
            return PDFParser()

        raise ValueError(f"Unsupported file type: {extension}")