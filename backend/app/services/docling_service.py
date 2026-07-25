from docling.document_converter import DocumentConverter


class DoclingService:
    def __init__(self):
        self.converter = DocumentConverter()

    def convert(self, file_path: str):
        """
        Convert a document using Docling.
        Returns the raw Docling document object.
        """
        result = self.converter.convert(file_path)
        return result.document