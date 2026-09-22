from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

from app.config.settings import settings


class DoclingService:
    def __init__(self):

        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = settings.DOCLING_DO_OCR
        pipeline_options.do_table_structure = (
            settings.DOCLING_DO_TABLE_STRUCTURE
        )

        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options
                ),
            }
        )

    def convert(self, file_path: str):
        """
        Convert a document using Docling.
        Returns the raw Docling document object.
        """
        result = self.converter.convert(file_path)
        return result.document
