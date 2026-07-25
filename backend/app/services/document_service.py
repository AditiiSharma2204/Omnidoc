from datetime import datetime
from pathlib import Path
from uuid import uuid4
import shutil

from fastapi import UploadFile

from app.config.settings import settings
from app.parser.factory import ParserFactory
from app.schemas.document_metadata import DocumentMetadata
from app.services.metadata_service import MetadataService
from app.services.parsed_document_service import ParsedDocumentService
from app.services.validation_service import ValidationService


class DocumentService:

    async def save_document(self, file: UploadFile):

        # -----------------------------------------
        # Step 1: Validate uploaded file
        # -----------------------------------------
        file_size = await ValidationService.validate(file)

        # -----------------------------------------
        # Step 2: Create document folder
        # -----------------------------------------
        document_id = str(uuid4())

        extension = Path(file.filename).suffix.lower()

        document_folder = Path(settings.DOCUMENTS_DIR) / document_id
        document_folder.mkdir(parents=True, exist_ok=True)

        save_path = document_folder / f"original{extension}"

        # -----------------------------------------
        # Step 3: Save original document
        # -----------------------------------------
        with save_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # -----------------------------------------
        # Step 4: Create initial metadata
        # -----------------------------------------
        metadata = DocumentMetadata(
            document_id=document_id,
            original_filename=file.filename,
            stored_filename=save_path.name,
            mime_type=file.content_type,
            size=file_size,
            upload_time=datetime.utcnow(),
            status="uploaded",
        )

        MetadataService.save(document_folder, metadata)

        parser_result = None

        # -----------------------------------------
        # Step 5: Parse document
        # -----------------------------------------
        try:

            parser = ParserFactory.get_parser(str(save_path))

            parser_result = parser.parse(str(save_path))

            ParsedDocumentService.save(
                document_folder,
                parser_result,
            )

            metadata.status = "parsed"

        except Exception as e:

            metadata.status = "parse_failed"
            metadata.parse_error = str(e)

        # -----------------------------------------
        # Step 6: Save updated metadata
        # -----------------------------------------
        MetadataService.save(document_folder, metadata)

        # -----------------------------------------
        # Step 7: Return response
        # -----------------------------------------
        return {
            "document_id": document_id,
            "filename": save_path.name,
            "original_filename": file.filename,
            "content_type": file.content_type,
            "file_size": file_size,
            "status": metadata.status,
            "message": "Document uploaded successfully",
        }