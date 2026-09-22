from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
import shutil

from fastapi import UploadFile

from app.config.settings import settings
from app.core.exceptions import DocumentNotFoundException
from app.parser.factory import ParserFactory
from app.schemas.document_metadata import DocumentMetadata
from app.services.metadata_service import MetadataService
from app.services.parsed_document_service import ParsedDocumentService
from app.services.validation_service import ValidationService
from app.services.chunking_service import ChunkingService
from app.services.embedding_service import EmbeddingService
from app.vectorstore.faiss_service import FAISSService


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
            upload_time=datetime.now(timezone.utc),
            status="uploaded",
        )

        MetadataService.save(document_folder, metadata)

        # -----------------------------------------
        # Step 5-7: Parse, chunk, embed, index
        # -----------------------------------------
        try:

            parser = ParserFactory.get_parser(str(save_path))
            parser_result = parser.parse(str(save_path))

            # The parser only sees the on-disk path (always
            # "original.<ext>"), so it can't know the user's
            # original filename. Stamp it here, once, so every
            # downstream consumer (chunks, sources, citations)
            # shows the real filename instead of "original.pdf".
            parser_result["title"] = file.filename

            ParsedDocumentService.save(
                document_folder,
                parser_result,
            )

            metadata.status = "parsed"
            MetadataService.save(document_folder, metadata)

            ChunkingService.chunk_document(
                document_folder
            )

            metadata.status = "chunked"
            MetadataService.save(document_folder, metadata)

            EmbeddingService.generate(
                document_folder
            )

            metadata.status = "embedded"
            MetadataService.save(document_folder, metadata)

            FAISSService.add_document(
                document_folder
            )

            metadata.status = "indexed"

        except Exception as e:

            metadata.status = "failed"
            metadata.parse_error = str(e)

        # -----------------------------------------
        # Step 8: Save final metadata
        # -----------------------------------------
        MetadataService.save(
            document_folder,
            metadata,
        )

        # -----------------------------------------
        # Step 9: Return response
        # -----------------------------------------
        return {
            "document_id": document_id,
            "filename": save_path.name,
            "original_filename": file.filename,
            "content_type": file.content_type,
            "file_size": file_size,
            "status": metadata.status,
            "message": (
                "Document uploaded successfully"
                if metadata.status == "indexed"
                else f"Document upload failed: {metadata.parse_error}"
            ),
        }

    @staticmethod
    def list_documents() -> list[DocumentMetadata]:
        """
        Returns metadata for every document currently on disk,
        newest first.
        """

        documents_dir = Path(settings.DOCUMENTS_DIR)

        if not documents_dir.exists():
            return []

        results = []

        for folder in documents_dir.iterdir():

            if not folder.is_dir():
                continue

            metadata = MetadataService.load(folder)

            if metadata is not None:
                results.append(metadata)

        results.sort(key=lambda m: m.upload_time, reverse=True)

        return results

    @staticmethod
    def delete_document(document_id: str) -> None:
        """
        Removes a document from disk and rebuilds the FAISS index
        without it.

        The index isn't ID-addressable (IndexFlatIP has no stable
        per-vector delete), so a full rebuild from the remaining
        documents' saved embeddings is the simplest correct option
        at this scale.
        """

        document_folder = Path(settings.DOCUMENTS_DIR) / document_id

        if not document_folder.exists():
            raise DocumentNotFoundException(document_id)

        shutil.rmtree(document_folder)

        FAISSService.rebuild_index(Path(settings.DOCUMENTS_DIR))
