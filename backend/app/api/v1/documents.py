from fastapi import APIRouter, UploadFile, File

from app.schemas.document import (
    DeleteResponse,
    DocumentListResponse,
    DocumentSummary,
    UploadResponse,
)
from app.services.document_service import DocumentService

router = APIRouter()

service = DocumentService()


@router.post(
    "/upload",
    response_model=UploadResponse,
)
async def upload_document(
    file: UploadFile = File(...)
):
    return await service.save_document(file)


@router.get(
    "",
    response_model=DocumentListResponse,
)
def list_documents():
    metadata = DocumentService.list_documents()

    return DocumentListResponse(
        documents=[
            DocumentSummary(**m.model_dump())
            for m in metadata
        ]
    )


@router.delete(
    "/{document_id}",
    response_model=DeleteResponse,
)
def delete_document(document_id: str):
    DocumentService.delete_document(document_id)

    return DeleteResponse(
        document_id=document_id,
        message="Document deleted successfully",
    )
