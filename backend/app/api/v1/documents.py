from fastapi import APIRouter, UploadFile, File

from app.schemas.document import UploadResponse
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