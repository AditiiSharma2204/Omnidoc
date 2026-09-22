from datetime import datetime

from pydantic import BaseModel


class UploadResponse(BaseModel):
    document_id: str
    filename: str
    original_filename: str
    content_type: str
    file_size: int
    status: str
    message: str


class DocumentSummary(BaseModel):
    document_id: str
    original_filename: str
    mime_type: str
    size: int
    upload_time: datetime
    status: str
    parse_error: str | None = None


class DocumentListResponse(BaseModel):
    documents: list[DocumentSummary]


class DeleteResponse(BaseModel):
    document_id: str
    message: str