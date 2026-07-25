from pydantic import BaseModel


class UploadResponse(BaseModel):
    document_id: str
    filename: str
    original_filename: str
    content_type: str
    file_size: int
    status: str
    message: str