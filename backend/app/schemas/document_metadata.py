from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class DocumentMetadata(BaseModel):
    document_id: str
    original_filename: str
    stored_filename: str
    mime_type: str
    size: int

    upload_time: datetime

    status: str

    parse_error: Optional[str] = None