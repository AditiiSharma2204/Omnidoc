from typing import List, Dict, Any

from pydantic import BaseModel


class ParsedDocument(BaseModel):
    document_id: str
    title: str | None = None
    pages: int
    text: str
    tables: List[Dict[str, Any]] = []
    figures: List[Dict[str, Any]] = []
    metadata: Dict[str, Any] = {}