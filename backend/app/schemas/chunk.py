from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    chunk_id: str = Field(default_factory=lambda: str(uuid4()))

    text: str

    page: int | None = None

    metadata: dict[str, Any] = Field(default_factory=dict)