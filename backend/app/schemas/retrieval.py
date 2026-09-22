from pydantic import BaseModel


class RetrievedChunk(BaseModel):
    score: float

    document_id: str

    chunk_id: str

    heading: str | None = None

    title: str | None = None

    section: str | None = None

    page: int | None = None

    text: str

    

class SearchResponse(BaseModel):
    query: str

    results: list[RetrievedChunk]