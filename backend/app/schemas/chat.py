from pydantic import BaseModel


class ChatRequest(BaseModel):
    question: str
    top_k: int = 5
    document_ids: list[str] | None = None


class Source(BaseModel):
    index: int
    document: str
    heading: str | None = None
    page: int | None = None
    cited: bool


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]