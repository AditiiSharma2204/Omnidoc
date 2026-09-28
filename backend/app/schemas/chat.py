from datetime import datetime

from pydantic import BaseModel


class ChatRequest(BaseModel):
    question: str
    top_k: int = 5
    document_ids: list[str] | None = None
    # Omit on the first turn; pass back the conversation_id the
    # response returned to continue that conversation on later turns.
    # An unknown or omitted id starts a new conversation.
    conversation_id: str | None = None


class Source(BaseModel):
    index: int
    document: str
    heading: str | None = None
    page: int | None = None
    cited: bool


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]
    conversation_id: str


class MessageOut(BaseModel):
    role: str
    content: str
    sources: list[Source] | None = None
    created_at: datetime


class ConversationHistoryResponse(BaseModel):
    conversation_id: str
    messages: list[MessageOut]