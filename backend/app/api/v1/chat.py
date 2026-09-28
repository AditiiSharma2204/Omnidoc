from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.core.exceptions import ConversationNotFoundException
from app.schemas.chat import (
    ChatRequest,
    ChatResponse,
    ConversationHistoryResponse,
)
from app.services.chat_service import ChatService
from app.services.conversation_service import ConversationService

router = APIRouter()


@router.post(
    "/chat",
    response_model=ChatResponse,
)
def chat(request: ChatRequest):

    return ChatService.chat(
        question=request.question,
        top_k=request.top_k,
        document_ids=request.document_ids,
        conversation_id=request.conversation_id,
    )


@router.post(
    "/chat/stream",
)
def chat_stream(request: ChatRequest):

    generator = ChatService.stream(
        question=request.question,
        top_k=request.top_k,
        document_ids=request.document_ids,
        conversation_id=request.conversation_id,
    )

    return StreamingResponse(
        generator,
        media_type="application/x-ndjson",
    )


@router.get(
    "/conversations/{conversation_id}",
    response_model=ConversationHistoryResponse,
)
def get_conversation(conversation_id: str):

    if not ConversationService.conversation_exists(conversation_id):
        raise ConversationNotFoundException(conversation_id)

    return ConversationHistoryResponse(
        conversation_id=conversation_id,
        messages=ConversationService.get_full_history(conversation_id),
    )
