from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.schemas.chat import ChatRequest, ChatResponse
from app.services.chat_service import ChatService

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
    )


@router.post(
    "/chat/stream",
)
def chat_stream(request: ChatRequest):

    generator = ChatService.stream(
        question=request.question,
        top_k=request.top_k,
        document_ids=request.document_ids,
    )

    return StreamingResponse(
        generator,
        media_type="text/plain",
    )