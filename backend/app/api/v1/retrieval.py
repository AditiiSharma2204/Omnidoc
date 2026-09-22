from fastapi import APIRouter

from app.schemas.search import SearchRequest
from app.schemas.retrieval import SearchResponse
from app.services.retrieval_service import RetrievalService

router = APIRouter(
    prefix="/search",
    tags=["Semantic Search"],
)


@router.post(
    "",
    response_model=SearchResponse,
)
async def search_documents(
    request: SearchRequest,
):
    """
    Perform semantic search over indexed documents.
    """

    results = RetrievalService.search(
        query=request.query,
        top_k=request.top_k,
    )

    return SearchResponse(
        query=request.query,
        results=results,
    )