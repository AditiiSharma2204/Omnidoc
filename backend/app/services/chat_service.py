from app.prompts.prompt_builder import PromptBuilder
from app.services.llm_service import LLMService
from app.services.retrieval_service import RetrievalService


class ChatService:
    """
    High-level service orchestrating Retrieval-Augmented Generation (RAG).
    """

    @classmethod
    def chat(
        cls,
        question: str,
        top_k: int = 5,
        document_ids: list[str] | None = None,
    ):
        """
        Normal (non-streaming) chat.
        """

        retrieved_chunks = RetrievalService.search(
            query=question,
            top_k=top_k,
            document_ids=document_ids,
        )

        system, user = PromptBuilder.build(
            question=question,
            retrieved_chunks=retrieved_chunks,
        )

        answer = LLMService.generate(system=system, user=user)

        sources = []

        seen = set()

        for chunk in retrieved_chunks:

            key = (
                chunk.document_id,
                chunk.heading,
                chunk.page,
            )

            if key in seen:
                continue

            seen.add(key)

            sources.append(
                {
                    "document": chunk.title,
                    "heading": chunk.heading,
                    "page": chunk.page,
                }
            )

        return {
            "answer": answer,
            "sources": sources,
        }

    @classmethod
    def stream(
        cls,
        question: str,
        top_k: int = 5,
        document_ids: list[str] | None = None,
    ):
        """
        Streaming chat.
        """

        retrieved_chunks = RetrievalService.search(
            query=question,
            top_k=top_k,
            document_ids=document_ids,
        )

        system, user = PromptBuilder.build(
            question=question,
            retrieved_chunks=retrieved_chunks,
        )

        return LLMService.stream(system=system, user=user)
