from app.prompts.prompt_builder import PromptBuilder
from app.services.citation_service import CitationService
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

        system, user, contexts = PromptBuilder.build(
            question=question,
            retrieved_chunks=retrieved_chunks,
        )

        answer = LLMService.generate(system=system, user=user)

        sources = cls._build_sources(answer, contexts)

        return {
            "answer": answer,
            "sources": sources,
        }

    @staticmethod
    def _build_sources(answer: str, contexts: list) -> list[dict]:
        """
        Builds the sources list with indices that line up exactly
        with the model's [N] citation markers (`contexts` is
        PromptBuilder's own "Context N" ordering, not re-derived), and
        flags which ones the model actually cited. Every retrieved
        context is included, not just cited ones, so the UI can show
        "used" vs. "available but not cited" -- and so a citation to
        an out-of-range number (a hallucinated reference) is visible
        as simply not matching any `index` here, rather than silently
        dropped.
        """
        cited_indices = CitationService.extract_cited_indices(answer)

        return [
            {
                "index": i,
                "document": chunk.title,
                "heading": chunk.heading,
                "page": chunk.page,
                "cited": i in cited_indices,
            }
            for i, chunk in enumerate(contexts, start=1)
        ]

    @classmethod
    def stream(
        cls,
        question: str,
        top_k: int = 5,
        document_ids: list[str] | None = None,
    ):
        """
        Streaming chat.

        Note: sources/citations aren't available here the way they
        are in `chat()` -- the caller only gets a token stream, with
        no way to attach the sources list once generation finishes.
        Wiring citations into the streaming response is tracked
        alongside "wire up streaming to the frontend" on the roadmap;
        both need the same change (a structured SSE response with a
        final sources event, not a bare text stream).
        """

        retrieved_chunks = RetrievalService.search(
            query=question,
            top_k=top_k,
            document_ids=document_ids,
        )

        system, user, _contexts = PromptBuilder.build(
            question=question,
            retrieved_chunks=retrieved_chunks,
        )

        return LLMService.stream(system=system, user=user)
