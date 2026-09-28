import json

from app.prompts.prompt_builder import PromptBuilder
from app.services.citation_service import CitationService
from app.services.conversation_service import ConversationService
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
        conversation_id: str | None = None,
    ):
        """
        Normal (non-streaming) chat.

        Retrieval runs on the raw `question` only, with or without a
        conversation -- condensing a follow-up like "what about her
        second job?" into a standalone retrieval query is a separate,
        not-yet-built piece (query rewriting for follow-ups, distinct
        from the HyDE query_rewrite feature already on RetrievalService).
        Conversation memory here only makes *generation* aware of
        prior turns: history is fetched before this question is
        answered and included as real chat turns, so the model can
        naturally reference what it already said.
        """

        conversation_id = ConversationService.get_or_create(
            conversation_id
        )
        history = ConversationService.get_recent_messages(
            conversation_id
        )

        retrieved_chunks = RetrievalService.search(
            query=question,
            top_k=top_k,
            document_ids=document_ids,
        )

        system, user, contexts = PromptBuilder.build(
            question=question,
            retrieved_chunks=retrieved_chunks,
        )

        answer = LLMService.generate(
            system=system, user=user, history=history
        )

        sources = cls._build_sources(answer, contexts)

        ConversationService.add_message(
            conversation_id, role="user", content=question
        )
        ConversationService.add_message(
            conversation_id,
            role="assistant",
            content=answer,
            sources=sources,
        )

        return {
            "answer": answer,
            "sources": sources,
            "conversation_id": conversation_id,
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
        conversation_id: str | None = None,
    ):
        """
        Streaming chat.

        Yields newline-delimited JSON events so the caller gets both
        the token stream AND, once generation finishes, the same
        sources/citations and conversation persistence `chat()`
        provides -- previously a documented gap (bare text stream,
        no sources, no history saved). Event shapes:
          {"type": "token", "token": "..."}
          {"type": "done", "conversation_id": "...", "sources": [...]}
        """

        conversation_id = ConversationService.get_or_create(
            conversation_id
        )
        history = ConversationService.get_recent_messages(
            conversation_id
        )

        retrieved_chunks = RetrievalService.search(
            query=question,
            top_k=top_k,
            document_ids=document_ids,
        )

        system, user, contexts = PromptBuilder.build(
            question=question,
            retrieved_chunks=retrieved_chunks,
        )

        def _generate():
            answer_parts = []

            for token in LLMService.stream(
                system=system, user=user, history=history
            ):
                answer_parts.append(token)
                yield json.dumps({"type": "token", "token": token}) + "\n"

            answer = "".join(answer_parts)
            sources = cls._build_sources(answer, contexts)

            ConversationService.add_message(
                conversation_id, role="user", content=question
            )
            ConversationService.add_message(
                conversation_id,
                role="assistant",
                content=answer,
                sources=sources,
            )

            yield json.dumps(
                {
                    "type": "done",
                    "conversation_id": conversation_id,
                    "sources": sources,
                }
            ) + "\n"

        return _generate()
