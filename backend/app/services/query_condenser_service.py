from app.services.llm_service import LLMService

SYSTEM_PROMPT = (
    "Given a conversation history and a follow-up question, rewrite the "
    "follow-up as a standalone question that makes sense without the "
    "history -- resolve pronouns and implicit references (e.g. \"her\", "
    "\"that\", \"the second one\") using the history. If the follow-up "
    "is already standalone, return it unchanged. Output ONLY the "
    "rewritten question and nothing else -- no quotes, no preamble."
)


class QueryCondenserService:
    """
    Condenses a follow-up like "what about her second job?" into a
    standalone retrieval query ("what was Aditii's second job?") using
    conversation history, since RetrievalService.search() only ever
    sees the raw current question and has no access to prior turns.

    Distinct from QueryRewriteService's HyDE rewrite: this runs BEFORE
    retrieval and rewrites the QUESTION itself using real conversation
    history, not a fabricated hypothetical answer to a standalone one.
    """

    @staticmethod
    def condense(question: str, history: list[dict]) -> str:
        if not history:
            return question

        transcript = "\n".join(
            f"{message['role']}: {message['content']}"
            for message in history
        )
        user = (
            f"Conversation history:\n{transcript}\n\n"
            f"Follow-up question: {question}"
        )

        rewritten = LLMService.generate(
            system=SYSTEM_PROMPT,
            user=user,
            temperature=0.0,
        )

        return rewritten.strip() or question
