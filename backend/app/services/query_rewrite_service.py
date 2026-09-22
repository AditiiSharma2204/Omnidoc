from app.services.llm_service import LLMService

SYSTEM_PROMPT = (
    "You write a single short, plausible sentence that WOULD answer "
    "the user's question, as if it were a sentence extracted directly "
    "from a document. Do not hedge, do not say you don't know, and do "
    "not add commentary -- just the hypothetical sentence itself. If "
    "you don't actually know the real answer, invent a generic but "
    "plausible one; it will only be used to help a search engine find "
    "similar real text, never shown to a user or treated as fact."
)


class QueryRewriteService:
    """
    HyDE-style query rewriting: asks the LLM to write a short
    hypothetical answer to a question, so THAT text (rather than the
    raw question) can be embedded for dense retrieval.

    Why this helps: a query like "What is this person's name?" shares
    no vocabulary with a chunk whose content is just "## Aditii
    Sharma" -- there's nothing for embedding similarity or lexical
    overlap to match on. A hypothetical answer like "The person's
    name is Aditii Sharma." is written in the same register and
    vocabulary as the real chunk, so it embeds much closer to it.

    The hypothetical text is fabricated by design and must never be
    shown to the user or used as an answer -- it exists purely to
    generate a better search vector.
    """

    @staticmethod
    def generate_hypothetical_answer(question: str) -> str:
        # Deliberately not routed through PromptBuilder: this is a
        # short, low-stakes generation step with its own narrow
        # system prompt, not a real user-facing answer.
        return LLMService.generate(
            system=SYSTEM_PROMPT,
            user=question,
            temperature=0.3,
        )
