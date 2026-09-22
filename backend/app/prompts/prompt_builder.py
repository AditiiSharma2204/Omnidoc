from app.config.settings import settings
from app.schemas.retrieval import RetrievedChunk

# Rough heuristic (no tokenizer dependency): ~4 characters/token for
# English text. Good enough to keep us safely under num_ctx instead
# of silently truncating mid-prompt inside Ollama.
CHARS_PER_TOKEN = 4
RESERVED_TOKENS_FOR_SYSTEM_AND_ANSWER = 1024


class PromptBuilder:
    """
    Builds prompts for the LLM using retrieved document chunks.

    Returns a (system, user) pair rather than one blob, so the
    instructions can be sent as an actual `system` message instead
    of being smuggled into the `user` turn.
    """

    SYSTEM_PROMPT = (
        "You are OmniDoc AI, a document question-answering assistant.\n\n"
        "Rules:\n"
        "1. Answer only using the supplied document context. Never invent facts.\n"
        "2. If the answer is not present in the context, say exactly: "
        "\"I couldn't find that information in the uploaded documents.\"\n"
        "3. Section headings organize the context; they are not answers "
        "themselves unless the question is literally about document structure.\n"
        "4. When multiple context passages describe the same thing, combine "
        "them into one coherent answer instead of repeating each separately.\n"
        "5. Ignore near-duplicate passages.\n"
        "6. Keep answers concise, factual and well formatted (use lists or "
        "short paragraphs where that helps readability).\n"
        "7. When you use a fact from a specific document, you may refer to it "
        "by its document title."
    )

    NO_CONTEXT_MESSAGE = (
        "I couldn't find that information in the uploaded documents."
    )

    @classmethod
    def build(
        cls,
        question: str,
        retrieved_chunks: list[RetrievedChunk],
    ) -> tuple[str, str]:
        """
        Build the (system, user) messages for the language model.
        """

        if not retrieved_chunks:
            # Nothing relevant was retrieved. Tell the model there is
            # no context instead of sending an empty context block,
            # which otherwise invites the model to answer from its
            # own (unverifiable) knowledge.
            user = (
                f"No document context was found for this question.\n\n"
                f"QUESTION:\n{question}\n\n"
                f"Respond with: \"{cls.NO_CONTEXT_MESSAGE}\""
            )
            return cls.SYSTEM_PROMPT, user

        retrieved_chunks = cls._fit_to_context_budget(retrieved_chunks)

        context_sections = []

        for i, chunk in enumerate(retrieved_chunks, start=1):

            lines = [
                "==============================",
                f"Context {i}",
                "",
                "Document:",
                str(chunk.title),
            ]

            if chunk.section:
                lines += ["", "Section:", chunk.section]

            lines += [
                "",
                "Heading:",
                str(chunk.heading),
                "",
                "Content:",
                chunk.text,
            ]

            context_sections.append("\n".join(lines))

        context = "\n\n".join(context_sections)

        user = (
            f"DOCUMENT CONTEXT\n\n"
            f"{context}\n\n"
            f"==============================\n"
            f"QUESTION\n\n"
            f"{question}"
        )

        return cls.SYSTEM_PROMPT, user

    @classmethod
    def _fit_to_context_budget(
        cls,
        retrieved_chunks: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        """
        Drops the lowest-ranked chunks (retrieved_chunks is assumed
        sorted best-first) until the estimated prompt size fits
        under `settings.LLM_NUM_CTX`.

        Without this, Ollama silently truncates prompts that exceed
        num_ctx, which can cut off the question itself and produce a
        confidently wrong answer with no error surfaced anywhere.
        """

        budget_chars = (
            settings.LLM_NUM_CTX - RESERVED_TOKENS_FOR_SYSTEM_AND_ANSWER
        ) * CHARS_PER_TOKEN

        budget_chars = max(budget_chars, 0)

        kept = []
        used_chars = 0

        for chunk in retrieved_chunks:

            chunk_chars = len(chunk.text) + len(str(chunk.title)) + 100

            if kept and used_chars + chunk_chars > budget_chars:
                break

            kept.append(chunk)
            used_chars += chunk_chars

        return kept or retrieved_chunks[:1]
