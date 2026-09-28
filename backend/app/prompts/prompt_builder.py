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

    Returns a (system, user, contexts) triple rather than one blob:
    system/user so the instructions can be sent as an actual `system`
    message instead of being smuggled into the `user` turn, and
    `contexts` (the exact, budget-trimmed list of chunks that ended
    up in the prompt, in "Context N" order) so a caller can build a
    sources list whose indices line up exactly with the [N] citation
    markers the model was told to use. Building `contexts` any other
    way (e.g. re-deriving it from the original retrieved_chunks list)
    risks a citation pointing at the wrong source if budget trimming
    dropped anything.
    """

    SYSTEM_PROMPT = (
        "You are OmniDoc AI, a document question-answering assistant.\n\n"
        "CITATION FORMAT (required on every answer): end each sentence "
        "or bullet that states a fact from the context with the context "
        "number in square brackets, e.g. [1]. Use [2][3] if two contexts "
        "both support it. Example:\n"
        "  \"She interned at Acme Corp in Chennai [1]. She also built a "
        "KYC pipeline [2].\"\n"
        "Every factual sentence needs a bracket. A sentence with no "
        "bracket is treated as unsupported.\n\n"
        "Other rules:\n"
        "1. Answer only using the supplied document context and/or the "
        "earlier conversation turns above (if any). Never invent facts "
        "beyond what's in either of those.\n"
        "2. If the answer is not present in the context AND not in the "
        "earlier conversation, say exactly: \"I couldn't find that "
        "information in the uploaded documents.\" Do not refuse just "
        "because the current context alone doesn't have it -- check "
        "the earlier conversation first.\n"
        "3. Section headings organize the context; they are not answers "
        "themselves unless the question is literally about document structure.\n"
        "4. When multiple context passages describe the same thing, combine "
        "them into one coherent answer instead of repeating each separately.\n"
        "5. Ignore near-duplicate passages.\n"
        "6. Keep answers concise, factual and well formatted (use lists or "
        "short paragraphs where that helps readability) -- every list item "
        "still needs its own [N] bracket.\n"
        "7. Do not cite a context number you did not actually use. A fact "
        "you're repeating from earlier conversation (not from the current "
        "context) doesn't need a new [N] -- it was already cited when "
        "first stated."
    )

    CITATION_REMINDER = (
        "Reminder: every factual sentence or bullet above must end with "
        "a [N] bracket citing the context number it came from."
    )

    NO_CONTEXT_MESSAGE = (
        "I couldn't find that information in the uploaded documents."
    )

    @classmethod
    def build(
        cls,
        question: str,
        retrieved_chunks: list[RetrievedChunk],
    ) -> tuple[str, str, list[RetrievedChunk]]:
        """
        Build the (system, user, contexts) triple for the language
        model. `contexts` is empty when there was no context to
        answer from.
        """

        if not retrieved_chunks:
            # Nothing relevant was retrieved. Tell the model there is
            # no NEW document context instead of sending an empty
            # context block, which otherwise invites the model to
            # answer from its own (unverifiable) knowledge. Still
            # allow answering from earlier conversation turns (passed
            # separately as real chat history, not part of this
            # string) -- an earlier version of this message flatly
            # commanded the refusal text regardless of history, which
            # made a trivial "what did you just say" follow-up refuse
            # even when the answer was sitting right there in the
            # previous turn.
            user = (
                f"No new document context was found for this specific "
                f"question.\n\n"
                f"QUESTION:\n{question}\n\n"
                f"If the earlier conversation above already answers "
                f"this, answer from that (no new [N] citation needed "
                f"for a fact you're just repeating). Otherwise, "
                f"respond with exactly: \"{cls.NO_CONTEXT_MESSAGE}\""
            )
            return cls.SYSTEM_PROMPT, user, []

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
            f"{question}\n\n"
            f"==============================\n"
            f"{cls.CITATION_REMINDER}"
        )

        return cls.SYSTEM_PROMPT, user, retrieved_chunks

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
