"""
Retrieval and answer-quality metrics for the evaluation harness.

Kept dependency-free (no numpy/sklearn) and pure so they're trivial
to unit test and to reuse from a notebook.
"""
from __future__ import annotations

from dataclasses import dataclass, field


def _chunk_is_relevant(chunk_text: str, keywords: list[str]) -> bool:
    """
    A retrieved chunk counts as "relevant" to a question if it
    contains at least one of the question's expected keywords
    (case-insensitive substring match).

    This is a deliberately simple proxy -- it doesn't require gold
    chunk IDs (which would mean hand-labeling exact chunk boundaries
    that shift every time chunking changes). It trades some
    precision for being cheap to maintain across chunking changes.
    """
    text_lower = chunk_text.lower()
    return any(kw.lower() in text_lower for kw in keywords)


def recall_at_k(
    retrieved_texts: list[str],
    expected_keywords: list[str],
    k: int,
) -> float:
    """
    1.0 if any of the top-k retrieved chunks contains an expected
    keyword, else 0.0. (Binary per-question recall; average across
    questions for the usual "Recall@k" summary number.)
    """
    if not expected_keywords:
        raise ValueError("expected_keywords must be non-empty")

    top_k = retrieved_texts[:k]
    return 1.0 if any(
        _chunk_is_relevant(t, expected_keywords) for t in top_k
    ) else 0.0


def reciprocal_rank(
    retrieved_texts: list[str],
    expected_keywords: list[str],
) -> float:
    """
    1 / rank of the first relevant chunk (1-indexed), or 0.0 if none
    of the retrieved chunks are relevant.
    """
    for i, text in enumerate(retrieved_texts, start=1):
        if _chunk_is_relevant(text, expected_keywords):
            return 1.0 / i
    return 0.0


def answer_contains_expected_facts(
    answer: str,
    expected_keywords: list[str],
    min_fraction: float = 1.0,
) -> bool:
    """
    Crude faithfulness/completeness proxy: did the generated answer
    mention at least `min_fraction` of the expected keywords?

    This is NOT a substitute for an LLM-judge or human review -- it
    will pass answers that happen to contain the right words in the
    wrong context, and fail correct answers that paraphrase instead
    of quoting. Treat it as a fast regression signal, not ground
    truth; the harness report flags every answer for the seed set to
    still be read by a human.
    """
    if not expected_keywords:
        raise ValueError("expected_keywords must be non-empty")

    answer_lower = answer.lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
    return (hits / len(expected_keywords)) >= min_fraction


def is_refusal(answer: str, refusal_marker: str) -> bool:
    return refusal_marker.lower() in answer.lower()


@dataclass
class AggregateResults:
    num_questions: int = 0
    recall_at_k_sum: float = 0.0
    mrr_sum: float = 0.0
    answer_facts_hit_sum: float = 0.0
    per_question: list[dict] = field(default_factory=list)

    @property
    def recall_at_k(self) -> float:
        return self.recall_at_k_sum / self.num_questions if self.num_questions else 0.0

    @property
    def mrr(self) -> float:
        return self.mrr_sum / self.num_questions if self.num_questions else 0.0

    @property
    def answer_fact_hit_rate(self) -> float:
        return (
            self.answer_facts_hit_sum / self.num_questions
            if self.num_questions
            else 0.0
        )
