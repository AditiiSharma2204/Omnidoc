from app.embeddings.reranker_model import RerankerModel
from app.schemas.retrieval import RetrievedChunk


class RerankerService:
    """
    Re-scores retrieved chunks with a cross-encoder that sees the
    (query, chunk) pair jointly, rather than comparing independently
    computed embeddings/lexical scores. Meant to run AFTER
    retrieval/fusion, on a modest-sized candidate pool -- it's O(n)
    forward passes, not something to run over an entire corpus.
    """

    @staticmethod
    def rerank(
        query: str,
        candidates: list[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:

        if not candidates:
            return []

        model = RerankerModel.get_model()

        pairs = [(query, c.text) for c in candidates]
        scores = model.predict(pairs)

        reranked = [
            RetrievedChunk(
                **{**c.model_dump(), "score": float(score)}
            )
            for c, score in zip(candidates, scores)
        ]

        reranked.sort(key=lambda r: r.score, reverse=True)

        return reranked[:top_k]
