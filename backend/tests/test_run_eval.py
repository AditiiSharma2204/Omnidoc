"""
Unit coverage for eval/run_eval.py's threshold-calibration math.

This doesn't load any real model -- RetrievalService.search is
monkeypatched to return fixed, known-scored chunks, so the stats/
threshold-sweep arithmetic can be verified without the live models
that have repeatedly SIGSEGV'd on this dev machine under memory
pressure (see README's Docker/threshold-calibration known-limitations
entries). It's a substitute for, not a replacement of, an eventual
live run.
"""
from app.schemas.retrieval import RetrievedChunk
from eval.run_eval import run_threshold_calibration


def _chunk(text: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        score=score,
        document_id="doc1",
        chunk_id="c1",
        heading="Heading",
        title="doc1.pdf",
        section=None,
        page=1,
        text=text,
    )


class TestRunThresholdCalibration:

    def test_splits_relevant_and_irrelevant_by_keyword(self, monkeypatch):
        def fake_search(query, top_k, mode, rerank):
            return [
                _chunk("mentions Python here", score=5.0),
                _chunk("totally unrelated content", score=-2.0),
            ]

        monkeypatch.setattr(
            "eval.run_eval.RetrievalService.search", fake_search
        )

        items = [
            {
                "id": "q1",
                "category": "factual",
                "question": "What language?",
                "expected_keywords": ["Python"],
            }
        ]

        result = run_threshold_calibration(items, top_k=2)

        assert result["relevant"]["n"] == 1
        assert result["relevant"]["min"] == 5.0
        assert result["irrelevant"]["n"] == 1
        assert result["irrelevant"]["min"] == -2.0

    def test_skips_unanswerable_items(self, monkeypatch):
        calls = []

        def fake_search(query, top_k, mode, rerank):
            calls.append(query)
            return [_chunk("irrelevant", score=1.0)]

        monkeypatch.setattr(
            "eval.run_eval.RetrievalService.search", fake_search
        )

        items = [
            {
                "id": "q1",
                "category": "unanswerable",
                "question": "Unanswerable?",
                "expected_keywords": [],
            }
        ]

        run_threshold_calibration(items, top_k=2)

        assert calls == []

    def test_threshold_sweep_counts_drops_correctly(self, monkeypatch):
        def fake_search(query, top_k, mode, rerank):
            return [
                _chunk("mentions Python here", score=-4.0),
                _chunk("mentions Python too", score=2.0),
                _chunk("unrelated", score=-6.0),
                _chunk("also unrelated", score=0.5),
            ]

        monkeypatch.setattr(
            "eval.run_eval.RetrievalService.search", fake_search
        )

        items = [
            {
                "id": "q1",
                "category": "factual",
                "question": "What language?",
                "expected_keywords": ["Python"],
            }
        ]

        result = run_threshold_calibration(items, top_k=4)

        # relevant_scores = [-4.0, 2.0], irrelevant_scores = [-6.0, 0.5]
        by_threshold = {
            row["threshold"]: row for row in result["threshold_sweep"]
        }

        # threshold -3.0: drops the -4.0 relevant chunk (1) and the
        # -6.0 irrelevant chunk (1); the 0.5 irrelevant chunk survives.
        assert by_threshold[-3.0]["relevant_dropped"] == 1
        assert by_threshold[-3.0]["irrelevant_dropped"] == 1

        # threshold 1.0: drops both relevant... no, only -4.0 (< 1.0),
        # 2.0 survives; drops both irrelevant (-6.0 and 0.5, both < 1.0).
        assert by_threshold[1.0]["relevant_dropped"] == 1
        assert by_threshold[1.0]["irrelevant_dropped"] == 2

        # The min relevant score itself (-4.0) is always included as a
        # candidate and drops zero relevant chunks by construction.
        assert by_threshold[-4.0]["relevant_dropped"] == 0

    def test_empty_relevant_scores_handled(self, monkeypatch):
        def fake_search(query, top_k, mode, rerank):
            return [_chunk("nothing matches", score=3.0)]

        monkeypatch.setattr(
            "eval.run_eval.RetrievalService.search", fake_search
        )

        items = [
            {
                "id": "q1",
                "category": "factual",
                "question": "What language?",
                "expected_keywords": ["Python"],
            }
        ]

        result = run_threshold_calibration(items, top_k=1)

        assert result["relevant"] == {"n": 0}
        assert result["irrelevant"]["n"] == 1
