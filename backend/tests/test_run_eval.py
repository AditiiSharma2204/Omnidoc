"""
Unit coverage for eval/run_eval.py's threshold-calibration and
document-scoped-eval math.

This doesn't load any real model -- RetrievalService.search (and, for
the document-scoped tests, DocumentService.list_documents) is
monkeypatched to return fixed, known values, so the stats/scoring
arithmetic can be verified without the live models that have
repeatedly SIGSEGV'd on this dev machine under memory pressure (see
README's Docker/threshold-calibration known-limitations entries).
It's a substitute for, not a replacement of, an eventual live run.
"""
from datetime import datetime, timezone

from app.schemas.document_metadata import DocumentMetadata
from app.schemas.retrieval import RetrievedChunk
from eval.run_eval import run_document_scoped_eval, run_threshold_calibration


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


def _doc(document_id: str, filename: str) -> DocumentMetadata:
    return DocumentMetadata(
        document_id=document_id,
        original_filename=filename,
        stored_filename="original.pdf",
        mime_type="application/pdf",
        size=1,
        upload_time=datetime.now(timezone.utc),
        status="indexed",
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


class TestRunDocumentScopedEval:
    """
    run_document_scoped_eval() compares unscoped vs. document_ids-
    scoped retrieval for questions tagged with source_document, to
    check whether scoping recovers recall the growing corpus was
    found to crowd out (see backend/eval/README.md). Fully mocked --
    RetrievalService.search and DocumentService.list_documents are
    both monkeypatched, no real models or DB.
    """

    def test_skips_items_without_source_document(self, monkeypatch):
        calls = []

        def fake_search(query, top_k, mode, rerank, document_ids=None):
            calls.append(document_ids)
            return [_chunk("Python here", score=1.0)]

        monkeypatch.setattr(
            "eval.run_eval.RetrievalService.search", fake_search
        )
        monkeypatch.setattr(
            "eval.run_eval.DocumentService.list_documents",
            lambda: [_doc("d1", "doc1.pdf")],
        )

        items = [
            {
                "id": "q1",
                "category": "factual",
                "question": "Q",
                "expected_keywords": ["Python"],
            }
        ]

        result = run_document_scoped_eval(items, top_k=1)

        assert result["num_questions"] == 0
        assert calls == []

    def test_scoped_search_receives_the_matching_document_id(
        self, monkeypatch
    ):
        captured = []

        def fake_search(query, top_k, mode, rerank, document_ids=None):
            captured.append(document_ids)
            return [_chunk("Python here", score=1.0)]

        monkeypatch.setattr(
            "eval.run_eval.RetrievalService.search", fake_search
        )
        monkeypatch.setattr(
            "eval.run_eval.DocumentService.list_documents",
            lambda: [_doc("d1", "doc1.pdf")],
        )

        items = [
            {
                "id": "q1",
                "category": "factual",
                "question": "Q",
                "expected_keywords": ["Python"],
                "source_document": "doc1.pdf",
            }
        ]

        run_document_scoped_eval(items, top_k=1)

        assert captured == [None, ["d1"]]

    def test_recovered_by_scoping_flagged_correctly(self, monkeypatch):
        def fake_search(query, top_k, mode, rerank, document_ids=None):
            if document_ids:
                return [_chunk("mentions Python", score=1.0)]
            return [_chunk("totally unrelated", score=1.0)]

        monkeypatch.setattr(
            "eval.run_eval.RetrievalService.search", fake_search
        )
        monkeypatch.setattr(
            "eval.run_eval.DocumentService.list_documents",
            lambda: [_doc("d1", "doc1.pdf")],
        )

        items = [
            {
                "id": "q1",
                "category": "factual",
                "question": "Q",
                "expected_keywords": ["Python"],
                "source_document": "doc1.pdf",
            }
        ]

        result = run_document_scoped_eval(items, top_k=1)
        q = result["per_question"][0]

        assert q["unscoped_hit"] is False
        assert q["scoped_hit"] is True
        assert q["recovered_by_scoping"] is True
        assert q["regressed_by_scoping"] is False
        assert result["unscoped_recall"] == 0.0
        assert result["scoped_recall"] == 1.0

    def test_unmatched_source_document_is_skipped_and_reported(
        self, monkeypatch
    ):
        monkeypatch.setattr(
            "eval.run_eval.RetrievalService.search",
            lambda **kwargs: [_chunk("Python here", score=1.0)],
        )
        monkeypatch.setattr(
            "eval.run_eval.DocumentService.list_documents",
            lambda: [_doc("d1", "other.pdf")],
        )

        items = [
            {
                "id": "q1",
                "category": "factual",
                "question": "Q",
                "expected_keywords": ["Python"],
                "source_document": "missing.pdf",
            }
        ]

        result = run_document_scoped_eval(items, top_k=1)

        assert result["num_questions"] == 0
        assert result["skipped_no_document_match"] == [
            {"id": "q1", "source_document": "missing.pdf"}
        ]
