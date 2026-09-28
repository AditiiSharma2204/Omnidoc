"""
Evaluation harness for OmniDoc's retrieval + generation pipeline.

Run from the `backend/` directory (so the `app` package resolves):

    python -m eval.run_eval                    # retrieval only (fast, no LLM calls)
    python -m eval.run_eval --with-generation   # also runs full chat (slow; needs Ollama running)
    python -m eval.run_eval --top-k 1 3 5       # report Recall@1, @3, @5
    python -m eval.run_eval --document-scoped   # scoped vs unscoped retrieval per source_document

Requires at least one document to already be uploaded/indexed (see
eval/dataset.json's _readme for what it currently assumes).

Writes a timestamped JSON report to eval/results/ and prints a
summary table, so results are comparable across runs (e.g. before/
after adding hybrid retrieval or a reranker).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Allow running as `python eval/run_eval.py` too, not just
# `python -m eval.run_eval`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.bootstrap import apply_local_first_env_fixes  # noqa: E402

apply_local_first_env_fixes()

from app.prompts.prompt_builder import PromptBuilder  # noqa: E402
from app.services.chat_service import ChatService  # noqa: E402
from app.services.document_service import DocumentService  # noqa: E402
from app.services.llm_service import LLMService  # noqa: E402
from app.services.retrieval_service import RetrievalService  # noqa: E402
from eval.metrics import (  # noqa: E402
    AggregateResults,
    _chunk_is_relevant,
    answer_contains_expected_facts,
    is_refusal,
    reciprocal_rank,
    recall_at_k,
)

DATASET_PATH = Path(__file__).parent / "dataset.json"
RESULTS_DIR = Path(__file__).parent / "results"


def load_dataset() -> list[dict]:
    data = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    return data["items"]


def run_retrieval_eval(
    items: list[dict],
    top_k_values: list[int],
    mode: str | None = None,
    rerank: bool | None = None,
    query_rewrite: bool | None = None,
) -> tuple[dict[int, AggregateResults], float]:
    """
    Returns (results_by_k, total_wall_clock_seconds). Wall-clock time
    is reported alongside quality because some options (query
    rewriting in particular) trade one for the other -- an ablation
    that only reports Recall@k would hide that cost.
    """

    max_k = max(top_k_values)
    results = {k: AggregateResults() for k in top_k_values}

    start = time.monotonic()

    for item in items:
        if item["category"] != "factual":
            continue

        retrieved = RetrievalService.search(
            query=item["question"],
            top_k=max_k,
            mode=mode,
            rerank=rerank,
            query_rewrite=query_rewrite,
        )
        retrieved_texts = [r.text for r in retrieved]
        keywords = item["expected_keywords"]

        for k in top_k_values:
            agg = results[k]
            agg.num_questions += 1
            agg.recall_at_k_sum += recall_at_k(
                retrieved_texts, keywords, k
            )
            agg.mrr_sum += reciprocal_rank(
                retrieved_texts[:k], keywords
            )
            agg.per_question.append(
                {
                    "id": item["id"],
                    "question": item["question"],
                    "recall_hit": bool(
                        recall_at_k(retrieved_texts, keywords, k)
                    ),
                    "top_result_heading": (
                        retrieved[0].heading if retrieved else None
                    ),
                }
            )

    elapsed = time.monotonic() - start

    return results, elapsed


def run_threshold_calibration(items: list[dict], top_k: int) -> dict:
    """
    Runs hybrid+rerank retrieval for every factual question and splits
    the resulting (post-rerank) `.score` values into two buckets --
    relevant vs. irrelevant chunk, per the same keyword-substring
    proxy the rest of the harness uses -- so `RERANK_SCORE_THRESHOLD`
    can be picked from real measured scores instead of guessed.

    Retrieval-only: no Ollama call, same as the default (no
    --with-generation) path.
    """
    relevant_scores: list[float] = []
    irrelevant_scores: list[float] = []
    per_question = []

    for item in items:
        if item["category"] != "factual":
            continue

        retrieved = RetrievalService.search(
            query=item["question"],
            top_k=top_k,
            mode="hybrid",
            rerank=True,
        )
        keywords = item["expected_keywords"]

        q_relevant = []
        q_irrelevant = []
        for chunk in retrieved:
            is_relevant = _chunk_is_relevant(chunk.text, keywords)
            if is_relevant:
                relevant_scores.append(chunk.score)
                q_relevant.append(chunk.score)
            else:
                irrelevant_scores.append(chunk.score)
                q_irrelevant.append(chunk.score)

        per_question.append(
            {
                "id": item["id"],
                "question": item["question"],
                "relevant_scores": q_relevant,
                "irrelevant_scores": q_irrelevant,
            }
        )

    def _stats(values: list[float]) -> dict:
        if not values:
            return {"n": 0}
        sorted_v = sorted(values)
        n = len(sorted_v)
        return {
            "n": n,
            "min": sorted_v[0],
            "max": sorted_v[-1],
            "mean": sum(sorted_v) / n,
            "median": sorted_v[n // 2],
        }

    # Candidate thresholds: how many relevant chunks would be wrongly
    # dropped, and how many irrelevant chunks would be correctly
    # dropped, at each of a few round-number cutoffs plus the min
    # relevant score itself (the highest threshold that drops zero
    # relevant chunks in this sample).
    candidates = sorted(
        {-5.0, -3.0, -1.0, 0.0, 1.0}
        | ({min(relevant_scores)} if relevant_scores else set())
    )
    threshold_sweep = [
        {
            "threshold": t,
            "relevant_dropped": sum(1 for s in relevant_scores if s < t),
            "irrelevant_dropped": sum(1 for s in irrelevant_scores if s < t),
        }
        for t in candidates
    ]

    return {
        "relevant": _stats(relevant_scores),
        "irrelevant": _stats(irrelevant_scores),
        "threshold_sweep": threshold_sweep,
        "per_question": per_question,
    }


def run_document_scoped_eval(items: list[dict], top_k: int) -> dict:
    """
    Re-runs retrieval for every factual question that names a
    `source_document`, scoped to just that document via
    RetrievalService's `document_ids` filter, alongside the same
    question's normal unscoped (whole-corpus) retrieval -- to check
    whether scoping recovers recall the existing eval already found
    the growing corpus crowding out (see backend/eval/README.md's
    "corpus crowds out" finding). Hybrid+rerank (the current default
    pipeline) both ways. Retrieval-only, no Ollama call.

    Items without a `source_document` (the unanswerable questions,
    and any factual one not yet tagged) are skipped, not counted as
    misses -- there's nothing to scope them to.
    """
    filename_to_id = {
        doc.original_filename: doc.document_id
        for doc in DocumentService.list_documents()
    }

    per_question = []
    scoped_hits = 0
    unscoped_hits = 0
    skipped = []

    for item in items:
        if item["category"] != "factual" or "source_document" not in item:
            continue

        filename = item["source_document"]
        document_id = filename_to_id.get(filename)
        if document_id is None:
            skipped.append(
                {"id": item["id"], "source_document": filename}
            )
            continue

        keywords = item["expected_keywords"]

        unscoped = RetrievalService.search(
            query=item["question"],
            top_k=top_k,
            mode="hybrid",
            rerank=True,
        )
        scoped = RetrievalService.search(
            query=item["question"],
            top_k=top_k,
            mode="hybrid",
            rerank=True,
            document_ids=[document_id],
        )

        unscoped_hit = bool(
            recall_at_k([r.text for r in unscoped], keywords, top_k)
        )
        scoped_hit = bool(
            recall_at_k([r.text for r in scoped], keywords, top_k)
        )
        unscoped_hits += int(unscoped_hit)
        scoped_hits += int(scoped_hit)

        per_question.append(
            {
                "id": item["id"],
                "question": item["question"],
                "source_document": filename,
                "unscoped_hit": unscoped_hit,
                "scoped_hit": scoped_hit,
                "recovered_by_scoping": scoped_hit and not unscoped_hit,
                "regressed_by_scoping": unscoped_hit and not scoped_hit,
            }
        )

    n = len(per_question)
    return {
        "num_questions": n,
        "unscoped_recall": unscoped_hits / n if n else None,
        "scoped_recall": scoped_hits / n if n else None,
        "skipped_no_document_match": skipped,
        "per_question": per_question,
    }


def warm_up_ollama_with_low_contention() -> None:
    """
    Ollama's llama-server can crash on its OWN CUDA init when a
    PyTorch process (BGE-M3/the reranker) is already resident on this
    project's dev hardware (see LLMService's retry-budget comment for
    the full diagnosis). Retrying alone doesn't reliably help, because
    every retry attempt happens under the same contention.

    Must be called as the FIRST thing in main(), before BGE-M3 or the
    reranker are loaded at all -- that's the only reliably low-
    contention moment available. (An earlier version of this function
    tried releasing/reloading the already-resident torch singletons
    mid-run instead; reloading SentenceTransformer a second time in
    the same process hit an unrelated transformers library bug, so
    that approach was abandoned in favor of this simpler one: warm up
    once, up front, and rely on Ollama's keep-alive -- several
    minutes by default -- to stay warm through the rest of the run.)
    """
    LLMService.generate(system="You are a test.", user="Say hi.")


def run_generation_eval(items: list[dict]) -> dict:
    factual_hits = 0
    factual_total = 0
    refusal_hits = 0
    refusal_total = 0
    per_question = []

    for item in items:
        response = ChatService.chat(question=item["question"], top_k=5)
        answer = response["answer"]

        if item["category"] == "factual":
            factual_total += 1
            hit = answer_contains_expected_facts(
                answer, item["expected_keywords"], min_fraction=0.5
            )
            factual_hits += int(hit)
            per_question.append(
                {
                    "id": item["id"],
                    "category": "factual",
                    "question": item["question"],
                    "answer": answer,
                    "expected_keywords": item["expected_keywords"],
                    "keyword_hit": hit,
                }
            )
        else:
            refusal_total += 1
            refused = is_refusal(
                answer, PromptBuilder.NO_CONTEXT_MESSAGE
            )
            refusal_hits += int(refused)
            per_question.append(
                {
                    "id": item["id"],
                    "category": "unanswerable",
                    "question": item["question"],
                    "answer": answer,
                    "correctly_refused": refused,
                }
            )

    return {
        "factual_keyword_hit_rate": (
            factual_hits / factual_total if factual_total else None
        ),
        "refusal_rate_on_unanswerable": (
            refusal_hits / refusal_total if refusal_total else None
        ),
        "per_question": per_question,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--top-k", nargs="+", type=int, default=[1, 3, 5]
    )
    parser.add_argument(
        "--with-generation",
        action="store_true",
        help="Also run full chat generation (slow; requires Ollama).",
    )
    parser.add_argument(
        "--compare-modes",
        action="store_true",
        help=(
            "Run retrieval-only eval for dense, bm25 and hybrid "
            "modes and print a comparison table (an ablation, not "
            "just a single-mode report). Ignores --with-generation."
        ),
    )
    parser.add_argument(
        "--calibrate-threshold",
        action="store_true",
        help=(
            "Run hybrid+rerank retrieval and report the real "
            "reranker score distribution for relevant vs. irrelevant "
            "chunks, to pick RERANK_SCORE_THRESHOLD from data instead "
            "of a guess. Retrieval-only (no Ollama). Ignores "
            "--with-generation/--compare-modes."
        ),
    )
    parser.add_argument(
        "--document-scoped",
        action="store_true",
        help=(
            "Re-run retrieval for every tagged factual question "
            "scoped to just its source_document (via document_ids), "
            "compared against the normal unscoped run -- checks "
            "whether scoping recovers recall the corpus-crowding "
            "finding lost. Retrieval-only (no Ollama). Ignores "
            "--with-generation/--compare-modes/--calibrate-threshold."
        ),
    )
    parser.add_argument(
        "--with-query-rewrite",
        action="store_true",
        help=(
            "With --compare-modes, also run a hybrid+rerank+rewrite "
            "variant (HyDE-style query rewriting adds one LLM call "
            "per question -- slow, budget real time for it)."
        ),
    )
    args = parser.parse_args()

    items = load_dataset()
    factual_count = sum(1 for i in items if i["category"] == "factual")
    unanswerable_count = len(items) - factual_count

    print(f"Loaded {len(items)} questions "
          f"({factual_count} factual, {unanswerable_count} unanswerable)")
    print()

    needs_ollama = args.with_generation or (
        args.compare_modes and args.with_query_rewrite
    )

    if needs_ollama:
        # Warm Ollama up FIRST, before BGE-M3/the reranker are loaded
        # at all -- see warm_up_ollama_with_low_contention()'s
        # docstring for why this ordering specifically matters on
        # this project's dev hardware.
        warm_up_ollama_with_low_contention()
    else:
        # Best-effort: free Ollama's resident model before loading
        # the embedding model in THIS process. Nothing in this run
        # needs Ollama at all, so there's no reason to keep it
        # resident and competing for memory.
        LLMService.unload()

    if args.calibrate_threshold:
        print("=== Reranker score threshold calibration (hybrid+rerank) ===")
        result = run_threshold_calibration(items, top_k=args.top_k[0])
        print(f"Relevant chunks:   {result['relevant']}")
        print(f"Irrelevant chunks: {result['irrelevant']}")
        print()
        print(f"{'threshold':>10} | {'relevant dropped':>17} | {'irrelevant dropped':>19}")
        for row in result["threshold_sweep"]:
            print(
                f"{row['threshold']:>10.2f} | {row['relevant_dropped']:>17} "
                f"| {row['irrelevant_dropped']:>19}"
            )

        RESULTS_DIR.mkdir(exist_ok=True)
        out_path = (
            RESULTS_DIR
            / f"threshold_calibration_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
        )
        out_path.write_text(
            json.dumps(
                {"timestamp": datetime.now(timezone.utc).isoformat(), **result},
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nFull report written to {out_path}")
        return

    if args.document_scoped:
        print("=== Document-scoped vs unscoped retrieval (hybrid+rerank) ===")
        result = run_document_scoped_eval(items, top_k=args.top_k[0])
        print(f"Questions compared: {result['num_questions']}")
        if result["skipped_no_document_match"]:
            print(
                f"Skipped (no matching indexed document): "
                f"{result['skipped_no_document_match']}"
            )
        if result["num_questions"]:
            print(f"Unscoped Recall@{args.top_k[0]}: {result['unscoped_recall']:.2%}")
            print(f"Scoped Recall@{args.top_k[0]}:   {result['scoped_recall']:.2%}")
            recovered = [
                q["id"] for q in result["per_question"]
                if q["recovered_by_scoping"]
            ]
            regressed = [
                q["id"] for q in result["per_question"]
                if q["regressed_by_scoping"]
            ]
            print(f"Recovered by scoping: {recovered or 'none'}")
            print(f"Regressed by scoping: {regressed or 'none'}")

        RESULTS_DIR.mkdir(exist_ok=True)
        out_path = (
            RESULTS_DIR
            / f"document_scoped_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
        )
        out_path.write_text(
            json.dumps(
                {"timestamp": datetime.now(timezone.utc).isoformat(), **result},
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nFull report written to {out_path}")
        return

    if args.compare_modes:
        print("=== Retrieval ablation: dense vs bm25 vs hybrid vs hybrid+rerank (vs +query_rewrite) ===")
        print(f"{'variant':>20} | {'k':>4} | {'Recall@k':>10} | {'MRR':>8}")
        comparison = {}
        variants = [
            ("dense", "dense", False, False),
            ("bm25", "bm25", False, False),
            ("hybrid", "hybrid", False, False),
            ("hybrid+rerank", "hybrid", True, False),
        ]
        if args.with_query_rewrite:
            variants.append(
                ("hybrid+rerank+rewrite", "hybrid", True, True)
            )
        for label, mode, rerank, query_rewrite in variants:
            mode_results, elapsed = run_retrieval_eval(
                items,
                args.top_k,
                mode=mode,
                rerank=rerank,
                query_rewrite=query_rewrite,
            )
            comparison[label] = {
                "seconds_total": elapsed,
                "seconds_per_question": (
                    elapsed / mode_results[args.top_k[0]].num_questions
                    if mode_results[args.top_k[0]].num_questions
                    else None
                ),
                **{
                    str(k): {
                        "recall_at_k": mode_results[k].recall_at_k,
                        "mrr": mode_results[k].mrr,
                    }
                    for k in args.top_k
                },
            }
            for k in args.top_k:
                agg = mode_results[k]
                print(
                    f"{label:>20} | {k:>4} | {agg.recall_at_k:>10.2%} "
                    f"| {agg.mrr:>8.3f}"
                )
            print(f"{'':>20}   ({elapsed:.1f}s total, "
                  f"{elapsed / mode_results[args.top_k[0]].num_questions:.2f}s/question)")

        RESULTS_DIR.mkdir(exist_ok=True)
        out_path = (
            RESULTS_DIR
            / f"ablation_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
        )
        out_path.write_text(
            json.dumps(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "comparison": comparison,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nFull comparison written to {out_path}")
        return

    retrieval_results, _elapsed = run_retrieval_eval(items, args.top_k)

    print("=== Retrieval ===")
    print(f"{'k':>4} | {'Recall@k':>10} | {'MRR':>8}")
    for k in args.top_k:
        agg = retrieval_results[k]
        print(f"{k:>4} | {agg.recall_at_k:>10.2%} | {agg.mrr:>8.3f}")
    print()

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "num_questions": len(items),
        "retrieval": {
            str(k): {
                "recall_at_k": retrieval_results[k].recall_at_k,
                "mrr": retrieval_results[k].mrr,
                "per_question": retrieval_results[k].per_question,
            }
            for k in args.top_k
        },
    }

    if args.with_generation:
        print("=== Generation (this calls the LLM, may take a while) ===")
        gen_results = run_generation_eval(items)
        report["generation"] = gen_results

        hit_rate = gen_results["factual_keyword_hit_rate"]
        refusal_rate = gen_results["refusal_rate_on_unanswerable"]
        if hit_rate is not None:
            print(f"Factual answer keyword-hit rate: {hit_rate:.2%}")
        if refusal_rate is not None:
            print(f"Correct refusal rate on unanswerable Qs: {refusal_rate:.2%}")
        print()
        print("NOTE: keyword-hit rate is a cheap proxy, not ground")
        print("truth. Read report['generation']['per_question'] by")
        print("hand before quoting these numbers anywhere.")

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = (
        RESULTS_DIR
        / f"eval_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    )
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Full report written to {out_path}")


if __name__ == "__main__":
    main()
