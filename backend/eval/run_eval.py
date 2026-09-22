"""
Evaluation harness for OmniDoc's retrieval + generation pipeline.

Run from the `backend/` directory (so the `app` package resolves):

    python -m eval.run_eval                    # retrieval only (fast, no LLM calls)
    python -m eval.run_eval --with-generation   # also runs full chat (slow; needs Ollama running)
    python -m eval.run_eval --top-k 1 3 5       # report Recall@1, @3, @5

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
from datetime import datetime, timezone
from pathlib import Path

# Allow running as `python eval/run_eval.py` too, not just
# `python -m eval.run_eval`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.bootstrap import apply_local_first_env_fixes  # noqa: E402

apply_local_first_env_fixes()

from app.prompts.prompt_builder import PromptBuilder  # noqa: E402
from app.services.chat_service import ChatService  # noqa: E402
from app.services.llm_service import LLMService  # noqa: E402
from app.services.retrieval_service import RetrievalService  # noqa: E402
from eval.metrics import (  # noqa: E402
    AggregateResults,
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
    items: list[dict], top_k_values: list[int]
) -> dict[int, AggregateResults]:

    max_k = max(top_k_values)
    results = {k: AggregateResults() for k in top_k_values}

    for item in items:
        if item["category"] != "factual":
            continue

        retrieved = RetrievalService.search(
            query=item["question"], top_k=max_k
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

    return results


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
    args = parser.parse_args()

    items = load_dataset()
    factual_count = sum(1 for i in items if i["category"] == "factual")
    unanswerable_count = len(items) - factual_count

    print(f"Loaded {len(items)} questions "
          f"({factual_count} factual, {unanswerable_count} unanswerable)")
    print()

    # Best-effort: free Ollama's resident model before loading the
    # embedding model in THIS process. On memory-constrained
    # hardware, both being resident at once is enough to OOM --
    # Ollama reloads transparently (one-time delay) on the first
    # real generation call below, if --with-generation is set.
    LLMService.unload()

    retrieval_results = run_retrieval_eval(items, args.top_k)

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
