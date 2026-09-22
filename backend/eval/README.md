# Evaluation harness

Measures retrieval quality and (optionally) generation quality against
`dataset.json`, and writes a timestamped JSON report to `results/` so
runs are comparable over time (e.g. dense vs. hybrid retrieval, or
before/after a reranker).

## Running it

From `backend/`, with at least one document already uploaded/indexed
and the API's dependencies importable (same conda env as the server):

```bash
# Retrieval only -- fast (seconds), no LLM calls
python -m eval.run_eval

# Retrieval + generation -- slow (each question is a real LLM call;
# budget minutes per question on modest hardware), requires Ollama
# running
python -m eval.run_eval --with-generation

# Report Recall@1/3/5 instead of the default
python -m eval.run_eval --top-k 1 3 5

# Ablation: dense vs bm25 vs hybrid vs hybrid+rerank, side by side
python -m eval.run_eval --compare-modes
```

## What it measures

**Retrieval** (`eval/metrics.py`):
- **Recall@k** — did any of the top-k retrieved chunks contain at least
  one of the question's expected keywords? Averaged across questions.
- **MRR** — mean reciprocal rank of the first relevant chunk.

Relevance is decided by simple case-insensitive keyword matching
against `expected_keywords`, not hand-labeled gold chunk IDs. That's a
deliberate trade-off: gold chunk IDs break every time chunking logic
changes, while keyword matching survives re-chunking and is easy for
a human to write/verify by just reading the source document.

**Generation** (only with `--with-generation`):
- **Factual keyword-hit rate** — does the generated answer mention
  enough of the expected keywords (default: half)?
- **Refusal rate on unanswerable questions** — for questions deliberately
  outside the document's scope, does the system correctly say it can't
  find the answer instead of hallucinating one?

Both generation metrics are **cheap proxies, not ground truth**. A
keyword-hit check will pass an answer that uses the right words in the
wrong context and fail a correct answer that paraphrases instead of
quoting. Read `report["generation"]["per_question"]` by hand before
quoting any generation number publicly (a README, a resume bullet,
etc.) — the harness exists to catch regressions fast, not to replace
that read.

## Known result (5 documents, 36 questions, 2026-09-22)

The corpus now covers 5 document types indexed together: a resume
(PDF), a career/salary analysis (XLSX), a conference presentation
(PPTX), a dataset-analysis report (DOCX), and an academic paper (PDF)
— ~50+ chunks total, up from the original single-document/10-chunk
seed. This matters: several findings below only showed up once the
corpus was large enough for retrieval and candidate-pool size to
actually matter.

**Retrieval ablation (dense vs bm25 vs hybrid vs hybrid+rerank), `--compare-modes`:**

| variant       | Recall@1 | Recall@3 | Recall@5 | MRR   |
|---------------|----------|----------|----------|-------|
| dense         | 71.0%    | 90.3%    | 96.8%    | 0.821 |
| bm25          | 74.2%    | 90.3%    | 93.5%    | 0.831 |
| hybrid        | 77.4%    | 96.8%    | 96.8%    | 0.866 |
| hybrid+rerank | 83.9%    | 96.8%    | 96.8%    | 0.903 |

Every stage improves Recall@1 and MRR monotonically. This is the
result of an actual, evidence-driven tuning pass, not the first
number produced — worth walking through because the process is the
real point:

**A regression was found and fixed, not just a win reported.** The
first hybrid+rerank run on this expanded corpus actually *regressed*
Recall@3/5 versus plain hybrid (93.5% vs 96.8%) even though Recall@1
and MRR improved. Diagnosed by checking exactly which question
newly failed (`RetrievalService._retrieve` gives the pre-rerank
candidate pool directly) rather than guessing: the correct chunk for
*"What is this person's name?"* wasn't merely ranked low by the
reranker — it wasn't in the pre-rerank candidate pool AT ALL.
`RERANK_CANDIDATE_MULTIPLIER=4` with `top_k=5` fetches only the top 20
hybrid-ranked candidates before reranking; on the old 10-chunk corpus
that was effectively the whole corpus, but on ~50+ chunks across 5
documents, a paper-heavy set of results crowded the right chunk out
before the reranker ever got a chance to see it (a reranker can only
reorder what it's given). Tested multiplier 4 vs 8 vs 12 directly:
8 fully recovered Recall@5 to 96.8% with no further gain from 12, so
the default is now 8 (was 4) — a measured value, not a guess.

**One residual miss, and it's the same underlying pattern as the
regression above, not a new bug:** *"Who are the authors of the
Re-MTKD paper?"* (`expected_keywords: ["Zeqin Yu"]`, which does appear
verbatim in the paper's byline chunk) still misses at every variant.
Same root cause as the original *"what is this person's name?"* case
from the single-document baseline: a query asking "who/what is X's
name" shares no vocabulary with a byline/title chunk that just states
the name without ever using words like "author" or "name". Neither
lexical nor dense retrieval has a way to bridge that gap, and
reranking can't invent a candidate that never made the pool. This
needs query rewriting (already on the roadmap), not a retrieval tweak.

**Historical, from the original 1-document baseline (not yet re-run on
the 5-document corpus — see below):** generation (`--with-generation`)
scored 100% factual keyword-hit and 100% correct refusal, but manual
review of `per_question` found real issues the automated metrics
missed: one answer dropped a fact ("MongoDB") from a correctly
retrieved chunk, one answer was genuinely garbled prose, and the name
question was answered correctly despite retrieval missing the target
chunk — most likely because the prompt's `Document: Aditii_Resume.pdf`
line let the model infer the name from the filename rather than
retrieved content, a real grounding risk that happened to work by
coincidence. **Re-running `--with-generation` on the expanded 36-question
corpus is a pending follow-up**, not yet done (each question is a real,
slow LLM call — budget significant time for 36 of them on this
hardware).

## Extending the dataset

Progress against the original plan: ✅ 5 documents of different types
(was 1), ✅ per-`expected_keywords` verification against actual parsed
content (see `dataset.json`'s `_readme`). Still short of the 50+
question target (currently 36) and still all personal/academic files
from one person — no contracts, legal documents, or genuinely
adversarial content (e.g. a document deliberately containing text that
looks like a prompt injection). Both are reasonable next additions.
