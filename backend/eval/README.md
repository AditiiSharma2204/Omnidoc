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

## Known result (baseline, one document)

Running the seed set against a single indexed resume PDF surfaced a
real, reproducible retrieval gap: the question *"What is this person's
name?"* misses at every k up to 5, because dense embedding similarity
ranks a semantically-generic "Summary" chunk above the short chunk
whose heading literally is the name. This is exactly what you'd expect
from dense-only retrieval on a short, low-context query — see the
top-level README's roadmap (hybrid BM25 + dense retrieval) for the
planned fix. This is a good example of why this harness is worth
having: it turns "retrieval feels fine" into a specific, fixable,
re-testable failure.

## Extending the dataset

`dataset.json`'s `_readme` field says what's still missing: right now
it covers one document type. Before trusting these numbers as a real
benchmark:

1. Upload 5-10 documents of different types (paper, contract/report,
   slides, spreadsheet) alongside the resume.
2. Add ~5 questions per document to `dataset.json`'s `items` list,
   following the existing `factual`/`unanswerable` shape.
3. Re-read each `expected_keywords` list against the actual source
   document — a wrong expected answer makes every downstream number
   wrong.
4. Aim for 50+ questions total per the improvement plan.
