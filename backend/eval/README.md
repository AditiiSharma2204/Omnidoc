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

# Ablation: dense vs bm25 vs hybrid retrieval, side by side
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

## Known result (baseline, one document, 2026-09-22)

**Retrieval ablation (dense vs bm25 vs hybrid), `--compare-modes`:**

| mode   | Recall@1 | Recall@3 | Recall@5 | MRR   |
|--------|----------|----------|----------|-------|
| dense  | 75.0%    | 91.7%    | 91.7%    | 0.833 |
| bm25   | 91.7%    | 91.7%    | 91.7%    | 0.917 |
| hybrid | 91.7%    | 91.7%    | 91.7%    | 0.917 |

Hybrid matches or beats dense-only at every k, with **zero
per-question regressions** (checked by hand, not just the aggregate).
On this seed set BM25 alone is actually the strongest single signal,
because most questions are exact-term technical lookups (company
names, tool names) that lexical search is naturally good at; hybrid
ties it rather than losing anything by also blending in dense scores.

**Important, and corrected from an earlier draft of this doc:** hybrid
retrieval does **not** fix every miss. *"What is this person's name?"*
is still wrong at every k in **all three modes** — dense ranks a
generic "Summary" chunk above the short chunk whose heading is the
name (as expected), but BM25 finds nothing either, because the query
("what... person... name") shares zero vocabulary with the document
(which never uses the word "name"). Hybrid can only fuse rankings that
already contain some signal; it can't invent relevance neither
retrieval mode found. This particular gap needs query rewriting/
expansion, not better fusion — don't claim a fix you haven't measured,
which is exactly the mistake an earlier version of this note made
before the ablation was actually run.

**Generation:** factual keyword-hit rate 100% (12/12), refusal rate on
unanswerable questions 100% (3/3) — but reading `per_question` by hand
(as this doc tells you to) surfaces real issues the headline numbers
hide:
- The database-technologies question's retrieved chunk lists
  `MongoDB, MySQL`; the answer said *"SQL and MySQL"*, dropping MongoDB
  and miscategorizing SQL as a database technology. Still scored a
  "hit" because the metric only required 1 of 2 keywords.
- The cloud/DevOps-tools answer is genuinely garbled ("lists Git/GitHub
  as a tool & technology related to Git/GitHub, and Docker as a
  framework & library related to Docker") — low-quality phrasing from
  the 3B model, invisible to a keyword check.
- The name question's answer is *correct* despite retrieval missing
  the right chunk (above) — most likely because every context block in
  the prompt includes `Document: Aditii_Resume.pdf`, and the model
  pattern-matched the filename into a name rather than reading it from
  retrieved content. That's a real, subtle grounding risk: it worked
  here by coincidence (the filename happens to be the person's name)
  and would silently fail on a document whose filename doesn't match
  its content.

This is exactly why this harness is worth having, and why its own
README tells you not to trust the summary numbers blindly: "100%/100%"
reads as a finished system; a five-minute manual read of the same
report finds two real answer-quality bugs and one grounding risk that
a keyword check can't see.

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
