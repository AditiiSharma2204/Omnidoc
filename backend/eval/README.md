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

# Also add a hybrid+rerank+rewrite variant (HyDE-style query rewriting --
# adds one LLM call per question, slow; see "Query rewriting" below)
python -m eval.run_eval --compare-modes --with-query-rewrite
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

| variant       | Recall@1 | Recall@3 | Recall@5 | MRR   | s/question |
|---------------|----------|----------|----------|-------|------------|
| dense         | 71.0%    | 90.3%    | 96.8%    | 0.821 | ~0.4s      |
| bm25          | 74.2%    | 90.3%    | 93.5%    | 0.831 | ~0.01s     |
| hybrid        | 77.4%    | 96.8%    | 96.8%    | 0.866 | ~0.2-0.3s  |
| hybrid+rerank | 83.9%    | 96.8%    | 96.8%    | 0.903 | ~3.0-3.4s  |

Quality and latency both matter for a design decision, not just
quality — the table includes both. Reranking is ~10-15x slower than
plain hybrid per question (one cross-encoder forward pass per
candidate, CPU-bound on this hardware) for a real quality gain; worth
knowing before assuming "on" is free.

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

**Generation, re-run on the full 5-document/36-question corpus:** 87.1%
factual keyword-hit (27/31), 100% correct refusal (5/5). Manually read
every one of the 4 reported misses (as this doc keeps insisting on) —
only 3 are real:

- **`q22` is a false miss, not a model error.** Asked which institute
  conducted the research (`expected_keywords: ["SRMIST"]`); the answer
  was *"SRM Institute of Science and Technology, Chennai conducted the
  sign language storytelling research"* — completely correct, just
  spelled out instead of using the abbreviation. The keyword metric is
  too strict here, not the system. Correcting for this: **28/31 = 90.3%**
  is the more honest number. (Lesson for extending the dataset: list
  keyword *alternatives* for questions with a known acronym/full-name
  pair, e.g. `["SRMIST", "SRM Institute"]`.)
- **`q34`** is the already-known residual retrieval miss from the
  ablation above (author lookup, vocabulary mismatch) — refused rather
  than hallucinated, which is the correct behavior given retrieval
  didn't find the chunk.
- **`q30`** (paper title) is a **new** miss that didn't exist in the
  single-document baseline: refused instead of answering, meaning
  retrieval didn't surface the title chunk for this query on the
  larger corpus. Same "meta-question about document identity" family
  as `q34` and the original `q1` — asking "what is the title of the
  paper" doesn't lexically or semantically anchor to the chunk that
  states the title, no matter which retrieval mode.
- **`q12`** (Docker/skills) is also **new**, and the most informative
  one: this exact question scored a hit on the 1-document baseline.
  With 4 more documents now competing for the top-5 slots, the correct
  resume chunk got crowded out, and the model correctly refused rather
  than guessing. This eval doesn't scope questions to their source
  document via `document_ids` (built on Day 1, never exercised here) —
  doing so would very likely fix this specific case and is a natural
  next harness improvement: measure global retrieval *and*
  document-scoped retrieval side by side.

## Query rewriting (HyDE) — measured, correctly kept off by default

`app/services/query_rewrite_service.py` implements HyDE-style query
rewriting: instead of embedding the raw question for dense retrieval,
ask the LLM to write a short hypothetical answer passage first, and
embed *that*. This directly targets the residual-miss pattern above
(a "who/what is X" query sharing no vocabulary with its target chunk)
-- a hypothetical answer like *"The person's name is Aditii Sharma."*
is written in the same register as the real chunk, so it should embed
much closer to it than the bare question does. BM25 still gets the
literal query (a fabricated hypothetical would just inject noise into
lexical matching); only the dense leg's embedding is affected.

**Getting a live measurement took a full session first (see the CUDA
section below for the diagnosis) -- once the machine was healthy
again, `--compare-modes --with-query-rewrite` finally completed:**

| variant               | Recall@1 | Recall@3 | Recall@5 | MRR   | s/question |
|------------------------|----------|----------|----------|-------|------------|
| hybrid+rerank          | 83.9%    | 96.8%    | 96.8%    | 0.903 | ~3.5s      |
| hybrid+rerank+rewrite  | 83.9%    | 96.8%    | 96.8%    | 0.903 | ~7.0s      |

**Zero measured quality gain, double the latency.** Query rewriting
adds one full real LLM call per question before retrieval even starts,
and on this corpus it didn't move Recall@k or MRR by a single
percentage point.

A follow-up per-question check made the picture worse, not better: a
second run showed 2 misses at k=5 instead of the 1 the ablation table
above implies -- same dataset, same settings, different result. The
rewrite step samples the hypothetical answer at `temperature=0.3`
(see `QueryRewriteService`), so the embedded text -- and therefore
which chunks rank where -- varies run to run in a way the rest of the
pipeline (dense/BM25/hybrid/rerank) simply doesn't. That's a second,
independent reason to leave it off by default: it doesn't just fail
to help, it makes retrieval quality non-deterministic for no benefit.

**Conclusion:** `QUERY_REWRITE_ENABLED=False` remains the default --
now because two rounds of real measurement say so, not because
measurement was blocked. The feature stays in the codebase (wired
into `RetrievalService.search(query_rewrite=...)`, 9 unit tests) as an
opt-in for a corpus where it might actually help (e.g. one with many
more "who/what is the name of X" style questions), but it is not a
recommended default on this evidence.

## Inline citations — live-verified, prompt tuned on real evidence

The first live test of the citation feature (`app/services/citation_service.py`,
`ChatService._build_sources`) was a clean miss: technically correct
end to end (retrieval, generation, sources API all worked), but the
model cited *nothing* -- zero `[N]` markers across 5 retrieved
sources, despite an explicit citation rule in the system prompt. Root
cause, on inspection: the rule was #7 of 7, competing with six other
instructions for a 3B model's limited instruction-following budget.

Rewrote the prompt around a citation-first format block with a worked
example, plus a short reminder restated immediately before generation
starts (recency helps small models). Re-tested across 3 questions:

- *"What internships did this person do?"* → cited exactly the 2
  sources about internships, out of 5 retrieved candidates, correctly
  leaving an unrelated career-salary spreadsheet chunk and a generic
  "Summary" chunk uncited.
- *"What programming languages does this person know?"* → cited
  exactly the 1 relevant source.
- *"What is the capital of France?"* (unanswerable) → still correctly
  refused with `NO_CONTEXT_MESSAGE`, unaffected by the prompt change.

Selective, correct citation across all 3 -- not just "citations
appear at all", but "citations point at the right sources and are
omitted for irrelevant ones." One cosmetic rough edge: bracket
placement is sometimes at the start of a sentence instead of the end
(`[1] She interned at...` rather than `She interned at... [1]`); it
doesn't affect which source gets credited, so left as a known,
low-priority polish item rather than over-engineering the prompt
further for a formatting detail.

## A real bug found getting this far: Ollama vs. PyTorch on this machine

Every earlier attempt at this generation run failed with the same
error, and it took real diagnosis (not a guess) to fix: Ollama's
`llama-server` subprocess crashed on its own CUDA initialization
(`"CUDA error: shared object initialization failed"`, a native stack-
buffer-overrun exit) whenever a PyTorch process (this app's embedding/
reranker models) was concurrently resident. Ruled out simpler
explanations one at a time before accepting this one:
- Not overall memory quantity — reproduced with >3GB system RAM free
  and VRAM at 0MiB/2048MiB used.
- Not our process touching CUDA — reproduced identically with
  `CUDA_VISIBLE_DEVICES=""` set for the Python process.
- Not specific to having two torch models loaded — reproduced with
  only BGE-M3 resident (no reranker).

It reproduces the same way every time: Ollama auto-respawns
`llama-server` after the crash, and a **20-second wait** (measured
directly, not guessed) reliably lets that respawn succeed while the
torch process stays resident. `LLMService.generate()` now retries
with that backoff (`app/services/llm_service.py`) instead of failing
the whole chat turn. This affects the real app too, not just this
eval script: any chat request that lands while the embedding model is
warm can hit this on this machine's Windows/WDDM + MX450 setup.

## Extending the dataset

Progress against the original plan: ✅ 5 documents of different types
(was 1), ✅ per-`expected_keywords` verification against actual parsed
content (see `dataset.json`'s `_readme`), ✅ generation re-run on the
full corpus with every miss manually read. Still short of the 50+
question target (currently 36) and still all personal/academic files
from one person — no contracts, legal documents, or genuinely
adversarial content (e.g. a document deliberately containing text that
looks like a prompt injection). Also worth doing next: list keyword
*alternatives* for acronym/full-name pairs (the `q22` false-miss
lesson above), and add a document-scoped retrieval eval alongside the
current global one (the `q12` finding above) using the `document_ids`
filter `RetrievalService.search` already supports.
