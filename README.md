# OmniDoc AI

A local-first document intelligence platform: upload PDFs, Word docs,
PowerPoint decks or Excel sheets and ask questions about them. Parsing,
embedding, retrieval and generation all run on your machine — nothing
is sent to an external API.

**Status: early, actively being rebuilt.** This is a portfolio/learning
project, not a finished product. See [Known limitations](#known-limitations)
below for an honest account of what doesn't work yet.

## Architecture

```
React (Vite/TS)  →  FastAPI  →  Docling (parse)  →  Chunker  →  BGE-M3 (embed)
                                                                      ↓
                                    FAISS (dense) + BM25 (lexical)  ←┘
                                                      ↓
                                     Reciprocal Rank Fusion (hybrid)
                                                      ↓
                                    Cross-encoder reranker (top candidates)
                                                      ↓
                                       Prompt Builder → Qwen2.5 (Ollama)
                                                      ↓
                                                   Answer + sources
```

- **Parsing** — [Docling](https://github.com/docling-project/docling) converts PDF/DOCX/PPTX/XLSX to markdown.
- **Chunking** — a custom markdown-hierarchy chunker with a size cap and overlap (`app/services/chunking_service.py`).
- **Retrieval** — hybrid by default: FAISS (dense, BGE-M3) + BM25 (lexical), fused with Reciprocal Rank Fusion, then re-scored by a cross-encoder reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`). Optional HyDE-style query rewriting (an LLM-generated hypothetical answer, embedded instead of the raw question) targets a specific measured gap but is off by default and not yet quality-measured — see below. A reranker-score refusal threshold (`RERANK_SCORE_THRESHOLD`) can drop weak matches before they reach the LLM at all, so a bad retrieval can't be confidently answered from — implemented and unit-tested, but off by default (uncalibrated; see below). Each stage is independently toggleable via settings (`RETRIEVAL_MODE`, `RERANK_ENABLED`, `QUERY_REWRITE_ENABLED`, `RERANK_SCORE_THRESHOLD`) — see `backend/eval/README.md` for what's actually been measured.
- **Embeddings** — [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) via `sentence-transformers`.
- **Vector store** — FAISS (`IndexFlatIP`, cosine similarity via L2-normalized vectors).
- **Generation** — [Ollama](https://ollama.com) running `qwen2.5:3b` locally, instructed to cite sources inline as `[1]`, `[2]`, etc. matching the prompt's context numbering. `ChatService` parses those markers and returns a `sources` list whose `index` fields line up exactly with them, each flagged `cited: true/false` — an out-of-range or hallucinated citation number simply matches nothing, rather than crashing or silently mapping to the wrong source.

## Setup

### Prerequisites

- Python 3.11 (a conda env is recommended — the ML stack here is heavy)
- Node.js 18+
- [Ollama](https://ollama.com) installed, with a model pulled: `ollama pull qwen2.5:3b`

### Backend

```bash
cd backend
conda create -n omnidoc python=3.11 -y
conda activate omnidoc
pip install -r requirements.txt
uvicorn app.main:app --reload
```

The API comes up on `http://127.0.0.1:8000` by default (`--port` to change it),
with interactive docs at `/docs`. First run downloads the BGE-M3 embedding
model (~2.2GB) and, the first time retrieval runs with reranking enabled
(the default), the cross-encoder reranker (~90MB) — after that both are
cached and the app runs fully offline (see `app/bootstrap.py`).

Configuration is via environment variables or a `backend/.env` file — see
`app/config/settings.py` for everything that's tunable (LLM model/context
size, chunk size, Docling OCR/table-structure toggles, storage paths).

### Frontend

```bash
cd frontend
npm install
npm run dev
```

## Known limitations

This section is deliberately blunt — see it as the project's honest changelog.

- **No streaming in the UI yet.** The `/chat/stream` endpoint exists; the frontend isn't wired to it.
- **No markdown rendering in the chat UI.**
- **Query rewriting is implemented but its quality impact is unmeasured.**
  `QueryRewriteService` (HyDE-style: embeds an LLM-generated hypothetical
  answer instead of the raw question) is wired into retrieval and unit
  tested, off by default (`QUERY_REWRITE_ENABLED=False`). The live
  before/after ablation is blocked, not skipped: every attempt this
  session hit escalating machine instability (see the Ollama/CUDA note
  below) ending in a raw allocation failure with 4GB+ RAM free — a strong
  signal the dev machine needs a restart before more heavy ML runs are
  reliable, not something more code changes fix. See
  `backend/eval/README.md`'s "Query rewriting" section for the full
  account before deciding whether to enable this by default.
- **Refusal threshold is implemented but uncalibrated.**
  `RetrievalService._apply_score_threshold` can drop reranked chunks below
  `RERANK_SCORE_THRESHOLD` before they ever reach the LLM (mechanism unit
  tested: `tests/test_reranker_service.py::TestScoreThreshold`), but the
  setting defaults to `None` (disabled) because a real cutoff needs
  sampling actual reranker scores across known relevant vs. irrelevant
  chunks, which needs the same live eval runs currently blocked (see
  above). Shipping a guessed threshold risks silently refusing correct
  answers, which is worse than not having the feature at all.
- **Inline citations are unit-tested (32 tests) but not yet live-verified
  end to end.** Every attempt to run a real chat request while writing
  this feature hit the same ongoing machine instability documented
  above and in `backend/eval/README.md` (a CUDA crash, then a
  transformers library error, then a raw SIGSEGV on a single BGE-M3
  load that has worked reliably dozens of times earlier this session) —
  a real machine restart is needed before this can be confirmed against
  a live model rather than mocked ones. The citation-parsing and
  sources-building logic itself (`CitationService`, `ChatService.
  _build_sources`) is plain Python string/dict handling with no ML
  dependency, so it's exactly as trustworthy as its test coverage; what's
  unverified is only whether Qwen2.5:3b reliably follows the "[N]" citing
  instruction in practice, which needs a real model to check.
- **No OCR fallback** — scanned (image-only) PDFs will parse to near-empty text.
- **No table/chart/image understanding** — Docling extracts tables as markdown text; nothing structures or reasons over them specially yet.
- **No conversation memory** — each question is answered independently of chat history.
- **No persistent chat history** — refreshing the page loses the conversation.
- **No auth, no multi-user support.**
- **Memory-constrained by design.** This targets modest hardware (built against
  an NVIDIA MX450, 2GB VRAM). Ollama, Docling's parsing models, the BGE-M3
  embedding model and the cross-encoder reranker each keep a resident model
  in RAM, and on a machine with limited free system RAM (~3GB or less)
  having more than one loaded at once can OOM-crash a process. Mitigations
  in place: Docling's OCR and table-structure models are disabled by
  default (`DOCLING_DO_OCR`, `DOCLING_DO_TABLE_STRUCTURE` in settings), the
  embedding and reranker models load with `low_cpu_mem_usage`, the reranker
  was deliberately picked small (~500MB resident vs. BGE-M3's ~2GB), and
  `LLMService.unload()` can release Ollama's resident model on demand
  before heavy local work. If you still hit a crash, free up RAM or reduce
  concurrent load; a proper fix (ingestion in an isolated worker process)
  is on the roadmap.
- **Ollama can crash on startup while the embedding model is warm, on some
  GPU/driver setups — and the fix is a mitigation, not a guarantee.**
  Diagnosed directly on the dev machine (Windows + WDDM + MX450): Ollama's
  `llama-server` can fail its own CUDA init with a native crash when a
  PyTorch process (this app's embedder/reranker) is concurrently resident
  — reproduced with plenty of free RAM and VRAM, so it isn't simply a
  memory problem; looks like a driver-level race. `LLMService.generate()`
  retries with backoff (up to 5 attempts, `RETRY_BACKOFF_SECONDS=20`) and
  that alone was enough the first several times this was hit. It was
  **not** reliably enough on every occasion, though (a later attempt still
  failed after the full retry budget) — the more robust workaround found
  was warming Ollama up *before* loading any PyTorch model in the process
  at all (see `eval/run_eval.py`'s `warm_up_ollama_with_low_contention`),
  which the eval harness does but the main app's own chat path does not
  yet. If this app's very first chat request after startup is unusually
  slow or fails outright on this kind of hardware, this is why — applying
  the same "warm Ollama first" ordering to `app/main.py`'s startup is a
  reasonable next fix, not yet done.
- **Evaluation harness: 5 document types, 36 questions, generation re-run
  end to end.** See `backend/eval/` — retrieval (Recall@k, MRR) and
  generation (keyword-hit, refusal-rate) metrics, plus a dense/bm25/hybrid/
  hybrid+rerank ablation (`--compare-modes`). Every retrieval stage
  improves Recall@1 and MRR monotonically (dense 71.0%/0.821 → bm25
  74.2%/0.831 → hybrid 77.4%/0.866 → hybrid+rerank 83.9%/0.903) — getting
  there meant finding and fixing a real regression (reranking initially
  *hurt* Recall@5 because its candidate pool was too narrow; measured 4x
  vs 8x vs 12x directly rather than guessing, 8x is now the default).
  Generation scores 87.1% factual keyword-hit, 100% refusal-on-unanswerable
  — but manually reading all 4 reported misses found only 3 are real: one
  is a false miss (the model gave a fully correct, just differently-worded
  answer — corrected rate is 90.3%), one is the already-known vocabulary-
  mismatch gap, and one is new evidence that growing the corpus can crowd
  a previously-correct answer out of the top-5 (a concrete case for
  scoping retrieval to `document_ids`, which the API already supports but
  this eval doesn't yet exercise). See `backend/eval/README.md` for the
  full writeup, including a real bug found and fixed along the way (the
  Ollama/CUDA crash above) — the harness explicitly warns against trusting
  its summary numbers without reading `per_question`, and that warning has
  already caught a wrong claim once (an earlier draft assumed hybrid alone
  would fix a retrieval gap; it didn't, and the docs were corrected once
  actually measured).

## Roadmap

Short-term priorities, roughly in order: once the dev machine is stable
again, live-verify citations against a real model and calibrate/measure
query rewriting and the refusal threshold (all three implemented, all
blocked on the same instability above); a document-scoped retrieval eval
alongside the current global one (the corpus-crowding finding above);
grow the dataset past 50 questions (contracts/legal documents,
adversarial content); page-level citation highlighting (jump to the
cited PDF page, not just show the source card); streaming + markdown in
the UI; and a Docker Compose setup that works from a clean clone.
