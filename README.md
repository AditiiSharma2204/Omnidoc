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
                        SQLite (conversation history) → Prompt Builder → Qwen2.5 (Ollama)
                                                      ↓
                                     Answer + sources + conversation_id
                                                      ↓
                                    SQLite (persist this turn)
```

- **Parsing** — [Docling](https://github.com/docling-project/docling) converts PDF/DOCX/PPTX/XLSX to markdown.
- **Chunking** — a custom markdown-hierarchy chunker with a size cap and overlap (`app/services/chunking_service.py`).
- **Retrieval** — hybrid by default: FAISS (dense, BGE-M3) + BM25 (lexical), fused with Reciprocal Rank Fusion, then re-scored by a cross-encoder reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`). Optional HyDE-style query rewriting exists but is off by default — measured to give zero quality gain while doubling latency on this corpus (see `backend/eval/README.md`). A reranker-score refusal threshold (`RERANK_SCORE_THRESHOLD`) can drop weak matches before they reach the LLM at all — implemented and unit-tested, off by default pending real score calibration. Each stage is independently toggleable via settings (`RETRIEVAL_MODE`, `RERANK_ENABLED`, `QUERY_REWRITE_ENABLED`, `RERANK_SCORE_THRESHOLD`).
- **Embeddings** — [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) via `sentence-transformers`.
- **Vector store** — FAISS (`IndexFlatIP`, cosine similarity via L2-normalized vectors).
- **Metadata & conversations** — SQLite (`storage/app.db`), replacing the original per-document JSON files. A one-time, idempotent startup migration imports any pre-existing `metadata.json` files. `ConversationService` persists every chat turn and feeds the last 6 messages back to the LLM as real chat history on the next turn (`POST /chat` accepts/returns `conversation_id`; `GET /conversations/{id}` fetches full history).
- **Generation** — [Ollama](https://ollama.com) running `qwen2.5:3b` locally, instructed to cite sources inline as `[1]`, `[2]`, etc. matching the prompt's context numbering, and to answer from conversation history when the current retrieval alone doesn't have it. `ChatService` parses citation markers and returns a `sources` list whose `index` fields line up exactly with them, each flagged `cited: true/false` — an out-of-range or hallucinated citation number simply matches nothing, rather than crashing or silently mapping to the wrong source. Both citations and conversation memory are live-verified against the real model, including one real bug each found and fixed during that verification (see `backend/eval/README.md` and the conversation-memory limitation below).

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

### Docker (one command, no local Python/Node/Ollama install needed)

```bash
docker compose up --build
# once it's up, pull the model into the ollama container (one-time):
docker compose exec ollama ollama pull qwen2.5:3b
```

Frontend on `http://localhost:5173`, backend on `http://localhost:8000`.
Runs CPU-only by default — GPU passthrough needs host-specific setup
(NVIDIA Container Toolkit / WSL2 GPU support) that can't be assumed to
exist; see the commented-out block in `docker-compose.yml` to enable
it if you have that configured. Storage (documents, vector index,
SQLite metadata), the HuggingFace model cache and Ollama's own models
all persist in named volumes across restarts.

## Known limitations

This section is deliberately blunt — see it as the project's honest changelog.

- **No streaming in the UI yet.** The `/chat/stream` endpoint exists; the frontend isn't wired to it.
- **Query rewriting, measured — and correctly kept off by default.**
  `QueryRewriteService` (HyDE-style: embeds an LLM-generated hypothetical
  answer instead of the raw question) is wired into retrieval and unit
  tested. After a machine restart resolved the earlier instability, the
  full `--compare-modes --with-query-rewrite` ablation finally completed:
  Recall@k and MRR came back **identical** to hybrid+rerank alone
  (83.9%/96.8%/96.8%, MRR 0.903) — zero measured quality gain — while
  **doubling** per-question latency (3.5s → 7.0s, one extra real LLM call
  per query). A follow-up re-run also showed the specific miss it was
  meant to fix varying between runs (1 miss vs. 2, on the same
  dataset), consistent with the rewrite step's own sampling temperature
  making retrieval quality non-deterministic in a way the rest of the
  pipeline isn't. `QUERY_REWRITE_ENABLED=False` remains the default —
  now because the data says so, not because it was never measured. See
  `backend/eval/README.md`'s "Query rewriting" section for the full
  numbers.
- **Refusal threshold is implemented but uncalibrated.**
  `RetrievalService._apply_score_threshold` can drop reranked chunks below
  `RERANK_SCORE_THRESHOLD` before they ever reach the LLM (mechanism unit
  tested: `tests/test_reranker_service.py::TestScoreThreshold`), but the
  setting defaults to `None` (disabled) because a real cutoff needs
  sampling actual reranker scores across known relevant vs. irrelevant
  chunks, which needs the same live eval runs currently blocked (see
  above). Shipping a guessed threshold risks silently refusing correct
  answers, which is worse than not having the feature at all.
- **Inline citations, live-verified against the real model.** The first
  live test (after a machine restart resolved the earlier instability)
  showed the mechanism working but the model citing nothing at all —
  the original citation rule was rule 7 of 7 in the system prompt,
  competing for attention with six other instructions. Rewrote the
  prompt around a citation-first format block with a worked example,
  and added a short reminder restated right before generation starts
  (recency helps small models follow instructions). Re-tested across 3
  questions: the model now cites correctly and *selectively* — e.g. for
  "what internships did this person do", it cited exactly the 2 sources
  about internships and correctly left an unrelated career-salary
  spreadsheet chunk and a generic "Summary" chunk uncited, out of 5
  retrieved candidates. The refusal path (a genuinely unanswerable
  question) still returns `NO_CONTEXT_MESSAGE` correctly with the new
  prompt. One known rough edge: citation bracket placement is sometimes
  imperfect (e.g. `[1]` at the start of a sentence instead of the end) —
  cosmetic, doesn't affect which source is credited.
- **No OCR fallback** — scanned (image-only) PDFs will parse to near-empty text.
- **No table/chart/image understanding** — Docling extracts tables as markdown text; nothing structures or reasons over them specially yet.
- **Conversation memory: real, live-verified, and caught a real bug during
  verification.** `ChatService` persists every turn to SQLite and includes
  the last 6 messages as real chat turns for generation. First live test
  showed a *hard* regression the mocked tests couldn't catch: the model
  refused a trivial follow-up ("which one came second?") that was
  answerable from its own immediately-preceding answer, because the
  system prompt's "answer only from document context" rule read as
  excluding conversation history entirely. Fixed by explicitly
  authorizing the model to answer from earlier turns, not just the
  current retrieval. Re-tested: it now correctly answers from history
  (`"[2] came second"` — factually correct) rather than refusing, though
  the response is more verbose than ideal (re-states the whole prior
  answer instead of just the new part) — a real but minor polish item,
  not a correctness one. Retrieval itself still runs on the raw current
  question only; a follow-up needing *new* document lookup (not just
  recalling what was already said) can still miss if it doesn't share
  vocabulary with the target chunk — condensing follow-ups into
  standalone retrieval queries is a distinct, not-yet-built piece (see
  roadmap).
- **No persistent chat history across page reloads.** Conversation state
  lives in SQLite server-side (survives a backend restart) and is
  addressable via `GET /api/v1/conversations/{id}`, but the frontend
  doesn't yet store the current `conversation_id` anywhere durable —
  refreshing the page starts a new conversation client-side even though
  the old one is still in the database.
- **No auth, no multi-user support.**
- **Docker Compose setup exists, syntax-validated, build-verified on this
  machine only partially.** `docker compose config` resolves cleanly
  (services, volumes, port mappings, build contexts all correct). A live
  `docker compose build backend` was attempted on the dev machine and hit
  the same class of instability documented above (Ollama/CUDA driver
  crashes, Docling/transformers native errors under sustained heavy
  workloads): Docker Desktop's own engine went unreachable partway through
  the (~13 minute) pip install of the ML dependency stack, independent of
  anything in the Dockerfile itself. Not yet re-verified end to end
  (`docker compose up` smoke test) on this machine; the Dockerfiles,
  compose file and nginx config are still real, reviewable artifacts and
  the compose config validates, but treat "builds cleanly from a clean
  clone" as unconfirmed until re-tested on a stabler machine or after a
  successful retry here.
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

Short-term priorities, roughly in order: query rewriting for
follow-ups (condense "what about the second one?" into a standalone
retrieval query, distinct from the HyDE feature — see the conversation-
memory limitation above); persist `conversation_id` client-side so a
page reload doesn't lose the active conversation; calibrate the
refusal threshold against real reranker scores; streaming in the UI
(markdown rendering is already done); a document-scoped retrieval eval
alongside the current global one (the corpus-crowding finding in
`backend/eval/README.md`); grow the evaluation dataset past 50
questions (contracts/legal documents, adversarial content); page-level
citation highlighting (jump to the cited PDF page, not just show the
source card); and a confirmed, end-to-end `docker compose up` run
(compose file and Dockerfiles exist and validate; a live build hit
this machine's known instability partway through — see Known
limitations).
