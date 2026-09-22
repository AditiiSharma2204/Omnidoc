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
- **Retrieval** — hybrid by default: FAISS (dense, BGE-M3) + BM25 (lexical), fused with Reciprocal Rank Fusion, then re-scored by a cross-encoder reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`). Each stage is independently toggleable via settings (`RETRIEVAL_MODE`, `RERANK_ENABLED`) and measured — see `backend/eval/README.md`.
- **Embeddings** — [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) via `sentence-transformers`.
- **Vector store** — FAISS (`IndexFlatIP`, cosine similarity via L2-normalized vectors).
- **Generation** — [Ollama](https://ollama.com) running `qwen2.5:3b` locally.

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
- **No query rewriting.** Retrieval (hybrid BM25+dense, cross-encoder
  reranked) still can't answer a "who/what is X" query whose target chunk
  shares no vocabulary with the question — see the evaluation harness bullet
  below for a specific, measured example.
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
- **Evaluation harness exists, covers 5 document types, still short of the
  50+ question target.** See `backend/eval/` — 36 questions (resume, career/
  salary spreadsheet, conference slides, a dataset report, an academic paper)
  with retrieval (Recall@k, MRR) and generation (keyword-hit, refusal-rate)
  metrics, plus a dense/bm25/hybrid/hybrid+rerank ablation
  (`--compare-modes`). Every retrieval stage improves Recall@1 and MRR
  monotonically (dense 71.0%/0.821 → bm25 74.2%/0.831 → hybrid 77.4%/0.866 →
  hybrid+rerank 83.9%/0.903). Getting there included finding and fixing a
  real regression: the first reranked run on this corpus actually
  *regressed* Recall@5 versus plain hybrid, diagnosed to the reranker's
  candidate pool being too narrow (a relevant chunk was crowded out before
  reranking ever saw it) and fixed by measuring pool sizes directly rather
  than guessing (`RERANK_CANDIDATE_MULTIPLIER` 4→8). One residual miss
  remains — a name/author lookup whose target chunk shares no vocabulary
  with the query, the same pattern a smaller-corpus run first surfaced;
  needs query rewriting, not a retrieval tweak. Generation hasn't been
  re-run on this expanded corpus yet (still reflects the original
  single-document run). See `backend/eval/README.md`'s "Known result"
  section for the full writeup — the harness explicitly warns against
  trusting its summary numbers without reading `per_question`, and that
  warning has already caught a wrong claim once (an earlier draft assumed
  hybrid alone would fix the query above; it didn't, and the docs were
  corrected once actually measured).

## Roadmap

Short-term priorities, roughly in order: query rewriting (to address the
vocabulary-mismatch gap above), re-run the generation eval on the expanded
corpus and grow the dataset past 50 questions (contracts/legal documents,
adversarial content), inline citations with page-level source highlighting,
streaming + markdown in the UI, and a Docker Compose setup that works from a
clean clone.
