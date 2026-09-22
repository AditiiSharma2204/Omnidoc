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
                                              FAISS (vector index)  ←┘
                                                      ↓
                                       Retriever → Prompt Builder → Qwen2.5 (Ollama)
                                                      ↓
                                                   Answer + sources
```

- **Parsing** — [Docling](https://github.com/docling-project/docling) converts PDF/DOCX/PPTX/XLSX to markdown.
- **Chunking** — a custom markdown-hierarchy chunker with a size cap and overlap (`app/services/chunking_service.py`).
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
model (~2.2GB) — after that it's cached and runs fully offline.

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
- **No cross-encoder reranker yet.** Retrieval is hybrid (BM25 + dense,
  RRF-fused, measured to match or beat dense-only — see `backend/eval/README.md`),
  but nothing re-scores the fused candidates with a stronger model.
- **No OCR fallback** — scanned (image-only) PDFs will parse to near-empty text.
- **No table/chart/image understanding** — Docling extracts tables as markdown text; nothing structures or reasons over them specially yet.
- **No conversation memory** — each question is answered independently of chat history.
- **No persistent chat history** — refreshing the page loses the conversation.
- **No auth, no multi-user support.**
- **Memory-constrained by design.** This targets modest hardware (built against
  an NVIDIA MX450, 2GB VRAM). Ollama, Docling's parsing models and the BGE-M3
  embedding model each keep a resident model in RAM, and on a machine with
  limited free system RAM (~3GB or less) having more than one loaded at once
  can OOM-crash a process. Mitigations in place: Docling's OCR and
  table-structure models are disabled by default (`DOCLING_DO_OCR`,
  `DOCLING_DO_TABLE_STRUCTURE` in settings), the embedding model loads with
  `low_cpu_mem_usage`, and `LLMService.unload()` can release Ollama's resident
  model on demand before heavy local work. If you still hit a crash, free up
  RAM or reduce concurrent load; a proper fix (ingestion in an isolated worker
  process) is on the roadmap.
- **Evaluation harness exists but is small and single-document.** See
  `backend/eval/` — a 15-question seed set with retrieval (Recall@k, MRR,
  and a dense/bm25/hybrid ablation via `--compare-modes`) and generation
  (keyword-hit, refusal-rate) metrics. Hybrid retrieval measurably beats
  dense-only (Recall@1 75%→91.7%, MRR 0.83→0.92, zero per-question
  regressions) — but it is not a universal fix: one question is still
  wrong in *every* retrieval mode because it shares no vocabulary with
  the document at all, which no amount of fusion can invent. Manual
  review of the generation run also found two real answer-quality issues
  and one grounding risk the automated metrics missed. See
  `backend/eval/README.md`'s "Known result" section for the full,
  unvarnished writeup — the harness explicitly warns against trusting its
  summary numbers without reading `per_question`, and that warning has
  already caught a wrong claim once (an earlier draft of this project
  assumed hybrid would fix the query above; it didn't, and the docs were
  corrected once actually measured). Needs expanding to 50+ questions
  across varied document types before it's a real benchmark.

## Roadmap

Short-term priorities, roughly in order: a cross-encoder reranker on top
of hybrid retrieval, query rewriting (to address the vocabulary-mismatch
gap above), expanding the evaluation harness (more documents, more
questions, LLM-as-judge for faithfulness), inline citations with
page-level source highlighting,
streaming + markdown in the UI, and a Docker Compose setup that works from a
clean clone.
