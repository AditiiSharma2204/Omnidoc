from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    APP_NAME: str = "OmniDoc AI"
    APP_VERSION: str = "1.0.0"

    API_PREFIX: str = "/api/v1"

    DOCUMENTS_DIR: str = "storage/documents"
    VECTORSTORE_DIR: str = "storage/vectorstore"
    DATABASE_PATH: str = "storage/app.db"

    # LLM (Ollama)
    OLLAMA_BASE_URL: str = "http://127.0.0.1:11434"
    LLM_MODEL_NAME: str = "qwen2.5:3b"
    LLM_NUM_CTX: int = 8192
    LLM_TEMPERATURE: float = 0.2

    # Embeddings
    EMBEDDING_MODEL_NAME: str = "BAAI/bge-m3"
    EMBEDDING_CACHE_DIR: str | None = r"D:\HF_CACHE"

    # Retrieval
    # "dense": FAISS/BGE-M3 only (original behavior).
    # "bm25": lexical-only (BM25Okapi), for comparison.
    # "hybrid": both, fused with Reciprocal Rank Fusion. Default --
    # measured on the seed eval set to match or beat dense-only at
    # every k (Recall@1 75%->91.7%, MRR 0.83->0.92), with no
    # regressions on any individual question. See
    # backend/eval/README.md for the full ablation and an honest
    # caveat: it does NOT fix every retrieval miss (a query that
    # shares no vocabulary with the target chunk defeats BM25 too).
    RETRIEVAL_MODE: str = "hybrid"
    RRF_K: int = 60

    # Cross-encoder reranking: re-scores the top RERANK_CANDIDATE_MULTIPLIER
    # * top_k fused candidates with a model that sees the actual
    # (query, chunk) pair jointly, instead of comparing independently
    # computed embeddings. Slower per-query (one forward pass per
    # candidate) but more accurate at the top of the ranking -- see
    # backend/eval/README.md for the measured before/after. Picked a
    # small model (~500MB resident, vs. ~2GB for BGE-M3) deliberately
    # given this project's memory constraints (see
    # DOCLING_DO_OCR/DOCLING_DO_TABLE_STRUCTURE above).
    RERANK_ENABLED: bool = True
    RERANKER_MODEL_NAME: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    # Measured, not guessed: on a 5-document/~50-chunk corpus, 4x
    # (top_k*4=20 candidates) was too narrow -- a paper-heavy corpus
    # crowded a genuinely relevant chunk out of the pre-rerank pool
    # entirely for one query, which reranking then had no chance to
    # recover (it can only reorder what it's given). 8x fixed it with
    # no further gain from 12x. See backend/eval/README.md.
    RERANK_CANDIDATE_MULTIPLIER: int = 8

    # Refusal threshold: chunks the reranker scores below this cutoff
    # are dropped before being sent to the LLM at all, so a weak
    # retrieval match can't be confidently answered from as if it
    # were solid grounding (see PromptBuilder.NO_CONTEXT_MESSAGE for
    # the resulting refusal). Scoped specifically to reranker scores
    # (cross-encoder logits) -- NOT applied to dense/BM25/hybrid
    # scores, whose scales are different and not calibrated here.
    # None (the default) disables filtering entirely: this needs
    # real calibration (sampling reranker scores across known
    # relevant vs. irrelevant chunks) before it's safe to set a
    # nonzero default -- see backend/eval/README.md. Shipping an
    # uncalibrated guess risks silently refusing correct answers.
    RERANK_SCORE_THRESHOLD: float | None = None

    # Query rewriting (HyDE-style): generates a short hypothetical
    # answer passage with the LLM and embeds THAT for dense retrieval
    # instead of the raw question. Targets a specific, measured gap:
    # short "what is X's name/title/author" queries share no
    # vocabulary with the chunk that states the answer, so neither
    # dense nor BM25 similarity has anything to match on (see
    # backend/eval/README.md's residual-miss writeup). A hypothetical
    # answer written in document-like prose closes that gap. Off by
    # default: it adds a real LLM call (measured latency cost) to
    # every query, and the benefit is narrow (a handful of questions
    # in the current eval set) -- see the eval writeup for the
    # measured before/after before turning this on.
    QUERY_REWRITE_ENABLED: bool = False

    # Chunking (character-based; ~4 chars/token heuristic)
    CHUNK_MAX_CHARS: int = 1600
    CHUNK_OVERLAP_CHARS: int = 200
    CHUNK_MIN_BODY_CHARS: int = 80

    # Docling parsing pipeline. OCR loads 3 extra CPU models (~1GB+)
    # for scanned-page detection we don't yet use (see roadmap: OCR
    # fallback), and on memory-constrained machines competes with
    # the layout + table-structure + embedding models for RAM in the
    # same process. Off by default; flip on once OCR is wired up or
    # you're not RAM-constrained.
    DOCLING_DO_OCR: bool = False
    DOCLING_DO_TABLE_STRUCTURE: bool = False

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )


settings = Settings()