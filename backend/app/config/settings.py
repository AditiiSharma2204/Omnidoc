from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    APP_NAME: str = "OmniDoc AI"
    APP_VERSION: str = "1.0.0"

    API_PREFIX: str = "/api/v1"

    DOCUMENTS_DIR: str = "storage/documents"
    VECTORSTORE_DIR: str = "storage/vectorstore"

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