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