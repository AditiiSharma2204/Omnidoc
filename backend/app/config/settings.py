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