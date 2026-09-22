from app.bootstrap import apply_local_first_env_fixes

apply_local_first_env_fixes()

from sentence_transformers import CrossEncoder  # noqa: E402

from app.config.settings import settings  # noqa: E402


class RerankerModel:
    """
    Lazy singleton for the cross-encoder reranker, mirroring
    EmbeddingModel's pattern.
    """

    _model = None

    @classmethod
    def get_model(cls) -> CrossEncoder:
        if cls._model is None:
            cls._model = CrossEncoder(
                settings.RERANKER_MODEL_NAME,
                cache_folder=settings.EMBEDDING_CACHE_DIR,
                model_kwargs={"low_cpu_mem_usage": True},
            )
        return cls._model
