from app.bootstrap import apply_local_first_env_fixes

# Defense in depth for any entry point that imports this module
# without going through app.main or eval/run_eval.py first (e.g. a
# one-off script or notebook). Harmless if already applied --
# os.environ.setdefault() no-ops on the second call.
apply_local_first_env_fixes()

from sentence_transformers import SentenceTransformer  # noqa: E402

from app.config.settings import settings  # noqa: E402


class EmbeddingModel:
    _model = None

    @classmethod
    def get_model(cls):
        if cls._model is None:
            cls._model = SentenceTransformer(
                settings.EMBEDDING_MODEL_NAME,
                cache_folder=settings.EMBEDDING_CACHE_DIR,
                trust_remote_code=True,
                # Reduces peak RSS during weight materialization by
                # not keeping a duplicate copy of the state dict
                # around mid-load. Matters on memory-constrained
                # machines (this app targets a 2GB-VRAM/CPU-inference
                # laptop) where loading BGE-M3 alongside Docling's
                # already-resident models can otherwise OOM.
                model_kwargs={"low_cpu_mem_usage": True},
            )

        return cls._model