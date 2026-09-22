import os

# OmniDoc is local-first: once the embedding model is cached, it
# must never make a network call to HuggingFace Hub just to check
# for updates. Besides being slow, it makes the app fail outright
# on a broken proxy/SSL config (as happened here) or with no
# internet at all, defeating the point of running locally. This
# must be set before `sentence_transformers` is imported.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

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