import json
from pathlib import Path

import numpy as np

from app.embeddings.model import EmbeddingModel


class EmbeddingService:

    @staticmethod
    def generate(document_folder: Path) -> np.ndarray:

        chunks_path = document_folder / "chunks.json"

        if not chunks_path.exists():
            raise FileNotFoundError("chunks.json not found")

        with chunks_path.open(
            "r",
            encoding="utf-8",
        ) as f:

            chunks = json.load(f)

        texts = [
            chunk["text"]
            for chunk in chunks
        ]

        model = EmbeddingModel.get_model()

        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        output = document_folder / "embeddings.npy"

        np.save(
            output,
            embeddings,
        )

        return embeddings