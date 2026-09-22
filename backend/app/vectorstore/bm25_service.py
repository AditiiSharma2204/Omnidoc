import json
import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from app.config.settings import settings

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """
    Simple, dependency-free tokenizer: lowercase, alphanumeric runs
    only. No stemming/stopword removal -- BM25's own idf weighting
    already discounts common words, and keeping this dependency-free
    avoids pulling in nltk/spacy (and their data downloads) for a
    local-first app.
    """
    return _TOKEN_PATTERN.findall(text.lower())


class BM25Service:
    """
    Lexical (BM25) sibling to FAISSService's dense index.

    Unlike FAISS, BM25's idf/avgdl statistics depend on the whole
    corpus, so there's no cheap incremental "add one document"
    operation -- every add or delete rebuilds the full index from
    every document's chunks.json on disk. That's fine at this
    project's scale (a handful of documents, tens of chunks each):
    a full rebuild is milliseconds, not a bottleneck worth
    engineering around yet.
    """

    INDEX_FILE = "bm25_index.pkl"
    METADATA_FILE = "bm25_metadata.json"

    @classmethod
    def get_index_path(cls) -> Path:
        return Path(settings.VECTORSTORE_DIR) / cls.INDEX_FILE

    @classmethod
    def get_metadata_path(cls) -> Path:
        return Path(settings.VECTORSTORE_DIR) / cls.METADATA_FILE

    @classmethod
    def load_index(cls) -> BM25Okapi | None:
        path = cls.get_index_path()
        if not path.exists():
            return None
        with path.open("rb") as f:
            return pickle.load(f)

    @classmethod
    def save_index(cls, index: BM25Okapi) -> None:
        with cls.get_index_path().open("wb") as f:
            pickle.dump(index, f)

    @classmethod
    def load_metadata(cls) -> list[dict]:
        path = cls.get_metadata_path()
        if not path.exists():
            return []
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    @classmethod
    def save_metadata(cls, metadata: list[dict]) -> None:
        with cls.get_metadata_path().open(
            "w", encoding="utf-8"
        ) as f:
            json.dump(metadata, f, indent=4, ensure_ascii=False)

    @classmethod
    def rebuild_index(cls, documents_dir: Path) -> None:
        """
        Rebuilds the BM25 index from every document's chunks.json on
        disk. Called after every add and delete (see module
        docstring for why there's no incremental add).
        """
        corpus_tokens: list[list[str]] = []
        metadata: list[dict] = []

        if not documents_dir.exists():
            folders = []
        else:
            folders = sorted(
                p for p in documents_dir.iterdir() if p.is_dir()
            )

        for document_folder in folders:
            chunks_path = document_folder / "chunks.json"
            if not chunks_path.exists():
                continue

            with chunks_path.open("r", encoding="utf-8") as f:
                chunks = json.load(f)

            document_id = document_folder.name

            for i, chunk in enumerate(chunks):
                corpus_tokens.append(tokenize(chunk["text"]))
                metadata.append(
                    {
                        "document_id": document_id,
                        "chunk_id": chunk["chunk_id"],
                        "chunk_index": i,
                        "heading": chunk["metadata"].get("heading"),
                        "page": chunk.get("page"),
                        "title": chunk["metadata"].get("title"),
                    }
                )

        if not corpus_tokens:
            cls.get_index_path().unlink(missing_ok=True)
            cls.get_metadata_path().unlink(missing_ok=True)
            return

        index = BM25Okapi(corpus_tokens)
        cls.save_index(index)
        cls.save_metadata(metadata)

    @classmethod
    def search(
        cls, query: str, top_k: int
    ) -> list[tuple[float, dict]]:
        """
        Returns up to top_k (score, metadata) pairs, sorted by score
        descending, skipping zero-score (no lexical overlap at all)
        results -- a zero BM25 score isn't "the 5th most relevant
        thing", it's "not relevant".
        """
        index = cls.load_index()
        if index is None:
            return []

        metadata = cls.load_metadata()
        scores = index.get_scores(tokenize(query))

        ranked = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )

        results = []
        for i in ranked[:top_k]:
            if scores[i] <= 0:
                break
            results.append((float(scores[i]), metadata[i]))

        return results
