import json

from app.services.chunking_service import ChunkingService


def _write_parsed(tmp_path, text, document_id="doc-1", title="Test.pdf"):
    parsed = {
        "document_id": document_id,
        "title": title,
        "pages": 1,
        "text": text,
        "tables": [],
        "figures": [],
        "metadata": {},
    }
    (tmp_path / "parsed.json").write_text(
        json.dumps(parsed), encoding="utf-8"
    )
    return tmp_path


def _chunk_texts(chunks):
    return [c.text for c in chunks]


class TestTinySectionMerging:
    """
    Regression tests for the original chunker bug: a tiny "parent"
    heading (e.g. "## Internship" followed by a couple of lines)
    must be merged into the next real section, not silently dropped.
    """

    def test_single_tiny_section_merges_into_next(self, tmp_path):
        text = (
            "## Internship\n\nChennai\nFeb 2026\n\n"
            "## Authenta.ai\n\n"
            + ("Research and development engineer intern. " * 5)
        )
        _write_parsed(tmp_path, text)

        chunks = ChunkingService.chunk_document(tmp_path)

        assert len(chunks) == 1
        assert chunks[0].metadata["section"] == "Internship"
        assert chunks[0].metadata["heading"] == "Authenta.ai"
        # The tiny section's own content must survive inside the
        # merged chunk, not be discarded.
        assert "Chennai" in chunks[0].text
        assert "Feb 2026" in chunks[0].text

    def test_consecutive_tiny_sections_both_survive(self, tmp_path):
        """
        Two tiny headings in a row used to overwrite each other
        (pending_heading/pending_text were reassigned, not
        accumulated), silently losing the first one.
        """
        text = (
            "## A\n\nshort a\n\n"
            "## B\n\nshort b\n\n"
            "## Real Section\n\n"
            + ("Enough real content here. " * 6)
        )
        _write_parsed(tmp_path, text)

        chunks = ChunkingService.chunk_document(tmp_path)

        assert len(chunks) == 1
        assert "short a" in chunks[0].text
        assert "short b" in chunks[0].text

    def test_trailing_tiny_section_is_not_dropped(self, tmp_path):
        """
        A tiny section with nothing after it in the document used to
        vanish entirely -- pending_heading/pending_text were set but
        never flushed.
        """
        text = (
            "## Real Section\n\n"
            + ("Enough real content here. " * 6)
            + "\n\n## Positions of Responsibility\n\nSponsorship lead"
        )
        _write_parsed(tmp_path, text)

        chunks = ChunkingService.chunk_document(tmp_path)

        all_text = " ".join(_chunk_texts(chunks))
        assert "Positions of Responsibility" in all_text
        assert "Sponsorship lead" in all_text


class TestSizeCapAndOverlap:

    def test_oversized_section_is_split_without_losing_words(
        self, tmp_path
    ):
        words = [f"word{i}" for i in range(1000)]
        body = " ".join(words)
        text = f"## Big Section\n\n{body}"
        _write_parsed(tmp_path, text)

        chunks = ChunkingService.chunk_document(tmp_path)

        assert len(chunks) > 1

        # Every original word must appear somewhere in the output
        # (overlap duplicates some words across boundaries, which is
        # fine -- nothing may be missing).
        combined = " ".join(_chunk_texts(chunks))
        for word in ("word0", "word500", "word999"):
            assert word in combined

    def test_overlap_does_not_split_mid_word(self):
        text = " ".join(f"token{i}" for i in range(600))

        pieces = ChunkingService._split_to_size(text)

        assert len(pieces) > 1

        for piece in pieces:
            for line in piece.split("\n\n"):
                for token in line.split(" "):
                    if token:
                        assert token.startswith("token"), (
                            f"found a fragment that isn't a whole "
                            f"token: {token!r}"
                        )

    def test_short_section_is_not_split(self, tmp_path):
        text = "## Short\n\n" + ("word " * 20)
        _write_parsed(tmp_path, text)

        chunks = ChunkingService.chunk_document(tmp_path)

        assert len(chunks) == 1


class TestMetadataAndPersistence:

    def test_chunk_metadata_carries_document_id_and_title(
        self, tmp_path
    ):
        text = "## Heading\n\n" + ("content " * 20)
        _write_parsed(
            tmp_path, text, document_id="abc-123", title="My Doc.pdf"
        )

        chunks = ChunkingService.chunk_document(tmp_path)

        assert chunks[0].metadata["document_id"] == "abc-123"
        assert chunks[0].metadata["title"] == "My Doc.pdf"

    def test_chunks_are_persisted_to_disk(self, tmp_path):
        text = "## Heading\n\n" + ("content " * 20)
        _write_parsed(tmp_path, text)

        ChunkingService.chunk_document(tmp_path)

        output_path = tmp_path / "chunks.json"
        assert output_path.exists()

        saved = json.loads(output_path.read_text(encoding="utf-8"))
        assert len(saved) == 1

    def test_empty_document_produces_no_chunks(self, tmp_path):
        _write_parsed(tmp_path, "")

        chunks = ChunkingService.chunk_document(tmp_path)

        assert chunks == []

    def test_missing_parsed_json_raises(self, tmp_path):
        import pytest

        with pytest.raises(FileNotFoundError):
            ChunkingService.chunk_document(tmp_path)
