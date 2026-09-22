import json
import re
from pathlib import Path

from app.config.settings import settings
from app.schemas.chunk import Chunk


class ChunkingService:
    """
    Splits markdown into semantic chunks.

    Small parent sections (e.g. a heading followed by a couple of
    lines) are merged forward into the next meaningful section so
    they are never emitted as orphan, context-free chunks. Sections
    that are too large for a single chunk are split further on
    paragraph boundaries with a small overlap, so retrieval doesn't
    have to stuff an entire chapter into one context slot.
    """

    HEADING_PATTERN = r"(?=^#{1,6}\s)"

    MIN_BODY_LENGTH = settings.CHUNK_MIN_BODY_CHARS
    MAX_CHUNK_CHARS = settings.CHUNK_MAX_CHARS
    OVERLAP_CHARS = settings.CHUNK_OVERLAP_CHARS

    @staticmethod
    def chunk_document(document_folder: Path) -> list[Chunk]:

        parsed_path = document_folder / "parsed.json"

        if not parsed_path.exists():
            raise FileNotFoundError("parsed.json not found")

        with parsed_path.open("r", encoding="utf-8") as f:
            parsed = json.load(f)

        markdown = parsed.get("text", "").strip()

        if not markdown:
            return []

        sections = re.split(
            ChunkingService.HEADING_PATTERN,
            markdown,
            flags=re.MULTILINE,
        )

        raw_sections = []

        # -------------------------------------------------
        # Pass 1: parse each markdown block into
        # (heading, level, body), merging any run of tiny
        # "parent" sections forward into the next real one.
        # A tiny section at the very end of the document is
        # flushed as its own chunk instead of being dropped.
        # -------------------------------------------------

        pending_heading = None
        pending_text = ""

        for section in sections:

            section = section.strip()

            if not section:
                continue

            lines = section.splitlines()
            first_line = lines[0].strip()

            heading = ""
            level = 0

            if first_line.startswith("#"):
                heading = first_line.lstrip("#").strip()
                level = len(first_line) - len(first_line.lstrip("#"))

            body = "\n".join(lines[1:]).strip()

            if len(body) < ChunkingService.MIN_BODY_LENGTH:

                # Accumulate consecutive tiny sections instead of
                # overwriting the previous pending one.
                pending_heading = pending_heading or heading

                pending_text = (
                    (pending_text + "\n\n" + section).strip()
                    if pending_text
                    else section
                )

                continue

            section_name = None

            if pending_heading:
                section_name = pending_heading
                section = pending_text + "\n\n" + section
                pending_heading = None
                pending_text = ""

            raw_sections.append((section_name, heading, level, section))

        # Flush a trailing tiny section instead of silently dropping it.
        if pending_text:
            raw_sections.append((None, pending_heading, 0, pending_text))

        # -------------------------------------------------
        # Pass 2: split any section that's too large for one
        # chunk, on paragraph boundaries with overlap so the
        # retriever never has to embed/stuff a giant blob.
        # -------------------------------------------------

        chunks: list[Chunk] = []

        for section_name, heading, level, text in raw_sections:

            parts = ChunkingService._split_to_size(text)

            for part_index, part in enumerate(parts):

                chunk = Chunk(
                    text=part,
                    page=None,
                    metadata={
                        "section": section_name,
                        "heading": heading,
                        "level": level,
                        "document_id": parsed.get("document_id"),
                        "title": parsed.get("title"),
                        "source": "docling",
                        "chunk_type": "section",
                        "parser": "markdown",
                        "part_index": part_index,
                        "part_count": len(parts),
                    },
                )

                chunks.append(chunk)

        ChunkingService.save(
            document_folder=document_folder,
            chunks=chunks,
        )

        return chunks

    @staticmethod
    def _split_to_size(text: str) -> list[str]:
        """
        Splits `text` into chunks of at most MAX_CHUNK_CHARS,
        preferring paragraph boundaries, with OVERLAP_CHARS of
        trailing context carried into the next piece so a fact
        split across a chunk boundary doesn't lose context.
        """

        max_chars = ChunkingService.MAX_CHUNK_CHARS

        if len(text) <= max_chars:
            return [text]

        paragraphs = re.split(r"\n\s*\n", text)

        pieces: list[str] = []
        current = ""

        for paragraph in paragraphs:

            paragraph = paragraph.strip()

            if not paragraph:
                continue

            # A single paragraph longer than the budget is split
            # on its own, on whitespace, so we never emit an
            # unbounded chunk.
            if len(paragraph) > max_chars:

                if current:
                    pieces.append(current)
                    current = ""

                words = paragraph.split(" ")
                piece = ""

                for word in words:
                    candidate = f"{piece} {word}".strip()

                    if len(candidate) > max_chars and piece:
                        pieces.append(piece)
                        piece = word
                    else:
                        piece = candidate

                if piece:
                    current = piece

                continue

            candidate = (
                f"{current}\n\n{paragraph}" if current else paragraph
            )

            if len(candidate) > max_chars:
                pieces.append(current)
                current = paragraph
            else:
                current = candidate

        if current:
            pieces.append(current)

        # Carry trailing overlap forward between adjacent pieces.
        overlapped = []

        for i, piece in enumerate(pieces):

            if i == 0:
                overlapped.append(piece)
                continue

            prev = pieces[i - 1]
            tail = prev[-ChunkingService.OVERLAP_CHARS:]

            # A raw character slice can land mid-word. Trim back to
            # the next space so the overlap starts on a word
            # boundary instead of splicing a fragment like "ing"
            # onto the front of the next chunk.
            space_index = tail.find(" ")
            if 0 < space_index < len(tail):
                tail = tail[space_index + 1:]

            overlapped.append(f"{tail}\n\n{piece}")

        return overlapped or [text]

    @staticmethod
    def save(
        document_folder: Path,
        chunks: list[Chunk],
    ) -> None:

        output_path = document_folder / "chunks.json"

        with output_path.open(
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                [chunk.model_dump() for chunk in chunks],
                f,
                indent=4,
                ensure_ascii=False,
            )
