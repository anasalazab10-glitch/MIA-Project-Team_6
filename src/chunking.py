from typing import Any
import re
from src.schemas import Chunk, ContentType, TableContent


DEFAULT_MAX_CHARS = 1000
DEFAULT_OVERLAP_CHARS = 150


def split_text(text: str,max_chars: int = DEFAULT_MAX_CHARS,overlap_chars: int = DEFAULT_OVERLAP_CHARS,) -> list[str]:
    """
    Split text into coherent chunks using sentence boundaries.

    Chunks are limited by max_chars whenever possible.
    A small overlap is kept between consecutive chunks
    to preserve context.
    """

    if not text.strip():
        return []

    sentences = re.split(r"(?<=[.!?])\s+", text.strip())

    chunks: list[str] = []
    current_sentences: list[str] = []
    current_length = 0

    for sentence in sentences:
        sentence = sentence.strip()

        if not sentence:
            continue

        sentence_length = len(sentence)

        # If adding the sentence exceeds the maximum size,
        # finish the current chunk first.
        if (
            current_sentences
            and current_length + sentence_length + 1 > max_chars):
            chunks.append(" ".join(current_sentences))
            # Keep the last sentences as overlap.
            overlap_sentences: list[str] = []
            overlap_length = 0

            for previous_sentence in reversed(current_sentences):
                previous_length = len(previous_sentence) + 1

                if overlap_length + previous_length > overlap_chars:
                    break

                overlap_sentences.insert(0, previous_sentence)
                overlap_length += previous_length

            current_sentences = overlap_sentences
            current_length = overlap_length

        current_sentences.append(sentence)
        current_length += sentence_length + 1

    # Add the final chunk.
    if current_sentences:
        chunks.append(" ".join(current_sentences))

    return chunks


def create_chunks(elements: list[dict[str, Any]],max_chars: int = DEFAULT_MAX_CHARS,overlap_chars: int = DEFAULT_OVERLAP_CHARS,) -> list[Chunk]:
    """
    Convert processed document elements into standardized retrieval chunks.

    Chunking strategy:
    - Section-aware: headings define section context.
    - Text from the same section is grouped together.
    - Text can continue across page boundaries.
    - Sentence boundaries are preferred when splitting text.
    - A maximum chunk size prevents excessively large chunks.
    - Consecutive text chunks have a small overlap.
    - Tables remain separate structured chunks.
    - Document, page, section, and content_type metadata are preserved.
    """

    chunks: list[Chunk] = []

    # Track the number of chunks created for each document.
    chunk_counters: dict[str, int] = {}

    # Text elements waiting to be converted into chunks.
    current_text_elements: list[dict[str, Any]] = []

    # Section determined by the most recent heading.
    current_section: str | None = None

    def next_chunk_id(document_id: str) -> str:
        """Generate a unique chunk ID within a document."""

        chunk_counters[document_id] = (
            chunk_counters.get(document_id, 0) + 1
        )

        return f"{document_id}_chunk{chunk_counters[document_id]}"

    def flush_text() -> None:
        """
        Convert accumulated text elements into retrieval chunks.
        """
        nonlocal current_text_elements
        if not current_text_elements:
            return
        first_element = current_text_elements[0]
        document_id = first_element["document_id"]
        # Combine text elements belonging to the same section.
        text = "\n\n".join(
            element["content"]
            for element in current_text_elements
            if isinstance(element["content"], str)
        )

        # Split the text using sentence boundaries and size limits.
        text_chunks = split_text(
            text,
            max_chars=max_chars,
            overlap_chars=overlap_chars,
        )

        # Preserve the original processed element IDs.
        source_element_ids = [
            element["element_id"]
            for element in current_text_elements
            if "element_id" in element
        ]

        # Preserve all pages touched by the text.
        source_pages = sorted(
            {
                element["page"]
                for element in current_text_elements
                if "page" in element
            }
        )

        for text_chunk in text_chunks:
            chunks.append(
                Chunk(
                    chunk_id=next_chunk_id(document_id),
                    document_id=document_id,
                    page=source_pages,
                    section=current_section,
                    content_type=ContentType.TEXT,
                    content=text_chunk,
                    bbox=first_element.get("bbox"),
                    metadata={
                        **first_element.get("metadata", {}),
                        "source_element_ids": source_element_ids,
                        "source_pages": source_pages,
                    },
                )
            )

        # Clear the temporary text elements.
        current_text_elements = []

    for element in elements:

        content_type = ContentType(element["content_type"])

        # Heading

        if content_type == ContentType.HEADING:

            # Finish text from the previous section.
            flush_text()

            # The new heading becomes the current section.
            current_section = element["content"]
        # Text

        elif content_type == ContentType.TEXT:

            if current_text_elements:
                previous = current_text_elements[-1]

                # Do not combine text from different documents.
                #
                # PAGE IS NOT checked here because a chunk is
                # allowed to continue across page boundaries.
                if previous["document_id"] != element["document_id"]:
                    flush_text()

            current_text_elements.append(element)

            # Prevent an excessively large amount of text from
            # accumulating before it is processed.
            current_length = sum(
                len(e["content"])
                for e in current_text_elements
                if isinstance(e["content"], str)
            )

            if current_length >= max_chars * 3:
                flush_text()

        # Table
        elif content_type == ContentType.TABLE:

            # Finish any text before the table.
            flush_text()

            document_id = element["document_id"]

            # Keep the table structured.
            table = TableContent.model_validate(
                element["content"]
            )

            chunks.append(
                Chunk(
                    chunk_id=next_chunk_id(document_id),
                    document_id=document_id,
                    page=[element["page"]],
                    section=current_section or element.get("section"),
                    content_type=ContentType.TABLE,
                    content=table,
                    bbox=element.get("bbox"),
                    metadata={
                        **element.get("metadata", {}),
                        "source_element_ids": (
                            [element["element_id"]]
                            if "element_id" in element
                            else []
                        ),
                        "source_pages": [element["page"]],
                    },
                )
            )

    # Process any remaining text.
    flush_text()

    return chunks

