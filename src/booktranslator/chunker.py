"""Split chapters into translation units (chunks) with overlap context.

A chunk is a sequence of contiguous paragraph elements from a single
chapter, roughly target_words long. Each chunk keeps references to its
paragraph elements (so we can put translated XHTML back in place) and
to the neighbouring paragraphs used as read-only context.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

from lxml import etree

from .epub_io import StructuredBook


def _element_to_xhtml_fragment(el: etree._Element) -> str:
    """Serialise an element + its children to an XHTML fragment string.

    Strips namespace prefixes by rendering the outer element's expanded
    tag as its local name. lxml's `tostring` on a namespaced element
    produces `<ns0:p xmlns:ns0="...">`, which is ugly in prompts.
    Rather than stripping namespaces from the live tree (which we must
    preserve for valid EPUB output), we serialize a shallow copy with
    namespaces cleaned up ONLY for the prompt.
    """
    # Deep copy so we don't mutate the live tree.
    import copy

    clone = copy.deepcopy(el)

    # Strip XHTML namespace prefixes from this subtree.
    for sub in clone.iter():
        if isinstance(sub.tag, str) and "}" in sub.tag:
            sub.tag = etree.QName(sub.tag).localname
    # Remove namespace declarations on the element itself.
    etree.cleanup_namespaces(clone)

    return etree.tostring(clone, encoding="unicode", with_tail=False)


def _element_word_count(el: etree._Element) -> int:
    return len(" ".join(el.itertext()).split())


@dataclass
class Chunk:
    id: str  # e.g. "ch03_c02"
    chapter_index: int  # 0-based index in StructuredBook.chapters
    paragraph_indexes: list[int]  # indexes within ChapterDoc.paragraphs
    word_count: int
    # Overlap (context only, NOT translated). Indices within ChapterDoc.paragraphs.
    prev_overlap_indexes: list[int] = field(default_factory=list)
    next_overlap_indexes: list[int] = field(default_factory=list)


@dataclass
class ChunkSet:
    """All chunks for a book plus the structured book they refer to."""

    book: StructuredBook
    chunks: list[Chunk]

    # Helpers to render chunks into prompt strings ----------------------

    def render_main(self, chunk: Chunk) -> list[str]:
        """XHTML fragments for each paragraph in the chunk (to translate)."""
        ch = self.book.chapters[chunk.chapter_index]
        return [_element_to_xhtml_fragment(ch.paragraphs[i]) for i in chunk.paragraph_indexes]

    def render_overlap(self, chunk: Chunk, side: str) -> list[str]:
        """XHTML fragments for the before/after overlap (context only)."""
        ch = self.book.chapters[chunk.chapter_index]
        indexes = chunk.prev_overlap_indexes if side == "prev" else chunk.next_overlap_indexes
        return [_element_to_xhtml_fragment(ch.paragraphs[i]) for i in indexes]


def chunk_book(
    book: StructuredBook,
    target_words: int = 2000,
    overlap_paragraphs: int = 1,
) -> ChunkSet:
    """Build a ChunkSet for the entire book."""
    all_chunks: list[Chunk] = []

    for ch_idx, chapter in enumerate(book.chapters):
        if not chapter.paragraphs:
            continue

        # Greedy pack paragraphs until we hit target_words, then close.
        current: list[int] = []
        current_words = 0
        chapter_chunks: list[list[int]] = []

        for p_idx, p in enumerate(chapter.paragraphs):
            w = _element_word_count(p)
            if current and current_words + w > target_words:
                chapter_chunks.append(current)
                current = []
                current_words = 0
            current.append(p_idx)
            current_words += w

        if current:
            chapter_chunks.append(current)

        # Promote packed lists into Chunk objects with overlap.
        for c_i, para_indexes in enumerate(chapter_chunks):
            wc = sum(_element_word_count(chapter.paragraphs[i]) for i in para_indexes)

            prev_ov: list[int] = []
            if c_i > 0 and overlap_paragraphs > 0:
                prev_pool = chapter_chunks[c_i - 1]
                prev_ov = prev_pool[-overlap_paragraphs:]

            next_ov: list[int] = []
            if c_i + 1 < len(chapter_chunks) and overlap_paragraphs > 0:
                next_pool = chapter_chunks[c_i + 1]
                next_ov = next_pool[:overlap_paragraphs]

            chunk_id = f"ch{ch_idx + 1:02d}_c{c_i + 1:02d}"
            all_chunks.append(
                Chunk(
                    id=chunk_id,
                    chapter_index=ch_idx,
                    paragraph_indexes=para_indexes,
                    word_count=wc,
                    prev_overlap_indexes=prev_ov,
                    next_overlap_indexes=next_ov,
                )
            )

    return ChunkSet(book=book, chunks=all_chunks)


def iter_chunks(chunk_set: ChunkSet) -> Iterator[Chunk]:
    return iter(chunk_set.chunks)
