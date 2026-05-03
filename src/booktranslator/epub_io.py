"""EPUB I/O. Iteration 1: read-only, flat text extraction."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

from ebooklib import ITEM_DOCUMENT, epub
from lxml import html as lxhtml

from .models import BookMeta


@dataclass
class ExtractedBook:
    """Plain-text view of an EPUB. Structural fidelity is not preserved
    at this stage — that comes in the translate iteration."""

    meta: BookMeta
    chapters: list[str]          # flat text per spine item
    source_path: Path

    def full_text(self) -> str:
        """Join all chapter texts for a whole-book prompt."""
        return "\n\n".join(self.chapters)


def _first_metadata(book: epub.EpubBook, field: str, default: str = "") -> str:
    """ebooklib returns metadata as a list of (value, attrs); take the first."""
    items = book.get_metadata("DC", field)
    if items and items[0] and items[0][0]:
        return items[0][0].strip()
    return default


def _chapter_text(document: epub.EpubHtml) -> str:
    """Extract readable text from a spine document."""
    content = document.get_content()
    if not content:
        return ""
    try:
        tree = lxhtml.fromstring(content)
    except Exception:
        return ""
    # text_content() joins all descendant text; good enough for glossary
    # extraction. A proper paragraph-aware extractor lives in chunker.py.
    return tree.text_content().strip()


def read_book(path: Path) -> ExtractedBook:
    """Read an EPUB and return metadata + flat chapter texts."""
    with warnings.catch_warnings():
        # ebooklib emits harmless warnings about missing properties
        # on legacy EPUB2 files; silence them for a clean CLI output.
        warnings.simplefilter("ignore")
        book = epub.read_epub(str(path))

    chapters: list[str] = []
    for spine_id, _ in book.spine:
        item = book.get_item_with_id(spine_id)
        if item is None or item.get_type() != ITEM_DOCUMENT:
            continue
        text = _chapter_text(item)
        if text:
            chapters.append(text)

    word_count = sum(len(c.split()) for c in chapters)

    meta = BookMeta(
        title=_first_metadata(book, "title", default=path.stem),
        author=_first_metadata(book, "creator", default="Unknown"),
        language=_first_metadata(book, "language", default="en"),
        word_count=word_count,
        chapters=len(chapters),
    )
    return ExtractedBook(meta=meta, chapters=chapters, source_path=path)
