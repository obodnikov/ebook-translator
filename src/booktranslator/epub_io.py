"""EPUB I/O with structural preservation.

- `read_book_flat` — flat text for glossary extraction (unchanged API).
- `read_book_structured` — keeps lxml trees per chapter, for translation.
- `write_translated_epub` — writes a new EPUB preserving the archive
  byte-for-byte except for the chapter XHTML trees we edited.

Iteration 3 scope: replace text inside <p>/<h1..6>/<blockquote> tags
while preserving inline formatting (<em>, <strong>, <i>, <b>, <a>, etc).
Other document content (e.g. <table>, <ul>, raw structural XHTML) is
left untouched in this first cut.
"""

from __future__ import annotations

import warnings
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from ebooklib import ITEM_DOCUMENT, epub
from lxml import etree
from lxml import html as lxhtml

from .models import BookMeta

# ---------------------------------------------------------------------------
# Flat read (used by glossary extraction)
# ---------------------------------------------------------------------------


@dataclass
class ExtractedBook:
    """Plain-text view of an EPUB, used for glossary extraction."""

    meta: BookMeta
    chapters: list[str]
    source_path: Path

    def full_text(self) -> str:
        return "\n\n".join(self.chapters)


def _first_metadata(book: epub.EpubBook, field: str, default: str = "") -> str:
    items = book.get_metadata("DC", field)
    if items and items[0] and items[0][0]:
        return items[0][0].strip()
    return default


def _chapter_text(document: epub.EpubHtml) -> str:
    content = document.get_content()
    if not content:
        return ""
    try:
        tree = lxhtml.fromstring(content)
    except Exception:
        return ""
    return tree.text_content().strip()


def read_book(path: Path) -> ExtractedBook:
    """Read an EPUB and return metadata + flat chapter texts."""
    with warnings.catch_warnings():
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


# ---------------------------------------------------------------------------
# Structured read (used by translation)
# ---------------------------------------------------------------------------


# Namespace map for XHTML docs. Most EPUB chapters declare xhtml as the
# default namespace; lxml auto-detects it when parsing html. Our XPath
# queries below strip namespaces via local-name() to tolerate both.


@dataclass
class ChapterDoc:
    """One XHTML document from the EPUB spine, with its parsed tree."""

    spine_index: int  # 0-based position in the spine
    archive_name: str  # path inside the EPUB zip
    original_bytes: bytes  # raw file bytes (for unchanged copy)
    tree: etree._ElementTree  # parsed lxml tree
    # Paragraphs are elements we can translate. We keep references to the
    # actual lxml elements so the translator can update them in place.
    paragraphs: list[etree._Element] = field(default_factory=list)


@dataclass
class StructuredBook:
    """Structural view of an EPUB ready for chunked translation."""

    meta: BookMeta
    source_path: Path
    chapters: list[ChapterDoc]


# Tags we consider translatable units. Headings get translated so the
# table of contents reads in the target language; block quotes carry
# prose; <p> is the default paragraph. <li> is excluded here — list
# items tend to be structural (ingredients, enumerations); we may enable
# them in a later iteration.
TRANSLATABLE_TAGS = {"p", "h1", "h2", "h3", "h4", "h5", "h6", "blockquote"}


def _is_translatable(el: etree._Element) -> bool:
    tag = etree.QName(el.tag).localname.lower() if isinstance(el.tag, str) else ""
    if tag not in TRANSLATABLE_TAGS:
        return False
    # Skip empty paragraphs.
    has_text = (el.text or "").strip() or any(
        (child.text or child.tail or "").strip() for child in el
    )
    return bool(has_text)


def _find_paragraphs(tree: etree._ElementTree) -> list[etree._Element]:
    """Return the list of translatable elements in document order.

    Uses local-name() so it works with or without XHTML namespace.
    """
    root = tree.getroot()
    out: list[etree._Element] = []
    for el in root.iter():
        if not isinstance(el.tag, str):
            continue
        if _is_translatable(el):
            out.append(el)
    return out


def _parse_xhtml(content: bytes) -> etree._ElementTree:
    """Parse an XHTML file into an lxml tree.

    We use lxml's XML parser (not HTML) to strictly preserve the input
    structure — including self-closing tags, entities, and namespaces.
    EPUB XHTML files are by spec well-formed XML.
    """
    parser = etree.XMLParser(
        resolve_entities=False,
        remove_blank_text=False,
        remove_comments=False,
        recover=True,  # tolerate minor issues, OceanofPDF builds vary
    )
    return etree.ElementTree(etree.fromstring(content, parser=parser))


def read_book_structured(path: Path) -> StructuredBook:
    """Read an EPUB preserving per-chapter lxml trees and raw bytes."""
    # Collect metadata via ebooklib (convenient for DC fields).
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        book = epub.read_epub(str(path))

    # But use zipfile directly for byte-exact reads, since ebooklib may
    # re-serialize XHTML and introduce subtle diffs.
    chapters: list[ChapterDoc] = []
    total_words = 0
    with zipfile.ZipFile(path, "r") as zf:
        archive_names = set(zf.namelist())

        for idx, (spine_id, _linear) in enumerate(book.spine):
            item = book.get_item_with_id(spine_id)
            if item is None or item.get_type() != ITEM_DOCUMENT:
                continue
            archive_name = item.get_name()
            # ebooklib resolves hrefs relative to OPF; zip may store under
            # OEBPS/... or root. Try a few candidates.
            candidate = _resolve_archive_path(archive_name, archive_names, book)
            if candidate is None:
                continue

            raw = zf.read(candidate)
            try:
                tree = _parse_xhtml(raw)
            except Exception:
                continue
            paragraphs = _find_paragraphs(tree)
            chapters.append(
                ChapterDoc(
                    spine_index=idx,
                    archive_name=candidate,
                    original_bytes=raw,
                    tree=tree,
                    paragraphs=paragraphs,
                )
            )
            for p in paragraphs:
                total_words += len(" ".join(p.itertext()).split())

    meta = BookMeta(
        title=_first_metadata(book, "title", default=path.stem),
        author=_first_metadata(book, "creator", default="Unknown"),
        language=_first_metadata(book, "language", default="en"),
        word_count=total_words,
        chapters=len(chapters),
    )
    return StructuredBook(meta=meta, source_path=path, chapters=chapters)


def _resolve_archive_path(href: str, archive_names: set[str], book: epub.EpubBook) -> str | None:
    """Map an ebooklib item href to the actual path inside the zip."""
    if href in archive_names:
        return href
    # ebooklib sometimes returns paths with the OPF dir stripped.
    for candidate in archive_names:
        if candidate.endswith("/" + href) or candidate.endswith(href):
            return candidate
    return None


# ---------------------------------------------------------------------------
# Writing a translated EPUB
# ---------------------------------------------------------------------------


def write_translated_epub(
    source_path: Path,
    dest_path: Path,
    modified_chapters: list[ChapterDoc],
    new_language: str | None = None,
    title_translation: str | None = None,
) -> None:
    """Write a new EPUB file, preserving everything from `source_path`
    except the XHTML trees in `modified_chapters`, which replace their
    originals, and optionally the `dc:language` / `dc:title` metadata.
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    # Build a lookup by archive name.
    replacements: dict[str, bytes] = {
        ch.archive_name: etree.tostring(
            ch.tree,
            xml_declaration=True,
            encoding="utf-8",
            pretty_print=False,
        )
        for ch in modified_chapters
    }

    # Optional metadata updates: rewrite the OPF file.
    opf_name: str | None = None
    opf_new_bytes: bytes | None = None
    if new_language or title_translation:
        opf_name, opf_new_bytes = _maybe_update_opf(source_path, new_language, title_translation)
        if opf_new_bytes is not None:
            replacements[opf_name] = opf_new_bytes

    with zipfile.ZipFile(source_path, "r") as src, zipfile.ZipFile(dest_path, "w") as dst:
        # mimetype first, uncompressed (EPUB spec).
        dst.writestr(
            zipfile.ZipInfo("mimetype"),
            "application/epub+zip",
            compress_type=zipfile.ZIP_STORED,
        )
        for info in src.infolist():
            if info.filename == "mimetype":
                continue
            if info.filename in replacements:
                dst.writestr(
                    info.filename,
                    replacements[info.filename],
                    compress_type=zipfile.ZIP_DEFLATED,
                )
            else:
                with src.open(info) as fh:
                    dst.writestr(
                        info,
                        fh.read(),
                    )


def _maybe_update_opf(
    source_path: Path,
    new_language: str | None,
    title_translation: str | None,
) -> tuple[str | None, bytes | None]:
    """Find the OPF file and rewrite dc:language / dc:title if asked."""
    if not (new_language or title_translation):
        return None, None
    with zipfile.ZipFile(source_path, "r") as zf:
        # container.xml points to the OPF
        try:
            container_xml = zf.read("META-INF/container.xml")
        except KeyError:
            return None, None
        ct = etree.fromstring(container_xml)
        ns_c = "urn:oasis:names:tc:opendocument:xmlns:container"
        rootfile = ct.find(f".//{{{ns_c}}}rootfile")
        if rootfile is None:
            return None, None
        opf_name = rootfile.get("full-path")

        opf_bytes = zf.read(opf_name)
        opf = etree.fromstring(opf_bytes)
        ns_opf = "http://www.idpf.org/2007/opf"
        ns_dc = "http://purl.org/dc/elements/1.1/"
        changed = False
        if new_language:
            lang_el = opf.find(f".//{{{ns_opf}}}metadata/{{{ns_dc}}}language")
            if lang_el is not None and (lang_el.text or "").strip() != new_language:
                lang_el.text = new_language
                changed = True
        if title_translation:
            title_el = opf.find(f".//{{{ns_opf}}}metadata/{{{ns_dc}}}title")
            if title_el is not None:
                title_el.text = title_translation
                changed = True
        if not changed:
            return opf_name, None
        new_bytes = etree.tostring(opf, xml_declaration=True, encoding="utf-8")
        return opf_name, new_bytes
