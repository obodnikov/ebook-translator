"""Inject reader footnotes into translated EPUB chapters.

Uses the series glossary `notes` field as footnote content.
Matches `translation` strings in the translated XHTML text and wraps
the first occurrence (per scope) in an EPUB footnote-ref / aside pair.

No LLM calls — purely deterministic string matching at assembly time.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from lxml import etree

from .models import Config, ReaderNotesConfig, SeriesGlossary, SeriesGlossaryEntry

# EPUB3 namespace for epub:type attribute
EPUB_NS = "http://www.idpf.org/2007/ops"
XHTML_NS = "http://www.w3.org/1999/xhtml"

NSMAP = {"epub": EPUB_NS, "xhtml": XHTML_NS}


class GlossaryLoadError(Exception):
    """Raised when a glossary file cannot be loaded or validated."""

    pass


@dataclass
class NoteCandidate:
    """A glossary entry eligible for footnoting."""

    entry: SeriesGlossaryEntry
    translation: str  # lowercased for matching
    note_text: str


@dataclass
class InjectionStats:
    """Statistics from a reader-notes injection pass."""

    total_candidates: int = 0
    notes_injected: int = 0
    chapters_modified: int = 0
    entries_matched: list[str] = field(default_factory=list)


def _build_candidates(
    glossary: SeriesGlossary,
    config: ReaderNotesConfig,
) -> list[NoteCandidate]:
    """Filter glossary entries that qualify for reader notes."""
    candidates = []
    for entry in glossary.entries:
        if entry.type not in config.types:
            continue
        if not entry.notes:
            continue
        if not entry.translation.strip():
            continue
        candidates.append(
            NoteCandidate(
                entry=entry,
                translation=entry.translation.lower(),
                note_text=entry.notes,
            )
        )
    # Sort by translation length descending so longer matches take priority
    # (e.g. "Detective Chief Inspector" before "Detective")
    candidates.sort(key=lambda c: len(c.translation), reverse=True)
    return candidates


def _get_text_content(element: etree._Element) -> str:
    """Get all text content from an element and its children."""
    return "".join(element.itertext())


def _find_and_wrap_first_match(
    element: etree._Element,
    search_lower: str,
    note_id: str,
    xhtml_ns: str,
) -> bool:
    """Find the first occurrence of `search_lower` in element's text tree
    and wrap it in a footnote-ref <sup><a>. Returns True if wrapped.

    Walks the text nodes (element.text and child.tail) looking for a
    case-insensitive match. When found, splits the text node and inserts
    the <sup><a epub:type="noteref" href="#note_id">[N]</a></sup> after
    the matched word.
    """
    # We need to find the match in the serialized text and figure out
    # which text node it lives in.
    # Strategy: walk text nodes in document order, accumulate offset,
    # find which node contains the match start.

    text_nodes: list[tuple[etree._Element | None, str, str]] = []
    # Each entry: (element_owning_text, "text"|"tail", the_text_value)

    def _is_forbidden_ancestor(el: etree._Element) -> bool:
        """Check if element is inside a link, footnote, or existing note."""
        tag = el.tag if isinstance(el.tag, str) else ""
        local = tag.split("}")[-1] if "}" in tag else tag
        # Skip <a> elements (would create nested links)
        if local == "a":
            return True
        # Skip <aside epub:type="footnote"> regions
        if local == "aside":
            epub_type = el.get(f"{{{EPUB_NS}}}type", "")
            if "footnote" in epub_type:
                return True
        # Skip existing reader-note <sup> elements
        return local == "sup" and el.get("class", "") == "reader-note"

    def _collect_text_nodes(el: etree._Element, inside_forbidden: bool = False) -> None:
        forbidden = inside_forbidden or _is_forbidden_ancestor(el)
        if el.text and not forbidden:
            text_nodes.append((el, "text", el.text))
        elif el.text and forbidden:
            # Still count the text length for offset tracking but mark as skip
            text_nodes.append((None, "skip", el.text))
        for child in el:
            _collect_text_nodes(child, forbidden)
            if child.tail:
                # Tail text belongs to the parent context, not the child.
                # But if the child itself is forbidden, its tail is still
                # in the parent's flow — only skip if parent is forbidden.
                if not inside_forbidden:
                    text_nodes.append((child, "tail", child.tail))
                else:
                    text_nodes.append((None, "skip", child.tail))

    _collect_text_nodes(element)

    # Build full text and find match (word-boundary aware)
    full_text = "".join(t for _, _, t in text_nodes)
    full_lower = full_text.lower()
    # Use left word boundary to avoid matching inside larger words
    # (e.g. "форма" should not match inside "информация").
    # For short terms (≤4 chars), use both boundaries to prevent
    # false positives (e.g. "art" matching "article").
    # For longer terms, right boundary is relaxed to allow inflected
    # forms in Russian (e.g. "вестигиум" matches "вестигиуме").
    escaped = re.escape(search_lower)
    if len(search_lower) <= 4:
        pattern = re.compile(r"(?<!\w)" + escaped + r"(?!\w)", re.IGNORECASE)
    else:
        pattern = re.compile(r"(?<!\w)" + escaped, re.IGNORECASE)
    m = pattern.search(full_lower)
    if m is None:
        return False
    match_pos = m.start()

    # Find which text node contains match_pos
    offset = 0
    for node_el, attr, text_val in text_nodes:
        node_start = offset
        node_end = offset + len(text_val)
        if node_start <= match_pos < node_end:
            # Skip matches that land in forbidden regions (inside <a>, footnotes)
            if attr == "skip" or node_el is None:
                return False

            # Match starts in this text node
            local_pos = match_pos - node_start
            match_len = len(search_lower)

            # Only handle matches that fit entirely within one text node
            # (simplification — covers vast majority of cases)
            if local_pos + match_len > len(text_val):
                # Match spans multiple nodes — skip for now
                return False

            # Split: before_match + matched_word + after_match
            before = text_val[: local_pos + match_len]
            after = text_val[local_pos + match_len :]

            # Create the superscript noteref element
            sup = etree.Element(f"{{{xhtml_ns}}}sup")
            sup.set("class", "reader-note")
            a = etree.SubElement(sup, f"{{{xhtml_ns}}}a")
            a.set(f"{{{EPUB_NS}}}type", "noteref")
            a.set("href", f"#{note_id}")
            # Extract the note number from note_id (e.g. "reader-note-3" -> "3")
            note_num = note_id.rsplit("-", 1)[-1]
            a.text = f"[{note_num}]"
            # Trailing text goes on sup.tail (after </sup>), NOT a.tail
            # (which would render inside <sup> after </a>).
            sup.tail = after

            if attr == "text":
                node_el.text = before
                # Insert sup as first child
                node_el.insert(0, sup)
            else:
                # attr == "tail" — node_el is a child element whose tail
                # contains the match
                node_el.tail = before
                parent = node_el.getparent()
                if parent is None:
                    return False
                # Insert sup right after node_el
                idx = list(parent).index(node_el)
                parent.insert(idx + 1, sup)

            return True
        offset = node_end

    return False


def inject_reader_notes(
    chapters: list,  # list of ChapterDoc
    glossary: SeriesGlossary,
    config: ReaderNotesConfig,
) -> InjectionStats:
    """Inject EPUB footnotes into chapter trees based on glossary.

    Modifies the lxml trees in-place. Call this AFTER rehydration and
    BEFORE write_translated_epub.

    Parameters
    ----------
    chapters : list of ChapterDoc
        The chapter documents (with .tree and .paragraphs).
    glossary : SeriesGlossary
        The series glossary to source notes from.
    config : ReaderNotesConfig
        Controls which types to annotate and scope.

    Returns
    -------
    InjectionStats
        Summary of what was injected.
    """
    candidates = _build_candidates(glossary, config)
    stats = InjectionStats(total_candidates=len(candidates))

    if not candidates:
        return stats

    # Track which entries have been noted (for first-in-book scope)
    noted_in_book: set[tuple] = set()
    note_counter = 0

    for chapter in chapters:
        if not chapter.paragraphs:
            continue

        # Detect the XHTML namespace from the chapter's root element
        root = chapter.tree.getroot()
        root_tag = root.tag if isinstance(root.tag, str) else ""
        xhtml_ns = root_tag.split("}", 1)[0].lstrip("{") if "}" in root_tag else XHTML_NS

        # Track which entries have been noted in this chapter
        noted_in_chapter: set[tuple] = set()
        chapter_modified = False

        # Collect footnote asides to append at end of body
        footnotes_for_chapter: list[etree._Element] = []

        for para in chapter.paragraphs:
            para_text_lower = _get_text_content(para).lower()

            for candidate in candidates:
                # Check if already noted per scope.
                # Use (original, translation, type) as unique key to handle
                # multiple entries with the same original but different senses.
                entry_key = (
                    candidate.entry.original,
                    candidate.entry.translation,
                    candidate.entry.type,
                )
                if config.scope == "first-in-book" and entry_key in noted_in_book:
                    continue
                if config.scope == "first-in-chapter" and entry_key in noted_in_chapter:
                    continue

                # Quick check: is the translation even in this paragraph?
                if candidate.translation not in para_text_lower:
                    continue

                # Try to wrap the first occurrence
                note_counter += 1
                note_id = f"reader-note-{note_counter}"

                success = _find_and_wrap_first_match(para, candidate.translation, note_id, xhtml_ns)

                if success:
                    # Create the footnote aside element
                    aside = etree.Element(f"{{{xhtml_ns}}}aside")
                    aside.set(f"{{{EPUB_NS}}}type", "footnote")
                    aside.set("id", note_id)
                    p = etree.SubElement(aside, f"{{{xhtml_ns}}}p")
                    p.text = candidate.note_text
                    footnotes_for_chapter.append(aside)

                    noted_in_chapter.add(entry_key)
                    noted_in_book.add(entry_key)
                    chapter_modified = True
                    stats.notes_injected += 1
                    stats.entries_matched.append(candidate.entry.original)
                else:
                    # Didn't manage to wrap — decrement counter
                    note_counter -= 1

        # Append all footnotes to the end of <body>
        if footnotes_for_chapter:
            body = root.find(f".//{{{xhtml_ns}}}body")
            if body is None:
                # Try without namespace
                body = root.find(".//body")
            if body is not None:
                for aside in footnotes_for_chapter:
                    body.append(aside)

        if chapter_modified:
            stats.chapters_modified += 1

    return stats


# ---------------------------------------------------------------------------
# CLI helper functions (orchestration extracted from assemble_cmd)
# ---------------------------------------------------------------------------


def load_notes_glossary(
    series_slug: str | None,
    glossary_path: Path | None,
    work_dir: Path,
    book_title: str,
    book_author: str,
    source_lang: str,
    target_lang: str,
) -> SeriesGlossary | None:
    """Load glossary for reader notes from --series or --glossary.

    Returns a SeriesGlossary if a valid source is found, or None if
    no glossary source is available (series not found, no path given).

    Raises GlossaryLoadError if --glossary file is malformed.
    """
    if series_slug:
        from .series import SeriesWorkDir
        from .series import load_series_glossary as _load

        swd = SeriesWorkDir.for_series(work_dir, series_slug)
        if swd.exists():
            return _load(swd.glossary_path)
        return None
    elif glossary_path:
        import json
        from pathlib import Path

        from pydantic import ValidationError

        from .models import Glossary, SeriesGlossaryEntry

        path = Path(glossary_path)
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            raise GlossaryLoadError(f"Cannot read glossary file {path}: {e}") from e

        try:
            book_glossary = Glossary.model_validate(raw)
        except ValidationError as e:
            raise GlossaryLoadError(f"Invalid glossary schema in {path}: {e}") from e

        return SeriesGlossary(
            series_slug=f"ad-hoc:{book_title}",
            title=book_title,
            author=book_author,
            source_lang=source_lang,
            target_lang=target_lang,
            entries=[
                SeriesGlossaryEntry(
                    original=e.original,
                    translation=e.translation,
                    type=e.type,
                    gender=e.gender,
                    plural=e.plural,
                    notes=e.notes,
                    origin_book=book_glossary.book,
                )
                for e in book_glossary.entries
            ],
        )
    return None


def inject_notes_for_cli(
    *,
    console,
    chapters: list,
    cfg: Config,
    notes_flag: bool | None,
    note_types: str | None,
    series: str | None,
    glossary_path: Path | None,
    work_dir: Path,
    book_title: str,
    book_author: str,
    preloaded_glossary: SeriesGlossary | None = None,
    explicit: bool = False,
) -> InjectionStats | None:
    """Resolve config + glossary and inject reader notes into chapter trees.

    Shared orchestration used by both ``translate`` and ``assemble`` commands.
    Returns InjectionStats if notes were attempted, None if disabled or skipped.

    Parameters
    ----------
    notes_flag:
        Explicit CLI override (True/False) or None to use cfg.reader_notes.enabled.
    preloaded_glossary:
        Pass the already-loaded SeriesGlossary from ``translate`` to avoid
        a redundant load. ``assemble`` passes None and loads via
        load_notes_glossary().
    explicit:
        True when the user passed reader-notes-specific flags (--notes or
        --note-types). Controls whether missing-glossary situations emit
        warnings (explicit=True) or are silently skipped (explicit=False).
        Note: --series/--glossary are for translation consistency and do NOT
        count as explicit notes intent — they are passed as preloaded_glossary
        when available, but their absence does not warn.
    """
    notes_enabled = notes_flag if notes_flag is not None else cfg.reader_notes.enabled
    if not notes_enabled:
        return None

    # Resolve glossary -------------------------------------------------------
    if preloaded_glossary is not None:
        glossary = preloaded_glossary
    else:
        glossary = load_notes_glossary(
            series_slug=series,
            glossary_path=glossary_path,
            work_dir=work_dir,
            book_title=book_title,
            book_author=book_author,
            source_lang=cfg.source_lang,
            target_lang=cfg.target_lang,
        )  # raises GlossaryLoadError on malformed file — caller handles

    if glossary is None and series:
        if explicit:
            console.print(
                f"[yellow]Warning:[/yellow] Series {series!r} not found. Skipping reader notes."
            )
        return None
    if glossary is None and not series and not glossary_path:
        if explicit:
            console.print(
                "[yellow]Warning:[/yellow] Reader notes enabled but no glossary available "
                "(pass --series or --glossary). Skipping."
            )
        return None
    if glossary is None:
        return None

    if glossary_path and preloaded_glossary is None:
        console.print(
            f"[dim]Glossary for notes: {len(glossary.entries)} entries "
            f"from {glossary_path.name}[/dim]"
        )

    # Inject -----------------------------------------------------------------
    notes_config = resolve_notes_config(cfg.reader_notes, note_types_override=note_types)
    note_stats = inject_reader_notes(
        chapters=chapters,
        glossary=glossary,
        config=notes_config,
    )

    if note_stats.notes_injected > 0:
        console.print(
            f"[bold]Reader notes:[/bold] {note_stats.notes_injected} "
            f"footnotes injected across "
            f"{note_stats.chapters_modified} chapters "
            f"(from {note_stats.total_candidates} candidates)"
        )
    else:
        console.print(
            f"[dim]Reader notes: 0 matches found "
            f"({note_stats.total_candidates} candidates checked).[/dim]"
        )

    return note_stats


def resolve_notes_config(
    base_config: ReaderNotesConfig,
    note_types_override: str | None = None,
) -> ReaderNotesConfig:
    """Build effective reader notes config with CLI overrides applied."""
    config = base_config.model_copy()
    config.enabled = True
    if note_types_override:
        config.types = [t.strip() for t in note_types_override.split(",")]
    return config
