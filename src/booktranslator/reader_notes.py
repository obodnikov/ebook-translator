"""Inject reader footnotes into translated EPUB chapters.

Uses the series glossary `notes` field as footnote content.
Matches `translation` strings in the translated XHTML text and wraps
the first occurrence (per scope) in an EPUB footnote-ref / aside pair.

No LLM calls — purely deterministic string matching at assembly time.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from lxml import etree

from .models import Config, ReaderNotesConfig, SeriesGlossary, SeriesGlossaryEntry

# EPUB3 namespace for epub:type attribute
EPUB_NS = "http://www.idpf.org/2007/ops"
XHTML_NS = "http://www.w3.org/1999/xhtml"

NSMAP = {"epub": EPUB_NS, "xhtml": XHTML_NS}

# ---------------------------------------------------------------------------
# Russian word forms
#
# A glossary term is stored in its dictionary form while the text uses it
# declined, so the match has to cover the declined forms. Accepting the
# dictionary form plus an arbitrary short tail does not work: it matches a
# sequence of letters rather than a word, and swallows unrelated words that
# merely begin the same way — "скрип" (company scrip) matched "скрипели",
# a form of the verb "скрипеть".
#
# Instead the term's own ending is stripped to get its stem, and only the
# endings that belong to its declension type are accepted. The whole form is
# part of the match, so the marker always lands after a complete word.
# ---------------------------------------------------------------------------

_VOWELS = "аеёиоуыэюя"
# after these consonants -ы is spelled -и (Russian spelling rule)
_SIBILANTS = "гкхжшчщ"

# Terms this short are matched exactly: a generated ending on a three-letter
# stem collides with ordinary words far too often.
_MIN_INFLECTED_LEN = 5

# The bare-stem form (genitive plural "ведьм") is deliberately absent from the
# vowel-stem tables: dropping a letter can expose a shorter, unrelated word —
# "Бойо" would otherwise match "бой". For consonant stems the dictionary form
# itself is the zero-ending form, so "" belongs there.
_MASC_HARD = ("", "а", "у", "ом", "е", "ы", "ов", "ам", "ами", "ах")
_MASC_HARD_SIB = ("", "а", "у", "ом", "е", "и", "ов", "ам", "ами", "ах")
_MASC_SOFT_J = ("й", "я", "ю", "ем", "е", "и", "ев", "ям", "ями", "ях")
_MASC_SOFT_SIGN = ("ь", "я", "ю", "ем", "е", "и", "ей", "ям", "ями", "ях")
_FEM_SIGN = ("ь", "и", "ью", "ей", "ям", "ями", "ях")
_FEM_A = ("а", "ы", "е", "у", "ой", "ою", "ам", "ами", "ах")
_FEM_A_SIB = ("а", "и", "е", "у", "ой", "ою", "ам", "ами", "ах")
_FEM_YA = ("я", "и", "е", "ю", "ей", "ею", "ям", "ями", "ях")
_FEM_IYA = ("ия", "ии", "ию", "ией", "иею", "ий", "иям", "иями", "иях")
_NEUT_O = ("о", "а", "у", "ом", "е", "ам", "ами", "ах")
_NEUT_E = ("е", "я", "ю", "ем", "и", "ям", "ями", "ях")
_ADJ_HARD = ("ый", "ой", "ого", "ому", "ым", "ом", "ая", "ую", "ою", "ое", "ые", "ых", "ыми")
_ADJ_SOFT = ("ий", "его", "ему", "им", "ем", "яя", "ей", "юю", "ею", "ее", "ие", "их", "ими")
_NOUN_IJ = ("ий", "ия", "ию", "ием", "ии", "иев", "иям", "иями", "иях")

# Looking left from a match, these are stepped over on the way to the
# punctuation that would mark the end of the previous sentence.
_SKIP_LEFT = " \t\r\n «»\"“”„‘’'()[]{}<>-–—*"
_SENTENCE_ENDERS = ".!?…:;"


class GlossaryLoadError(Exception):
    """Raised when a glossary file cannot be loaded or validated."""

    pass


@dataclass
class NoteCandidate:
    """A glossary entry eligible for footnoting."""

    entry: SeriesGlossaryEntry
    translation: str  # as written in the glossary — the case carries meaning
    note_text: str
    probe: str  # lowercased stem, for the cheap "is it in this paragraph" test


def _stem_and_endings(word: str, gender: str | None) -> tuple[str, tuple[str, ...]]:
    """Split a Russian word into its stem and the endings of its declension.

    The declension type is read off the word's own ending; for words in -ь,
    which can be either masculine or feminine, the glossary `gender` field
    settles it, and both sets are accepted when it is missing.
    """
    t = word.lower()
    if len(t) < _MIN_INFLECTED_LEN:
        return t, ("",)

    last, prev = t[-1], t[-2]

    if t.endswith(("ый", "ой")):
        return t[:-2], _ADJ_HARD
    if t.endswith("ий"):
        # "-ий" ends both adjectives ("Управляющий") and nouns ("Апиарий").
        # Accept the union: a form from the wrong set is not a word at all,
        # so it never occurs in the text.
        return t[:-2], _ADJ_SOFT + tuple(e for e in _NOUN_IJ if e not in _ADJ_SOFT)
    if t.endswith("ия"):
        return t[:-2], _FEM_IYA
    if last == "я":
        return t[:-1], _FEM_YA
    if last == "а":
        return t[:-1], _FEM_A_SIB if prev in _SIBILANTS else _FEM_A
    if last == "й":
        return t[:-1], _MASC_SOFT_J
    if last == "ь":
        if gender == "f":
            return t[:-1], _FEM_SIGN
        if gender == "m":
            return t[:-1], _MASC_SOFT_SIGN
        return t[:-1], _MASC_SOFT_SIGN + tuple(e for e in _FEM_SIGN if e not in _MASC_SOFT_SIGN)
    if last == "о":
        return t[:-1], _NEUT_O
    if last == "е":
        return t[:-1], _NEUT_E
    if last in _VOWELS:
        # -у, -и, -ю and the like: borrowed names that do not decline
        return t, ("",)
    return t, _MASC_HARD_SIB if last in _SIBILANTS else _MASC_HARD


@lru_cache(maxsize=2048)
def _compile_term_pattern(term: str, gender: str | None) -> re.Pattern[str]:
    """Build the regex that finds `term` in any of its grammatical forms.

    In a multi-word term only the last word is inflected; the rest has to
    appear verbatim, since agreement across the phrase is beyond what an
    ending table can do.
    """
    words = term.split()
    stem, endings = _stem_and_endings(words[-1], gender)
    body = re.escape(stem)
    if endings != ("",):
        # longest ending first, so the match covers the whole word
        alt = "|".join(re.escape(e) for e in sorted(set(endings), key=len, reverse=True))
        body += f"(?:{alt})"
    head = "".join(re.escape(w) + r"\s+" for w in words[:-1])
    return re.compile(rf"(?<!\w){head}{body}(?!\w)", re.IGNORECASE)


@lru_cache(maxsize=2048)
def _term_probe(term: str, gender: str | None) -> str:
    """A lowercased substring that any match of the term must contain."""
    words = term.split()
    if len(words) > 1:
        return words[0].lower()
    return _stem_and_endings(words[0], gender)[0]


def _requires_capital(term: str) -> bool:
    """Whether the term may only match text that is capitalised too.

    A term the glossary writes with a capital is a proper noun of the book's
    world — "Облако" the computing system, not "облако" in the sky. Russian
    does not capitalise common nouns mid-sentence, so that capital is the one
    signal separating the two senses, and it must not be discarded.
    """
    return term[:1].isupper()


def _is_sentence_start(text: str, pos: int) -> bool:
    """Whether the word at `pos` opens a sentence.

    There every word is capitalised, so a capital says nothing about whether
    this is the term or an ordinary word, and such occurrences are passed over
    in favour of an unambiguous one later in the text.
    """
    i = pos - 1
    while i >= 0 and text[i] in _SKIP_LEFT:
        i -= 1
    return i < 0 or text[i] in _SENTENCE_ENDERS


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
                translation=entry.translation,
                note_text=entry.notes,
                probe=_term_probe(entry.translation, entry.gender),
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
    term: str,
    note_id: str,
    xhtml_ns: str,
    gender: str | None = None,
) -> bool:
    """Find the first usable occurrence of `term` in element's text tree
    and wrap it in a footnote-ref <sup><a>. Returns True if wrapped.

    Walks the text nodes (element.text and child.tail) looking for the term
    in any of its grammatical forms. When found, splits the text node and
    inserts the <sup><a epub:type="noteref" href="#note_id">[N]</a></sup>
    after the matched word.

    `term` is taken as the glossary writes it: a capitalised term only
    matches capitalised text, and only away from the start of a sentence.
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

    # Build full text and find the term in any of its forms. The text keeps
    # its original case — for a capitalised term the case is the meaning.
    full_text = "".join(t for _, _, t in text_nodes)
    pattern = _compile_term_pattern(term, gender)
    needs_capital = _requires_capital(term)

    for m in pattern.finditer(full_text):
        match_pos = m.start()
        match_len = m.end() - m.start()

        if needs_capital and (
            not m.group(0)[:1].isupper() or _is_sentence_start(full_text, match_pos)
        ):
            # Either an ordinary word ("тяжёлое облако", not the Cloud), or a
            # position where the capital proves nothing. Try the next one.
            continue

        # Find which text node contains match_pos
        offset = 0
        for node_el, attr, text_val in text_nodes:
            node_start = offset
            node_end = offset + len(text_val)
            offset = node_end
            if not (node_start <= match_pos < node_end):
                continue

            # Skip matches that land in forbidden regions (inside <a>, footnotes)
            if attr == "skip" or node_el is None:
                break

            # Match starts in this text node
            local_pos = match_pos - node_start

            # Only handle matches that fit entirely within one text node
            # (simplification — covers vast majority of cases)
            if local_pos + match_len > len(text_val):
                # Match spans multiple nodes — try the next occurrence
                break

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
                    break
                # Insert sup right after node_el
                idx = list(parent).index(node_el)
                parent.insert(idx + 1, sup)

            return True

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

                # Quick check: is the term's stem even in this paragraph?
                if candidate.probe not in para_text_lower:
                    continue

                # Try to wrap the first usable occurrence
                note_counter += 1
                note_id = f"reader-note-{note_counter}"

                success = _find_and_wrap_first_match(
                    para,
                    candidate.translation,
                    note_id,
                    xhtml_ns,
                    gender=candidate.entry.gender,
                )

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
