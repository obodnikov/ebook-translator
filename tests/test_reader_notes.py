"""Tests for reader_notes footnote injection."""

from __future__ import annotations

from dataclasses import dataclass, field
from io import BytesIO

from lxml import etree

from booktranslator.models import ReaderNotesConfig, SeriesGlossary, SeriesGlossaryEntry
from booktranslator.reader_notes import (
    NOTE_STYLE_ID,
    _build_candidates,
    _find_and_wrap_first_match,
    inject_reader_notes,
    is_epub3,
)

XHTML_NS = "http://www.w3.org/1999/xhtml"
EPUB_NS = "http://www.idpf.org/2007/ops"


def _make_xhtml_tree(body_content: str) -> etree._ElementTree:
    """Create a minimal XHTML tree with given body content."""
    xhtml = f"""<?xml version="1.0" encoding="utf-8"?>
<html xmlns="{XHTML_NS}" xmlns:epub="{EPUB_NS}">
<head><title>Test</title></head>
<body>
{body_content}
</body>
</html>"""
    return etree.parse(BytesIO(xhtml.encode("utf-8")))


@dataclass
class FakeChapterDoc:
    """Minimal ChapterDoc-like object for testing."""

    spine_index: int = 0
    archive_name: str = "ch01.xhtml"
    original_bytes: bytes = b""
    tree: etree._ElementTree = None
    paragraphs: list = field(default_factory=list)


def _make_chapter(body_content: str) -> FakeChapterDoc:
    """Create a fake chapter with paragraphs extracted from body."""
    tree = _make_xhtml_tree(body_content)
    root = tree.getroot()
    paragraphs = root.findall(f".//{{{XHTML_NS}}}p")
    return FakeChapterDoc(tree=tree, paragraphs=paragraphs)


def _make_epub2_chapter(body_content: str) -> FakeChapterDoc:
    """A chapter shaped like the EPUB2 books we translate.

    The difference that matters: <html> does not declare xmlns:epub. The
    other fixture does, which is why it never caught lxml inventing a
    prefix of its own for the epub:type attribute.
    """
    xhtml = f"""<?xml version="1.0" encoding="utf-8"?>
<html xmlns="{XHTML_NS}">
<head><title>Test</title></head>
<body class="calibre2">
{body_content}
</body>
</html>"""
    tree = etree.parse(BytesIO(xhtml.encode("utf-8")))
    root = tree.getroot()
    return FakeChapterDoc(tree=tree, paragraphs=root.findall(f".//{{{XHTML_NS}}}p"))


def _serialize(chapter: FakeChapterDoc) -> str:
    return etree.tostring(chapter.tree, encoding="unicode")


def _note_items(chapter: FakeChapterDoc) -> list[etree._Element]:
    root = chapter.tree.getroot()
    return [el for el in root.iter() if el.get("class") == "reader-note-item"]


def _make_glossary(entries: list[dict]) -> SeriesGlossary:
    """Create a SeriesGlossary from a list of dicts."""
    return SeriesGlossary(
        series_slug="test-series",
        title="Test Series",
        author="Test Author",
        source_lang="en",
        target_lang="ru",
        entries=[SeriesGlossaryEntry(**e) for e in entries],
    )


# ---------------------------------------------------------------------------
# _build_candidates
# ---------------------------------------------------------------------------


class TestBuildCandidates:
    def test_filters_by_type(self):
        glossary = _make_glossary(
            [
                {
                    "original": "vestigium",
                    "translation": "вестигиум",
                    "type": "concept",
                    "notes": "Magical trace left by spells.",
                },
                {
                    "original": "Peter Grant",
                    "translation": "Питер Грант",
                    "type": "person",
                    "notes": "Main character.",
                },
                {
                    "original": "DCI",
                    "translation": "DCI",
                    "type": "term",
                    "notes": "Detective Chief Inspector.",
                },
            ]
        )
        config = ReaderNotesConfig(enabled=True, types=["concept", "term"])
        candidates = _build_candidates(glossary, config)
        assert len(candidates) == 2
        originals = {c.entry.original for c in candidates}
        assert originals == {"vestigium", "DCI"}

    def test_skips_entries_without_notes(self):
        glossary = _make_glossary(
            [
                {"original": "forma", "translation": "форма", "type": "concept", "notes": None},
                {
                    "original": "vestigium",
                    "translation": "вестигиум",
                    "type": "concept",
                    "notes": "Magical trace.",
                },
            ]
        )
        config = ReaderNotesConfig(enabled=True, types=["concept"])
        candidates = _build_candidates(glossary, config)
        assert len(candidates) == 1
        assert candidates[0].entry.original == "vestigium"

    def test_skips_empty_translation(self):
        glossary = _make_glossary(
            [
                {
                    "original": "vestigium",
                    "translation": "",
                    "type": "concept",
                    "notes": "Magical trace.",
                },
            ]
        )
        config = ReaderNotesConfig(enabled=True, types=["concept"])
        candidates = _build_candidates(glossary, config)
        assert len(candidates) == 0

    def test_sorted_by_length_descending(self):
        glossary = _make_glossary(
            [
                {
                    "original": "DCI",
                    "translation": "DCI",
                    "type": "term",
                    "notes": "Detective Chief Inspector.",
                },
                {
                    "original": "Detective Chief Inspector",
                    "translation": "старший инспектор уголовного розыска",
                    "type": "term",
                    "notes": "Senior detective rank.",
                },
            ]
        )
        config = ReaderNotesConfig(enabled=True, types=["term"])
        candidates = _build_candidates(glossary, config)
        assert len(candidates) == 2
        # Longer translation first
        assert len(candidates[0].translation) > len(candidates[1].translation)


# ---------------------------------------------------------------------------
# _find_and_wrap_first_match
# ---------------------------------------------------------------------------


class TestFindAndWrap:
    def test_wraps_simple_text(self):
        tree = _make_xhtml_tree(
            '<p xmlns="http://www.w3.org/1999/xhtml">Он почувствовал вестигиум в воздухе.</p>'
        )
        root = tree.getroot()
        para = root.find(f".//{{{XHTML_NS}}}p")

        result = _find_and_wrap_first_match(para, "вестигиум", "reader-note-1", XHTML_NS)
        assert result is True

        # Check that the noteref was inserted
        serialized = etree.tostring(para, encoding="unicode")
        assert "noteref" in serialized
        assert "reader-note-1" in serialized
        assert "[1]" in serialized

    def test_trailing_text_outside_sup(self):
        """Text after the matched term must be outside <sup>, not inside it."""
        tree = _make_xhtml_tree(
            '<p xmlns="http://www.w3.org/1999/xhtml">Он почувствовал вестигиум в воздухе.</p>'
        )
        root = tree.getroot()
        para = root.find(f".//{{{XHTML_NS}}}p")

        _find_and_wrap_first_match(para, "вестигиум", "reader-note-1", XHTML_NS)

        serialized = etree.tostring(para, encoding="unicode")
        # Structure should be: ...вестигиум<sup><a ...>[1]</a></sup> в воздухе.
        # The trailing " в воздухе." must be AFTER </sup>, not inside it.
        assert "</a></sup> в воздухе." in serialized.replace(f"{{{XHTML_NS}}}", "").replace(
            "ns0:", ""
        ).replace(":ns0", "")

        # More robust: check sup element's tail contains trailing text
        sup = para.find(f".//{{{XHTML_NS}}}sup")
        assert sup is not None
        assert sup.tail is not None
        assert "в воздухе." in sup.tail

        # And <a> inside sup should have no tail
        a = sup.find(f".//{{{XHTML_NS}}}a")
        assert a is not None
        assert a.tail is None or a.tail.strip() == ""

    def test_case_insensitive_match(self):
        tree = _make_xhtml_tree('<p xmlns="http://www.w3.org/1999/xhtml">Он видел Вестигиум.</p>')
        root = tree.getroot()
        para = root.find(f".//{{{XHTML_NS}}}p")

        result = _find_and_wrap_first_match(para, "вестигиум", "reader-note-1", XHTML_NS)
        assert result is True

    def test_no_match_returns_false(self):
        tree = _make_xhtml_tree(
            '<p xmlns="http://www.w3.org/1999/xhtml">Обычный текст без терминов.</p>'
        )
        root = tree.getroot()
        para = root.find(f".//{{{XHTML_NS}}}p")

        result = _find_and_wrap_first_match(para, "вестигиум", "reader-note-1", XHTML_NS)
        assert result is False

    def test_no_match_inside_larger_word(self):
        """Should NOT match 'форма' inside 'информация' (word boundary)."""
        tree = _make_xhtml_tree(
            '<p xmlns="http://www.w3.org/1999/xhtml">Он получил информацию.</p>'
        )
        root = tree.getroot()
        para = root.find(f".//{{{XHTML_NS}}}p")

        result = _find_and_wrap_first_match(para, "форма", "reader-note-1", XHTML_NS)
        assert result is False

    def test_matches_standalone_word(self):
        """Should match 'форма' when it's a standalone word."""
        tree = _make_xhtml_tree(
            '<p xmlns="http://www.w3.org/1999/xhtml">Он применил форма заклинания.</p>'
        )
        root = tree.getroot()
        para = root.find(f".//{{{XHTML_NS}}}p")

        result = _find_and_wrap_first_match(para, "форма", "reader-note-1", XHTML_NS)
        assert result is True

    def test_short_term_no_prefix_match(self):
        """Short terms (≤4 chars) should NOT match as prefix of longer word."""
        tree = _make_xhtml_tree(
            '<p xmlns="http://www.w3.org/1999/xhtml">He read the article carefully.</p>'
        )
        root = tree.getroot()
        para = root.find(f".//{{{XHTML_NS}}}p")

        # "art" should NOT match inside "article"
        result = _find_and_wrap_first_match(para, "art", "reader-note-1", XHTML_NS)
        assert result is False

    def test_short_term_matches_standalone(self):
        """Short terms match when they are standalone words."""
        tree = _make_xhtml_tree(
            '<p xmlns="http://www.w3.org/1999/xhtml">The art of magic is complex.</p>'
        )
        root = tree.getroot()
        para = root.find(f".//{{{XHTML_NS}}}p")

        result = _find_and_wrap_first_match(para, "art", "reader-note-1", XHTML_NS)
        assert result is True

    def test_long_term_matches_declined_form(self):
        """Long terms (>4 chars) are found in their declined forms."""
        tree = _make_xhtml_tree(
            '<p xmlns="http://www.w3.org/1999/xhtml">Он почувствовал вестигиуме.</p>'
        )
        root = tree.getroot()
        para = root.find(f".//{{{XHTML_NS}}}p")

        # "вестигиум" (8 chars) should match "вестигиуме" (inflected)
        result = _find_and_wrap_first_match(para, "вестигиум", "reader-note-1", XHTML_NS)
        assert result is True

    def test_marker_goes_after_inflected_ending(self):
        """The marker must follow the whole word, not split off its ending."""
        tree = _make_xhtml_tree(
            '<p xmlns="http://www.w3.org/1999/xhtml">Биоформы носят скафандры.</p>'
        )
        root = tree.getroot()
        para = root.find(f".//{{{XHTML_NS}}}p")

        result = _find_and_wrap_first_match(para, "биоформ", "reader-note-2", XHTML_NS)
        assert result is True

        # Text before <sup> ends with the full word, tail starts with a space
        sup = para.find(f".//{{{XHTML_NS}}}sup")
        assert para.text == "Биоформы"
        assert sup.tail == " носят скафандры."

    def test_marker_after_multi_letter_ending(self):
        """A three-letter case ending is still part of the matched word."""
        tree = _make_xhtml_tree(
            '<p xmlns="http://www.w3.org/1999/xhtml">Он говорил с биоформами вчера.</p>'
        )
        root = tree.getroot()
        para = root.find(f".//{{{XHTML_NS}}}p")

        result = _find_and_wrap_first_match(para, "биоформ", "reader-note-1", XHTML_NS)
        assert result is True

        sup = para.find(f".//{{{XHTML_NS}}}sup")
        assert para.text == "Он говорил с биоформами"
        assert sup.tail == " вчера."

    def test_no_match_when_tail_is_not_an_ending(self):
        """A tail that is not a case ending means a different word entirely."""
        tree = _make_xhtml_tree(
            '<p xmlns="http://www.w3.org/1999/xhtml">Марсианская адаптационная инженерия.</p>'
        )
        root = tree.getroot()
        para = root.find(f".//{{{XHTML_NS}}}p")

        # Company name "АдАпт" must not attach itself to "адаптационная"
        result = _find_and_wrap_first_match(para, "адапт", "reader-note-1", XHTML_NS)
        assert result is False

    def test_no_match_on_unrelated_word_with_same_start(self):
        """The tail has to be a case ending, not any three letters.

        "скрип" is the company currency; "скрипели" is the verb "скрипеть"
        and "скрипку" is a violin. Neither is the term.
        """
        for text in ("Ворота уже скрипели, раскрываясь.", "Он скрипит.", "Он взял скрипку."):
            tree = _make_xhtml_tree(f'<p xmlns="http://www.w3.org/1999/xhtml">{text}</p>')
            para = tree.getroot().find(f".//{{{XHTML_NS}}}p")

            result = _find_and_wrap_first_match(para, "скрип", "reader-note-1", XHTML_NS)
            assert result is False, text

    def test_matches_form_that_replaces_the_ending(self):
        """Declension replaces the ending as often as it adds one."""
        cases = [
            ("кровь", "Нуждались они в Крови.", "Крови"),
            ("апиарий", "Он монах Апиария.", "Апиария"),
            ("ведьма", "Он увидел ведьму.", "ведьму"),
            ("иерархия", "Диктат вживлённой иерархии.", "иерархии"),
            ("управляющий", "Он был бывшим управляющего.", "управляющего"),
        ]
        for term, text, expected in cases:
            tree = _make_xhtml_tree(f'<p xmlns="http://www.w3.org/1999/xhtml">{text}</p>')
            para = tree.getroot().find(f".//{{{XHTML_NS}}}p")

            assert _find_and_wrap_first_match(para, term, "reader-note-1", XHTML_NS) is True, term
            sup = para.find(f".//{{{XHTML_NS}}}sup")
            assert para.text.endswith(expected), (term, para.text)
            assert sup.tail == "."

    def test_bare_stem_is_not_a_form(self):
        """Stripping the ending must not expose a shorter, unrelated word."""
        tree = _make_xhtml_tree('<p xmlns="http://www.w3.org/1999/xhtml">Начался бой за город.</p>')
        para = tree.getroot().find(f".//{{{XHTML_NS}}}p")

        # The name "Бойо" must not attach itself to "бой"
        result = _find_and_wrap_first_match(para, "Бойо", "reader-note-1", XHTML_NS)
        assert result is False

    def test_capitalised_term_skips_lowercase_word(self):
        """ "Облако" the computing system is not "облако" in the sky."""
        tree = _make_xhtml_tree(
            '<p xmlns="http://www.w3.org/1999/xhtml">'
            "Я отступаю — тяжёлое облако остаётся позади."
            "</p>"
        )
        para = tree.getroot().find(f".//{{{XHTML_NS}}}p")

        result = _find_and_wrap_first_match(para, "Облако", "reader-note-1", XHTML_NS)
        assert result is False

    def test_capitalised_term_matches_capitalised_word(self):
        tree = _make_xhtml_tree(
            '<p xmlns="http://www.w3.org/1999/xhtml">Все данные ушли в Облако.</p>'
        )
        para = tree.getroot().find(f".//{{{XHTML_NS}}}p")

        assert _find_and_wrap_first_match(para, "Облако", "reader-note-1", XHTML_NS) is True
        assert para.text == "Все данные ушли в Облако"

    def test_capitalised_term_skips_sentence_start(self):
        """At the start of a sentence a capital proves nothing.

        The first "Кровь" is ordinary blood, the second is the Griffin term.
        """
        tree = _make_xhtml_tree(
            '<p xmlns="http://www.w3.org/1999/xhtml">'
            "Кровь у Крикета похолодела. Чужак там, откуда берётся Кровь."
            "</p>"
        )
        para = tree.getroot().find(f".//{{{XHTML_NS}}}p")

        assert _find_and_wrap_first_match(para, "Кровь", "reader-note-1", XHTML_NS) is True
        assert para.text.endswith("откуда берётся Кровь")

    def test_lowercase_term_matches_at_sentence_start(self):
        """A term the glossary writes lowercase keeps matching either case."""
        tree = _make_xhtml_tree('<p xmlns="http://www.w3.org/1999/xhtml">Биоформы носят броню.</p>')
        para = tree.getroot().find(f".//{{{XHTML_NS}}}p")

        assert _find_and_wrap_first_match(para, "биоформ", "reader-note-1", XHTML_NS) is True

    def test_gender_narrows_soft_sign_endings(self):
        """For a term in -ь the glossary gender picks the right paradigm."""
        tree = _make_xhtml_tree('<p xmlns="http://www.w3.org/1999/xhtml">Он истекал кровью.</p>')
        para = tree.getroot().find(f".//{{{XHTML_NS}}}p")

        # "кровью" is feminine instrumental; the masculine set has no such form
        assert _find_and_wrap_first_match(para, "кровь", "n1", XHTML_NS, gender="f") is True
        para2 = (
            _make_xhtml_tree('<p xmlns="http://www.w3.org/1999/xhtml">Он истекал кровью.</p>')
            .getroot()
            .find(f".//{{{XHTML_NS}}}p")
        )
        assert _find_and_wrap_first_match(para2, "кровь", "n1", XHTML_NS, gender="m") is False

    def test_multi_word_term_inflects_last_word_only(self):
        tree = _make_xhtml_tree(
            '<p xmlns="http://www.w3.org/1999/xhtml">'
            "Это была Заводская Администрация, не голос."
            "</p>"
        )
        para = tree.getroot().find(f".//{{{XHTML_NS}}}p")

        result = _find_and_wrap_first_match(
            para, "Заводская Администрация", "reader-note-1", XHTML_NS
        )
        assert result is True
        assert para.text.endswith("Заводская Администрация")

    def test_falls_back_to_later_occurrence(self):
        """When the first occurrence is unusable, a later one is used."""
        tree = _make_xhtml_tree(
            '<p xmlns="http://www.w3.org/1999/xhtml">'
            '<a href="ch02.xhtml">См. главу вестигиум</a>, где вестигиум описан.'
            "</p>"
        )
        root = tree.getroot()
        para = root.find(f".//{{{XHTML_NS}}}p")

        result = _find_and_wrap_first_match(para, "вестигиум", "reader-note-1", XHTML_NS)
        assert result is True

        # The noteref must sit outside the link, on the second occurrence
        link = para.find(f".//{{{XHTML_NS}}}a[@href='ch02.xhtml']")
        assert link.find(f".//{{{XHTML_NS}}}sup") is None
        sup = para.find(f".//{{{XHTML_NS}}}sup")
        assert link.tail == ", где вестигиум"
        assert sup.tail == " описан."

    def test_wraps_in_element_with_children(self):
        tree = _make_xhtml_tree(
            '<p xmlns="http://www.w3.org/1999/xhtml">Он <i>почувствовал</i> вестигиум.</p>'
        )
        root = tree.getroot()
        para = root.find(f".//{{{XHTML_NS}}}p")

        result = _find_and_wrap_first_match(para, "вестигиум", "reader-note-1", XHTML_NS)
        assert result is True
        serialized = etree.tostring(para, encoding="unicode")
        assert "noteref" in serialized

    def test_skips_match_inside_existing_link(self):
        """Should NOT inject noteref inside an existing <a> element."""
        tree = _make_xhtml_tree(
            '<p xmlns="http://www.w3.org/1999/xhtml">'
            '<a href="ch02.xhtml">See вестигиум chapter</a> for details.'
            "</p>"
        )
        root = tree.getroot()
        para = root.find(f".//{{{XHTML_NS}}}p")

        result = _find_and_wrap_first_match(para, "вестигиум", "reader-note-1", XHTML_NS)
        assert result is False

    def test_skips_match_inside_existing_footnote(self):
        """Should NOT inject noteref inside an existing footnote aside."""
        tree = _make_xhtml_tree(
            f'<p xmlns="{XHTML_NS}" xmlns:epub="{EPUB_NS}">'
            f'<aside epub:type="footnote">вестигиум is magic</aside> text.'
            f"</p>"
        )
        root = tree.getroot()
        para = root.find(f".//{{{XHTML_NS}}}p")

        result = _find_and_wrap_first_match(para, "вестигиум", "reader-note-1", XHTML_NS)
        assert result is False

    def test_skips_match_inside_existing_reader_note(self):
        """Should NOT inject noteref inside an existing reader-note sup."""
        tree = _make_xhtml_tree(
            f'<p xmlns="{XHTML_NS}">'
            f'Term<sup class="reader-note"><a href="#n1">вестигиум [1]</a></sup> text.'
            f"</p>"
        )
        root = tree.getroot()
        para = root.find(f".//{{{XHTML_NS}}}p")

        result = _find_and_wrap_first_match(para, "вестигиум", "reader-note-1", XHTML_NS)
        assert result is False


# ---------------------------------------------------------------------------
# inject_reader_notes (integration)
# ---------------------------------------------------------------------------


class TestInjectReaderNotes:
    def test_basic_injection(self):
        chapter = _make_chapter(
            '<p xmlns="http://www.w3.org/1999/xhtml">Он почувствовал вестигиум в комнате.</p>'
            '<p xmlns="http://www.w3.org/1999/xhtml">Это был сильный вестигиум.</p>'
        )
        glossary = _make_glossary(
            [
                {
                    "original": "vestigium",
                    "translation": "вестигиум",
                    "type": "concept",
                    "notes": "Magical trace left by spells.",
                },
            ]
        )
        config = ReaderNotesConfig(enabled=True, types=["concept"], scope="first-in-chapter")

        stats = inject_reader_notes([chapter], glossary, config, epub_version="3.0")
        assert stats.notes_injected == 1
        assert stats.chapters_modified == 1
        assert "vestigium" in stats.entries_matched

        # An EPUB3 book gets the aside a reader can pop up
        root = chapter.tree.getroot()
        asides = root.findall(f".//{{{XHTML_NS}}}aside")
        assert len(asides) == 1
        assert asides[0].get("id") == "reader-note-1"

    def test_first_in_chapter_scope(self):
        ch1 = _make_chapter('<p xmlns="http://www.w3.org/1999/xhtml">Вестигиум здесь.</p>')
        ch2 = _make_chapter('<p xmlns="http://www.w3.org/1999/xhtml">Ещё один вестигиум.</p>')
        glossary = _make_glossary(
            [
                {
                    "original": "vestigium",
                    "translation": "вестигиум",
                    "type": "concept",
                    "notes": "Magical trace.",
                },
            ]
        )
        config = ReaderNotesConfig(enabled=True, types=["concept"], scope="first-in-chapter")

        stats = inject_reader_notes([ch1, ch2], glossary, config)
        # Should inject in both chapters (first in each chapter)
        assert stats.notes_injected == 2
        assert stats.chapters_modified == 2

    def test_first_in_book_scope(self):
        ch1 = _make_chapter('<p xmlns="http://www.w3.org/1999/xhtml">Вестигиум здесь.</p>')
        ch2 = _make_chapter('<p xmlns="http://www.w3.org/1999/xhtml">Ещё один вестигиум.</p>')
        glossary = _make_glossary(
            [
                {
                    "original": "vestigium",
                    "translation": "вестигиум",
                    "type": "concept",
                    "notes": "Magical trace.",
                },
            ]
        )
        config = ReaderNotesConfig(enabled=True, types=["concept"], scope="first-in-book")

        stats = inject_reader_notes([ch1, ch2], glossary, config)
        # Should inject only once in the whole book
        assert stats.notes_injected == 1
        assert stats.chapters_modified == 1

    def test_all_scope(self):
        chapter = _make_chapter(
            '<p xmlns="http://www.w3.org/1999/xhtml">Вестигиум здесь.</p>'
            '<p xmlns="http://www.w3.org/1999/xhtml">Ещё один вестигиум.</p>'
        )
        glossary = _make_glossary(
            [
                {
                    "original": "vestigium",
                    "translation": "вестигиум",
                    "type": "concept",
                    "notes": "Magical trace.",
                },
            ]
        )
        config = ReaderNotesConfig(enabled=True, types=["concept"], scope="all")

        stats = inject_reader_notes([chapter], glossary, config)
        # Should inject in every paragraph where found
        assert stats.notes_injected == 2

    def test_no_candidates_returns_empty_stats(self):
        chapter = _make_chapter('<p xmlns="http://www.w3.org/1999/xhtml">Обычный текст.</p>')
        glossary = _make_glossary(
            [
                {
                    "original": "Peter",
                    "translation": "Питер",
                    "type": "person",
                    "notes": "Main character.",
                },
            ]
        )
        config = ReaderNotesConfig(enabled=True, types=["concept"], scope="first-in-chapter")

        stats = inject_reader_notes([chapter], glossary, config)
        assert stats.notes_injected == 0
        assert stats.total_candidates == 0

    def test_multiple_terms_in_same_paragraph(self):
        chapter = _make_chapter(
            '<p xmlns="http://www.w3.org/1999/xhtml">'
            "Он почувствовал вестигиум и применил форму."
            "</p>"
        )
        glossary = _make_glossary(
            [
                {
                    "original": "vestigium",
                    "translation": "вестигиум",
                    "type": "concept",
                    "notes": "Magical trace.",
                },
                {
                    "original": "forma",
                    "translation": "форму",
                    "type": "concept",
                    "notes": "A spell shape.",
                },
            ]
        )
        config = ReaderNotesConfig(enabled=True, types=["concept"], scope="first-in-chapter")

        stats = inject_reader_notes([chapter], glossary, config)
        assert stats.notes_injected == 2

    def test_empty_chapters_skipped(self):
        chapter = FakeChapterDoc(
            tree=_make_xhtml_tree(""),
            paragraphs=[],
        )
        glossary = _make_glossary(
            [
                {
                    "original": "vestigium",
                    "translation": "вестигиум",
                    "type": "concept",
                    "notes": "Magical trace.",
                },
            ]
        )
        config = ReaderNotesConfig(enabled=True, types=["concept"], scope="first-in-chapter")

        stats = inject_reader_notes([chapter], glossary, config)
        assert stats.notes_injected == 0


# ---------------------------------------------------------------------------
# Markup chosen by the source book's package version
# ---------------------------------------------------------------------------

_VESTIGIUM = {
    "original": "vestigium",
    "translation": "вестигиум",
    "type": "concept",
    "notes": "Magical trace left by spells.",
}
_FORMA = {
    "original": "forma",
    "translation": "форма",
    "type": "concept",
    "notes": "Basic unit of magic.",
}


def _config(scope: str = "first-in-chapter") -> ReaderNotesConfig:
    return ReaderNotesConfig(enabled=True, types=["concept"], scope=scope)


class TestIsEpub3:
    def test_reads_the_version(self):
        assert is_epub3("3.0") is True
        assert is_epub3("3.2") is True
        assert is_epub3("2.0") is False
        assert is_epub3("") is False
        assert is_epub3(None) is False


class TestEpubPrefix:
    """epub:type, never a prefix lxml invented for itself."""

    def test_marker_uses_the_epub_prefix_when_the_book_declares_none(self):
        chapter = _make_epub2_chapter(
            f'<p xmlns="{XHTML_NS}">Он почувствовал вестигиум в комнате.</p>'
        )
        inject_reader_notes([chapter], _make_glossary([_VESTIGIUM]), _config(), "2.0")

        out = _serialize(chapter)
        assert 'epub:type="noteref"' in out
        assert 'epub:type="footnote"' in out
        assert "ns0:" not in out

    def test_epub3_book_keeps_its_own_declaration(self):
        chapter = _make_chapter(f'<p xmlns="{XHTML_NS}">Он почувствовал вестигиум.</p>')
        inject_reader_notes([chapter], _make_glossary([_VESTIGIUM]), _config(), "3.0")

        out = _serialize(chapter)
        assert 'epub:type="noteref"' in out
        assert "ns0:" not in out


class TestEpub2Endnotes:
    """EPUB2 has no footnote mechanism, and no <aside> either."""

    def test_note_is_a_div_inside_a_notes_block(self):
        chapter = _make_epub2_chapter(
            f'<p xmlns="{XHTML_NS}">Он почувствовал вестигиум в комнате.</p>'
        )
        inject_reader_notes([chapter], _make_glossary([_VESTIGIUM]), _config(), "2.0")

        root = chapter.tree.getroot()
        assert root.findall(f".//{{{XHTML_NS}}}aside") == []

        blocks = [el for el in root.iter(f"{{{XHTML_NS}}}div") if el.get("class") == "reader-notes"]
        assert len(blocks) == 1
        block = blocks[0]
        assert block.get("role") == "doc-endnotes"
        title = block[0]
        assert title.get("class") == "reader-notes-title"
        assert title.text == "Примечания"
        notes = [el for el in block if el.get("class") == "reader-note-item"]
        assert len(notes) == 1
        assert notes[0].get("id") == "reader-note-1"
        assert notes[0].get("role") == "doc-footnote"

    def test_heading_comes_from_config(self):
        chapter = _make_epub2_chapter(f'<p xmlns="{XHTML_NS}">Он почувствовал вестигиум.</p>')
        config = ReaderNotesConfig(
            enabled=True, types=["concept"], scope="first-in-chapter", heading="Notes"
        )
        inject_reader_notes([chapter], _make_glossary([_VESTIGIUM]), config, "2.0")

        assert '<div class="reader-notes-title">Notes</div>' in _serialize(chapter)

    def test_note_text_is_not_a_translatable_paragraph(self):
        """A <p> here would be picked up as prose if the output is re-read."""
        chapter = _make_epub2_chapter(f'<p xmlns="{XHTML_NS}">Он почувствовал вестигиум.</p>')
        inject_reader_notes([chapter], _make_glossary([_VESTIGIUM]), _config(), "2.0")

        root = chapter.tree.getroot()
        paragraphs = root.findall(f".//{{{XHTML_NS}}}p")
        assert len(paragraphs) == 1  # only the prose paragraph
        assert "Magical trace" not in (paragraphs[0].text or "")


class TestBacklink:
    """A reader without popups follows the link and has to get back."""

    def test_note_links_back_to_its_marker(self):
        chapter = _make_epub2_chapter(f'<p xmlns="{XHTML_NS}">Он почувствовал вестигиум.</p>')
        inject_reader_notes([chapter], _make_glossary([_VESTIGIUM]), _config(), "2.0")

        root = chapter.tree.getroot()
        marker = root.find(f".//{{{XHTML_NS}}}sup/{{{XHTML_NS}}}a")
        assert marker.get("id") == "reader-note-1-ref"
        assert marker.get("href") == "#reader-note-1"

        back = root.find(f".//{{{XHTML_NS}}}a[@class='reader-note-back']")
        assert back is not None
        assert back.get("href") == "#reader-note-1-ref"
        assert back.text == "↩"

    def test_epub3_notes_get_a_backlink_too(self):
        chapter = _make_chapter(f'<p xmlns="{XHTML_NS}">Он почувствовал вестигиум.</p>')
        inject_reader_notes([chapter], _make_glossary([_VESTIGIUM]), _config(), "3.0")

        root = chapter.tree.getroot()
        back = root.find(f".//{{{XHTML_NS}}}a[@class='reader-note-back']")
        assert back is not None
        assert back.get("href") == "#reader-note-1-ref"


class TestNoteOrder:
    """Numbers follow the text, not the order the glossary was walked in."""

    def test_two_notes_in_one_paragraph_are_numbered_in_reading_order(self):
        # "форма" comes first in the text, "вестигиум" second; the glossary
        # is walked in the other order.
        chapter = _make_epub2_chapter(
            f'<p xmlns="{XHTML_NS}">Сначала форма, потом вестигиум в комнате.</p>'
        )
        stats = inject_reader_notes(
            [chapter], _make_glossary([_VESTIGIUM, _FORMA]), _config(), "2.0"
        )
        assert stats.notes_injected == 2

        root = chapter.tree.getroot()
        markers = [
            sup.find(f"{{{XHTML_NS}}}a")
            for sup in root.iter(f"{{{XHTML_NS}}}sup")
            if sup.get("class") == "reader-note"
        ]
        assert [m.text for m in markers] == ["[1]", "[2]"]

        # Every marker still points at a note that exists, and the notes
        # are listed in the same order.
        ids = [el.get("id") for el in root.iter() if el.get("class") == "reader-note-item"]
        assert ids == ["reader-note-1", "reader-note-2"]
        assert [m.get("href") for m in markers] == ["#reader-note-1", "#reader-note-2"]

        # And each note still leads back to its own marker, not the other's.
        marker_ids = [m.get("id") for m in markers]
        assert marker_ids == ["reader-note-1-ref", "reader-note-2-ref"]
        assert len(set(marker_ids)) == len(marker_ids)
        backs = [
            note.find(f".//{{{XHTML_NS}}}a[@class='reader-note-back']").get("href")
            for note in _note_items(chapter)
        ]
        assert backs == [f"#{marker_id}" for marker_id in marker_ids]

    def test_numbering_keeps_rising_across_chapters(self):
        ch1 = _make_epub2_chapter(f'<p xmlns="{XHTML_NS}">Сначала форма, потом вестигиум.</p>')
        ch2 = _make_epub2_chapter(f'<p xmlns="{XHTML_NS}">Снова форма здесь.</p>')
        inject_reader_notes([ch1, ch2], _make_glossary([_VESTIGIUM, _FORMA]), _config(), "2.0")

        def numbers(chapter):
            root = chapter.tree.getroot()
            return [
                sup.find(f"{{{XHTML_NS}}}a").text
                for sup in root.iter(f"{{{XHTML_NS}}}sup")
                if sup.get("class") == "reader-note"
            ]

        assert numbers(ch1) == ["[1]", "[2]"]
        assert numbers(ch2) == ["[3]"]


class TestNoteStyles:
    def test_styles_land_in_head_once(self):
        chapter = _make_epub2_chapter(
            f'<p xmlns="{XHTML_NS}">Он почувствовал вестигиум.</p>'
            f'<p xmlns="{XHTML_NS}">И ещё форма рядом.</p>'
        )
        inject_reader_notes([chapter], _make_glossary([_VESTIGIUM, _FORMA]), _config(), "2.0")

        root = chapter.tree.getroot()
        styles = [el for el in root.iter(f"{{{XHTML_NS}}}style") if el.get("id") == NOTE_STYLE_ID]
        assert len(styles) == 1
        assert "reader-notes" in styles[0].text

    def test_no_styles_when_nothing_was_injected(self):
        chapter = _make_epub2_chapter(f'<p xmlns="{XHTML_NS}">Ничего подходящего здесь.</p>')
        inject_reader_notes([chapter], _make_glossary([_VESTIGIUM]), _config(), "2.0")

        root = chapter.tree.getroot()
        assert root.findall(f".//{{{XHTML_NS}}}style") == []


# The note text mentions the term it explains, so a pass that sees the note
# in the tree must not mark the term inside it.
_SELF_REFERRING = {
    "original": "vestigium",
    "translation": "вестигиум",
    "type": "concept",
    "notes": "Тот самый вестигиум, след магии.",
}


class TestSecondPassLeavesNotesAlone:
    """The guard has to hold when the note is already in the document.

    The first pass appends notes after matching, so the note body is not in
    the tree while that pass runs — only a second pass can reach it. These
    tests refresh the paragraph list in between, the way reading the output
    back in would, and run the injection again.
    """

    def test_epub3_note_paragraph_is_not_marked(self):
        chapter = _make_chapter(f'<p xmlns="{XHTML_NS}">Он почувствовал вестигиум в комнате.</p>')
        glossary = _make_glossary([_SELF_REFERRING])
        inject_reader_notes([chapter], glossary, _config("all"), "3.0")

        # An EPUB3 note body is a <p>, so a re-read really would pick it up.
        root = chapter.tree.getroot()
        chapter.paragraphs = root.findall(f".//{{{XHTML_NS}}}p")
        assert any(el.get("class") == "reader-note-text" for el in chapter.paragraphs)

        inject_reader_notes([chapter], glossary, _config("all"), "3.0")

        notes = _note_items(chapter)
        for note in notes:
            assert note.findall(f".//{{{XHTML_NS}}}sup") == []

    def test_epub2_note_is_not_marked_and_the_block_is_not_duplicated(self):
        chapter = _make_epub2_chapter(
            f'<p xmlns="{XHTML_NS}">Он почувствовал вестигиум в комнате.</p>'
        )
        glossary = _make_glossary([_SELF_REFERRING])
        inject_reader_notes([chapter], glossary, _config("all"), "2.0")

        # An EPUB2 note body is a <div>, which a re-read would not treat as
        # prose. Hand it in anyway: the guard must not depend on that.
        root = chapter.tree.getroot()
        chapter.paragraphs = root.findall(f".//{{{XHTML_NS}}}p") + [
            el for el in root.iter(f"{{{XHTML_NS}}}div") if el.get("class") == "reader-note-text"
        ]

        inject_reader_notes([chapter], glossary, _config("all"), "2.0")

        for note in _note_items(chapter):
            assert note.findall(f".//{{{XHTML_NS}}}sup") == []

        blocks = [el for el in root.iter(f"{{{XHTML_NS}}}div") if el.get("class") == "reader-notes"]
        assert len(blocks) == 1
        titles = [el for el in root.iter() if el.get("class") == "reader-notes-title"]
        assert len(titles) == 1


class TestRenumberFailsClosed:
    """A chapter it cannot fully account for is left exactly as it was."""

    def test_a_marker_without_a_note_stops_the_renumbering(self):
        chapter = _make_epub2_chapter(
            f'<p xmlns="{XHTML_NS}">Сначала форма, потом вестигиум в комнате.</p>'
        )
        inject_reader_notes([chapter], _make_glossary([_VESTIGIUM, _FORMA]), _config(), "2.0")

        # Break one link the way a hand-edited book might: the marker now
        # names a note that is not there.
        root = chapter.tree.getroot()
        markers = [
            sup.find(f"{{{XHTML_NS}}}a")
            for sup in root.iter(f"{{{XHTML_NS}}}sup")
            if sup.get("class") == "reader-note"
        ]
        markers[0].set("href", "#reader-note-99")
        before = etree.tostring(chapter.tree, encoding="unicode")

        from booktranslator.reader_notes import _renumber_notes_in_reading_order

        _renumber_notes_in_reading_order(root, XHTML_NS)

        assert etree.tostring(chapter.tree, encoding="unicode") == before
