"""Tests for reader_notes footnote injection."""

from __future__ import annotations

from dataclasses import dataclass, field
from io import BytesIO

from lxml import etree

from booktranslator.models import ReaderNotesConfig, SeriesGlossary, SeriesGlossaryEntry
from booktranslator.reader_notes import (
    _build_candidates,
    _find_and_wrap_first_match,
    inject_reader_notes,
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

        stats = inject_reader_notes([chapter], glossary, config)
        assert stats.notes_injected == 1
        assert stats.chapters_modified == 1
        assert "vestigium" in stats.entries_matched

        # Check footnote aside was added to body
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
