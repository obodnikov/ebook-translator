"""Markup the model dropped or changed comes back from the source book.

The shapes are from A Drop of Corruption (six page markers gone, a doubled class),
The Tainted Cup (`<i>` without its class) and Shroud (markers one paragraph off).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from lxml import etree
from typer.testing import CliRunner

from booktranslator.cli import app
from booktranslator.markup import restore_book_markup, restore_markup
from tests.epub_fixtures import write_minimal_epub

XHTML = "http://www.w3.org/1999/xhtml"
OPS = "http://www.idpf.org/2007/ops"


def _el(inner: str) -> etree._Element:
    root = etree.fromstring(f'<body xmlns="{XHTML}" xmlns:epub="{OPS}">{inner}</body>')
    return root[0]


def _restored(source: str, translated: str) -> tuple[etree._Element, list[str]]:
    element = _el(translated)
    done = restore_markup(_el(source), element)
    return element, done


def _text(el: etree._Element) -> str:
    return "".join(el.itertext())


def _before(el: etree._Element, marker_id: str) -> str:
    """The text in front of the element with this id."""
    [marker] = el.xpath(f".//*[@id='{marker_id}']")
    return "".join(marker.xpath("preceding::text()"))


def _page(number: int) -> str:
    return f'<span epub:type="pagebreak" id="page_{number}" role="doc-pagebreak" title="{number}"/>'


class TestLostMarkers:
    def test_a_page_marker_comes_back_whole_and_the_text_is_untouched(self):
        text = "Раз два три четыре пять шесть семь."
        el, done = _restored(f"<p>One two three {_page(168)}four five six.</p>", f"<p>{text}</p>")

        [marker] = el
        assert dict(marker.attrib) == {
            f"{{{OPS}}}type": "pagebreak",
            "id": "page_168",
            "role": "doc-pagebreak",
            "title": "168",
        }
        assert marker.tag == f"{{{XHTML}}}span"
        assert _text(el) == text
        assert done

    def test_it_stands_between_words_about_where_the_source_has_it(self):
        el, _ = _restored(
            f"<p>One two three {_page(2)}four five six.</p>", "<p>Раз два три четыре пять.</p>"
        )
        assert _before(el, "page_2") == "Раз два три "

    def test_at_the_start_it_stays_at_the_start(self):
        el, _ = _restored(
            f'<p>{_page(3)}<i class="c">One</i> two.</p>', '<p><i class="c">Раз</i> два.</p>'
        )
        assert el[0].get("id") == "page_3"

    def test_at_the_end_it_stays_at_the_end(self):
        el, _ = _restored(f"<p>One two.{_page(4)}</p>", "<p>Раз два.</p>")
        assert _before(el, "page_4") == "Раз два."

    def test_it_can_stand_inside_emphasis(self):
        el, _ = _restored(
            f"<p>A <i>one two {_page(5)}three four five</i> B</p>",
            "<p>А <i>раз два три четыре пять</i> Б</p>",
        )
        assert el.xpath(".//*[@id='page_5']")
        assert _text(el) == "А раз два три четыре пять Б"

    def test_never_inside_a_link(self):
        el, _ = _restored(
            f'<p><a href="#n">one two {_page(6)}three four</a></p>',
            '<p><a href="#n">раз два три четыре</a></p>',
        )
        [marker] = el.xpath(".//*[@id='page_6']")
        assert marker.getparent() is el

    def test_two_markers_keep_their_order(self):
        el, _ = _restored(f"<p>{_page(7)}{_page(8)}One two.</p>", "<p>Раз два.</p>")
        assert [e.get("id") for e in el] == ["page_7", "page_8"]

    def test_an_anchor_that_held_text_comes_back_as_an_empty_target(self):
        el, _ = _restored(
            '<p>See <span id="term">the word</span> and <em>more</em>.</p>',
            "<p>См. слово и <em>прочее</em>.</p>",
        )
        [target] = el.xpath(".//*[@id='term']")
        assert dict(target.attrib) == {"id": "term"}
        assert not target.text
        assert _text(el) == "См. слово и прочее."


class TestForeignMarkers:
    def test_a_marker_of_another_paragraph_is_removed_and_its_text_kept(self):
        el, done = _restored("<p>One two.</p>", f"<p>Раз {_page(42)}два.</p>")
        assert len(el) == 0
        assert _text(el) == "Раз два."
        assert done

    def test_text_after_a_removed_marker_joins_the_element_before_it(self):
        el, _ = _restored("<p><i>One</i> two.</p>", f"<p><i>Раз</i>{_page(42)} два.</p>")
        assert _text(el) == "Раз два."
        assert el[0].tail == " два."

    def test_a_foreign_id_on_text_loses_the_id_only(self):
        el, _ = _restored("<p>One <b>two</b>.</p>", '<p>Раз <b id="made-up">два</b>.</p>')
        assert el[0].get("id") is None
        assert el[0].text == "два"

    def test_a_marker_printed_twice_is_kept_once(self):
        el, _ = _restored(f"<p>One {_page(9)}two.</p>", f"<p>{_page(9)}Раз {_page(9)}два.</p>")
        assert len(el.xpath(".//*[@id='page_9']")) == 1
        assert _text(el) == "Раз два."


class TestAttributes:
    def test_a_paragraph_gets_the_class_of_the_source(self):
        el, done = _restored('<p class="para-sp">One.</p>', '<p class="para-p para-sp">Раз.</p>')
        assert el.get("class") == "para-sp"
        assert done

    def test_an_attribute_the_model_made_up_goes(self):
        el, _ = _restored("<p>One.</p>", '<p class="new">Раз.</p>')
        assert not el.attrib

    def test_a_translated_text_attribute_stays_translated(self):
        source = '<p title="A note" class="x">One <img src="a.png" alt="A map"/></p>'
        el, _ = _restored(source, '<p title="Заметка">Раз <img src="b.png" alt="Карта"/></p>')
        assert el.get("title") == "Заметка"
        assert el.get("class") == "x"
        assert el[0].get("alt") == "Карта"
        assert el[0].get("src") == "a.png"

    def test_emphasis_gets_its_class_back(self):
        el, _ = _restored(
            '<p><i class="char-i">One</i> and <i class="char-i">two</i>.</p>',
            "<p><i>Раз</i> и <i>два</i>.</p>",
        )
        assert [e.get("class") for e in el] == ["char-i", "char-i"]

    def test_inline_elements_that_differ_are_not_guessed_at(self):
        """A translation may reorder them, so position says nothing."""
        translated = '<p><span class="b">два</span> и <span>раз</span>.</p>'
        el, _ = _restored(
            '<p><span class="a">One</span> and <span class="b">two</span>.</p>', translated
        )
        assert [e.get("class") for e in el] == ["b", None]

    def test_an_element_with_an_id_is_matched_by_it(self):
        el, _ = _restored(
            '<p><a id="x" href="#1">One</a> <a id="y" href="#2">two</a></p>',
            '<p><a id="y">два</a> <a id="x">раз</a></p>',
        )
        assert [e.get("href") for e in el] == ["#2", "#1"]

    def test_paragraphs_inside_a_quote_are_matched_in_order(self):
        el, _ = _restored(
            '<blockquote>Lead<p class="a">One</p><p class="b">Two</p></blockquote>',
            "<blockquote>Начало<p>Раз</p><p>Два</p></blockquote>",
        )
        assert [p.get("class") for p in el] == ["a", "b"]

    def test_sound_markup_is_left_exactly_as_it_is(self):
        translated = f'<p class="p">Раз <i class="c">два</i> {_page(1)}три.</p>'
        el = _el(translated)
        before = etree.tostring(el)

        done = restore_markup(_el(f'<p class="p">One {_page(1)}<i class="c">two</i>.</p>'), el)

        assert done == []
        assert etree.tostring(el) == before


def test_a_marker_moved_to_the_next_paragraph_ends_up_once_in_its_own():
    source = [_el(f"<p>One {_page(43)}two.</p>"), _el("<p>Three.</p>")]
    translated = [_el("<p>Раз два.</p>"), _el(f"<p>{_page(43)}Три.</p>")]
    chapter = SimpleNamespace(archive_name="ch.xhtml", paragraphs=translated)
    original = SimpleNamespace(archive_name="ch.xhtml", paragraphs=source)

    report = restore_book_markup([chapter], [original])

    assert [len(el.xpath(".//*[@id='page_43']")) for el in translated] == [1, 0]
    assert [(name, number) for name, number, _ in report] == [("ch.xhtml", 1), ("ch.xhtml", 2)]


class TestInTheCommands:
    BOOK = (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        f'<html xmlns="{XHTML}" xmlns:epub="{OPS}"><head><title>Test</title></head><body>\n'
        f'<p class="first">A short paragraph {_page(11)}about vestigium.</p>\n'
        "</body></html>"
    )

    @pytest.fixture
    def translated(self, tmp_path: Path):
        epub = write_minimal_epub(tmp_path / "book.epub", self.BOOK)
        provider = MagicMock()
        provider.complete.return_value = MagicMock(
            text='===PARAGRAPH 1===\n<p class="other">Короткий абзац про вестигиум.</p>',
            input_tokens=1,
            output_tokens=1,
            finish_reason="stop",
        )
        args = ["translate", str(epub), "--work", str(tmp_path / "work"), "--no-judge"]
        args += ["--no-proofread", "--no-style", "--no-verify", "--no-notes"]
        args += ["--out", str(tmp_path / "out.epub")]
        with patch("booktranslator.cli.create_stage_provider", return_value=provider):
            result = CliRunner().invoke(app, args)
        assert result.exit_code == 0, result.output
        return epub, result

    @staticmethod
    def _chapter(path: Path) -> str:
        import zipfile

        with zipfile.ZipFile(path) as zf:
            return zf.read("ch01.xhtml").decode("utf-8")

    def test_translate_writes_the_book_with_the_markup_of_the_source(self, translated, tmp_path):
        _, result = translated
        chapter = self._chapter(tmp_path / "out.epub")
        assert 'id="page_11"' in chapter
        assert 'class="first"' in chapter
        assert chapter.count("xmlns:epub") == 1
        assert "Markup restored" in result.output

    def test_assemble_restores_it_from_the_cache_without_a_model(self, translated, tmp_path):
        epub, _ = translated
        out = tmp_path / "again.epub"
        work = tmp_path / "work" / "test-book"

        result = CliRunner().invoke(
            app, ["assemble", str(work), "--epub", str(epub), "--out", str(out), "--no-notes"]
        )

        assert result.exit_code == 0, result.output
        chapter = self._chapter(out)
        assert 'id="page_11"' in chapter
        assert 'class="first"' in chapter
        assert "Markup restored" in result.output
