"""What `epub_io` hands to translation: every piece of text once, however deep it sits.

Two books showed the two halves of this. A Drop of Corruption has block quotes made
of paragraphs: both the quote and each paragraph in it were listed, so the model was
sent the text twice and the copy under the paragraph's own number never reached the
book. The Faith of Beasts sets whole paragraphs in <em><span>, two levels down: those
were not listed at all and stayed in English.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from lxml import etree

from booktranslator.cache import Cache
from booktranslator.chunker import chunk_book
from booktranslator.epub_io import _find_paragraphs, read_book_structured
from booktranslator.pipeline_helpers import rehydrate_book_from_waterfall
from booktranslator.translator import TranslateStats, Translator
from tests.epub_fixtures import write_minimal_epub

XHTML_NS = "http://www.w3.org/1999/xhtml"


def _units(body: str) -> list[tuple[str, str]]:
    root = etree.fromstring(f'<html xmlns="{XHTML_NS}"><body>{body}</body></html>')
    return [
        (etree.QName(el.tag).localname, " ".join("".join(el.itertext()).split()))
        for el in _find_paragraphs(etree.ElementTree(root))
    ]


class TestNestedBlocks:
    def test_a_quote_made_of_paragraphs_gives_its_paragraphs(self):
        body = "<blockquote>\n  <p>One.</p>\n  <p>Two.</p>\n</blockquote>"
        assert _units(body) == [("p", "One."), ("p", "Two.")]

    def test_a_quote_with_text_of_its_own_is_one_unit(self):
        assert _units("<blockquote>Plain words.</blockquote>") == [("blockquote", "Plain words.")]

    @pytest.mark.parametrize(
        "body",
        [
            "<blockquote>Lead in.<p>One.</p></blockquote>",
            "<blockquote><p>One.</p> After it.</blockquote>",
            "<blockquote><p>One.</p><footer>The author</footer></blockquote>",
            "<blockquote><div>Lead in.</div><p>One.</p></blockquote>",
        ],
    )
    def test_text_outside_the_inner_paragraphs_keeps_the_quote_whole(self, body):
        units = _units(body)
        assert [tag for tag, _ in units] == ["blockquote"]
        assert "One." in units[0][1]

    def test_a_quote_in_a_quote_is_listed_once(self):
        assert _units("<blockquote><blockquote><p>One.</p></blockquote></blockquote>") == [
            ("p", "One.")
        ]

    def test_a_heading_inside_a_quote_is_a_unit(self):
        assert _units("<blockquote><h3>Head</h3><p>One.</p></blockquote>") == [
            ("h3", "Head"),
            ("p", "One."),
        ]

    def test_a_comment_is_not_text_of_the_quote(self):
        assert _units("<blockquote><!-- note --><p>One.</p></blockquote>") == [("p", "One.")]

    def test_a_deep_paragraph_inside_a_quote_is_a_unit_of_its_own(self):
        body = "<blockquote><p>One.</p><p><em><b>Two.</b></em></p></blockquote>"
        assert _units(body) == [("p", "One."), ("p", "Two.")]

    def test_paragraphs_around_a_quote_keep_document_order(self):
        body = "<p>Before.</p><blockquote><p>Inside.</p></blockquote><p>After.</p>"
        assert [text for _, text in _units(body)] == ["Before.", "Inside.", "After."]


class TestTextDeepInside:
    @pytest.mark.parametrize(
        "body, expected",
        [
            (
                '<p><em><span class="dropcap"><span>T</span></span>he swarm walks.</em></p>',
                ("p", "The swarm walks."),
            ),
            ('<h1><a href="#toc"><span>Meet the Author</span></a></h1>', ("h1", "Meet the Author")),
            ("<p><strong><em>Financial Times</em></strong></p>", ("p", "Financial Times")),
        ],
    )
    def test_text_two_levels_down_is_translated(self, body, expected):
        assert _units(body) == [expected]

    @pytest.mark.parametrize(
        "body",
        [
            '<p><a href="https://example.com"><i>Example.com</i></a></p>',
            "<p><a><span>WWW.EXAMPLE.CO.UK</span></a></p>",
            "<p><a><span>https://example.com/books/12</span></a></p>",
            "<h2><a><strong>12</strong></a></h2>",
            "<p><span><b> </b></span></p>",
        ],
    )
    def test_an_address_or_a_bare_number_deep_inside_is_left_alone(self, body):
        assert _units(body) == []


class TestQuoteThroughThePipeline:
    XHTML = (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        f'<html xmlns="{XHTML_NS}"><head><title>Test</title></head><body>\n'
        '<blockquote class="letter">\n'
        "<p>To the representatives of the office, with respect and some urgency.</p>\n"
        "<p>Please come quickly, for I cannot say how safe any of us are tonight.</p>\n"
        "</blockquote>\n</body></html>"
    )
    REPLY = (
        "===PARAGRAPH 1===\n<p>Представителям ведомства, с уважением и некоторой спешкой.</p>\n"
        "===PARAGRAPH 2===\n"
        "<p>Приезжайте скорее: не могу сказать, насколько мы все сегодня в безопасности.</p>"
    )

    @pytest.fixture
    def translated(self, tmp_path: Path):
        epub = write_minimal_epub(tmp_path / "quote.epub", self.XHTML)
        chunk_set = chunk_book(read_book_structured(epub), target_words=2000, overlap_paragraphs=1)
        prompt = tmp_path / "translate.md"
        prompt.write_text(
            "---\nname: t\nversion: 1\ntemperature: 0.1\nmax_tokens: 100\n---\n\n"
            "# System\n\nTranslate.\n\n# User\n\n{{ main_fragments | join('\\n') }}\n",
            encoding="utf-8",
        )
        provider = MagicMock()
        provider.complete.return_value = MagicMock(
            text=self.REPLY, input_tokens=1, output_tokens=1, finish_reason="stop"
        )
        cache = Cache(tmp_path / "cache.sqlite")
        translator = Translator(provider, prompt, cache, None, model="m")
        translator.translate_chunk(chunk_set, chunk_set.chunks[0], TranslateStats(chunks_total=1))
        yield chunk_set, cache, provider
        cache.close()

    def test_the_model_is_sent_each_sentence_once(self, translated):
        _, _, provider = translated
        assert provider.complete.call_count == 1
        assert provider.complete.call_args.kwargs["user"].count("Please come quickly") == 1

    def test_the_quote_itself_stays_in_the_book(self, translated):
        chunk_set, _, _ = translated
        [quote] = chunk_set.book.chapters[0].tree.getroot().iter(f"{{{XHTML_NS}}}blockquote")
        assert quote.get("class") == "letter"
        assert "Представителям ведомства" in "".join(quote.itertext())

    def test_a_later_edit_inside_the_quote_reaches_the_book(self, translated):
        chunk_set, cache, _ = translated
        edited = self.REPLY.replace("Приезжайте скорее", "Прошу, приезжайте")
        cache.put("p", "proofread", "m", "1", edited, meta={"chunk_id": "ch01_c01"})

        rehydrate_book_from_waterfall(cache, chunk_set)

        text = "".join(chunk_set.book.chapters[0].tree.getroot().itertext())
        assert "Прошу, приезжайте" in text
