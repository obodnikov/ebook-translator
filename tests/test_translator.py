"""Tests for translator.py: a reply is checked in full before it is cached or spliced.

The broken replies below are shaped after the three chunks of Foxglove Summer
that passed the paragraph count and still could not go into the book.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from lxml import etree

from booktranslator.cache import Cache
from booktranslator.chunker import chunk_book
from booktranslator.epub_io import read_book_structured
from booktranslator.pipeline_helpers import collect_waterfall_paragraphs
from booktranslator.prompts import load_prompt, render_prompt
from booktranslator.provider import CompletionResult
from booktranslator.translator import TranslateStats, Translator
from tests.epub_fixtures import write_minimal_epub, xhtml_with_paragraphs

TRANSLATE_PROMPT = Path(__file__).resolve().parents[1] / "prompts" / "translate.md"

GOOD = "===PARAGRAPH 1===\n<p>Абзац про вестигиум.</p>\n===PARAGRAPH 2===\n<p>Абзац с Сиуоллом.</p>"

BROKEN = {
    # ch04_c06: no closing tag anywhere.
    "unclosed": (
        "===PARAGRAPH 1===\n<p>Абзац про вестигиум.\n===PARAGRAPH 2===\n<p>Абзац с Сиуоллом."
    ),
    # Two elements under one marker that are not one paragraph split in two.
    "two_different_under_one": (
        "===PARAGRAPH 1===\n<p>Абзац про вестигиум.</p>\n<h2>Лишний.</h2>\n"
        "===PARAGRAPH 2===\n<p>Абзац с Сиуоллом.</p>"
    ),
    # Two <p> under one marker, but far more text than the source paragraph holds.
    "two_too_long_under_one": (
        "===PARAGRAPH 1===\n<p>Абзац про вестигиум.</p>\n"
        f"<p>{'Текст, которого в этом абзаце оригинала нет. ' * 3}</p>\n"
        "===PARAGRAPH 2===\n<p>Абзац с Сиуоллом.</p>"
    ),
    # ch05_c02: a mistyped marker and the model's own correction between paragraphs.
    "commentary": (
        "===PARAGRAPH 1===\n<p>Абзац про вестигиум.</p>\n===PARAGRAPH 2</p>\n"
        "<p>Абзац с Сиуоллом.</p>\n\nПостойте, я ошибся в номере — сейчас исправлю.\n\n"
        "===PARAGRAPH 2===\n<p>Абзац с Сиуоллом.</p>"
    ),
}


def _reply(text: str) -> CompletionResult:
    return CompletionResult(
        text=text,
        input_tokens=100,
        output_tokens=50,
        total_tokens=150,
        model="m",
        raw={},
        finish_reason="stop",
    )


@pytest.fixture
def chunk_set(tmp_path: Path):
    epub = write_minimal_epub(tmp_path / "book.epub")
    return chunk_book(read_book_structured(epub), target_words=2000, overlap_paragraphs=1)


@pytest.fixture
def cache(tmp_path: Path) -> Cache:
    return Cache(tmp_path / "cache.sqlite")


def _translator(provider, cache: Cache) -> Translator:
    return Translator(provider, TRANSLATE_PROMPT, cache, None, model="m")


def _texts(chunk_set) -> list[str]:
    return ["".join(p.itertext()) for p in chunk_set.book.chapters[0].paragraphs]


@pytest.mark.parametrize("shape", sorted(BROKEN))
def test_rejected_reply_is_asked_again_and_the_good_one_cached(chunk_set, cache, shape):
    provider = MagicMock()
    provider.complete.side_effect = [_reply(BROKEN[shape]), _reply(GOOD)]
    stats = TranslateStats(chunks_total=1)

    _translator(provider, cache).translate_chunk(chunk_set, chunk_set.chunks[0], stats)

    assert provider.complete.call_count == 2
    assert "Your previous reply was rejected" in provider.complete.call_args.kwargs["user"]
    assert stats.retries == 1
    assert stats.input_tokens == 200, "both calls were paid for"
    [row] = cache.get_chunk_stages("ch01_c01")
    assert row.content == GOOD
    assert _texts(chunk_set) == ["Абзац про вестигиум.", "Абзац с Сиуоллом."]


@pytest.mark.parametrize("shape", sorted(BROKEN))
def test_rejected_twice_caches_nothing_and_leaves_the_chunk_whole(
    chunk_set, cache, tmp_path, shape
):
    provider = MagicMock()
    provider.complete.side_effect = [_reply(BROKEN[shape]), _reply(BROKEN[shape])]
    before = _texts(chunk_set)

    stats = _translator(provider, cache).translate_book(chunk_set)

    assert stats.chunks_failed == 1
    assert cache.list_stages() == {}
    # Not even the paragraphs before the broken one went in.
    assert _texts(chunk_set) == before
    dump = (tmp_path / "failed" / "translate-ch01_c01.txt").read_text(encoding="utf-8")
    assert "--- reply ---" in dump and "--- reply after the retry ---" in dump


def test_retry_budget_spent_means_no_second_call(chunk_set, cache, tmp_path):
    provider = MagicMock()
    provider.complete.return_value = _reply(BROKEN["unclosed"])
    stats = TranslateStats(chunks_total=1, retries=3)

    with pytest.raises(ValueError, match="Paragraphs 1, 2 of 2"):
        _translator(provider, cache).translate_chunk(chunk_set, chunk_set.chunks[0], stats)

    assert provider.complete.call_count == 1
    assert (tmp_path / "failed" / "translate-ch01_c01.txt").is_file()


def test_broken_cached_reply_is_reported_not_rebought(chunk_set, cache):
    provider = MagicMock()
    translator = _translator(provider, cache)
    chunk = chunk_set.chunks[0]
    system, user = render_prompt(translator.prompt, translator._build_context(chunk_set, chunk))
    key = Cache.make_key("translate", "m", translator.prompt.version, system, user)
    cache.put(key, "translate", "m", "3", BROKEN["commentary"], meta={"chunk_id": chunk.id})

    with pytest.raises(ValueError, match="btrans cache clear"):
        translator.translate_chunk(chunk_set, chunk, TranslateStats(chunks_total=1))

    provider.complete.assert_not_called()


def _dialogue_chunk_set(tmp_path: Path, *paragraphs: str):
    epub = write_minimal_epub(tmp_path / "dialogue.epub", xhtml_with_paragraphs(*paragraphs))
    return chunk_book(read_book_structured(epub), target_words=2000, overlap_paragraphs=1)


# Shaped after ch32_c01 of The Tainted Cup: the count holds and the speech is gone.
SPEECH_AND_SILENCE = (
    "She shrugged. “Maybe someone out there wants us to stop looking, and that would "
    "explain why the man who sold the paper had to die.”",
    "There was an uneasy silence in the room.",
)
LOST_SPEECH = (
    "===PARAGRAPH 1===\n<p>Она пожала плечами:</p>\n"
    "===PARAGRAPH 2===\n<p>В комнате повисла тревожная тишина.</p>"
)
KEPT_SPEECH = (
    "===PARAGRAPH 1===\n<p>Она пожала плечами. — Возможно, кто-то хочет, чтобы мы перестали "
    "искать, и тогда понятно, почему продавец бумаги должен был умереть.</p>\n"
    "===PARAGRAPH 2===\n<p>В комнате повисла тревожная тишина.</p>"
)


def test_reply_that_lost_speech_is_rejected_though_the_count_holds(cache, tmp_path):
    chunk_set = _dialogue_chunk_set(tmp_path, *SPEECH_AND_SILENCE)
    provider = MagicMock()
    provider.complete.side_effect = [_reply(LOST_SPEECH), _reply(KEPT_SPEECH)]
    stats = TranslateStats(chunks_total=1)

    _translator(provider, cache).translate_chunk(chunk_set, chunk_set.chunks[0], stats)

    assert provider.complete.call_count == 2
    second_ask = provider.complete.call_args.kwargs["user"]
    assert "Your previous reply was rejected" in second_ask
    assert "Paragraph 1 of 2" in second_ask
    [row] = cache.get_chunk_stages("ch01_c01")
    assert row.content == KEPT_SPEECH


def test_reply_that_lost_speech_twice_is_never_cached(cache, tmp_path):
    chunk_set = _dialogue_chunk_set(tmp_path, *SPEECH_AND_SILENCE)
    provider = MagicMock()
    provider.complete.return_value = _reply(LOST_SPEECH)

    stats = _translator(provider, cache).translate_book(chunk_set)

    assert stats.chunks_failed == 1
    assert cache.list_stages() == {}
    assert (tmp_path / "failed" / "translate-ch01_c01.txt").is_file()


def test_paragraph_split_under_one_marker_is_joined_without_paying_again(cache, tmp_path, caplog):
    chunk_set = _dialogue_chunk_set(tmp_path, *SPEECH_AND_SILENCE)
    split = KEPT_SPEECH.replace("плечами. — Возможно", "плечами.</p>\n<p>— Возможно")
    provider = MagicMock()
    provider.complete.return_value = _reply(split)
    stats = TranslateStats(chunks_total=1)

    with caplog.at_level("WARNING"):
        _translator(provider, cache).translate_chunk(chunk_set, chunk_set.chunks[0], stats)

    assert provider.complete.call_count == 1
    assert stats.retries == 0
    [row] = cache.get_chunk_stages("ch01_c01")
    assert row.content == KEPT_SPEECH, "later stages read this row: it must hold one element"
    [meta_json] = cache.conn.execute("SELECT meta_json FROM cache").fetchone()
    assert json.loads(meta_json)["joined_paragraphs"] == [1], "a join is recorded on the row"
    assert "joined" in caplog.text and "ch01_c01" in caplog.text, "a join is never silent"
    [first, second] = _texts(chunk_set)
    assert first.startswith("Она пожала плечами. — Возможно")
    assert second == "В комнате повисла тревожная тишина."


def test_a_complete_over_split_reply_is_rebuilt_without_paying_again(cache, tmp_path, caplog):
    """ch09_c02 and ch33_c01: the first reply had every line, under too many markers."""
    chunk_set = _dialogue_chunk_set(tmp_path, *SPEECH_AND_SILENCE)
    over_split = (
        "===PARAGRAPH 1===\n<p>Она пожала плечами.</p>\n"
        "===PARAGRAPH 2===\n<p>— Возможно, кто-то хочет, чтобы мы перестали искать, и тогда "
        "понятно, почему продавец бумаги должен был умереть.</p>\n"
        "===PARAGRAPH 3===\n<p>В комнате повисла тревожная тишина.</p>"
    )
    provider = MagicMock()
    provider.complete.return_value = _reply(over_split)
    stats = TranslateStats(chunks_total=1)

    with caplog.at_level("WARNING"):
        _translator(provider, cache).translate_chunk(chunk_set, chunk_set.chunks[0], stats)

    assert provider.complete.call_count == 1
    assert stats.retries == 0
    [row] = cache.get_chunk_stages("ch01_c01")
    assert row.content == KEPT_SPEECH, "later stages read this row: one element per marker"
    [meta_json] = cache.conn.execute("SELECT meta_json FROM cache").fetchone()
    assert json.loads(meta_json)["rebuilt_paragraphs"] == [1], "a rebuild is recorded on the row"
    assert "rebuilt" in caplog.text and "ch01_c01" in caplog.text, "a rebuild is never silent"


def test_a_rebuilt_reply_still_has_to_pass_every_check(cache, tmp_path):
    """Rebuilding fits the count; it must not let a reply that lost its speech through."""
    chunk_set = _dialogue_chunk_set(
        tmp_path,
        SPEECH_AND_SILENCE[0],
        "She shrugged. “Perhaps nobody really knows who started all of this, sir, not even me.”",
        SPEECH_AND_SILENCE[1],
    )
    still_lost = (
        "===PARAGRAPH 1===\n<p>Она пожала плечами.</p>\n"
        "===PARAGRAPH 2===\n<p>— Возможно, кто-то хочет, чтобы мы перестали искать, и тогда "
        "понятно, почему продавец бумаги должен был умереть.</p>\n"
        "===PARAGRAPH 3===\n<p>Она пожала плечами:</p>\n"
        "===PARAGRAPH 4===\n<p>В комнате повисла тревожная тишина.</p>"
    )
    provider = MagicMock()
    provider.complete.return_value = _reply(still_lost)

    stats = _translator(provider, cache).translate_book(chunk_set)

    assert stats.chunks_failed == 1
    assert cache.list_stages() == {}


def _rendered_system(target_lang: str, target_lang_name: str) -> str:
    system, _ = render_prompt(
        load_prompt(TRANSLATE_PROMPT),
        {
            "source_lang_name": "English",
            "target_lang": target_lang,
            "target_lang_name": target_lang_name,
            "glossary_block": "",
            "main_fragments": [],
            "prev_overlap": "",
            "next_overlap": "",
            "reflection_notes": "",
        },
    )
    return " ".join(system.split())


def test_russian_speech_is_marked_with_an_em_dash_in_the_prompt():
    system = _rendered_system("ru", "Russian")
    assert "opens with `— `" in system
    assert "never with guillemets" in system
    assert "never three dots" in system


def test_another_language_is_not_given_the_russian_dash_rule():
    system = _rendered_system("hu", "Hungarian")
    assert "guillemets" not in system
    assert "the way Hungarian fiction does" in system
    assert "never three dots" in system, "this rule is not about Russian"


def test_a_reply_that_lost_the_epub_declaration_goes_all_the_way_through(cache, tmp_path):
    """ch36_c04: asked twice, failed twice, over one attribute. No text was wrong."""
    ops = "http://www.idpf.org/2007/ops"
    span = '<span epub:type="pagebreak" id="page_245" title="245"/>'
    epub = write_minimal_epub(
        tmp_path / "pages.epub",
        '<?xml version="1.0" encoding="utf-8"?>\n'
        f'<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="{ops}">\n'
        f"<head><title>Test</title></head>\n<body>\n<p>{span}A short paragraph.</p>\n"
        "</body>\n</html>",
    )
    chunk_set = chunk_book(read_book_structured(epub), target_words=2000, overlap_paragraphs=1)
    reply = f"===PARAGRAPH 1===\n<p>{span}Короткий абзац.</p>"
    provider = MagicMock()
    provider.complete.return_value = _reply(reply)
    stats = TranslateStats(chunks_total=1)

    _translator(provider, cache).translate_chunk(chunk_set, chunk_set.chunks[0], stats)

    assert provider.complete.call_count == 1
    assert stats.chunks_translated == 1
    [paragraph] = chunk_set.book.chapters[0].paragraphs
    assert "epub:type" in etree.tostring(paragraph, encoding="unicode")
    assert "".join(paragraph.itertext()) == "Короткий абзац."
    # The row holds the model's reply, as with a bare `&`, and later readers can use it.
    counts = {"ch01_c01": 1}
    assert collect_waterfall_paragraphs(cache, ["ch01_c01"], "proofread", counts) == {
        "ch01_c01": [f"<p>{span}Короткий абзац.</p>"]
    }


def test_the_second_ask_names_every_broken_paragraph(cache, tmp_path):
    chunk_set = _dialogue_chunk_set(tmp_path, *SPEECH_AND_SILENCE)
    both_broken = "===PARAGRAPH 1===\n<p>Один\n===PARAGRAPH 2===\n<p>Два"
    provider = MagicMock()
    provider.complete.side_effect = [_reply(both_broken), _reply(KEPT_SPEECH)]

    _translator(provider, cache).translate_chunk(
        chunk_set, chunk_set.chunks[0], TranslateStats(chunks_total=1)
    )

    assert "Paragraphs 1, 2 of 2" in provider.complete.call_args.kwargs["user"]


def test_failed_chunks_are_named_in_the_stats(chunk_set, cache):
    provider = MagicMock()
    provider.complete.return_value = _reply(BROKEN["unclosed"])

    stats = _translator(provider, cache).translate_book(chunk_set)

    assert stats.failed_chunk_ids == ["ch01_c01"]


# ---------------------------------------------------------------------------
# Overlap is source text, so a chunk's prompt does not depend on its neighbours
# ---------------------------------------------------------------------------


def _two_chunks(tmp_path: Path):
    """The fixture book cut so that each paragraph is a chunk, each the other's overlap."""
    epub = tmp_path / "book.epub"
    if not epub.exists():
        write_minimal_epub(epub)
    chunk_set = chunk_book(read_book_structured(epub), target_words=1, overlap_paragraphs=1)
    assert [c.id for c in chunk_set.chunks] == ["ch01_c01", "ch01_c02"]
    return chunk_set


def _translate_in_order(provider, cache, tmp_path, order):
    chunk_set = _two_chunks(tmp_path)
    stats = TranslateStats(chunks_total=2)
    translator = _translator(provider, cache)
    for index in order:
        translator.translate_chunk(chunk_set, chunk_set.chunks[index], stats)
    return chunk_set


ONE_PARAGRAPH = "===PARAGRAPH 1===\n<p>Перевод этого абзаца, уже на русском языке.</p>"


def test_overlap_stays_source_text_after_the_neighbour_is_translated(cache, tmp_path):
    provider = MagicMock()
    provider.complete.return_value = _reply(ONE_PARAGRAPH)
    chunk_set = _translate_in_order(provider, cache, tmp_path, [0])

    second = chunk_set.chunks[1]
    assert _texts(chunk_set)[0].startswith("Перевод"), "the neighbour really was spliced"
    assert "vestigium" in chunk_set.render_overlap(second, "prev")[0]
    assert "Перевод" not in chunk_set.render_overlap(second, "prev")[0]


@pytest.mark.parametrize("first_run_order", [[0, 1], [1, 0]])
def test_a_second_run_finds_every_chunk_whichever_finished_first(cache, tmp_path, first_run_order):
    """Parallel chunks finish in any order; the second run must not pay for any of them."""
    provider = MagicMock()
    provider.complete.return_value = _reply(ONE_PARAGRAPH)
    _translate_in_order(provider, cache, tmp_path, first_run_order)
    assert provider.complete.call_count == 2

    rerun = MagicMock()
    stats = _translator(rerun, cache).translate_book(_two_chunks(tmp_path))

    rerun.complete.assert_not_called()
    assert stats.chunks_cached == 2
