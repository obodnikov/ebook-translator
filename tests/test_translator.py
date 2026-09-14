"""Tests for translator.py: a reply is checked in full before it is cached or spliced.

The broken replies below are shaped after the three chunks of Foxglove Summer
that passed the paragraph count and still could not go into the book.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from booktranslator.cache import Cache
from booktranslator.chunker import chunk_book
from booktranslator.epub_io import read_book_structured
from booktranslator.prompts import render_prompt
from booktranslator.provider import CompletionResult
from booktranslator.translator import TranslateStats, Translator
from tests.epub_fixtures import write_minimal_epub

TRANSLATE_PROMPT = Path(__file__).resolve().parents[1] / "prompts" / "translate.md"

GOOD = "===PARAGRAPH 1===\n<p>Абзац про вестигиум.</p>\n===PARAGRAPH 2===\n<p>Абзац с Сиуоллом.</p>"

BROKEN = {
    # ch04_c06: no closing tag anywhere.
    "unclosed": (
        "===PARAGRAPH 1===\n<p>Абзац про вестигиум.\n===PARAGRAPH 2===\n<p>Абзац с Сиуоллом."
    ),
    # ch09_c01: two paragraphs under one marker.
    "two_under_one": (
        "===PARAGRAPH 1===\n<p>Абзац про вестигиум.</p>\n<p>Лишний.</p>\n"
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

    with pytest.raises(ValueError, match="Paragraph 1 of 2"):
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
