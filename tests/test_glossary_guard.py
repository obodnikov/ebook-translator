"""Tests for glossary.py bookend guard and error handling.

Covers:
- TruncationError on wrong bookstart (truncation from beginning)
- TruncationError on wrong bookend (truncation from end)
- TruncationError on missing markers (top-level list response)
- Clear error message on empty content (with finish_reason hint)
- Happy path: correct bookstart+bookend -> glossary parsed, cache written
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from booktranslator.glossary import TruncationError, extract_glossary
from booktranslator.models import Glossary
from booktranslator.provider import CompletionResult

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def prompt_path() -> Path:
    return Path(__file__).parent.parent / "prompts" / "glossary_extract.md"


@pytest.fixture
def mock_book():
    book = MagicMock()
    book.full_text.return_value = "Once upon a time there lived a wizard named Merlin."
    book.meta.title = "Test Book"
    book.meta.author = "Test Author"
    return book


@pytest.fixture
def mock_cache(tmp_path):
    from booktranslator.cache import Cache

    return Cache(tmp_path / "test.sqlite")


def _make_provider(response_text: str, finish_reason: str = "stop") -> MagicMock:
    """Return a mock OpenRouterProvider whose complete() returns the given text."""
    result = CompletionResult(
        text=response_text,
        input_tokens=100,
        output_tokens=50,
        total_tokens=150,
        model="anthropic/claude-sonnet-4.6",
        raw={},
        finish_reason=finish_reason,
    )
    provider = MagicMock()
    provider.complete.return_value = result
    return provider


def _make_response(
    bookstart: str | None,
    bookend: str | None,
    entries: list[dict],
) -> str:
    """Build a JSON response string as the model would return it."""
    if bookstart is None and bookend is None:
        # Top-level list (old format, no markers)
        return json.dumps(entries)
    data: dict = {"entries": entries}
    if bookstart is not None:
        data["bookstart"] = bookstart
    if bookend is not None:
        data["bookend"] = bookend
    return json.dumps(data)


SAMPLE_ENTRIES = [
    {
        "original": "Merlin",
        "translation": "Мерлин",
        "type": "person",
        "gender": "m",
        "plural": None,
        "notes": None,
    }
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestTruncationDetected:
    """Wrong bookstart or bookend raises TruncationError; cache not written."""

    def test_wrong_bookend_raises(self, mock_book, mock_cache, prompt_path):
        """Wrong bookend (truncation from end) raises TruncationError."""
        # bookstart correct, bookend wrong
        wrong_response = _make_response("aabbccdd", "deadbeef", SAMPLE_ENTRIES)
        provider = _make_provider(wrong_response)

        with (
            patch(
                "booktranslator.glossary.secrets.token_hex",
                side_effect=["aabbccdd", "11223344"],
            ),
            pytest.raises(TruncationError, match="bookend"),
        ):
            extract_glossary(
                book=mock_book,
                provider=provider,
                prompt_path=prompt_path,
                cache=mock_cache,
            )

    def test_wrong_bookstart_raises(self, mock_book, mock_cache, prompt_path):
        """Wrong bookstart (truncation from beginning) raises TruncationError."""
        # bookstart wrong, bookend correct
        wrong_response = _make_response("deadbeef", "11223344", SAMPLE_ENTRIES)
        provider = _make_provider(wrong_response)

        with (
            patch(
                "booktranslator.glossary.secrets.token_hex",
                side_effect=["aabbccdd", "11223344"],
            ),
            pytest.raises(TruncationError, match="bookstart"),
        ):
            extract_glossary(
                book=mock_book,
                provider=provider,
                prompt_path=prompt_path,
                cache=mock_cache,
            )

    def test_wrong_bookend_cache_not_written(self, mock_book, mock_cache, prompt_path):
        wrong_response = _make_response("aabbccdd", "deadbeef", SAMPLE_ENTRIES)
        provider = _make_provider(wrong_response)

        with (
            patch(
                "booktranslator.glossary.secrets.token_hex",
                side_effect=["aabbccdd", "11223344"],
            ),
            pytest.raises(TruncationError),
        ):
            extract_glossary(
                book=mock_book,
                provider=provider,
                prompt_path=prompt_path,
                cache=mock_cache,
            )

        # Cache must be empty — truncated response must not be persisted
        all_keys = mock_cache.conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
        assert all_keys == 0


class TestMissingBookend:
    """Top-level list response (no markers) -> TruncationError."""

    def test_list_response_raises_truncation(self, mock_book, mock_cache, prompt_path):
        list_response = _make_response(None, None, SAMPLE_ENTRIES)  # top-level list
        provider = _make_provider(list_response)

        with pytest.raises(TruncationError):
            extract_glossary(
                book=mock_book,
                provider=provider,
                prompt_path=prompt_path,
                cache=mock_cache,
            )

    def test_list_response_cache_not_written(self, mock_book, mock_cache, prompt_path):
        list_response = _make_response(None, None, SAMPLE_ENTRIES)
        provider = _make_provider(list_response)

        with pytest.raises(TruncationError):
            extract_glossary(
                book=mock_book,
                provider=provider,
                prompt_path=prompt_path,
                cache=mock_cache,
            )

        all_keys = mock_cache.conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
        assert all_keys == 0


class TestEmptyContentMessage:
    """Empty model response raises ValueError with finish_reason hint, not JSON error."""

    def test_empty_content_clear_error(self, mock_book, mock_cache, prompt_path):
        provider = _make_provider("", finish_reason="stop")

        with pytest.raises(ValueError, match="empty content") as exc_info:
            extract_glossary(
                book=mock_book,
                provider=provider,
                prompt_path=prompt_path,
                cache=mock_cache,
            )

        # Must NOT be a bare JSON parse error
        assert "Expecting value" not in str(exc_info.value)

    def test_empty_content_includes_finish_reason(self, mock_book, mock_cache, prompt_path):
        provider = _make_provider("", finish_reason="stop")

        with pytest.raises(ValueError, match="stop"):
            extract_glossary(
                book=mock_book,
                provider=provider,
                prompt_path=prompt_path,
                cache=mock_cache,
            )

    def test_whitespace_only_content_treated_as_empty(self, mock_book, mock_cache, prompt_path):
        provider = _make_provider("   \n  ", finish_reason="length")

        with pytest.raises(ValueError, match="empty content"):
            extract_glossary(
                book=mock_book,
                provider=provider,
                prompt_path=prompt_path,
                cache=mock_cache,
            )


class TestBookendOk:
    """Correct bookstart+bookend -> glossary parsed, cache written, nonces not in entries."""

    def test_correct_markers_returns_glossary(self, mock_book, mock_cache, prompt_path):
        good_response = _make_response("aabbccdd", "11223344", SAMPLE_ENTRIES)
        provider = _make_provider(good_response)

        with patch(
            "booktranslator.glossary.secrets.token_hex",
            side_effect=["aabbccdd", "11223344"],
        ):
            glossary, result, raw = extract_glossary(
                book=mock_book,
                provider=provider,
                prompt_path=prompt_path,
                cache=mock_cache,
            )

        assert isinstance(glossary, Glossary)
        assert len(glossary.entries) == 1
        assert glossary.entries[0].original == "Merlin"

    def test_correct_markers_writes_cache(self, mock_book, mock_cache, prompt_path):
        good_response = _make_response("aabbccdd", "11223344", SAMPLE_ENTRIES)
        provider = _make_provider(good_response)

        with patch(
            "booktranslator.glossary.secrets.token_hex",
            side_effect=["aabbccdd", "11223344"],
        ):
            extract_glossary(
                book=mock_book,
                provider=provider,
                prompt_path=prompt_path,
                cache=mock_cache,
            )

        count = mock_cache.conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
        assert count == 1

    def test_markers_not_in_entries(self, mock_book, mock_cache, prompt_path):
        """The nonce values must not leak into glossary entries."""
        good_response = _make_response("aabbccdd", "11223344", SAMPLE_ENTRIES)
        provider = _make_provider(good_response)

        with patch(
            "booktranslator.glossary.secrets.token_hex",
            side_effect=["aabbccdd", "11223344"],
        ):
            glossary, _, _ = extract_glossary(
                book=mock_book,
                provider=provider,
                prompt_path=prompt_path,
                cache=mock_cache,
            )

        originals = [e.original for e in glossary.entries]
        assert not any("aabbccdd" in o or "11223344" in o for o in originals)

    def test_cache_hit_skips_provider(self, mock_book, mock_cache, prompt_path):
        """Second call with same inputs uses cache without patching nonce.

        The cache key is built from the prompt rendered WITHOUT the nonces,
        so it is stable across calls even though each live call uses fresh
        random nonces. The second call must be a cache hit.
        """
        good_response = _make_response("aabbccdd", "11223344", SAMPLE_ENTRIES)
        provider = _make_provider(good_response)

        # First call: live provider call, nonces patched to match the response.
        with patch(
            "booktranslator.glossary.secrets.token_hex",
            side_effect=["aabbccdd", "11223344"],
        ):
            extract_glossary(
                book=mock_book,
                provider=provider,
                prompt_path=prompt_path,
                cache=mock_cache,
            )

        # Second call: no nonce patch needed — cache key is stable, so this
        # hits the cache without calling the provider again.
        extract_glossary(
            book=mock_book,
            provider=provider,
            prompt_path=prompt_path,
            cache=mock_cache,
        )

        # Provider called only once (first call); second was a cache hit.
        assert provider.complete.call_count == 1


class TestPromptBookendContract:
    """Assert that the glossary prompt enforces the bookend echo contract."""

    def test_prompt_requires_bookend_field(self, prompt_path):
        """The rendered prompt must instruct the model to return bookstart and bookend fields."""
        from booktranslator.prompts import load_prompt, render_prompt

        prompt = load_prompt(prompt_path)
        context = {
            "title": "Test",
            "author": "Author",
            "source_lang": "en",
            "source_lang_name": "English",
            "target_lang": "ru",
            "target_lang_name": "Russian",
            "book_text": "[[BOOKSTART::aabb]] Some text [[BOOKEND::ccdd]]",
            "known_terms": "",
        }
        system, user = render_prompt(prompt, context)
        full_prompt = system + "\n" + user

        assert "bookstart" in full_prompt.lower(), (
            "Glossary prompt must instruct model to return a 'bookstart' field"
        )
        assert "bookend" in full_prompt.lower(), (
            "Glossary prompt must instruct model to return a 'bookend' field"
        )

    def test_prompt_requires_object_not_list(self, prompt_path):
        """The rendered prompt must require object response, not a top-level list."""
        from booktranslator.prompts import load_prompt, render_prompt

        prompt = load_prompt(prompt_path)
        context = {
            "title": "Test",
            "author": "Author",
            "source_lang": "en",
            "source_lang_name": "English",
            "target_lang": "ru",
            "target_lang_name": "Russian",
            "book_text": "Some text [[BOOKEND::aabbccdd]]",
            "known_terms": "",
        }
        system, user = render_prompt(prompt, context)
        full_prompt = system + "\n" + user

        # Must mention entries array inside an object
        assert '"entries"' in full_prompt, (
            'Glossary prompt must require {"bookend": ..., "entries": [...]} format'
        )

    def test_prompt_reasoning_effort_is_none(self, prompt_path):
        """The glossary prompt must have reasoning_effort: none to avoid empty content."""
        from booktranslator.prompts import load_prompt

        prompt = load_prompt(prompt_path)
        assert prompt.reasoning_effort == "none", (
            "glossary_extract.md must set reasoning_effort: none"
        )
