"""Tests for the empty-response guard and the response-ceiling warning.

Background (docs/design/2026-08-03-judge-repair-stage-and-reasoning-on-gateway.md §4):
kiro-gateway caps a response at roughly 23 KB, injected reasoning shares that
budget with the answer, and `finish_reason` comes back as "stop" even when the
answer was truncated away entirely. So an empty `content` is the only reliable
failure signal, and it must never fall back to the reasoning text.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest

from booktranslator.provider import EmptyCompletionError, OpenRouterProvider


def _make_provider(max_response_bytes: int | None = None) -> OpenRouterProvider:
    with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
        return OpenRouterProvider(max_response_bytes=max_response_bytes)


def _make_mock_response(
    content: str = "hello",
    finish_reason: str = "stop",
    reasoning_content: str | None = None,
    reasoning: str | None = None,
) -> MagicMock:
    choice = MagicMock()
    choice.message.content = content
    choice.finish_reason = finish_reason
    # MagicMock invents attributes, so absent fields must be set to None
    choice.message.reasoning_content = reasoning_content
    choice.message.reasoning = reasoning

    usage = MagicMock()
    usage.prompt_tokens = 10
    usage.completion_tokens = 5
    usage.total_tokens = 15

    response = MagicMock()
    response.choices = [choice]
    response.usage = usage
    response.model = "test-model"
    response.model_dump.return_value = {}
    return response


class TestEmptyResponse:
    def test_empty_content_raises(self):
        provider = _make_provider()
        response = _make_mock_response(content="", finish_reason="stop")
        with (
            patch.object(provider.client.chat.completions, "create", return_value=response),
            pytest.raises(EmptyCompletionError),
        ):
            provider.complete(model="m", system="s", user="u")

    def test_whitespace_only_content_raises(self):
        provider = _make_provider()
        response = _make_mock_response(content="   \n\t ")
        with (
            patch.object(provider.client.chat.completions, "create", return_value=response),
            pytest.raises(EmptyCompletionError),
        ):
            provider.complete(model="m", system="s", user="u")

    def test_reasoning_is_never_used_as_the_answer(self):
        """The thinking draft must not be passed off as the model's answer."""
        provider = _make_provider()
        response = _make_mock_response(
            content="", reasoning_content="Let me think about this at length..."
        )
        with (
            patch.object(provider.client.chat.completions, "create", return_value=response),
            pytest.raises(EmptyCompletionError) as exc,
        ):
            provider.complete(model="m", system="s", user="u")
        assert "Let me think" not in str(exc.value)

    def test_error_names_the_reasoning_size_and_finish_reason(self):
        """The message has to be actionable: gateway 'stop' hides the truncation."""
        provider = _make_provider()
        response = _make_mock_response(
            content="", finish_reason="stop", reasoning_content="x" * 22998
        )
        with (
            patch.object(provider.client.chat.completions, "create", return_value=response),
            pytest.raises(EmptyCompletionError) as exc,
        ):
            provider.complete(model="m", system="s", user="u")
        message = str(exc.value)
        assert "22998" in message
        assert "stop" in message
        assert "target_words" in message

    def test_not_retried(self):
        """A permanent failure must not be retried five times with backoff."""
        provider = _make_provider()
        response = _make_mock_response(content="")
        with (
            patch.object(
                provider.client.chat.completions, "create", return_value=response
            ) as create,
            pytest.raises(EmptyCompletionError),
        ):
            provider.complete(model="m", system="s", user="u")
        assert create.call_count == 1

    def test_transient_errors_are_still_retried(self):
        provider = _make_provider()
        ok = _make_mock_response(content="fine")
        with patch.object(
            provider.client.chat.completions,
            "create",
            side_effect=[ConnectionError("boom"), ok],
        ) as create:
            result = provider.complete(model="m", system="s", user="u")
        assert result.text == "fine"
        assert create.call_count == 2


class TestReasoningField:
    def test_reads_reasoning_content_field(self):
        """kiro-gateway names it `reasoning_content`."""
        provider = _make_provider()
        response = _make_mock_response(content="answer", reasoning_content="gateway thinking")
        with patch.object(provider.client.chat.completions, "create", return_value=response):
            result = provider.complete(model="m", system="s", user="u")
        assert result.reasoning == "gateway thinking"
        assert result.text == "answer"

    def test_reads_reasoning_field(self):
        """OpenRouter names it `reasoning`."""
        provider = _make_provider()
        response = _make_mock_response(content="answer", reasoning="openrouter thinking")
        with patch.object(provider.client.chat.completions, "create", return_value=response):
            result = provider.complete(model="m", system="s", user="u")
        assert result.reasoning == "openrouter thinking"

    def test_absent_reasoning_is_empty_string(self):
        provider = _make_provider()
        response = _make_mock_response(content="answer")
        with patch.object(provider.client.chat.completions, "create", return_value=response):
            result = provider.complete(model="m", system="s", user="u")
        assert result.reasoning == ""


class TestResponseCeilingWarning:
    def test_warns_when_near_the_ceiling(self, caplog):
        provider = _make_provider(max_response_bytes=1000)
        response = _make_mock_response(content="x" * 900)
        with (
            patch.object(provider.client.chat.completions, "create", return_value=response),
            caplog.at_level(logging.WARNING),
        ):
            provider.complete(model="m", system="s", user="u")
        assert "ceiling" in caplog.text
        assert "target_words" in caplog.text

    def test_reasoning_counts_towards_the_ceiling(self, caplog):
        """Reasoning and answer share one budget, so both must be counted."""
        provider = _make_provider(max_response_bytes=1000)
        response = _make_mock_response(content="x" * 100, reasoning_content="y" * 800)
        with (
            patch.object(provider.client.chat.completions, "create", return_value=response),
            caplog.at_level(logging.WARNING),
        ):
            provider.complete(model="m", system="s", user="u")
        assert "ceiling" in caplog.text

    def test_silent_when_well_under(self, caplog):
        provider = _make_provider(max_response_bytes=1000)
        response = _make_mock_response(content="x" * 100)
        with (
            patch.object(provider.client.chat.completions, "create", return_value=response),
            caplog.at_level(logging.WARNING),
        ):
            provider.complete(model="m", system="s", user="u")
        assert caplog.text == ""

    def test_silent_when_no_ceiling_configured(self, caplog):
        """OpenRouter has no known ceiling — it must not be warned about."""
        provider = _make_provider(max_response_bytes=None)
        response = _make_mock_response(content="x" * 100_000)
        with (
            patch.object(provider.client.chat.completions, "create", return_value=response),
            caplog.at_level(logging.WARNING),
        ):
            provider.complete(model="m", system="s", user="u")
        assert caplog.text == ""

    def test_counts_bytes_not_characters(self):
        """Cyrillic costs two bytes a character; the ceiling is in bytes."""
        provider = _make_provider(max_response_bytes=1000)
        # 400 Cyrillic characters = 800 bytes = 80% of the ceiling
        assert len(("я" * 400).encode("utf-8")) == 800
        response = _make_mock_response(content="я" * 400)
        with patch.object(provider.client.chat.completions, "create", return_value=response):
            result = provider.complete(model="m", system="s", user="u")
        assert result.text == "я" * 400
