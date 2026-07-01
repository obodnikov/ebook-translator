"""Tests for reasoning_effort passthrough in OpenRouterProvider.complete().

Verifies:
- reasoning_effort is passed via extra_body when set
- extra_body is absent when reasoning_effort is None
- finish_reason is populated in CompletionResult
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from booktranslator.provider import OpenRouterProvider


def _make_provider() -> OpenRouterProvider:
    with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
        return OpenRouterProvider()


def _make_mock_response(
    content: str = "hello",
    finish_reason: str = "stop",
    model: str = "test-model",
) -> MagicMock:
    """Build a minimal mock that mimics openai.ChatCompletion structure."""
    choice = MagicMock()
    choice.message.content = content
    choice.finish_reason = finish_reason

    usage = MagicMock()
    usage.prompt_tokens = 10
    usage.completion_tokens = 5
    usage.total_tokens = 15

    response = MagicMock()
    response.choices = [choice]
    response.usage = usage
    response.model = model
    response.model_dump.return_value = {}
    return response


class TestReasoningEffortPassthrough:
    def test_reasoning_effort_none_no_extra_body(self):
        """When reasoning_effort is None, extra_body must not be sent."""
        provider = _make_provider()
        mock_resp = _make_mock_response()

        with patch.object(
            provider.client.chat.completions, "create", return_value=mock_resp
        ) as mock_create:
            provider.complete("m", "sys", "user", reasoning_effort=None)

        call_kwargs = mock_create.call_args[1]
        assert "extra_body" not in call_kwargs

    def test_reasoning_effort_none_default(self):
        """Default call (no reasoning_effort arg) also omits extra_body."""
        provider = _make_provider()
        mock_resp = _make_mock_response()

        with patch.object(
            provider.client.chat.completions, "create", return_value=mock_resp
        ) as mock_create:
            provider.complete("m", "sys", "user")

        call_kwargs = mock_create.call_args[1]
        assert "extra_body" not in call_kwargs

    def test_reasoning_effort_none_string_sends_extra_body(self):
        """reasoning_effort='none' sends extra_body={'reasoning_effort': 'none'}."""
        provider = _make_provider()
        mock_resp = _make_mock_response()

        with patch.object(
            provider.client.chat.completions, "create", return_value=mock_resp
        ) as mock_create:
            provider.complete("m", "sys", "user", reasoning_effort="none")

        call_kwargs = mock_create.call_args[1]
        assert call_kwargs["extra_body"] == {"reasoning_effort": "none"}

    def test_reasoning_effort_low_sends_extra_body(self):
        """reasoning_effort='low' is forwarded correctly."""
        provider = _make_provider()
        mock_resp = _make_mock_response()

        with patch.object(
            provider.client.chat.completions, "create", return_value=mock_resp
        ) as mock_create:
            provider.complete("m", "sys", "user", reasoning_effort="low")

        call_kwargs = mock_create.call_args[1]
        assert call_kwargs["extra_body"] == {"reasoning_effort": "low"}

    def test_finish_reason_populated(self):
        """finish_reason from the API response is stored in CompletionResult."""
        provider = _make_provider()
        mock_resp = _make_mock_response(finish_reason="stop")

        with patch.object(provider.client.chat.completions, "create", return_value=mock_resp):
            result = provider.complete("m", "sys", "user")

        assert result.finish_reason == "stop"

    def test_finish_reason_length(self):
        """finish_reason='length' (output truncated) is preserved."""
        provider = _make_provider()
        mock_resp = _make_mock_response(finish_reason="length")

        with patch.object(provider.client.chat.completions, "create", return_value=mock_resp):
            result = provider.complete("m", "sys", "user")

        assert result.finish_reason == "length"

    def test_finish_reason_missing_is_none(self):
        """If finish_reason attribute is absent on choice, result is None."""
        provider = _make_provider()
        mock_resp = _make_mock_response()
        # Remove finish_reason attribute entirely
        del mock_resp.choices[0].finish_reason

        with patch.object(provider.client.chat.completions, "create", return_value=mock_resp):
            result = provider.complete("m", "sys", "user")

        assert result.finish_reason is None

    def test_result_text_populated(self):
        """Sanity: text content is still returned correctly."""
        provider = _make_provider()
        mock_resp = _make_mock_response(content="translated text")

        with patch.object(provider.client.chat.completions, "create", return_value=mock_resp):
            result = provider.complete("m", "sys", "user")

        assert result.text == "translated text"
