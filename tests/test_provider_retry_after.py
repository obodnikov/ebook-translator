"""Tests for waiting as long as the provider asks before a retry.

OpenRouter answers 402 `in_flight_budget_exhausted` when the requests already
running have reserved the account's credit, and says to retry after 120 s. The
old exponential backoff spent all five attempts in about half a minute, and the
proofread chunks of Foxglove Summer were lost with the wait not half over.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest
from openai import APIStatusError

from booktranslator.provider import (
    MAX_RETRY_AFTER_SECONDS,
    _wait_before_retry,
    retry_after_seconds,
)

IN_FLIGHT_BODY = {
    "message": "This request would exceed your available credits given your current "
    "in-flight requests.",
    "code": 402,
    "metadata": {
        "reason": "in_flight_budget_exhausted",
        "limit_source": "openrouter_in_flight_budget",
        "headers": {"Retry-After": "120"},
    },
}


def _status_error(status: int, headers: dict | None = None, body: object = None) -> APIStatusError:
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    response = httpx.Response(status, headers=headers or {}, request=request)
    return APIStatusError(f"Error code: {status}", response=response, body=body)


def _state(error: BaseException | None, attempt: int = 1) -> SimpleNamespace:
    outcome = MagicMock()
    outcome.exception.return_value = error
    return SimpleNamespace(outcome=outcome, attempt_number=attempt)


def test_reads_retry_after_from_the_openrouter_body():
    assert retry_after_seconds(_status_error(402, body=IN_FLIGHT_BODY)) == 120.0


def test_header_is_read_too():
    assert retry_after_seconds(_status_error(429, headers={"Retry-After": "7"})) == 7.0


def test_an_absurd_wait_is_capped():
    error = _status_error(402, headers={"Retry-After": "86400"})
    assert retry_after_seconds(error) == MAX_RETRY_AFTER_SECONDS


@pytest.mark.parametrize(
    "error",
    [
        _status_error(402, body={"message": "Insufficient credits", "code": 402}),
        _status_error(503, headers={"Retry-After": "soon"}),
        ConnectionError("dropped"),
        None,
    ],
)
def test_no_usable_wait_means_none(error):
    assert retry_after_seconds(error) is None


def test_waits_the_time_the_provider_asked_for():
    assert _wait_before_retry(_state(_status_error(402, body=IN_FLIGHT_BODY))) == 120.0


def test_other_failures_keep_the_exponential_backoff():
    first = _wait_before_retry(_state(ConnectionError("dropped"), attempt=1))
    later = _wait_before_retry(_state(ConnectionError("dropped"), attempt=4))
    assert 2 <= first < later <= 60


def test_complete_sleeps_the_asked_time_then_succeeds(monkeypatch):
    from booktranslator.provider import OpenRouterProvider
    from tests.test_provider_response_guard import _make_mock_response, _make_provider

    slept: list[float] = []
    monkeypatch.setattr(OpenRouterProvider.complete.retry, "sleep", slept.append)
    provider = _make_provider()
    provider.client.chat.completions.create = MagicMock(
        side_effect=[_status_error(402, body=IN_FLIGHT_BODY), _make_mock_response("готово")]
    )

    result = provider.complete(model="m", system="s", user="u")

    assert result.text == "готово"
    assert slept == [120.0]
