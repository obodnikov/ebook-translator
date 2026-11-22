"""Tests for running stages at a chosen reasoning level on kiro-gateway.

Background (docs/design/2026-11-01-native-reasoning-on-gateway.md): the level now
reaches the model, so it has to be chosen per stage and per endpoint, must not
serve answers cached at another level, and has failure modes of its own: a
request that outlasts the client, a model that cannot stop reasoning, a refusal.
"""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest
from openai import APITimeoutError
from pydantic import ValidationError

from booktranslator.glossary import extract_glossary
from booktranslator.judge import Judge
from booktranslator.models import Config, ProviderConfig, ProvidersConfig, ReasoningConfig
from booktranslator.pipeline_helpers import create_stage_provider
from booktranslator.postprocess import PostProcessor
from booktranslator.prompts import load_prompt
from booktranslator.provider import (
    CompletionTimeoutError,
    EmptyCompletionError,
    OpenRouterProvider,
    RefusalError,
    credits_used,
)
from booktranslator.reflect import Reflector
from booktranslator.repair import Repairer
from booktranslator.translator import Translator

PROMPTS = Path(__file__).resolve().parent.parent / "prompts"


def _provider(**kwargs) -> OpenRouterProvider:
    with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
        return OpenRouterProvider(**kwargs)


def _response(
    content: str = "answer",
    finish_reason: str = "stop",
    reasoning_content: str | None = None,
    refusal: str | None = None,
    credits: float | None = None,
) -> MagicMock:
    choice = MagicMock()
    choice.message.content = content
    choice.finish_reason = finish_reason
    # MagicMock invents attributes, so absent fields must be set to None
    choice.message.reasoning_content = reasoning_content
    choice.message.reasoning = None
    choice.message.refusal = refusal

    usage = MagicMock()
    usage.prompt_tokens = 10
    usage.completion_tokens = 5
    usage.total_tokens = 15
    usage.credits_used = credits

    response = MagicMock()
    response.choices = [choice]
    response.usage = usage
    response.model = "test-model"
    response.model_dump.return_value = {}
    return response


def _complete(provider: OpenRouterProvider, response, **kwargs):
    with patch.object(provider.client.chat.completions, "create", return_value=response) as create:
        return provider.complete(model="m", system="s", user="u", **kwargs), create


class TestClientSettings:
    def test_the_sdk_does_not_retry_on_its_own(self):
        """Its two retries inside our five attempts made 15 requests out of one call."""
        assert _provider().client.max_retries == 0

    def test_timeout_comes_from_the_config(self):
        cfg = Config(
            providers=ProvidersConfig(
                text=ProviderConfig(api_key_env="OPENROUTER_API_KEY", timeout_seconds=123.0)
            )
        )
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
            provider = create_stage_provider(cfg, "translate")
        assert provider.client.timeout == 123.0


class TestTimeout:
    def test_is_final_and_names_what_to_change(self):
        provider = _provider(timeout_seconds=600.0)
        timeout = APITimeoutError(request=httpx.Request("POST", "http://gateway/v1"))
        with (
            patch.object(provider.client.chat.completions, "create", side_effect=timeout) as create,
            pytest.raises(CompletionTimeoutError) as exc,
        ):
            provider.complete(model="m", system="s", user="u", reasoning_effort="max")
        assert create.call_count == 1
        assert "600" in str(exc.value)
        assert "'max'" in str(exc.value)
        assert "timeout_seconds" in str(exc.value)


class TestRefusal:
    def test_reports_the_explanation_and_is_not_retried(self):
        provider = _provider()
        response = _response(
            content="",
            finish_reason="content_filter",
            refusal="The selected model cannot continue. (Kiro refusal category: CYBER)",
        )
        with (
            patch.object(
                provider.client.chat.completions, "create", return_value=response
            ) as create,
            pytest.raises(RefusalError) as exc,
        ):
            provider.complete(model="m", system="s", user="u")
        assert create.call_count == 1
        assert "CYBER" in str(exc.value)
        assert "target_words" not in str(exc.value)

    def test_is_an_empty_completion_to_the_callers(self):
        """Stages already handle an empty answer: a refusal must take the same path."""
        assert issubclass(RefusalError, EmptyCompletionError)

    def test_content_filter_without_an_explanation(self):
        provider = _provider()
        with pytest.raises(RefusalError, match="no explanation given"):
            _complete(provider, _response(content="", finish_reason="content_filter"))

    def test_an_empty_answer_without_a_refusal_is_not_called_one(self):
        provider = _provider()
        with pytest.raises(EmptyCompletionError) as exc:
            _complete(provider, _response(content="", reasoning_content="y" * 30000))
        assert not isinstance(exc.value, RefusalError)
        assert "30000" in str(exc.value)


class TestReasoningDespiteNone:
    def test_warns_once_per_model(self, caplog):
        provider = _provider()
        response = _response(reasoning_content="thinking anyway")
        with caplog.at_level(logging.WARNING):
            _complete(provider, response, reasoning_effort="none")
            _complete(provider, response, reasoning_effort="none")
        assert caplog.text.count("despite reasoning_effort=none") == 1

    def test_silent_when_the_model_obeys(self, caplog):
        with caplog.at_level(logging.WARNING):
            _complete(_provider(), _response(), reasoning_effort="none")
        assert caplog.text == ""

    def test_silent_when_reasoning_was_asked_for(self, caplog):
        with caplog.at_level(logging.WARNING):
            _complete(_provider(), _response(reasoning_content="ok"), reasoning_effort="low")
        assert caplog.text == ""


class TestCredits:
    def test_adds_up_what_the_provider_reports(self):
        before = credits_used()
        provider = _provider()
        _complete(provider, _response(credits=0.5))
        _complete(provider, _response(credits=0.25))
        assert credits_used() == pytest.approx(before + 0.75)

    def test_counts_a_call_that_came_back_empty(self):
        """It was paid for, even though it produced nothing."""
        before = credits_used()
        with pytest.raises(EmptyCompletionError):
            _complete(_provider(), _response(content="", credits=2.5))
        assert credits_used() == pytest.approx(before + 2.5)

    def test_no_report_changes_nothing(self):
        before = credits_used()
        _complete(_provider(), _response(credits=None))
        assert credits_used() == before


class TestLoadPromptLevel:
    @pytest.fixture
    def prompt_path(self, tmp_path) -> Path:
        path = tmp_path / "p.md"
        path.write_text("---\nversion: 3\nreasoning_effort: none\n---\n\n# User\n\nhi\n")
        return path

    def test_without_a_level_the_prompt_is_as_written(self, prompt_path):
        prompt = load_prompt(prompt_path)
        assert prompt.reasoning_effort == "none"
        assert prompt.version == "3"

    def test_a_level_replaces_the_frontmatter_and_joins_the_version(self, prompt_path):
        """The version is part of the cache key, so no answer from another level is reused."""
        prompt = load_prompt(prompt_path, "high")
        assert prompt.reasoning_effort == "high"
        assert prompt.version == "3+high"

    def test_the_same_level_keeps_existing_cache_rows_valid(self, prompt_path):
        assert load_prompt(prompt_path, "none").version == "3"


class TestStagesTakeTheLevel:
    @pytest.mark.parametrize(
        "build",
        [
            lambda: Translator(
                MagicMock(), PROMPTS / "translate.md", MagicMock(), None, model="m",
                reasoning_effort="xhigh",
            ).prompt,
            lambda: Judge(
                MagicMock(), PROMPTS / "judge.md", MagicMock(), None, model="m",
                reasoning_effort="xhigh",
            ).prompt,
            lambda: PostProcessor(
                MagicMock(), PROMPTS / "style.md", MagicMock(), None, stage="style", model="m",
                reasoning_effort="xhigh",
            ).prompt,
            lambda: Repairer(
                MagicMock(), PROMPTS / "repair.md", MagicMock(), None, model="m",
                reasoning_effort="xhigh",
            ).prompt,
        ],
        ids=["translate", "judge", "style", "repair"],
    )  # fmt: skip
    def test_stage(self, build):
        prompt = build()
        assert prompt.reasoning_effort == "xhigh"
        assert prompt.version.endswith("+xhigh")

    def test_reflect_applies_it_to_both_of_its_prompts(self):
        reflector = Reflector(
            MagicMock(),
            PROMPTS / "reflect.md",
            PROMPTS / "translate.md",
            MagicMock(),
            None,
            model="m",
            reasoning_effort="high",
        )
        assert reflector.reflect_prompt.reasoning_effort == "high"
        assert reflector.translate_prompt.reasoning_effort == "high"

    def test_glossary_sends_it(self):
        provider = MagicMock()
        provider.complete.return_value = MagicMock(text="not json", finish_reason="stop")
        book = MagicMock()
        book.meta.title = "T"
        book.meta.author = "A"
        book.full_text.return_value = "book text"
        cache = MagicMock()
        cache.get.return_value = None
        with pytest.raises(ValueError):
            extract_glossary(
                book, provider, PROMPTS / "glossary_extract.md", cache, reasoning_effort="high"
            )
        assert provider.complete.call_args.kwargs["reasoning_effort"] == "high"


class TestReasoningConfig:
    def test_unset_leaves_the_prompts_alone(self):
        assert Config().reasoning.translate is None

    def test_a_typo_in_a_stage_name_fails_loudly(self):
        with pytest.raises(ValidationError):
            ReasoningConfig.model_validate({"judgee": "high"})

    def test_a_typo_in_a_level_fails_loudly(self):
        with pytest.raises(ValidationError):
            ReasoningConfig.model_validate({"judge": "hgih"})


class TestTranslatePromptKeepsParagraphsWhole:
    def test_forbids_splitting_a_paragraph(self):
        """Sonnet 5.5 split dialogue off into a second element, breaking the count."""
        prompt = load_prompt(PROMPTS / "translate.md")
        assert "Never split a paragraph" in prompt.system_tmpl
        assert int(prompt.version) >= 4
