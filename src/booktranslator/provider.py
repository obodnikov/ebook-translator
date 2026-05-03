"""OpenRouter LLM provider via the OpenAI-compatible SDK."""

from __future__ import annotations

import os
from dataclasses import dataclass

from openai import OpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


@dataclass
class CompletionResult:
    text: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    model: str
    raw: dict


class OpenRouterProvider:
    """Thin wrapper around the OpenAI SDK pointed at OpenRouter."""

    def __init__(
        self,
        api_key: str | None = None,
        app_name: str | None = None,
        site_url: str | None = None,
    ):
        key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set. "
                "Add it to your .env file or environment."
            )

        default_headers: dict[str, str] = {}
        if site_url or os.environ.get("OPENROUTER_SITE_URL"):
            default_headers["HTTP-Referer"] = (
                site_url or os.environ["OPENROUTER_SITE_URL"]
            )
        if app_name or os.environ.get("OPENROUTER_APP_NAME"):
            default_headers["X-OpenRouter-Title"] = (
                app_name or os.environ["OPENROUTER_APP_NAME"]
            )

        self.client = OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=key,
            default_headers=default_headers or None,
        )

    @retry(
        reraise=True,
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=2, max=60),
    )
    def complete(
        self,
        model: str,
        system: str,
        user: str,
        *,
        temperature: float = 0.3,
        max_tokens: int | None = None,
        response_format: dict | None = None,
    ) -> CompletionResult:
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})

        kwargs: dict = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if response_format is not None:
            kwargs["response_format"] = response_format

        response = self.client.chat.completions.create(**kwargs)

        choice = response.choices[0]
        text = choice.message.content or ""

        usage = response.usage
        input_tokens = getattr(usage, "prompt_tokens", 0) or 0
        output_tokens = getattr(usage, "completion_tokens", 0) or 0
        total_tokens = getattr(usage, "total_tokens", input_tokens + output_tokens)

        return CompletionResult(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            model=response.model or model,
            raw=response.model_dump() if hasattr(response, "model_dump") else {},
        )
