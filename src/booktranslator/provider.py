"""OpenAI-compatible LLM provider with configurable endpoint."""

from __future__ import annotations

import base64
import logging
import os
from dataclasses import dataclass, field

from openai import OpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"

# Warn once a response gets this close to the provider's response ceiling.
# There is no earlier signal: kiro-gateway returns finish_reason="stop" even
# when it truncates (docs/design/2026-08-03-...md §4.4).
RESPONSE_SIZE_WARN_RATIO = 0.85


class EmptyCompletionError(ValueError):
    """The provider returned no content.

    Permanent, not transient: retrying the identical request cannot help.
    The usual cause is the response ceiling — injected reasoning is ordinary
    output and competes with the answer for the same budget, so a verbose
    reasoning pass can leave nothing for the answer itself.

    Subclasses ValueError because that is what the stages raised for this
    failure before the check moved here, and callers catch it that way.
    """


@dataclass
class CompletionResult:
    text: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    model: str
    raw: dict
    finish_reason: str | None = None  # e.g. "stop", "length", "content_filter"
    # Reasoning text, when the provider exposes it. Diagnostics only — it is a
    # draft of the model's thinking, never the answer.
    reasoning: str = ""


@dataclass
class ImageGenerationResult:
    """Result of an image generation/edit call."""

    image_bytes: bytes
    mime_type: str
    text: str  # any accompanying text from the model
    model: str
    raw: dict = field(default_factory=dict)


class OpenRouterProvider:
    """Thin wrapper around the OpenAI SDK pointed at any OpenAI-compatible endpoint."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        app_name: str | None = None,
        site_url: str | None = None,
        api_key_env: str = "OPENROUTER_API_KEY",
        extra_headers: dict[str, str] | None = None,
        max_response_bytes: int | None = None,
    ):
        key = api_key or os.environ.get(api_key_env)
        if not key:
            raise RuntimeError(
                f"{api_key_env} is not set. Add it to your .env file or environment."
            )

        resolved_base_url = base_url or DEFAULT_BASE_URL

        default_headers: dict[str, str] = {}
        if extra_headers:
            default_headers.update(extra_headers)
        if site_url or os.environ.get("OPENROUTER_SITE_URL"):
            default_headers["HTTP-Referer"] = site_url or os.environ["OPENROUTER_SITE_URL"]
        if app_name or os.environ.get("OPENROUTER_APP_NAME"):
            default_headers["X-OpenRouter-Title"] = app_name or os.environ["OPENROUTER_APP_NAME"]

        self._api_key = key
        self._base_url = resolved_base_url
        self._extra_headers = default_headers.copy()
        self._max_response_bytes = max_response_bytes

        self.client = OpenAI(
            base_url=resolved_base_url,
            api_key=key,
            default_headers=default_headers or None,
        )

    @retry(
        reraise=True,
        # Retry anything transient, but not EmptyCompletionError: it is
        # permanent, and the identical request would fail the same way.
        retry=retry_if_not_exception_type(EmptyCompletionError),
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
        reasoning_effort: str | None = None,
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
        # Pass reasoning_effort via extra_body so it merges into the request
        # body as a non-standard field understood by kiro-gateway and some
        # OpenRouter models (e.g. reasoning_effort: "none" disables thinking).
        if reasoning_effort is not None:
            # Merge into extra_body rather than overwriting, so other callers
            # can also set extra_body fields without conflict.
            extra = dict(kwargs.get("extra_body") or {})
            extra["reasoning_effort"] = reasoning_effort
            kwargs["extra_body"] = extra

        response = self.client.chat.completions.create(**kwargs)

        choice = response.choices[0]
        message = choice.message
        text = message.content or ""
        finish_reason = getattr(choice, "finish_reason", None)
        # Providers disagree on the field name: kiro-gateway uses
        # `reasoning_content`, OpenRouter uses `reasoning`.
        reasoning = (
            getattr(message, "reasoning_content", None) or getattr(message, "reasoning", None) or ""
        )

        if not text.strip():
            # Do NOT fall back to `reasoning`: it is a draft of the model's
            # thinking, not the answer. Caching it would put nonsense into the
            # book. finish_reason is no help either — kiro-gateway reports
            # "stop" even when it truncates.
            raise EmptyCompletionError(
                f"{model} returned no content "
                f"(finish_reason={finish_reason!r}, reasoning={len(reasoning)} chars). "
                "Reasoning shares the response budget with the answer: shrink the chunk "
                "(chunker.target_words) or lower reasoning_effort in the prompt. "
                "Raising max_tokens helps on OpenRouter but not on kiro-gateway, "
                "which never forwards it upstream."
            )

        self._warn_if_near_response_ceiling(text, reasoning, model)

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
            finish_reason=finish_reason,
            reasoning=reasoning,
            raw=response.model_dump() if hasattr(response, "model_dump") else {},
        )

    def _warn_if_near_response_ceiling(self, text: str, reasoning: str, model: str) -> None:
        """Log a warning when a response approaches the configured size limit.

        `max_response_bytes` is a cautious threshold, not the provider's exact
        limit: a 23 644-byte response came back whole while a 26 825-byte one
        was cut off mid-JSON, and the real limit looks to be counted in tokens
        rather than bytes. Reasoning and answer share the budget, so both are
        counted here. Silence means nothing on its own — the next, slightly
        longer chunk may be the one that comes back empty.
        """
        if not self._max_response_bytes:
            return
        used = len(text.encode("utf-8")) + len(reasoning.encode("utf-8"))
        if used < self._max_response_bytes * RESPONSE_SIZE_WARN_RATIO:
            return
        logger.warning(
            "Response from %s is %d bytes, past the %d-byte warning threshold "
            "set for this provider (%d of them reasoning). Shrink "
            "chunker.target_words or lower reasoning_effort before responses "
            "start coming back empty.",
            model,
            used,
            self._max_response_bytes,
            len(reasoning.encode("utf-8")),
        )

    @retry(
        reraise=True,
        retry=retry_if_exception_type((TimeoutError, ConnectionError, OSError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=3, min=5, max=120),
    )
    def generate_image(
        self,
        model: str,
        prompt: str,
        *,
        input_image: bytes | None = None,
        input_mime_type: str = "image/jpeg",
        aspect_ratio: str = "2:3",
        image_size: str = "1K",
    ) -> ImageGenerationResult:
        """Generate or edit an image via OpenRouter's image generation API.

        Uses the chat completions endpoint with modalities=["image", "text"]
        as documented by OpenRouter.

        Retries only on transient failures (timeouts, connection errors, 429,
        5xx). Fails fast on permanent errors (400, 401, 403, 404).

        Args:
            model: OpenRouter model ID (e.g. google/gemini-3.1-flash-image-preview).
            prompt: Text prompt describing the desired image/edit.
            input_image: Optional source image bytes for editing.
            input_mime_type: MIME type of the input image.
            aspect_ratio: Output aspect ratio (default 2:3 for book covers).
            image_size: Output resolution (1K, 2K, 4K).

        Returns:
            ImageGenerationResult with the generated image bytes.
        """
        import httpx

        # Build the message content
        content: list[dict] = []

        if input_image is not None:
            b64_data = base64.b64encode(input_image).decode("ascii")
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{input_mime_type};base64,{b64_data}",
                    },
                }
            )

        content.append({"type": "text", "text": prompt})

        messages = [{"role": "user", "content": content}]

        # OpenRouter image generation uses a custom payload structure
        payload: dict = {
            "model": model,
            "messages": messages,
            "modalities": ["image", "text"],
            "image_config": {
                "aspect_ratio": aspect_ratio,
                "image_size": image_size,
            },
        }

        # Use httpx directly since the OpenAI SDK doesn't natively support
        # the modalities + image_config parameters for image generation.
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        # Propagate extra headers (referer, app name)
        if self._extra_headers:
            headers.update(self._extra_headers)

        with httpx.Client(timeout=180.0) as http:
            resp = http.post(
                f"{self._base_url}/chat/completions",
                json=payload,
                headers=headers,
            )
            # Distinguish transient vs permanent HTTP errors.
            # Transient (429, 5xx): will be retried by tenacity decorator.
            # Permanent (400, 401, 403, 404): fail fast with clear message.
            if resp.status_code == 429 or resp.status_code >= 500:
                raise TimeoutError(
                    f"Transient HTTP {resp.status_code} from OpenRouter "
                    f"(will retry): {resp.text[:200]}"
                )
            if resp.status_code >= 400:
                raise RuntimeError(
                    f"HTTP {resp.status_code} from OpenRouter (non-retryable): {resp.text[:500]}"
                )
            data = resp.json()

        # Parse the response — images come in choices[0].message.images
        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})
        images = message.get("images", [])
        text_content = message.get("content", "")

        if not images:
            raise RuntimeError(
                f"No images returned from {model}. "
                f"Response: {data.get('error', text_content or 'empty')}"
            )

        # First image — extract base64 data URL
        image_url = images[0].get("image_url", {}).get("url", "")
        if not image_url.startswith("data:"):
            raise RuntimeError(f"Unexpected image URL format from {model}: {image_url[:80]}...")

        # Parse data URL: data:<mime>;base64,<data>
        try:
            header, b64_payload = image_url.split(",", 1)
        except ValueError as e:
            raise RuntimeError(f"Malformed data URL from {model}: missing comma separator") from e

        try:
            mime = header.split(":")[1].split(";")[0]
        except (IndexError, ValueError) as e:
            raise RuntimeError(f"Malformed data URL header from {model}: {header[:80]}") from e

        # Validate MIME is an image type
        if not mime.startswith("image/"):
            raise RuntimeError(
                f"Non-image MIME type returned from {model}: {mime}. Expected image/* format."
            )

        # Decode base64 payload with explicit error handling
        if not b64_payload:
            raise RuntimeError(f"Empty image payload returned from {model}")

        try:
            image_bytes = base64.b64decode(b64_payload)
        except Exception as e:
            raise RuntimeError(f"Failed to decode base64 image from {model}: {e}") from e

        if not image_bytes:
            raise RuntimeError(f"Decoded image is empty (0 bytes) from {model}")

        return ImageGenerationResult(
            image_bytes=image_bytes,
            mime_type=mime,
            text=text_content or "",
            model=data.get("model", model),
            raw=data,
        )
