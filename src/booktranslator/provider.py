"""OpenAI-compatible LLM provider with configurable endpoint."""

from __future__ import annotations

import base64
import logging
import os
import threading
from dataclasses import dataclass, field

from openai import APIStatusError, APITimeoutError, OpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"

# How long to wait for one answer unless the config says otherwise: the SDK's
# own default, made explicit because a timeout now ends the call.
DEFAULT_TIMEOUT_SECONDS = 600.0

# Warn once a response gets this close to a limit configured for the provider.
RESPONSE_SIZE_WARN_RATIO = 0.85


# Longest pause a server may ask for before a retry. OpenRouter asks for 120 s
# when the requests already running have reserved all the account's credit.
MAX_RETRY_AFTER_SECONDS = 300.0

_backoff = wait_exponential(multiplier=2, min=2, max=60)

_credits_lock = threading.Lock()
_credits_used = 0.0


def credits_used() -> float:
    """Credits this process has spent so far, for providers that report them.

    kiro-gateway does (`usage.credits_used`), and it is the one reliable cost
    figure there: its token counts read 0 on large requests. Calls that came
    back empty are included, they were paid for too.
    """
    with _credits_lock:
        return _credits_used


def _add_credits(usage: object) -> None:
    global _credits_used
    spent = getattr(usage, "credits_used", None)
    if isinstance(spent, int | float):
        with _credits_lock:
            _credits_used += spent


def retry_after_seconds(error: BaseException | None) -> float | None:
    """How long the server asked us to wait before retrying, if it said.

    Read from the `Retry-After` header, or from `metadata.headers` in the error
    body, where OpenRouter repeats it for its 402 `in_flight_budget_exhausted`.
    """
    if not isinstance(error, APIStatusError):
        return None
    value = error.response.headers.get("retry-after")
    if value is None and isinstance(error.body, dict):
        headers = (error.body.get("metadata") or {}).get("headers") or {}
        value = next((v for k, v in headers.items() if k.lower() == "retry-after"), None)
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return None
    return min(max(seconds, 0.0), MAX_RETRY_AFTER_SECONDS)


def _wait_before_retry(retry_state) -> float:
    """Wait as long as the server asked, or back off exponentially if it did not.

    Backing off 2-16 s is right for a dropped connection, but a server that
    says "retry after 120 s" gets asked five times inside half a minute and
    the chunk is lost with its wait not even half over.
    """
    error = retry_state.outcome.exception() if retry_state.outcome else None
    asked = retry_after_seconds(error)
    if asked is None:
        return _backoff(retry_state)
    logger.warning(
        "Provider asked to retry after %.0f s (HTTP %s); waiting before attempt %d.",
        asked,
        getattr(error, "status_code", "?"),
        retry_state.attempt_number + 1,
    )
    return asked


class EmptyCompletionError(ValueError):
    """The provider returned no content.

    Permanent, not transient: retrying the identical request cannot help.
    The usual cause is reasoning that never ended in an answer. Measured on the
    translate prompt: Sonnet 4.6 at `high` and Haiku 5.5 at `xhigh` and `max`
    wrote about 30 KB of reasoning and no answer, with finish_reason "stop".

    Subclasses ValueError because that is what the stages raised for this
    failure before the check moved here, and callers catch it that way.
    """


class RefusalError(EmptyCompletionError):
    """The provider declined the request and sent no answer.

    An empty answer whose cause is known, so it fails the same way, with the
    provider's own explanation instead of advice about reasoning.
    """


class CompletionTimeoutError(RuntimeError):
    """No answer arrived within the provider's timeout.

    Final, not retried: the cause is nearly always the reasoning level, so the
    same request would time out again, and the endpoint keeps working on an
    abandoned request. Sonnet 5 at `max` took 10 minutes per answer and paid for
    the attempt the client had given up on.
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
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
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
        self._timeout_seconds = timeout_seconds
        self._warned_about_none: set[str] = set()

        self.client = OpenAI(
            base_url=resolved_base_url,
            api_key=key,
            default_headers=default_headers or None,
            timeout=timeout_seconds,
            # Retries belong to `complete` alone. The SDK's own two, nested
            # inside its five attempts, made up to 15 requests out of one call.
            max_retries=0,
        )

    @retry(
        reraise=True,
        # Retry anything transient, but not these: they are permanent, and the
        # identical request would fail the same way.
        retry=retry_if_not_exception_type((EmptyCompletionError, CompletionTimeoutError)),
        stop=stop_after_attempt(5),
        wait=_wait_before_retry,
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
        # OpenRouter models. "none" disables thinking only on models that can
        # stop it; see _warn_if_reasoned_despite_none.
        if reasoning_effort is not None:
            # Merge into extra_body rather than overwriting, so other callers
            # can also set extra_body fields without conflict.
            extra = dict(kwargs.get("extra_body") or {})
            extra["reasoning_effort"] = reasoning_effort
            kwargs["extra_body"] = extra

        try:
            response = self.client.chat.completions.create(**kwargs)
        except APITimeoutError as e:
            raise CompletionTimeoutError(
                f"{model} did not answer within {self._timeout_seconds:.0f} s at "
                f"reasoning_effort={reasoning_effort!r}. Lower the level (`reasoning:` in the "
                "config) or raise timeout_seconds for this provider."
            ) from e

        usage = response.usage
        _add_credits(usage)

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
            refusal = getattr(message, "refusal", None)
            if not isinstance(refusal, str):
                refusal = ""
            if refusal or finish_reason == "content_filter":
                raise RefusalError(
                    f"{model} declined the request (finish_reason={finish_reason!r}): "
                    f"{refusal or 'no explanation given'}"
                )
            # Do NOT fall back to `reasoning`: it is a draft of the model's
            # thinking, not the answer. Caching it would put nonsense into the
            # book. finish_reason is no help either — kiro-gateway reports
            # "stop" even for an empty answer.
            raise EmptyCompletionError(
                f"{model} returned no content "
                f"(finish_reason={finish_reason!r}, reasoning={len(reasoning)} chars). "
                "If it reasoned at length, lower the level (`reasoning:` in the config or "
                "reasoning_effort in the prompt) or shrink the chunk (chunker.target_words). "
                "Raising max_tokens helps on OpenRouter but not on kiro-gateway, "
                "which never forwards it upstream."
            )

        self._warn_if_reasoned_despite_none(reasoning_effort, reasoning, model)
        self._warn_if_near_response_ceiling(text, reasoning, model)

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

    def _warn_if_reasoned_despite_none(
        self, reasoning_effort: str | None, reasoning: str, model: str
    ) -> None:
        """Say once per model that `none` did not stop it from reasoning.

        Sonnet 5.5 cannot stop: kiro-gateway drops the setting and the model runs
        at its default level, at about twice the credits of `low` on a translation.
        Nothing else reports this.
        """
        if reasoning_effort != "none" or not reasoning or model in self._warned_about_none:
            return
        self._warned_about_none.add(model)
        logger.warning(
            "%s reasoned (%d chars) despite reasoning_effort=none: it cannot switch reasoning "
            "off, so `none` means its default level here. Set the level under `reasoning:` "
            "in the config; `low` is the cheapest.",
            model,
            len(reasoning),
        )

    def _warn_if_near_response_ceiling(self, text: str, reasoning: str, model: str) -> None:
        """Log a warning when a response approaches the configured size limit.

        `max_response_bytes` is a cautious threshold for an endpoint known to
        cut long responses. Both reasoning and answer are counted, since a limit
        on output may cover both. kiro-gateway once cut responses near 23 KB;
        that no longer reproduces, so its shipped config sets no threshold.
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
