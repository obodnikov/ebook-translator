"""Post-processing passes: proofread, style, verify.

Each pass takes the translation from the previous stage (via waterfall),
runs it through a dedicated prompt, and stores the result as its own
stage in the cache. The original stages are never modified.

Waterfall input order:
  - proofread reads from: reflect > translate
  - style reads from: proofread > reflect > translate
  - verify reads from: style > proofread > reflect > translate

Delta mode (v2 prompts):
  The model returns only changed paragraphs as JSON patches or
  NO_CHANGES, drastically reducing output tokens. The full text is
  reconstructed before caching so downstream stages work unchanged.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from .cache import Cache
from .models import SeriesGlossary
from .prompts import Prompt, load_prompt, render_prompt
from .provider import OpenRouterProvider
from .series import render_for_prompt

logger = logging.getLogger(__name__)

LANG_NAMES = {
    "en": "English",
    "ru": "Russian",
    "hu": "Hungarian",
}

_PARAGRAPH_MARKER_RE = re.compile(r"^===PARAGRAPH\s+(\d+)===\s*$", re.MULTILINE)

# Matches a `&` that does NOT start a valid XML entity.
_BARE_AMPERSAND_RE = re.compile(r"&(?!(?:amp|lt|gt|quot|apos|#\d+|#x[0-9a-fA-F]+);)")

PostprocessStage = Literal["proofread", "style", "verify"]


@dataclass
class PostprocessResult:
    """Result of post-processing one chunk."""

    chunk_id: str
    stage: PostprocessStage
    changed: bool  # True if output differs from input
    content: str  # The processed text


@dataclass
class PostprocessStats:
    stage: PostprocessStage
    chunks_total: int = 0
    chunks_cached: int = 0
    chunks_processed: int = 0
    chunks_failed: int = 0
    chunks_changed: int = 0
    chunks_unchanged: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    results: list[PostprocessResult] = field(default_factory=list)


class PostProcessor:
    """Run a single post-processing pass (proofread, style, or verify).

    Each instance is configured for one specific stage. Create separate
    instances for each pass.
    """

    def __init__(
        self,
        provider: OpenRouterProvider,
        prompt_path: Path,
        cache: Cache,
        glossary: SeriesGlossary | None,
        *,
        stage: PostprocessStage,
        model: str,
        source_lang: str = "en",
        target_lang: str = "ru",
    ):
        self.provider = provider
        self.prompt: Prompt = load_prompt(prompt_path)
        self.cache = cache
        self.glossary = glossary
        self.stage: PostprocessStage = stage
        self.model = model
        self.source_lang = source_lang
        self.target_lang = target_lang

        self._glossary_block = render_for_prompt(glossary) if glossary else "(no glossary provided)"
        self._lock = threading.Lock()

    def _build_context(
        self,
        translated_paragraphs: list[str],
        original_text: str | None = None,
    ) -> dict:
        """Build template context for the prompt.

        Args:
            translated_paragraphs: List of XHTML paragraph strings to process.
            original_text: Original source text (only used by verify stage).
        """
        ctx = {
            "source_lang": self.source_lang,
            "source_lang_name": LANG_NAMES.get(self.source_lang, self.source_lang),
            "target_lang": self.target_lang,
            "target_lang_name": LANG_NAMES.get(self.target_lang, self.target_lang),
            "glossary_block": self._glossary_block,
            "main_fragments": translated_paragraphs,
        }
        if original_text is not None:
            ctx["original_text"] = original_text
        return ctx

    def _parse_response(self, text: str, expected_n: int) -> list[str]:
        """Split the LLM response into N XHTML fragments by paragraph markers.

        Returns exactly `expected_n` fragments or raises ValueError.
        """
        text = text.strip()
        # Strip optional code fences
        if text.startswith("```"):
            lines = text.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines)

        matches = list(_PARAGRAPH_MARKER_RE.finditer(text))
        if not matches:
            raise ValueError(
                f"No '===PARAGRAPH N===' markers found in response. First 200 chars: {text[:200]!r}"
            )

        fragments: list[str] = []
        for i, m in enumerate(matches):
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            fragment = text[start:end].strip()
            fragments.append(fragment)

        if len(fragments) != expected_n:
            raise ValueError(f"Expected {expected_n} paragraphs, got {len(fragments)}.")
        return fragments

    def _parse_delta_response(
        self, raw_text: str, input_paragraphs: list[str]
    ) -> tuple[list[str], bool]:
        """Parse a delta-mode response (NO_CHANGES or JSON patches).

        Returns:
            (output_paragraphs, changed) — the full list of paragraphs
            with patches applied, and whether anything changed.

        Raises ValueError if the response can't be parsed as delta.
        """
        text = raw_text.strip()

        # Strip code fences if present
        if text.startswith("```"):
            lines = text.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        # Check for NO_CHANGES
        if text.upper().replace("_", "").replace(" ", "") == "NOCHANGES":
            return list(input_paragraphs), False

        # Parse JSON array of patches
        # First, try to find JSON array in the text (model may prepend reasoning)
        json_start = text.find("[")
        if json_start > 0:
            # There's text before the JSON — try parsing from the [ onwards
            text_from_bracket = text[json_start:]
            try:
                patches = json.loads(text_from_bracket)
                if isinstance(patches, list):
                    logger.debug(
                        "Found JSON array at offset %d (skipped %d chars of preamble)",
                        json_start,
                        json_start,
                    )
                    return self._apply_patches(patches, input_paragraphs)
            except json.JSONDecodeError:
                pass

        try:
            patches = json.loads(text)
        except json.JSONDecodeError:
            # Try raw_decode in case of trailing text
            decoder = json.JSONDecoder()
            try:
                patches, _ = decoder.raw_decode(text)
            except (json.JSONDecodeError, ValueError):
                # Last resort: try to salvage truncated JSON array.
                # If the response was cut off mid-entry, try closing it.
                # Also try from the first [ if there's preamble text
                salvage_text = text
                if json_start > 0:
                    salvage_text = text[json_start:]
                salvaged = self._try_salvage_truncated_json(salvage_text)
                if salvaged is not None:
                    patches = salvaged
                else:
                    raise ValueError(
                        f"Delta response is not valid JSON or NO_CHANGES. "
                        f"First 200 chars: {text[:200]!r}"
                    ) from None

        if not isinstance(patches, list):
            raise ValueError(f"Delta response is not a JSON array. Got: {type(patches).__name__}")

        return self._apply_patches(patches, input_paragraphs)

    def _apply_patches(self, patches: list, input_paragraphs: list[str]) -> tuple[list[str], bool]:
        """Apply a list of patch dicts to input paragraphs.

        Returns (output_paragraphs, changed).
        """
        if not isinstance(patches, list):
            raise ValueError(f"Delta response is not a JSON array. Got: {type(patches).__name__}")

        if len(patches) == 0:
            return list(input_paragraphs), False

        # Apply patches
        output = list(input_paragraphs)
        n = len(input_paragraphs)
        for patch in patches:
            if not isinstance(patch, dict):
                raise ValueError(f"Patch entry is not an object: {patch!r}")
            p_idx = patch.get("p")
            p_text = patch.get("text")
            if p_idx is None or p_text is None:
                raise ValueError(f"Patch missing 'p' or 'text': {patch!r}")
            # p is 1-based
            idx = int(p_idx) - 1
            if idx < 0 or idx >= n:
                raise ValueError(f"Patch paragraph index {p_idx} out of range 1..{n}")
            output[idx] = str(p_text).strip()

        return output, True

    def _is_delta_response(self, text: str) -> bool:
        """Heuristic: does this response look like delta format?

        Delta responses are either NO_CHANGES or start with [ (JSON array).
        Legacy responses contain ===PARAGRAPH markers.
        """
        stripped = text.strip()
        # Strip code fences for detection
        if stripped.startswith("```"):
            lines = stripped.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            stripped = "\n".join(lines).strip()

        if stripped.upper().replace("_", "").replace(" ", "") == "NOCHANGES":
            return True
        if stripped.startswith("["):
            return True
        # If it has paragraph markers, it's legacy full-text format;
        # otherwise (ambiguous) treat as delta.
        return not _PARAGRAPH_MARKER_RE.search(stripped)

    def _reconstruct_full_text(self, paragraphs: list[str]) -> str:
        """Rebuild the full ===PARAGRAPH N=== format for cache storage."""
        parts = []
        for i, p in enumerate(paragraphs, 1):
            parts.append(f"===PARAGRAPH {i}===")
            parts.append(p)
        return "\n".join(parts)

    @staticmethod
    def _try_salvage_truncated_json(text: str) -> list[dict] | None:
        """Try to recover patches from a truncated JSON array.

        If the model hit max_tokens mid-response, the JSON array may be
        cut off. We try to find the last complete object and parse up to
        there.

        Returns the list of complete patch objects, or None if salvage fails.
        """
        # Find the last complete "}" that could end a patch object
        # Strategy: progressively trim from the end until we get valid JSON
        text = text.rstrip()

        # Must start with [
        if not text.startswith("["):
            return None

        # Try closing the array at each } from the end
        last_brace = text.rfind("}")
        while last_brace > 0:
            candidate = text[: last_brace + 1] + "]"
            try:
                result = json.loads(candidate)
                if isinstance(result, list) and all(
                    isinstance(p, dict) and "p" in p and "text" in p for p in result
                ):
                    logger.info(
                        "Salvaged %d patches from truncated JSON (%d/%d chars used)",
                        len(result),
                        last_brace + 1,
                        len(text),
                    )
                    return result
            except json.JSONDecodeError:
                pass
            last_brace = text.rfind("}", 0, last_brace)

        return None

    def process_chunk(
        self,
        chunk_id: str,
        translated_paragraphs: list[str],
        stats: PostprocessStats,
        original_text: str | None = None,
    ) -> PostprocessResult | None:
        """Process a single chunk through this stage.

        Args:
            chunk_id: Unique chunk identifier.
            translated_paragraphs: List of XHTML paragraph strings
                (from the previous stage in the waterfall).
            stats: Mutable stats object.
            original_text: Original source text (required for verify stage).

        Returns:
            PostprocessResult or None on failure.
        """
        context = self._build_context(translated_paragraphs, original_text)
        system, user = render_prompt(self.prompt, context)

        cache_key = Cache.make_key(
            self.stage,
            self.model,
            self.prompt.version,
            system,
            user,
        )

        with self._lock:
            cached = self.cache.get(cache_key)

        if cached is not None:
            raw_text = cached.content
            with self._lock:
                stats.chunks_cached += 1
        else:
            result = self.provider.complete(
                model=self.model,
                system=system,
                user=user,
                temperature=self.prompt.temperature,
                max_tokens=self.prompt.max_tokens,
                reasoning_effort=self.prompt.reasoning_effort,
            )
            raw_text = result.text
            # Guard: empty content means the provider routed the answer
            # elsewhere (tool call / reasoning_content). Raise BEFORE the parse
            # and cache below so a poisoned empty response is never stored.
            if not raw_text.strip():
                raise ValueError(
                    f"{self.stage} model returned empty content "
                    f"(finish_reason={result.finish_reason!r}). Check "
                    "WEB_SEARCH_ENABLED / FAKE_REASONING on the gateway."
                )
            with self._lock:
                stats.chunks_processed += 1
                stats.input_tokens += result.input_tokens
                stats.output_tokens += result.output_tokens

        # Parse response: detect delta vs legacy full-text format
        expected = len(translated_paragraphs)

        if self._is_delta_response(raw_text):
            # Delta mode: parse patches and reconstruct full text
            try:
                output_paragraphs, changed = self._parse_delta_response(
                    raw_text, translated_paragraphs
                )
            except ValueError:
                # Delta parse failed — log raw response length for debugging
                logger.debug(
                    "%s chunk delta parse failed, raw response %d chars "
                    "(max_tokens=%s). Last 100: %r",
                    self.stage,
                    len(raw_text),
                    self.prompt.max_tokens,
                    raw_text[-100:],
                )
                raise
            # Store reconstructed full text in cache (not the raw delta)
            # so downstream stages can read it as normal ===PARAGRAPH=== format
            full_text = self._reconstruct_full_text(output_paragraphs)
            if cached is None:
                with self._lock:
                    self.cache.put(
                        key=cache_key,
                        stage=self.stage,
                        model=self.model,
                        prompt_version=self.prompt.version,
                        content=full_text,
                        input_tokens=result.input_tokens,
                        output_tokens=result.output_tokens,
                        meta={"chunk_id": chunk_id, "delta": True},
                    )
        else:
            # Legacy full-text format: parse paragraph markers
            _fragments = self._parse_response(raw_text, expected)
            output_paragraphs = _fragments

            # Determine if content changed
            input_joined = "\n".join(translated_paragraphs).strip()
            output_joined = "\n".join(_fragments).strip()
            changed = input_joined != output_joined

            # Store raw text in cache (legacy format, already has markers)
            if cached is None:
                with self._lock:
                    self.cache.put(
                        key=cache_key,
                        stage=self.stage,
                        model=self.model,
                        prompt_version=self.prompt.version,
                        content=raw_text,
                        input_tokens=result.input_tokens,
                        output_tokens=result.output_tokens,
                        meta={"chunk_id": chunk_id},
                    )

        with self._lock:
            if changed:
                stats.chunks_changed += 1
            else:
                stats.chunks_unchanged += 1

        pp_result = PostprocessResult(
            chunk_id=chunk_id,
            stage=self.stage,
            changed=changed,
            content=self._reconstruct_full_text(output_paragraphs)
            if self._is_delta_response(raw_text)
            else raw_text,
        )
        with self._lock:
            stats.results.append(pp_result)
        return pp_result

    def process_chunks(
        self,
        chunks_data: list[dict],
        on_progress=None,
        parallelism: int = 4,
    ) -> PostprocessStats:
        """Process multiple chunks through this stage.

        Args:
            chunks_data: List of dicts with keys:
                - chunk_id: str
                - translated_paragraphs: list[str]
                - original_text: str | None (required for verify)
            on_progress: callback(done, total, chunk_id, stats)
            parallelism: concurrent calls
        """
        stats = PostprocessStats(
            stage=self.stage,
            chunks_total=len(chunks_data),
        )

        if parallelism <= 1:
            for i, chunk_data in enumerate(chunks_data):
                try:
                    self.process_chunk(
                        chunk_id=chunk_data["chunk_id"],
                        translated_paragraphs=chunk_data["translated_paragraphs"],
                        stats=stats,
                        original_text=chunk_data.get("original_text"),
                    )
                except Exception as e:
                    with self._lock:
                        stats.chunks_failed += 1
                    logger.warning(
                        "%s chunk %s failed: %s",
                        self.stage,
                        chunk_data["chunk_id"],
                        e,
                    )
                if on_progress:
                    on_progress(
                        i + 1,
                        len(chunks_data),
                        chunk_data["chunk_id"],
                        stats,
                    )
            return stats

        done_count = 0
        with ThreadPoolExecutor(max_workers=parallelism) as pool:
            futures = {
                pool.submit(self._process_chunk_safe, chunk_data, stats): chunk_data["chunk_id"]
                for chunk_data in chunks_data
            }
            for fut in as_completed(futures):
                chunk_id = futures[fut]
                err = fut.result()
                if err is not None:
                    with self._lock:
                        stats.chunks_failed += 1
                    logger.warning(
                        "%s chunk %s failed: %s",
                        self.stage,
                        chunk_id,
                        err,
                    )
                done_count += 1
                if on_progress:
                    on_progress(
                        done_count,
                        len(chunks_data),
                        chunk_id,
                        stats,
                    )

        return stats

    def _process_chunk_safe(self, chunk_data: dict, stats: PostprocessStats) -> Exception | None:
        try:
            self.process_chunk(
                chunk_id=chunk_data["chunk_id"],
                translated_paragraphs=chunk_data["translated_paragraphs"],
                stats=stats,
                original_text=chunk_data.get("original_text"),
            )
            return None
        except Exception as e:  # noqa: BLE001
            return e
