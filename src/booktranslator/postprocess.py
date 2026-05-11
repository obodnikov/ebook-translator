"""Post-processing passes: proofread, style, verify.

Each pass takes the translation from the previous stage (via waterfall),
runs it through a dedicated prompt, and stores the result as its own
stage in the cache. The original stages are never modified.

Waterfall input order:
  - proofread reads from: reflect > translate
  - style reads from: proofread > reflect > translate
  - verify reads from: style > proofread > reflect > translate
"""

from __future__ import annotations

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

_PARAGRAPH_MARKER_RE = re.compile(
    r"^===PARAGRAPH\s+(\d+)===\s*$", re.MULTILINE
)

# Matches a `&` that does NOT start a valid XML entity.
_BARE_AMPERSAND_RE = re.compile(
    r"&(?!(?:amp|lt|gt|quot|apos|#\d+|#x[0-9a-fA-F]+);)"
)

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

        self._glossary_block = (
            render_for_prompt(glossary) if glossary else "(no glossary provided)"
        )
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
                "No '===PARAGRAPH N===' markers found in response. "
                f"First 200 chars: {text[:200]!r}"
            )

        fragments: list[str] = []
        for i, m in enumerate(matches):
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            fragment = text[start:end].strip()
            fragments.append(fragment)

        if len(fragments) != expected_n:
            raise ValueError(
                f"Expected {expected_n} paragraphs, got {len(fragments)}."
            )
        return fragments

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
            )
            raw_text = result.text
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
                stats.chunks_processed += 1
                stats.input_tokens += result.input_tokens
                stats.output_tokens += result.output_tokens

        # Parse and validate paragraph count
        expected = len(translated_paragraphs)
        _fragments = self._parse_response(raw_text, expected)

        # Determine if content changed
        input_joined = "\n".join(translated_paragraphs).strip()
        output_joined = "\n".join(_fragments).strip()
        changed = input_joined != output_joined

        with self._lock:
            if changed:
                stats.chunks_changed += 1
            else:
                stats.chunks_unchanged += 1

        pp_result = PostprocessResult(
            chunk_id=chunk_id,
            stage=self.stage,
            changed=changed,
            content=raw_text,
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
                        self.stage, chunk_data["chunk_id"], e,
                    )
                if on_progress:
                    on_progress(
                        i + 1, len(chunks_data),
                        chunk_data["chunk_id"], stats,
                    )
            return stats

        done_count = 0
        with ThreadPoolExecutor(max_workers=parallelism) as pool:
            futures = {
                pool.submit(
                    self._process_chunk_safe, chunk_data, stats
                ): chunk_data["chunk_id"]
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
                        self.stage, chunk_id, err,
                    )
                done_count += 1
                if on_progress:
                    on_progress(
                        done_count, len(chunks_data), chunk_id, stats,
                    )

        return stats

    def _process_chunk_safe(
        self, chunk_data: dict, stats: PostprocessStats
    ) -> Exception | None:
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
