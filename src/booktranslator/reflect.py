"""Reflect pass: re-translate chunks that scored poorly.

Implements the Andrew Ng reflection method:
1. Critique the translation (prompts/reflect.md) → structured notes.
2. Re-translate with the original translate prompt + reflection_notes.

The improved translation is stored as stage='reflect' in the cache.
The original stage='translate' entry is never modified.
"""

from __future__ import annotations

import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

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


@dataclass
class ReflectResult:
    """Result of reflecting on one chunk."""

    chunk_id: str
    improved: bool  # True if reflection produced a different translation
    reflection_notes: str  # The critique JSON
    new_translation: str  # The re-translated text (or empty if not improved)


@dataclass
class ReflectStats:
    chunks_total: int = 0
    chunks_cached: int = 0
    chunks_reflected: int = 0
    chunks_failed: int = 0
    chunks_improved: int = 0
    chunks_unchanged: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    results: list[ReflectResult] = field(default_factory=list)


class Reflector:
    """Perform reflection-based improvement on translated chunks."""

    def __init__(
        self,
        provider: OpenRouterProvider,
        reflect_prompt_path: Path,
        translate_prompt_path: Path,
        cache: Cache,
        glossary: SeriesGlossary | None,
        *,
        model: str,
        source_lang: str = "en",
        target_lang: str = "ru",
    ):
        self.provider = provider
        self.reflect_prompt: Prompt = load_prompt(reflect_prompt_path)
        self.translate_prompt: Prompt = load_prompt(translate_prompt_path)
        self.cache = cache
        self.glossary = glossary
        self.model = model
        self.source_lang = source_lang
        self.target_lang = target_lang

        self._glossary_block = render_for_prompt(glossary) if glossary else "(no glossary provided)"
        self._lock = threading.Lock()

    def _build_reflect_context(
        self,
        original_text: str,
        translated_text: str,
        judge_score: int,
        judge_issues: list[str],
    ) -> dict:
        return {
            "source_lang": self.source_lang,
            "source_lang_name": LANG_NAMES.get(self.source_lang, self.source_lang),
            "target_lang": self.target_lang,
            "target_lang_name": LANG_NAMES.get(self.target_lang, self.target_lang),
            "glossary_block": self._glossary_block,
            "original_text": original_text,
            "translated_text": translated_text,
            "judge_score": str(judge_score),
            "judge_issues": "; ".join(judge_issues) if judge_issues else "none specified",
        }

    def _build_retranslate_context(
        self,
        original_paragraphs: list[str],
        reflection_notes: str,
        prev_overlap: str = "",
        next_overlap: str = "",
    ) -> dict:
        """Build context for re-translation with reflection notes."""
        return {
            "source_lang": self.source_lang,
            "source_lang_name": LANG_NAMES.get(self.source_lang, self.source_lang),
            "target_lang": self.target_lang,
            "target_lang_name": LANG_NAMES.get(self.target_lang, self.target_lang),
            "glossary_block": self._glossary_block,
            "main_fragments": original_paragraphs,
            "prev_overlap": prev_overlap,
            "next_overlap": next_overlap,
            "reflection_notes": reflection_notes,
        }

    def _get_reflection_notes(
        self,
        chunk_id: str,
        original_text: str,
        translated_text: str,
        judge_score: int,
        judge_issues: list[str],
        stats: ReflectStats,
    ) -> str:
        """Step 1: Get critique/reflection notes from the reflect prompt."""
        context = self._build_reflect_context(
            original_text, translated_text, judge_score, judge_issues
        )
        system, user = render_prompt(self.reflect_prompt, context)

        cache_key = Cache.make_key(
            "reflect_notes",
            self.model,
            self.reflect_prompt.version,
            system,
            user,
        )

        with self._lock:
            cached = self.cache.get(cache_key)

        if cached is not None:
            return cached.content

        result = self.provider.complete(
            model=self.model,
            system=system,
            user=user,
            temperature=self.reflect_prompt.temperature,
            max_tokens=self.reflect_prompt.max_tokens,
            reasoning_effort=self.reflect_prompt.reasoning_effort,
        )

        # Guard: empty content means the provider routed the answer elsewhere
        # (tool call / reasoning_content). Raise BEFORE caching so an empty
        # response is never written to the cache.
        if not (result.text or "").strip():
            raise ValueError(
                "Reflect (notes) model returned empty content "
                f"(finish_reason={result.finish_reason!r}). Check "
                "WEB_SEARCH_ENABLED / FAKE_REASONING on the gateway."
            )

        # Store reflection notes in cache (not in waterfall — internal artifact)
        with self._lock:
            self.cache.put(
                key=cache_key,
                stage="reflect_notes",
                model=self.model,
                prompt_version=self.reflect_prompt.version,
                content=result.text,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                meta={"chunk_id": chunk_id},
            )
            stats.input_tokens += result.input_tokens
            stats.output_tokens += result.output_tokens

        return result.text

    def _retranslate_with_notes(
        self,
        chunk_id: str,
        original_paragraphs: list[str],
        reflection_notes: str,
        stats: ReflectStats,
    ) -> str:
        """Step 2: Re-translate using the translate prompt + reflection notes."""
        context = self._build_retranslate_context(original_paragraphs, reflection_notes)
        system, user = render_prompt(self.translate_prompt, context)

        cache_key = Cache.make_key(
            "reflect",
            self.model,
            self.translate_prompt.version,
            system,
            user,
        )

        with self._lock:
            cached = self.cache.get(cache_key)

        if cached is not None:
            with self._lock:
                stats.chunks_cached += 1
            return cached.content

        result = self.provider.complete(
            model=self.model,
            system=system,
            user=user,
            temperature=self.translate_prompt.temperature,
            max_tokens=self.translate_prompt.max_tokens,
            reasoning_effort=self.translate_prompt.reasoning_effort,
        )

        # Guard: empty content means the provider routed the answer elsewhere
        # (tool call / reasoning_content). Raise BEFORE caching so an empty
        # response is never written to the cache.
        if not (result.text or "").strip():
            raise ValueError(
                "Reflect (re-translate) model returned empty content "
                f"(finish_reason={result.finish_reason!r}). Check "
                "WEB_SEARCH_ENABLED / FAKE_REASONING on the gateway."
            )

        with self._lock:
            self.cache.put(
                key=cache_key,
                stage="reflect",
                model=self.model,
                prompt_version=self.translate_prompt.version,
                content=result.text,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                meta={"chunk_id": chunk_id},
            )
            stats.chunks_reflected += 1
            stats.input_tokens += result.input_tokens
            stats.output_tokens += result.output_tokens

        return result.text

    def reflect_chunk(
        self,
        chunk_id: str,
        original_text: str,
        original_paragraphs: list[str],
        translated_text: str,
        judge_score: int,
        judge_issues: list[str],
        stats: ReflectStats,
    ) -> ReflectResult:
        """Reflect on and re-translate a single chunk.

        Args:
            chunk_id: Unique chunk identifier.
            original_text: Full original text (all paragraphs concatenated).
            original_paragraphs: List of individual XHTML paragraph strings.
            translated_text: Current translation text.
            judge_score: Score from judge (1-5).
            judge_issues: Issues identified by judge.
            stats: Mutable stats object.
        """
        # Step 1: Get reflection notes (critique)
        reflection_notes = self._get_reflection_notes(
            chunk_id, original_text, translated_text, judge_score, judge_issues, stats
        )

        # Step 2: Re-translate with notes
        new_translation = self._retranslate_with_notes(
            chunk_id, original_paragraphs, reflection_notes, stats
        )

        # Determine if the translation actually changed
        improved = new_translation.strip() != translated_text.strip()
        with self._lock:
            if improved:
                stats.chunks_improved += 1
            else:
                stats.chunks_unchanged += 1

        result = ReflectResult(
            chunk_id=chunk_id,
            improved=improved,
            reflection_notes=reflection_notes,
            new_translation=new_translation,
        )
        with self._lock:
            stats.results.append(result)
        return result

    def reflect_chunks(
        self,
        chunks_to_reflect: list[dict],
        on_progress=None,
        parallelism: int = 2,
    ) -> ReflectStats:
        """Reflect on multiple chunks.

        Args:
            chunks_to_reflect: List of dicts with keys:
                chunk_id, original_text, original_paragraphs,
                translated_text, judge_score, judge_issues
            on_progress: callback(done, total, chunk_id, stats)
            parallelism: concurrent reflect calls (lower than judge —
                reflection uses Sonnet which is heavier)
        """
        stats = ReflectStats(chunks_total=len(chunks_to_reflect))

        if parallelism <= 1:
            for i, chunk_data in enumerate(chunks_to_reflect):
                try:
                    self.reflect_chunk(
                        chunk_id=chunk_data["chunk_id"],
                        original_text=chunk_data["original_text"],
                        original_paragraphs=chunk_data["original_paragraphs"],
                        translated_text=chunk_data["translated_text"],
                        judge_score=chunk_data["judge_score"],
                        judge_issues=chunk_data["judge_issues"],
                        stats=stats,
                    )
                except Exception as e:
                    stats.chunks_failed += 1
                    logger.warning(
                        "Reflect chunk %s failed: %s",
                        chunk_data["chunk_id"],
                        e,
                    )
                if on_progress:
                    on_progress(i + 1, len(chunks_to_reflect), chunk_data["chunk_id"], stats)
            return stats

        done_count = 0
        with ThreadPoolExecutor(max_workers=parallelism) as pool:
            futures = {
                pool.submit(self._reflect_chunk_safe, chunk_data, stats): chunk_data["chunk_id"]
                for chunk_data in chunks_to_reflect
            }
            for fut in as_completed(futures):
                chunk_id = futures[fut]
                err = fut.result()
                if err is not None:
                    with self._lock:
                        stats.chunks_failed += 1
                    logger.warning("Reflect chunk %s failed: %s", chunk_id, err)
                done_count += 1
                if on_progress:
                    on_progress(done_count, len(chunks_to_reflect), chunk_id, stats)

        return stats

    def _reflect_chunk_safe(self, chunk_data: dict, stats: ReflectStats) -> Exception | None:
        try:
            self.reflect_chunk(
                chunk_id=chunk_data["chunk_id"],
                original_text=chunk_data["original_text"],
                original_paragraphs=chunk_data["original_paragraphs"],
                translated_text=chunk_data["translated_text"],
                judge_score=chunk_data["judge_score"],
                judge_issues=chunk_data["judge_issues"],
                stats=stats,
            )
            return None
        except Exception as e:  # noqa: BLE001
            return e
