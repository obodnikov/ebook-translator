"""Judge pass: score each translated chunk for quality.

The judge model (default: Haiku 4.5) receives the original text and
translation, returns a JSON score (1-5) with categorized issues.
Results are stored in the cache as stage='judge' — metadata only,
not part of the assembly waterfall.
"""

from __future__ import annotations

import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from . import model_json
from .cache import Cache
from .models import SeriesGlossary
from .prompts import Prompt, load_prompt, render_prompt
from .provider import CompletionResult, EmptyCompletionError, OpenRouterProvider
from .series import render_for_prompt

logger = logging.getLogger(__name__)

# Sits next to the judge prompt. Used only when a verdict cannot be read even
# after repair — see `Judge._refetch_valid_json`.
JSON_FIX_PROMPT_NAME = "json_fix.md"

LANG_NAMES = {
    "en": "English",
    "ru": "Russian",
    "hu": "Hungarian",
}


@dataclass
class JudgeResult:
    """Parsed result from judging one chunk."""

    chunk_id: str
    score: int
    issues: list[str]
    raw_json: str


@dataclass
class JudgeStats:
    chunks_total: int = 0
    chunks_cached: int = 0
    chunks_judged: int = 0
    chunks_failed: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    results: list[JudgeResult] = field(default_factory=list)


class Judge:
    """Evaluate translation quality for cached chunks."""

    def __init__(
        self,
        provider: OpenRouterProvider,
        prompt_path: Path,
        cache: Cache,
        glossary: SeriesGlossary | None,
        *,
        model: str,
        source_lang: str = "en",
        target_lang: str = "ru",
        judged_stage: str = "translate",
        run_id: str | None = None,
    ):
        self.provider = provider
        self.prompt_path = prompt_path
        self.prompt: Prompt = load_prompt(prompt_path)
        self.cache = cache
        self.glossary = glossary
        self.model = model
        self.source_lang = source_lang
        self.target_lang = target_lang
        # Which stage's text these verdicts describe. Recorded on every row so
        # scoring a second stage does not leave the cache holding two verdicts
        # per chunk with no way to tell them apart.
        self.judged_stage = judged_stage
        # Set to start a fresh measurement: it joins the cache key, so this
        # run neither reads nor overwrites earlier verdicts on the same text.
        # Reusing the same id resumes that run from its own cached rows, which
        # is what makes an interrupted variance measurement restartable.
        self.run_id = run_id

        self._glossary_block = render_for_prompt(glossary) if glossary else "(no glossary provided)"
        self._lock = threading.Lock()
        # Loaded on first use; absent file means the re-ask is simply skipped.
        self._json_fix_prompt: Prompt | None = None
        self._json_fix_loaded = False

    def _build_context(
        self,
        original_text: str,
        translated_text: str,
    ) -> dict:
        return {
            "source_lang": self.source_lang,
            "source_lang_name": LANG_NAMES.get(self.source_lang, self.source_lang),
            "target_lang": self.target_lang,
            "target_lang_name": LANG_NAMES.get(self.target_lang, self.target_lang),
            "glossary_block": self._glossary_block,
            "original_text": original_text,
            "translated_text": translated_text,
        }

    def _parse_judge_response(self, text: str) -> tuple[int, list[str]]:
        """Parse the judge's JSON response into score and issues.

        Raises ValueError if the response is not valid JSON or if the
        score field is missing/invalid. Does NOT auto-clamp — malformed
        responses should be treated as failures, not low scores.
        """
        text = text.strip()
        try:
            data = model_json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Judge response is not valid JSON: {e}. First 200 chars: {text[:200]!r}"
            ) from e

        if not isinstance(data, dict):
            raise ValueError(
                f"Judge response is not a JSON object. Got: {type(data).__name__}. "
                f"First 200 chars: {text[:200]!r}"
            )

        raw_score = data.get("score")
        if raw_score is None:
            raise ValueError(
                f"Judge response missing required 'score' field. Got keys: {list(data.keys())}"
            )

        try:
            score = int(raw_score)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Judge 'score' is not a valid integer: {raw_score!r}") from e

        if not 1 <= score <= 5:
            raise ValueError(f"Judge 'score' out of range 1-5: {score}")

        raw_issues = data.get("issues", [])
        if isinstance(raw_issues, list):
            issues = [str(i) for i in raw_issues]
        elif isinstance(raw_issues, str):
            issues = [raw_issues]
        else:
            issues = []

        return score, issues

    def _load_json_fix_prompt(self) -> Prompt | None:
        """Load the fix-up prompt from beside the judge prompt, once."""
        with self._lock:
            if not self._json_fix_loaded:
                self._json_fix_loaded = True
                path = self.prompt_path.parent / JSON_FIX_PROMPT_NAME
                if path.exists():
                    self._json_fix_prompt = load_prompt(path)
                else:
                    logger.debug("No fix-up prompt at %s; malformed verdicts will fail", path)
            return self._json_fix_prompt

    def _refetch_valid_json(self, broken: str, error: Exception) -> CompletionResult | None:
        """Ask the model to re-emit its own answer as valid JSON.

        Sends only the broken answer — not the chunk — so this costs a small
        fraction of a judging call: a verdict is about 1.5 KB either way.
        The model is told to change nothing but the escaping, so the verdict
        it already formed is preserved rather than formed anew.

        Returns None if there is no fix-up prompt or the call fails, leaving
        the caller to report the original parse failure.
        """
        prompt = self._load_json_fix_prompt()
        if prompt is None:
            return None

        system, user = render_prompt(prompt, {"broken_json": broken, "error": str(error)})
        try:
            result = self.provider.complete(
                model=self.model,
                system=system,
                user=user,
                temperature=prompt.temperature,
                max_tokens=prompt.max_tokens,
                reasoning_effort=prompt.reasoning_effort,
            )
        except Exception as e:  # noqa: BLE001 — the original failure is the one to report
            logger.warning("Fix-up call for a malformed verdict failed: %s", e)
            return None
        if not result.text.strip():
            return None
        return result

    def judge_chunk(
        self,
        chunk_id: str,
        original_text: str,
        translated_text: str,
        stats: JudgeStats,
    ) -> JudgeResult | None:
        """Judge a single chunk. Returns JudgeResult or None on failure."""
        context = self._build_context(original_text, translated_text)
        system, user = render_prompt(self.prompt, context)

        key_parts = ["judge", self.model, self.prompt.version, system, user]
        if self.run_id:
            key_parts.append(self.run_id)
        cache_key = Cache.make_key(*key_parts)

        with self._lock:
            cached = self.cache.get(cache_key)

        if cached is not None:
            raw_text = cached.content
            result = None
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
            # Belt and braces: provider.complete() already raises on an empty
            # response, but a stubbed provider might not, and an empty verdict
            # must never reach the cache.
            if not raw_text.strip():
                raise EmptyCompletionError(
                    "Judge model returned empty content "
                    f"(finish_reason={result.finish_reason!r}). Reasoning shares the "
                    "response budget with the answer — lower reasoning_effort."
                )

        # Validate BEFORE caching: _parse_judge_response raises on invalid JSON,
        # so a malformed/empty response is never written to the cache.
        fixup: CompletionResult | None = None
        try:
            score, issues = self._parse_judge_response(raw_text)
        except ValueError as e:
            # A cached row is never worth money: if it no longer reads, that
            # is a failure to surface, not a call to make.
            if result is None:
                raise
            logger.warning(
                "Verdict for %s did not parse (%s); asking the model to fix it", chunk_id, e
            )
            fixup = self._refetch_valid_json(raw_text, e)
            if fixup is None:
                raise
            # Store the text that parses, so `btrans status --scores` reads the
            # same verdict later without repeating the fix-up call.
            score, issues = self._parse_judge_response(fixup.text)
            raw_text = fixup.text

        # Cache only fresh, validated results (never on a cache hit).
        if result is not None:
            fix_in = fixup.input_tokens if fixup else 0
            fix_out = fixup.output_tokens if fixup else 0
            with self._lock:
                self.cache.put(
                    key=cache_key,
                    stage="judge",
                    model=self.model,
                    prompt_version=self.prompt.version,
                    content=raw_text,
                    input_tokens=result.input_tokens + fix_in,
                    output_tokens=result.output_tokens + fix_out,
                    meta={
                        "chunk_id": chunk_id,
                        "judged_stage": self.judged_stage,
                        **({"run": self.run_id} if self.run_id else {}),
                    },
                )
                stats.chunks_judged += 1
                stats.input_tokens += result.input_tokens + fix_in
                stats.output_tokens += result.output_tokens + fix_out
        judge_result = JudgeResult(
            chunk_id=chunk_id,
            score=score,
            issues=issues,
            raw_json=raw_text,
        )
        with self._lock:
            stats.results.append(judge_result)
        return judge_result

    def judge_chunks(
        self,
        chunk_originals: dict[str, str],
        chunk_translations: dict[str, str],
        on_progress=None,
        parallelism: int = 4,
    ) -> JudgeStats:
        """Judge all provided chunks.

        Args:
            chunk_originals: {chunk_id: original XHTML text}
            chunk_translations: {chunk_id: translated XHTML text}
            on_progress: callback(done, total, chunk_id, stats)
            parallelism: concurrent judge calls
        """
        chunk_ids = sorted(chunk_originals.keys())
        stats = JudgeStats(chunks_total=len(chunk_ids))

        if parallelism <= 1:
            for i, chunk_id in enumerate(chunk_ids):
                original = chunk_originals[chunk_id]
                translated = chunk_translations.get(chunk_id, "")
                try:
                    self.judge_chunk(chunk_id, original, translated, stats)
                except Exception as e:
                    stats.chunks_failed += 1
                    logger.warning("Judge chunk %s failed: %s", chunk_id, e)
                if on_progress:
                    on_progress(i + 1, len(chunk_ids), chunk_id, stats)
            return stats

        done_count = 0
        with ThreadPoolExecutor(max_workers=parallelism) as pool:
            futures = {
                pool.submit(
                    self._judge_chunk_safe,
                    chunk_id,
                    chunk_originals[chunk_id],
                    chunk_translations.get(chunk_id, ""),
                    stats,
                ): chunk_id
                for chunk_id in chunk_ids
            }
            for fut in as_completed(futures):
                chunk_id = futures[fut]
                err = fut.result()
                if err is not None:
                    with self._lock:
                        stats.chunks_failed += 1
                    logger.warning("Judge chunk %s failed: %s", chunk_id, err)
                done_count += 1
                if on_progress:
                    on_progress(done_count, len(chunk_ids), chunk_id, stats)

        return stats

    def _judge_chunk_safe(
        self,
        chunk_id: str,
        original_text: str,
        translated_text: str,
        stats: JudgeStats,
    ) -> Exception | None:
        try:
            self.judge_chunk(chunk_id, original_text, translated_text, stats)
            return None
        except Exception as e:  # noqa: BLE001
            return e
