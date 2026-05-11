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
    ):
        self.provider = provider
        self.prompt: Prompt = load_prompt(prompt_path)
        self.cache = cache
        self.glossary = glossary
        self.model = model
        self.source_lang = source_lang
        self.target_lang = target_lang

        self._glossary_block = (
            render_for_prompt(glossary) if glossary else "(no glossary provided)"
        )
        self._lock = threading.Lock()

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
        # Strip code fences if present
        if text.startswith("```"):
            lines = text.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines)

        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Judge response is not valid JSON: {e}. "
                f"First 200 chars: {text[:200]!r}"
            ) from e

        raw_score = data.get("score")
        if raw_score is None:
            raise ValueError(
                "Judge response missing required 'score' field. "
                f"Got keys: {list(data.keys())}"
            )

        try:
            score = int(raw_score)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"Judge 'score' is not a valid integer: {raw_score!r}"
            ) from e

        if not 1 <= score <= 5:
            raise ValueError(
                f"Judge 'score' out of range 1-5: {score}"
            )

        raw_issues = data.get("issues", [])
        if isinstance(raw_issues, list):
            issues = [str(i) for i in raw_issues]
        elif isinstance(raw_issues, str):
            issues = [raw_issues]
        else:
            issues = []

        return score, issues

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

        cache_key = Cache.make_key(
            "judge",
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
                    stage="judge",
                    model=self.model,
                    prompt_version=self.prompt.version,
                    content=raw_text,
                    input_tokens=result.input_tokens,
                    output_tokens=result.output_tokens,
                    meta={"chunk_id": chunk_id},
                )
                stats.chunks_judged += 1
                stats.input_tokens += result.input_tokens
                stats.output_tokens += result.output_tokens

        score, issues = self._parse_judge_response(raw_text)
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
