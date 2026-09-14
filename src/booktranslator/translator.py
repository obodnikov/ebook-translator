"""Translate chunks of a book and splice the result back into the EPUB trees.

Per-chunk flow:
  1. Build prompt context (glossary block, main paragraphs as XHTML
     fragments, optional overlap).
  2. Check cache. Miss -> call provider.
  3. Check the reply before caching it: exactly N paragraphs, every one a
     well-formed XHTML element. A rejected fresh reply is asked for once
     more, with the reason appended (see `replies.RETRY_NOTE`).
  4. Replace the corresponding paragraph subtrees in the chapter's
     lxml tree with the translated fragments — all of them or none.
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from lxml import etree

from .cache import Cache
from .chunker import Chunk, ChunkSet
from .models import SeriesGlossary
from .prompts import Prompt, load_prompt, render_prompt
from .provider import CompletionResult, OpenRouterProvider
from .replies import ReplyChecker, parse_cached_paragraphs
from .series import render_for_prompt

logger = logging.getLogger(__name__)


LANG_NAMES = {
    "en": "English",
    "ru": "Russian",
    "hu": "Hungarian",
}


@dataclass
class TranslateStats:
    chunks_total: int = 0
    chunks_cached: int = 0
    chunks_translated: int = 0
    chunks_failed: int = 0
    retries: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


class Translator:
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

        self._glossary_block = render_for_prompt(glossary) if glossary else "(no glossary provided)"
        # Serialize stats updates and live-tree mutations when running
        # chunks in parallel. The lxml tree itself is not thread-safe
        # for writes even across disjoint elements on some builds.
        self._lock = threading.Lock()
        self._checker = ReplyChecker(
            stage="translate",
            model=model,
            prompt_version=str(self.prompt.version),
            cache_path=getattr(cache, "path", None),
            lock=self._lock,
        )

    # -- prompt building ---------------------------------------------------

    def _build_context(
        self,
        chunk_set: ChunkSet,
        chunk: Chunk,
        reflection_notes: str = "",
    ) -> dict:
        main_fragments = chunk_set.render_main(chunk)
        prev_list = chunk_set.render_overlap(chunk, "prev")
        next_list = chunk_set.render_overlap(chunk, "next")
        return {
            "source_lang": self.source_lang,
            "source_lang_name": LANG_NAMES.get(self.source_lang, self.source_lang),
            "target_lang": self.target_lang,
            "target_lang_name": LANG_NAMES.get(self.target_lang, self.target_lang),
            "glossary_block": self._glossary_block,
            "main_fragments": main_fragments,
            "prev_overlap": "\n".join(prev_list),
            "next_overlap": "\n".join(next_list),
            "reflection_notes": reflection_notes,
        }

    # -- asking the model ----------------------------------------------------

    def _complete(self, system: str, user: str) -> CompletionResult:
        result = self.provider.complete(
            model=self.model,
            system=system,
            user=user,
            temperature=self.prompt.temperature,
            max_tokens=self.prompt.max_tokens,
            reasoning_effort=self.prompt.reasoning_effort,
        )
        # Guard: empty content means the provider routed the answer
        # elsewhere (tool call / reasoning_content) — e.g. kiro-gateway with
        # web_search or fake reasoning enabled. Raise BEFORE caching so a
        # poisoned empty response is never written to the cache.
        if not result.text.strip():
            raise ValueError(
                "Model returned empty content "
                f"(finish_reason={result.finish_reason!r}). The provider may have "
                "returned a tool call or put the answer in reasoning_content — "
                "check WEB_SEARCH_ENABLED / FAKE_REASONING on the gateway."
            )
        return result

    # -- splicing fragments back into the chapter tree --------------------

    @staticmethod
    def _replace_element(old_el: etree._Element, new_el: etree._Element) -> None:
        """Replace `old_el` in its parent with `new_el`, preserving tail.

        The incoming `new_el` has no namespace prefix. The live tree uses
        the XHTML namespace. To keep the serialization consistent, we
        rewrite tags on `new_el` to match the original element's namespace.
        """
        original_tag = old_el.tag  # e.g. '{http://www.w3.org/1999/xhtml}p'
        if isinstance(original_tag, str) and "}" in original_tag:
            ns_uri = original_tag.split("}", 1)[0].lstrip("{")
            # Retag new_el and all its descendants into the same namespace.
            for sub in new_el.iter():
                if isinstance(sub.tag, str) and "}" not in sub.tag:
                    sub.tag = f"{{{ns_uri}}}{sub.tag}"

        new_el.tail = old_el.tail
        parent = old_el.getparent()
        if parent is None:
            raise RuntimeError("Cannot replace root element of a chapter tree.")
        parent.replace(old_el, new_el)

    # -- core translate one chunk -----------------------------------------

    def translate_chunk(
        self,
        chunk_set: ChunkSet,
        chunk: Chunk,
        stats: TranslateStats,
    ) -> None:
        """Translate one chunk and update the live chapter tree in place."""
        context = self._build_context(chunk_set, chunk)
        system, user = render_prompt(self.prompt, context)

        cache_key = Cache.make_key(
            "translate",
            self.model,
            self.prompt.version,
            system,
            user,
        )
        expected = len(chunk.paragraph_indexes)
        with self._lock:
            cached = self.cache.get(cache_key)

        if cached is not None:
            elements = parse_cached_paragraphs(
                cached.content, expected, stage="translate", chunk_id=chunk.id
            )
            with self._lock:
                stats.chunks_cached += 1
        else:
            # Validate BEFORE caching: paragraph count and every paragraph
            # well-formed (AI_PIPELINE.md). A rejected reply is never cached.
            reply = self._checker.ask(
                self._complete, system, user, chunk_id=chunk.id, expected=expected, stats=stats
            )
            elements = reply.elements
            with self._lock:
                self.cache.put(
                    key=cache_key,
                    stage="translate",
                    model=self.model,
                    prompt_version=self.prompt.version,
                    content=reply.text,
                    input_tokens=reply.input_tokens,
                    output_tokens=reply.output_tokens,
                    meta={"chunk_id": chunk.id},
                )
                stats.chunks_translated += 1

        # Splice the paragraphs into the live tree. Every one parsed above, so
        # the chunk goes in whole — never half translated. Guarded by the lock
        # so parallel chunks don't race when mutating lxml elements.
        #
        # NOTE: this replaces the source paragraphs in place, so once
        # translation has run, `chunk_set` no longer holds the original text —
        # asking it for originals returns the translation. Any stage that
        # compares against the source (judge, reflect, verify) must read from a
        # separately built ChunkSet; see `source_chunk_set` in cli.translate.
        with self._lock:
            chapter = chunk_set.book.chapters[chunk.chapter_index]
            for para_idx, new_el in zip(chunk.paragraph_indexes, elements, strict=True):
                self._replace_element(chapter.paragraphs[para_idx], new_el)
                chapter.paragraphs[para_idx] = new_el

    # -- full book translation loop ---------------------------------------

    def translate_book(
        self,
        chunk_set: ChunkSet,
        on_progress=None,
        parallelism: int = 1,
    ) -> TranslateStats:
        stats = TranslateStats(chunks_total=len(chunk_set.chunks))
        total = len(chunk_set.chunks)

        if parallelism <= 1:
            # Sequential path: simplest, matches v1 behaviour.
            for i, chunk in enumerate(chunk_set.chunks):
                try:
                    self.translate_chunk(chunk_set, chunk, stats)
                except Exception as e:
                    stats.chunks_failed += 1
                    logger.warning("Chunk %s failed: %s", chunk.id, e)
                if on_progress:
                    on_progress(i + 1, total, chunk, stats)
            return stats

        # Parallel path. Chunks execute on a thread pool; the provider
        # call is blocking I/O so threads are fine here (no GIL issues
        # for network waits).
        done_count = 0
        with ThreadPoolExecutor(max_workers=parallelism) as pool:
            futures = {
                pool.submit(self._translate_chunk_safe, chunk_set, chunk, stats): chunk
                for chunk in chunk_set.chunks
            }
            for fut in as_completed(futures):
                chunk = futures[fut]
                err = fut.result()
                if err is not None:
                    with self._lock:
                        stats.chunks_failed += 1
                    logger.warning("Chunk %s failed: %s", chunk.id, err)
                done_count += 1
                if on_progress:
                    on_progress(done_count, total, chunk, stats)
        return stats

    def _translate_chunk_safe(
        self, chunk_set: ChunkSet, chunk: Chunk, stats: TranslateStats
    ) -> Exception | None:
        """Translate a single chunk, returning any exception raised."""
        try:
            self.translate_chunk(chunk_set, chunk, stats)
            return None
        except Exception as e:  # noqa: BLE001
            return e
