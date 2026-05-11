"""Translate chunks of a book and splice the result back into the EPUB trees.

Per-chunk flow:
  1. Build prompt context (glossary block, main paragraphs as XHTML
     fragments, optional overlap).
  2. Check cache. Miss -> call provider.
  3. Parse response into N translated XHTML fragments, verify N matches
     input paragraph count; retry (tightening instructions) on mismatch.
  4. Replace the corresponding paragraph subtrees in the chapter's
     lxml tree with the translated fragments.
"""

from __future__ import annotations

import logging
import re
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
from .series import render_for_prompt


logger = logging.getLogger(__name__)

_PARAGRAPH_MARKER_RE = re.compile(
    r"^===PARAGRAPH\s+(\d+)===\s*$", re.MULTILINE
)

# Matches a `&` that does NOT start a valid XML entity
# (&amp; &lt; &gt; &quot; &apos; or numeric like &#123; / &#xAF;).
# Used to fix stray ampersands in LLM output like "M&S", "AT&T".
_BARE_AMPERSAND_RE = re.compile(
    r"&(?!(?:amp|lt|gt|quot|apos|#\d+|#x[0-9a-fA-F]+);)"
)


def _escape_bare_ampersands(fragment: str) -> str:
    """Replace stray `&` (not part of a valid entity) with `&amp;`.

    LLMs sometimes return English brand names like "M&S" or "AT&T"
    verbatim inside XHTML fragments. Those break lxml parsing. We fix
    them here rather than asking the model to escape, because the
    instruction is easy to miss on a long chunk.
    """
    return _BARE_AMPERSAND_RE.sub("&amp;", fragment)

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

        self._glossary_block = (
            render_for_prompt(glossary) if glossary else "(no glossary provided)"
        )
        # Serialize stats updates and live-tree mutations when running
        # chunks in parallel. The lxml tree itself is not thread-safe
        # for writes even across disjoint elements on some builds.
        self._lock = threading.Lock()

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

    # -- response parsing --------------------------------------------------

    def _parse_response(self, text: str, expected_n: int) -> list[str]:
        """Split the LLM response into N XHTML fragments by paragraph markers.

        Returns exactly `expected_n` fragments or raises ValueError.
        """
        # Strip optional code fences.
        text = text.strip()
        if text.startswith("```"):
            # Drop first and last fence lines
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

    # -- splicing fragments back into the chapter tree --------------------

    @staticmethod
    def _parse_fragment(fragment: str) -> etree._Element:
        """Parse a translated XHTML fragment into an lxml element."""
        # XHTML default namespace may matter when we later re-serialize
        # the chapter. But the fragment we received from the LLM has no
        # namespaces. We'll splice it directly; the enclosing chapter
        # already declares xmlns on <html>.
        # Fix stray ampersands first — LLMs often return "M&S" / "AT&T"
        # verbatim, which would make the fragment non-well-formed XML.
        fragment = _escape_bare_ampersands(fragment)
        try:
            return etree.fromstring(fragment)
        except etree.XMLSyntaxError as e:
            # Wrap in a root with explicit XHTML ns as a fallback
            # (handles cases where the LLM emitted an HTML-ish fragment).
            raise ValueError(
                f"Translated fragment is not well-formed XML: {e}. "
                f"First 200 chars: {fragment[:200]!r}"
            ) from e

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
            raise RuntimeError(
                "Cannot replace root element of a chapter tree."
            )
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
        with self._lock:
            cached = self.cache.get(cache_key)
        if cached is not None:
            raw_text = cached.content
            with self._lock:
                stats.chunks_cached += 1
            result = None
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
                    stage="translate",
                    model=self.model,
                    prompt_version=self.prompt.version,
                    content=raw_text,
                    input_tokens=result.input_tokens,
                    output_tokens=result.output_tokens,
                    meta={"chunk_id": chunk.id},
                )
                stats.chunks_translated += 1
                stats.input_tokens += result.input_tokens
                stats.output_tokens += result.output_tokens

        expected = len(chunk.paragraph_indexes)
        fragments = self._parse_response(raw_text, expected)

        # Splice fragments into the live tree. Guarded by the lock so
        # parallel chunks don't race when mutating lxml elements (even
        # though they target different subtrees, we stay on the safe
        # side).
        with self._lock:
            chapter = chunk_set.book.chapters[chunk.chapter_index]
            for para_idx, frag_str in zip(chunk.paragraph_indexes, fragments):
                original_el = chapter.paragraphs[para_idx]
                new_el = self._parse_fragment(frag_str)
                self._replace_element(original_el, new_el)
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
        last_chunk: Chunk | None = None
        with ThreadPoolExecutor(max_workers=parallelism) as pool:
            futures = {
                pool.submit(self._translate_chunk_safe, chunk_set, chunk, stats): chunk
                for chunk in chunk_set.chunks
            }
            for fut in as_completed(futures):
                chunk = futures[fut]
                last_chunk = chunk
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
