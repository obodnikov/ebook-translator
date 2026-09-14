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

from . import model_json
from .cache import Cache
from .models import SeriesGlossary
from .prompts import Prompt, load_prompt, render_prompt
from .provider import CompletionResult, OpenRouterProvider
from .replies import parse_fragment
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

# The delta prompts print NO_CHANGES inside a code fence and refer to it as
# `NO_CHANGES` in the rules, so models return it fenced, back-quoted, or after
# a sentence of commentary. All of those mean the same thing, and rejecting
# them threw away correct verdicts: style ch19_c05 and verify ch25_c03 both
# failed this way on the Bear Head run.
_FENCED_BLOCK_RE = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)
_NO_CHANGES_RE = re.compile(r"\A`*\s*NO[_ ]?CHANGES\s*`*\Z", re.IGNORECASE)


# A paragraph of dialogue opens with an em dash in Russian; guillemets belong
# to speech quoted inline. The proofread prompt used to ask for « » outright
# and models obliged, costing 8 paragraphs of Bear Head. The prompt says the
# opposite now, and this refuses the change if a model does it anyway.
_OPENS_WITH_DASH = re.compile(r"^\s*(?:<[^>]+>\s*)*[—–]\s")
_OPENS_WITH_GUILLEMET = re.compile(r"^\s*(?:<[^>]+>\s*)*«")


# Appended to the user message when a reply ignored the output format. Kept in
# code, not in the prompt file: it is only ever sent on the second ask, so the
# prompt version — and every cached result under it — stays untouched.
_FORMAT_RETRY_NOTE = """

## Your previous reply was rejected

It did not follow the output format: {reason}

Answer again for the same input, with ONLY `NO_CHANGES` or the JSON array of
patches. No analysis, no explanation, nothing before or after it."""

# Format retries are full-price calls, so a run gets a small budget: enough to
# absorb the odd bad reply, not enough to pay twice for a whole stage.
_MIN_FORMAT_RETRIES = 3

# How many `[` to try when hunting for the patch array inside a reply that
# also contains prose. Enough for any real preamble, bounded so a reply full
# of brackets cannot turn the parse into a long scan.
_MAX_BRACKET_STARTS = 20


def _looks_like_patches(value: object) -> bool:
    """True when a parsed value is plausibly the patch array.

    A bracket inside the commentary — `[14]`, `[note]` — can parse as a
    perfectly valid list, so parsing is not enough to identify the array we
    are after. An empty list qualifies: it means "nothing to change".
    """
    return isinstance(value, list) and all(
        isinstance(p, dict) and "p" in p and "text" in p for p in value
    )


def _bracket_positions(text: str) -> list[int]:
    """Offsets of the first few `[` in `text`, in order."""
    positions: list[int] = []
    at = text.find("[")
    while at != -1 and len(positions) < _MAX_BRACKET_STARTS:
        positions.append(at)
        at = text.find("[", at + 1)
    return positions


def _breaks_dialogue_dash(before: str, after: str) -> bool:
    """True when a patch replaces a paragraph's opening dash with guillemets."""
    return bool(_OPENS_WITH_DASH.match(before)) and bool(_OPENS_WITH_GUILLEMET.match(after))


# A sentence has to be at least this long before finding it inside another
# paragraph's patch means the paragraphs were mixed up. Short lines — "Да.",
# "– Хорошо, – сказал я." — recur in any novel and prove nothing.
_MIN_BORROWED_SENTENCE = 40
_TAG_RE = re.compile(r"<[^>]+>")
_SENTENCE_END_RE = re.compile(r"(?<=[.!?…»])\s+")


def _plain_text(fragment: str) -> str:
    return " ".join(_TAG_RE.sub(" ", fragment).split())


def _borrowed_paragraph(idx: int, new_text: str, paragraphs: list[str]) -> int | None:
    """The 1-based number of another paragraph whose sentence a patch copies, if any.

    A patch rewrites one paragraph; it has no business carrying a sentence of a
    different one. When it does, the model has mixed paragraphs up — on Foxglove
    Summer a proofread patch for paragraph 21 came back as paragraph 22 plus the
    second half of 21, its first half gone. Only sentences that paragraph `idx`
    does not already contain count, so a line the author really repeats is fine.
    """
    patch = _plain_text(new_text)
    own = _plain_text(paragraphs[idx])
    for other, paragraph in enumerate(paragraphs):
        if other == idx:
            continue
        for sentence in _SENTENCE_END_RE.split(_plain_text(paragraph)):
            if (
                len(sentence) >= _MIN_BORROWED_SENTENCE
                and sentence in patch
                and sentence not in own
            ):
                return other + 1
    return None


def _is_no_changes(text: str) -> bool:
    """True when the response says "nothing to change", however it is wrapped.

    Accepts the bare word, any number of surrounding backticks, and a fenced
    NO_CHANGES that follows commentary. A response that also carries a JSON
    array is never read this way: patches win, so a contradictory answer
    cannot silently drop edits the model asked for.
    """
    if _NO_CHANGES_RE.match(text.strip()):
        return True
    if "[" in text:
        return False
    return any(_NO_CHANGES_RE.match(block.strip()) for block in _FENCED_BLOCK_RE.findall(text))


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
    # Second asks spent on replies that ignored the output format.
    format_retries: int = 0
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
        self._retry_budget_warned = False

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

        # Check for NO_CHANGES before touching fences: the helper understands
        # every wrapping the prompts invite, while the fence stripper below
        # would flatten a single-line ```NO_CHANGES``` to nothing.
        if _is_no_changes(text):
            return list(input_paragraphs), False

        text = model_json.strip_code_fences(text)

        # Parse the JSON array of patches. The model sometimes puts reasoning
        # in front of it despite the prompt, and that reasoning can itself
        # contain a bracket — so try every `[` in turn, not only the first: a
        # false start must not hide the real array behind it.
        json_start = text.find("[")
        first_list: list | None = None
        for offset in _bracket_positions(text):
            try:
                patches = model_json.loads(text[offset:])
            except (json.JSONDecodeError, ValueError):
                continue
            if _looks_like_patches(patches):
                if offset:
                    logger.debug(
                        "Found JSON array at offset %d (skipped %d chars of preamble)",
                        offset,
                        offset,
                    )
                return self._apply_patches(patches, input_paragraphs)
            if first_list is None and isinstance(patches, list):
                first_list = patches

        # An array that parsed but holds the wrong shape: hand it to the patch
        # applier anyway, so the failure names what was wrong with it rather
        # than reporting the reply as unreadable.
        if first_list is not None:
            return self._apply_patches(first_list, input_paragraphs)

        try:
            patches = model_json.loads(text)
        except json.JSONDecodeError:
            # A reply cut off mid-entry is the one break model_json leaves
            # alone: close the array after the last complete entry, dropping
            # the incomplete one. Also try from the first [ if there's preamble.
            salvage_text = text[json_start:] if json_start > 0 else text
            salvaged = self._try_salvage_truncated_json(salvage_text)
            if salvaged is None:
                raise ValueError(
                    f"Delta response is not valid JSON or NO_CHANGES. "
                    f"First 200 chars: {text[:200]!r}"
                ) from None
            patches = salvaged

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
            new_text = str(p_text).strip()
            if _breaks_dialogue_dash(input_paragraphs[idx], new_text):
                # Refuse this one patch and keep the paragraph. Rejecting the
                # whole chunk would throw away the pass's good work over a
                # punctuation slip, and accepting it prints the slip.
                logger.warning(
                    "%s: patch for paragraph %s turns a dialogue dash into "
                    "guillemets — keeping the original paragraph",
                    self.stage,
                    p_idx,
                )
                continue
            # The same reasoning for the two checks below: one bad patch costs
            # that paragraph's edit, never the rest of the chunk's work.
            try:
                parse_fragment(new_text)
            except ValueError as e:
                logger.warning(
                    "%s: patch for paragraph %s is not well-formed XHTML (%s) — "
                    "keeping the original paragraph",
                    self.stage,
                    p_idx,
                    str(e).split(". First 200 chars")[0],
                )
                continue
            borrowed = _borrowed_paragraph(idx, new_text, input_paragraphs)
            if borrowed is not None:
                logger.warning(
                    "%s: patch for paragraph %s carries a sentence of paragraph %d — "
                    "paragraphs mixed up, keeping the original paragraph",
                    self.stage,
                    p_idx,
                    borrowed,
                )
                continue
            output[idx] = new_text

        return output, output != list(input_paragraphs)

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

    def _retry_budget(self, chunks_total: int) -> int:
        """How many format retries this run may buy.

        A retry is a full-price call. One model having a bad day is worth
        paying for; a model that has stopped following the format at all is
        not — that would silently double the cost of the stage.
        """
        return max(_MIN_FORMAT_RETRIES, chunks_total // 10)

    def _retry_for_format(
        self,
        system: str,
        user: str,
        error: Exception,
        stats: PostprocessStats,
    ) -> CompletionResult | None:
        """Ask the same question again, insisting on the output format.

        Not a repair of the broken reply: when a model answers with analysis
        instead of patches there is no JSON to repair, and turning prose into
        patches would let it invent text nobody vetted. So the chunk is asked
        for again, with the format restated.

        Returns None when the budget is spent or the call fails, leaving the
        caller to report the original failure.
        """
        with self._lock:
            if stats.format_retries >= self._retry_budget(stats.chunks_total):
                if not self._retry_budget_warned:
                    self._retry_budget_warned = True
                    logger.warning(
                        "%s: format-retry budget spent (%d). The model is not following "
                        "the delta format; remaining bad chunks will be skipped, not re-asked.",
                        self.stage,
                        stats.format_retries,
                    )
                return None
            stats.format_retries += 1

        try:
            result = self.provider.complete(
                model=self.model,
                system=system,
                user=user + _FORMAT_RETRY_NOTE.format(reason=str(error)[:200]),
                temperature=self.prompt.temperature,
                max_tokens=self.prompt.max_tokens,
                reasoning_effort=self.prompt.reasoning_effort,
            )
        except Exception as e:  # noqa: BLE001 — the original failure is the one to report
            logger.warning("%s: format retry failed: %s", self.stage, e)
            return None

        if not result.text.strip():
            return None
        with self._lock:
            stats.input_tokens += result.input_tokens
            stats.output_tokens += result.output_tokens
        return result

    def _report_delta_failure(
        self,
        chunk_id: str,
        raw_text: str,
        result: CompletionResult | None,
        error: Exception,
        retry: CompletionResult | None = None,
    ) -> None:
        """Log the failure and keep the reply on disk to look at later.

        The reply that fails to parse is never cached — without this it is
        gone the moment the run ends, and the next such failure has to be
        diagnosed from the 200 characters that fit in the error message.
        """
        path = self._dump_failed_reply(chunk_id, raw_text, result, error, retry)
        logger.warning(
            "%s chunk %s: reply does not parse as delta (%s). %d chars, "
            "finish_reason=%r, max_tokens=%s.%s",
            self.stage,
            chunk_id,
            error,
            len(raw_text),
            result.finish_reason if result else None,
            self.prompt.max_tokens,
            f" Full reply: {path}" if path else "",
        )

    def _dump_failed_reply(
        self,
        chunk_id: str,
        raw_text: str,
        result: CompletionResult | None,
        error: Exception,
        retry: CompletionResult | None = None,
    ) -> Path | None:
        """Write the failing reply next to the cache. Never raises."""
        base = getattr(self.cache, "path", None)
        if base is None:
            return None
        try:
            out_dir = Path(base).parent / "failed"
            out_dir.mkdir(parents=True, exist_ok=True)
            path = out_dir / f"{self.stage}-{chunk_id}.txt"
            parts = [
                f"stage: {self.stage}",
                f"chunk: {chunk_id}",
                f"model: {self.model}",
                f"prompt version: {self.prompt.version}",
                f"max_tokens: {self.prompt.max_tokens}",
                f"finish_reason: {result.finish_reason if result else None!r}",
                f"error: {error}",
                "",
                "--- reply ---",
                raw_text,
            ]
            if retry is not None:
                parts += ["", "--- reply after the format retry ---", retry.text]
            path.write_text("\n".join(parts), encoding="utf-8")
            return path
        except OSError as e:
            logger.debug("Could not write the failing reply for %s: %s", chunk_id, e)
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

        result: CompletionResult | None = None
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
            retry: CompletionResult | None = None
            try:
                output_paragraphs, changed = self._parse_delta_response(
                    raw_text, translated_paragraphs
                )
            except ValueError as e:
                # The model answered with commentary instead of the format the
                # prompt demands. Ask once more rather than dropping the chunk.
                # Never for a cached row: re-reading the cache costs nothing,
                # so a row that no longer parses is a failure, not a purchase.
                retry = None if result is None else self._retry_for_format(system, user, e, stats)
                if retry is None:
                    self._report_delta_failure(chunk_id, raw_text, result, e)
                    raise
                try:
                    output_paragraphs, changed = self._parse_delta_response(
                        retry.text, translated_paragraphs
                    )
                except ValueError as e2:
                    self._report_delta_failure(chunk_id, raw_text, result, e, retry=retry)
                    raise e2 from e
                logger.info("%s chunk %s recovered on the second ask", self.stage, chunk_id)
            # Store reconstructed full text in cache (not the raw delta)
            # so downstream stages can read it as normal ===PARAGRAPH=== format
            full_text = self._reconstruct_full_text(output_paragraphs)
            if result is not None:
                with self._lock:
                    self.cache.put(
                        key=cache_key,
                        stage=self.stage,
                        model=self.model,
                        prompt_version=self.prompt.version,
                        content=full_text,
                        input_tokens=result.input_tokens + (retry.input_tokens if retry else 0),
                        output_tokens=result.output_tokens + (retry.output_tokens if retry else 0),
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
