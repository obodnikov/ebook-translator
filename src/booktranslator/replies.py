"""Check the `===PARAGRAPH N===` replies of the stages that write whole paragraphs.

Translate and reflect both ask a model for N paragraphs, each an XHTML
element that gets spliced into the book. A reply can carry the right number
of markers and still be unusable: a paragraph missing its `</p>`, two `<p>`
under one marker, or the model's own commentary ("wait, I got the number
wrong") pasted between them. Those have to be caught before the reply is
cached — once cached, every later run reads the same broken text back
without asking the model again.
"""

from __future__ import annotations

import logging
import re
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from lxml import etree

from .provider import CompletionResult

logger = logging.getLogger(__name__)

PARAGRAPH_MARKER_RE = re.compile(r"^===PARAGRAPH\s+(\d+)===\s*$", re.MULTILINE)

# Matches a `&` that does NOT start a valid XML entity
# (&amp; &lt; &gt; &quot; &apos; or numeric like &#123; / &#xAF;).
# Used to fix stray ampersands in LLM output like "M&S", "AT&T".
_BARE_AMPERSAND_RE = re.compile(r"&(?!(?:amp|lt|gt|quot|apos|#\d+|#x[0-9a-fA-F]+);)")

# Appended to the user message when a reply is rejected. Kept in code, not in
# the prompt file: it is only ever sent on the second ask, so the prompt
# version — and every cached result under it — stays untouched.
RETRY_NOTE = """

## Your previous reply was rejected

{reason}

Answer again for the same paragraphs. Output exactly {expected} `===PARAGRAPH N===`
markers, numbered 1 to {expected}, each followed by exactly ONE well-formed XHTML
element with every tag closed. No commentary, no corrections, nothing before,
between or after the paragraphs."""

# Retries are full-price calls, so a run gets a small budget: enough to absorb
# the odd bad reply, not enough to pay twice for a whole stage.
MIN_RETRIES = 3


def retry_budget(chunks_total: int) -> int:
    """How many rejected replies a run may ask for again."""
    return max(MIN_RETRIES, chunks_total // 10)


def escape_bare_ampersands(fragment: str) -> str:
    """Replace stray `&` (not part of a valid entity) with `&amp;`.

    LLMs sometimes return English brand names like "M&S" or "AT&T"
    verbatim inside XHTML fragments. Those break lxml parsing. We fix
    them here rather than asking the model to escape, because the
    instruction is easy to miss on a long chunk.
    """
    return _BARE_AMPERSAND_RE.sub("&amp;", fragment)


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines)
    return text


def split_markers(text: str) -> list[str] | None:
    """Split a reply into fragments by `===PARAGRAPH N===` markers, however many.

    Returns None when the reply has no markers at all.
    """
    text = _strip_code_fence(text)
    matches = list(PARAGRAPH_MARKER_RE.finditer(text))
    if not matches:
        return None
    return [
        text[m.end() : matches[i + 1].start() if i + 1 < len(matches) else len(text)].strip()
        for i, m in enumerate(matches)
    ]


def split_paragraphs(text: str, expected_n: int) -> list[str]:
    """Split a reply into exactly `expected_n` paragraph fragments, or raise ValueError."""
    fragments = split_markers(text)
    if fragments is None:
        raise ValueError(
            "No '===PARAGRAPH N===' markers found in response. "
            f"First 200 chars: {_strip_code_fence(text)[:200]!r}"
        )
    if len(fragments) != expected_n:
        raise ValueError(f"Expected {expected_n} paragraphs, got {len(fragments)}.")
    return fragments


def parse_fragment(fragment: str) -> etree._Element:
    """Parse one translated paragraph into an lxml element, or raise ValueError."""
    fragment = escape_bare_ampersands(fragment)
    try:
        return etree.fromstring(fragment)
    except etree.XMLSyntaxError as e:
        raise ValueError(
            f"Translated fragment is not well-formed XML: {e}. First 200 chars: {fragment[:200]!r}"
        ) from e


def parse_paragraphs(text: str, expected_n: int) -> list[etree._Element]:
    """Split a reply and parse every paragraph; all of them or ValueError.

    The error names the paragraph, so the log says where the reply broke.
    """
    elements: list[etree._Element] = []
    for number, fragment in enumerate(split_paragraphs(text, expected_n), start=1):
        try:
            elements.append(parse_fragment(fragment))
        except ValueError as e:
            raise ValueError(f"Paragraph {number} of {expected_n}: {e}") from e
    return elements


def parse_cached_paragraphs(
    text: str, expected_n: int, *, stage: str, chunk_id: str
) -> list[etree._Element]:
    """Check a cached reply without ever paying to replace it.

    Reading the cache is free, so a cached reply that no longer passes is a
    failure to surface, not a reason to buy a new one. Rows cached before
    replies were checked can be broken; the error names the command that
    removes them.
    """
    try:
        return parse_paragraphs(text, expected_n)
    except ValueError as e:
        raise ValueError(
            f"{e} The cached {stage} reply is broken; remove it with `btrans cache clear "
            f"<book workdir> --stage {stage} --chunk {chunk_id}` and run again."
        ) from e


class RetryStats(Protocol):
    """The stats fields `ReplyChecker` reads and updates."""

    chunks_total: int
    retries: int
    input_tokens: int
    output_tokens: int


@dataclass
class CheckedReply:
    """A reply that passed: its text, parsed paragraphs, and what it cost."""

    text: str
    elements: list[etree._Element]
    input_tokens: int
    output_tokens: int


class ReplyChecker:
    """Ask for a stage's paragraphs, check the reply, ask once more if rejected.

    Shared by translate and reflect so the two cannot drift apart. Every call
    made is added to the stats, including rejected ones: they were paid for.
    A reply that fails twice is written to `failed/` and the error re-raised.
    """

    def __init__(
        self,
        *,
        stage: str,
        model: str,
        prompt_version: str,
        cache_path: Path | None,
        lock: threading.Lock,
    ):
        self.stage = stage
        self.model = model
        self.prompt_version = prompt_version
        self.cache_path = cache_path
        self._lock = lock
        self._budget_warned = False

    def ask(
        self,
        complete: Callable[[str, str], CompletionResult],
        system: str,
        user: str,
        *,
        chunk_id: str,
        expected: int,
        stats: RetryStats,
    ) -> CheckedReply:
        result = self._paid(complete(system, user), stats)
        try:
            return self._checked(result.text, expected, result)
        except ValueError as e:
            retry = self._retry(complete, system, user, expected, e, stats)
            if retry is None:
                self._report(chunk_id, result, e)
                raise
            try:
                reply = self._checked(retry.text, expected, result, retry)
            except ValueError as e2:
                self._report(chunk_id, result, e2, retry)
                raise e2 from e
            logger.info("%s chunk %s recovered on the second ask", self.stage, chunk_id)
            return reply

    def _paid(self, result: CompletionResult, stats: RetryStats) -> CompletionResult:
        with self._lock:
            stats.input_tokens += result.input_tokens
            stats.output_tokens += result.output_tokens
        return result

    @staticmethod
    def _checked(
        text: str,
        expected: int,
        result: CompletionResult,
        retry: CompletionResult | None = None,
    ) -> CheckedReply:
        elements = parse_paragraphs(text, expected)
        return CheckedReply(
            text=text,
            elements=elements,
            input_tokens=result.input_tokens + (retry.input_tokens if retry else 0),
            output_tokens=result.output_tokens + (retry.output_tokens if retry else 0),
        )

    def _retry(
        self,
        complete: Callable[[str, str], CompletionResult],
        system: str,
        user: str,
        expected: int,
        error: Exception,
        stats: RetryStats,
    ) -> CompletionResult | None:
        """Returns None when the run's budget is spent or the call itself fails."""
        with self._lock:
            if stats.retries >= retry_budget(stats.chunks_total):
                if not self._budget_warned:
                    self._budget_warned = True
                    logger.warning(
                        "%s: retry budget spent (%d). Remaining rejected chunks "
                        "will be skipped, not re-asked.",
                        self.stage,
                        stats.retries,
                    )
                return None
            stats.retries += 1
        try:
            note = RETRY_NOTE.format(reason=str(error)[:300], expected=expected)
            return self._paid(complete(system, user + note), stats)
        except Exception as e:  # noqa: BLE001 — the original rejection is the one to report
            logger.warning("%s: retry failed: %s", self.stage, e)
            return None

    def _report(
        self,
        chunk_id: str,
        result: CompletionResult,
        error: Exception,
        retry: CompletionResult | None = None,
    ) -> None:
        path = dump_failed_reply(
            self.cache_path,
            stage=self.stage,
            chunk_id=chunk_id,
            model=self.model,
            prompt_version=self.prompt_version,
            error=error,
            reply=result.text,
            finish_reason=result.finish_reason,
            retry_reply=retry.text if retry else None,
        )
        # The reply the error is about: the retry's, when there was one.
        last = retry or result
        logger.warning(
            "%s chunk %s: reply rejected%s (%s). %d chars, finish_reason=%r.%s",
            self.stage,
            chunk_id,
            " after the retry" if retry else "",
            error,
            len(last.text),
            last.finish_reason,
            f" Full reply: {path}" if path else "",
        )


def dump_failed_reply(
    cache_path: Path | None,
    *,
    stage: str,
    chunk_id: str,
    model: str,
    prompt_version: str,
    error: Exception,
    reply: str,
    finish_reason: str | None = None,
    retry_reply: str | None = None,
) -> Path | None:
    """Write a rejected reply to `failed/<stage>-<chunk>.txt` beside the cache.

    A rejected reply is never cached, so without this it is gone when the run
    ends. Never raises.
    """
    if cache_path is None:
        return None
    try:
        out_dir = Path(cache_path).parent / "failed"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{stage}-{chunk_id}.txt"
        parts = [
            f"stage: {stage}",
            f"chunk: {chunk_id}",
            f"model: {model}",
            f"prompt version: {prompt_version}",
            f"finish_reason: {finish_reason!r}",
            f"error: {error}",
            "",
            "--- reply ---",
            reply,
        ]
        if retry_reply is not None:
            parts += ["", "--- reply after the retry ---", retry_reply]
        path.write_text("\n".join(parts), encoding="utf-8")
        return path
    except OSError as e:
        logger.debug("Could not write the failing reply for %s: %s", chunk_id, e)
        return None
