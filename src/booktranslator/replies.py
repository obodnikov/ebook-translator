"""Check the `===PARAGRAPH N===` replies of the stages that write whole paragraphs.

Translate and reflect both ask a model for N paragraphs, each an XHTML
element that gets spliced into the book. A reply can carry the right number
of markers and still be unusable: a paragraph missing its `</p>`, two `<p>`
under one marker, or the model's own commentary ("wait, I got the number
wrong") pasted between them. It can also have N well-formed elements and a
hole in the text: a model that split a paragraph and noticed the count was off
keeps the count right by dropping a line of speech or the end of the chunk.
Those have to be caught before the reply is cached — once cached, every later
run reads the same broken text back without asking the model again.

Two things are repaired instead of rejected, because all the text is there and
a re-ask costs a whole chunk: a paragraph split under its one marker
(`join_split_paragraphs`) and a paragraph split over two markers
(`rebuild_over_split_reply`). Nothing else is, and both results go through every
check above.
"""

from __future__ import annotations

import logging
import math
import re
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from itertools import pairwise
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

_EPUB_NS = "http://www.idpf.org/2007/ops"
_EPUB_ATTRIBUTE_RE = re.compile(r"\sepub:[\w-]+\s*=")
_FIRST_TAG_RE = re.compile(r"^\s*<[A-Za-z][\w.-]*")

_TAG_RE = re.compile(r"<[^>]+>")
# How direct speech shows in the source, and in a translation of it.
_SOURCE_SPEECH_RE = re.compile(r'[“"‘]')
_REPLY_SPEECH_RE = re.compile(r'[—–―«»„“”"]')

# A paragraph and its translation run about as long as each other: the median
# chunk came out at 0.97-1.01 of the source on five books. The limits below were
# measured on those books: they flag the 8 chunks of The Tainted Cup that really
# lost text, and none of the 468 chunks of Bear Head, Bee Speaker, Shroud and
# Foxglove Summer. Measure again on real caches before changing them. A false
# alarm is a paid re-ask, a miss is a hole in the book.
_MIN_COMPARED_CHARS = 30  # shorter lines vary too much to say anything
_MIN_LENGTH_RATIO = 0.55
_MAX_LENGTH_RATIO = 1.8
_MAX_OFF_PARAGRAPHS = 2  # a translator may be free with a paragraph or two
_LOST_SPEECH_RATIO = 0.6
# Real splits joined back came to 0.82-1.27 of their source paragraph. The one
# false shape seen, The Tainted Cup ch52_c02, came to 1.83: under one marker the
# model wrote that paragraph and the next one, then the next one again.
_MAX_JOINED_RATIO = 1.4
# A sentence has to be at least this long before finding it inside another
# paragraph means the paragraphs were mixed up. Short lines — "Да.",
# "– Хорошо, – сказал я." — recur in any novel and prove nothing.
MIN_TELLING_SENTENCE = 40
SENTENCE_END_RE = re.compile(r"(?<=[.!?…»])\s+")

# Rebuilding an over-split reply: a source paragraph that opens with narration and
# then speech may take two reply paragraphs, when the first is narration alone, the
# second opens as speech, and each is about as long as the same part of the source
# (a ratio below e^0.75, about twice, either way). Nothing else. Measured on The
# Tainted Cup: 235 of 235 good translations split at random came back exactly, and
# with a looser rule that compared only the whole paragraph, 52% did.
_NARRATION_THEN_SPEECH_RE = re.compile(r'^([^“"‘]{3,}?)\s*([“"‘].*)$', re.DOTALL)
_OPENS_AS_SPEECH_RE = re.compile(r'^[—–―«„“"]')
_MAX_PART_MISMATCH = 0.75
_SPLIT_PENALTY = 1.0

# Appended to the user message when a reply is rejected. Kept in code, not in
# the prompt file: it is only ever sent on the second ask, so the prompt
# version — and every cached result under it — stays untouched.
RETRY_NOTE = """

## Your previous reply was rejected

{reason}

Answer again for the same paragraphs. Output exactly {expected} `===PARAGRAPH N===`
markers, numbered 1 to {expected}, each followed by exactly ONE well-formed XHTML
element with every tag closed. Each source paragraph is translated in full under its
own marker, in the order of the source: narration and speech stay together in one
element, you never drop a line to make the count fit, and marker {expected} holds the
last source paragraph, never an earlier one. No commentary, no corrections, nothing
before, between or after the paragraphs."""

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


def declare_epub_prefix(fragment: str) -> str:
    """Add `xmlns:epub` when a model kept an `epub:` attribute and lost its declaration.

    A page-break marker comes with the declaration on its paragraph, and now and then a
    model copies the marker without it. The text is fine and lxml refuses the whole
    fragment, so as with a bare `&`, this is fixed here rather than paid for again.
    """
    if "xmlns:epub" in fragment or not _EPUB_ATTRIBUTE_RE.search(fragment):
        return fragment
    declaration = f' xmlns:epub="{_EPUB_NS}"'
    return _FIRST_TAG_RE.sub(lambda tag: tag.group(0) + declaration, fragment, count=1)


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
    fragment = declare_epub_prefix(escape_bare_ampersands(fragment))
    try:
        return etree.fromstring(fragment)
    except etree.XMLSyntaxError as e:
        raise ValueError(
            f"Translated fragment is not well-formed XML: {e}. First 200 chars: {fragment[:200]!r}"
        ) from e


def _numbered(numbers: list[int]) -> str:
    label = "Paragraph" if len(numbers) == 1 else "Paragraphs"
    return f"{label} {', '.join(map(str, numbers))}"


def parse_paragraphs(text: str, expected_n: int) -> list[etree._Element]:
    """Split a reply and parse every paragraph; all of them or ValueError.

    The error names every broken paragraph, numbers first: it goes into the second
    ask cut to `RETRY_NOTE`'s limit, and a model told of only the first one leaves
    the others as they were.
    """
    elements: list[etree._Element] = []
    broken: list[tuple[int, ValueError]] = []
    for number, fragment in enumerate(split_paragraphs(text, expected_n), start=1):
        try:
            elements.append(parse_fragment(fragment))
        except ValueError as e:
            broken.append((number, e))
    if broken:
        numbers = _numbered([number for number, _ in broken])
        first_error = broken[0][1]
        if len(broken) == 1:
            raise ValueError(f"{numbers} of {expected_n}: {first_error}") from first_error
        raise ValueError(f"{numbers} of {expected_n} are broken. First: {first_error}") from (
            first_error
        )
    return elements


def _visible_text(fragment: str) -> str:
    return " ".join(_TAG_RE.sub("", fragment).split())


def _with_markers(fragments: list[str]) -> str:
    return "\n".join(
        f"===PARAGRAPH {number}===\n{fragment}" for number, fragment in enumerate(fragments, 1)
    )


def _lost_speech(source: str, reply: str) -> bool:
    """The source paragraph has direct speech; its translation is narration alone."""
    if not _SOURCE_SPEECH_RE.search(source) or _REPLY_SPEECH_RE.search(reply):
        return False
    return reply.endswith(":") or len(reply) < _LOST_SPEECH_RATIO * len(source)


def check_alignment(source_fragments: list[str], reply_fragments: list[str]) -> None:
    """Raise ValueError when paragraphs no longer match their source paragraphs.

    Both lists have the same length: this looks for what the count cannot see, a
    reply whose paragraphs moved down the source or lost part of their text.
    """
    pairs = [
        (_visible_text(source), _visible_text(reply))
        for source, reply in zip(source_fragments, reply_fragments, strict=True)
    ]
    total = len(pairs)
    off = [
        number
        for number, (source, reply) in enumerate(pairs, start=1)
        if len(source) >= _MIN_COMPARED_CHARS
        and not _MIN_LENGTH_RATIO <= len(reply) / len(source) <= _MAX_LENGTH_RATIO
    ]
    if len(off) > _MAX_OFF_PARAGRAPHS:
        raise ValueError(
            f"{_numbered(off)} of {total} are far from their source paragraph in length: "
            "the paragraphs no longer line up, so text was moved or dropped."
        )
    repeats = [
        number
        for number, ((_, reply), (_, following)) in enumerate(pairwise(pairs), start=1)
        if any(
            len(sentence) >= MIN_TELLING_SENTENCE and sentence in following
            for sentence in SENTENCE_END_RE.split(reply)
        )
    ]
    if repeats:
        raise ValueError(
            f"{_numbered(repeats)} of {total}: a sentence here comes again in the next "
            "paragraph, so a paragraph holds the translation of its neighbour as well."
        )
    lost = [
        number
        for number, (source, reply) in enumerate(pairs, start=1)
        if _lost_speech(source, reply)
    ]
    if lost:
        raise ValueError(
            f"{_numbered(lost)} of {total}: the source has direct speech here and the "
            f"translation left it out. First: {pairs[lost[0] - 1][1][:80]!r}"
        )


def _append_text(element: etree._Element, text: str) -> None:
    if len(element):
        element[-1].tail = (element[-1].tail or "") + text
    else:
        element.text = (element.text or "") + text


def _sibling_parts(fragment: str) -> list[etree._Element] | None:
    """The same-looking sibling elements in `fragment`; None if it holds anything else."""
    declaration = f' xmlns:epub="{_EPUB_NS}"' if _EPUB_ATTRIBUTE_RE.search(fragment) else ""
    try:
        holder = etree.fromstring(
            f"<holder{declaration}>{escape_bare_ampersands(fragment)}</holder>"
        )
    except etree.XMLSyntaxError:
        return None
    parts = list(holder)
    if len(parts) < 2 or (holder.text or "").strip() or not isinstance(parts[0].tag, str):
        return None
    first = parts[0]
    for part in parts:
        same_look = part.tag == first.tag and dict(part.attrib) == dict(first.attrib)
        if not same_look or (part.tail or "").strip():
            return None
    return parts


def _are_one_paragraph(parts: list[etree._Element], source: str, neighbours: list[str]) -> bool:
    """The parts fit one source paragraph and repeat nothing of the neighbouring markers."""
    texts = [" ".join("".join(part.itertext()).split()) for part in parts]
    limit = _MAX_JOINED_RATIO * max(len(_visible_text(source)), _MIN_COMPARED_CHARS)
    if sum(map(len, texts)) > limit:
        return False
    nearby = " ".join(_visible_text(neighbour) for neighbour in neighbours)
    return not any(
        len(text) >= MIN_TELLING_SENTENCE and text[:MIN_TELLING_SENTENCE] in nearby
        for text in texts
    )


def _merged(parts: list[etree._Element]) -> str:
    first = parts[0]
    for part in parts[1:]:
        _append_text(first, f" {part.text or ''}")
        for child in list(part):
            first.append(child)
    first.tail = None
    return etree.tostring(first, encoding="unicode")


def join_split_paragraphs(text: str, source_fragments: list[str]) -> tuple[str, list[int]]:
    """Join a paragraph the model split in two under its one marker.

    Returns the reply and the numbers of the paragraphs joined; a reply with nothing
    to join comes back as it was.

    This is what the prompt asks for, nothing is lost, and a re-ask costs a whole
    chunk. It is deliberately narrow, because two elements under one marker can also
    be a second source paragraph in the wrong place, and joining that would print it
    twice: only same-looking siblings, only if the result fits the source paragraph,
    and only if no part repeats a neighbouring marker. A wrong number of markers is
    left to be rejected: it can't be repaired without guessing.
    """
    fragments = split_markers(text)
    if fragments is None or len(fragments) != len(source_fragments):
        return text, []
    joined: list[int] = []
    for number, source in enumerate(source_fragments, start=1):
        parts = _sibling_parts(fragments[number - 1])
        neighbours = fragments[max(number - 2, 0) : number - 1] + fragments[number : number + 1]
        if parts is not None and _are_one_paragraph(parts, source, neighbours):
            fragments[number - 1] = _merged(parts)
            joined.append(number)
    if not joined:
        return text, []
    return _with_markers(fragments), joined


def _mismatch(source: str, reply: str) -> float:
    return abs(math.log(max(len(reply), 1) / max(len(source), 1)))


def _one_part_cost(source: str, reply: str) -> float:
    """How badly a reply paragraph fits; a long paragraph off by a factor counts for more."""
    return _mismatch(source, reply) * math.sqrt(max(len(source), MIN_TELLING_SENTENCE))


def _two_part_cost(source: str, narration: str, speech: str) -> float | None:
    """How badly narration and speech fit one source paragraph; None if they can't be it."""
    parts = _NARRATION_THEN_SPEECH_RE.match(source)
    if parts is None or _REPLY_SPEECH_RE.search(narration):
        return None
    if not _OPENS_AS_SPEECH_RE.match(speech):
        return None
    if max(_mismatch(parts[1], narration), _mismatch(parts[2], speech)) >= _MAX_PART_MISMATCH:
        return None
    return _SPLIT_PENALTY + _one_part_cost(parts[1], narration) + _one_part_cost(parts[2], speech)


def _keep_cheaper(
    states: dict[int, tuple[float, list[int]]], used: int, cost: float, sizes: list[int]
) -> None:
    if used not in states or cost < states[used][0]:
        states[used] = (cost, sizes)


def _group_sizes(sources: list[str], replies: list[str]) -> list[int] | None:
    """How many reply paragraphs each source paragraph takes, 1 or 2; None if nothing fits.

    The cheapest way to use every reply paragraph exactly once and in order. A state
    is the number of extra markers used up so far: with that and the source paragraph,
    the position in the reply is known.
    """
    extra = len(replies) - len(sources)
    best: dict[int, tuple[float, list[int]]] = {0: (0.0, [])}
    for index, source in enumerate(sources):
        following: dict[int, tuple[float, list[int]]] = {}
        for used, (cost, sizes) in best.items():
            position = index + used
            if position < len(replies):
                one = cost + _one_part_cost(source, replies[position])
                _keep_cheaper(following, used, one, [*sizes, 1])
            if used < extra and position + 1 < len(replies):
                split = _two_part_cost(source, replies[position], replies[position + 1])
                if split is not None:
                    _keep_cheaper(following, used + 1, cost + split, [*sizes, 2])
        best = following
    return best[extra][1] if extra in best else None


def rebuild_over_split_reply(text: str, source_fragments: list[str]) -> tuple[str, list[int]]:
    """Put a complete reply with too many markers back under one marker per paragraph.

    Returns the reply and the numbers of the source paragraphs rebuilt; a reply this
    can't place comes back as it was, to be rejected for its count.

    The model's first answer often has every line and splits narration from speech,
    as Russian typography would. Asked again for "exactly N", it stops at marker N and
    loses the end of the chunk, so this first answer is the best one there will be.
    Only that one shape is put back: see the limits at the top of this module. Numbering
    that isn't 1 to M means two answers in one reply or a lost marker, not a split.
    """
    fragments = split_markers(text)
    if fragments is None or len(fragments) <= len(source_fragments):
        return text, []
    numbers = [int(n) for n in PARAGRAPH_MARKER_RE.findall(_strip_code_fence(text))]
    if numbers != list(range(1, len(fragments) + 1)):
        return text, []
    sizes = _group_sizes(
        [_visible_text(source) for source in source_fragments],
        [_visible_text(fragment) for fragment in fragments],
    )
    if sizes is None:
        return text, []
    grouped: list[str] = []
    rebuilt: list[int] = []
    position = 0
    for number, size in enumerate(sizes, start=1):
        if size == 1:
            grouped.append(fragments[position])
        else:
            parts = _sibling_parts("\n".join(fragments[position : position + size]))
            if parts is None:
                return text, []
            grouped.append(_merged(parts))
            rebuilt.append(number)
        position += size
    return _with_markers(grouped), rebuilt


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
    joined: list[int] = field(default_factory=list)
    rebuilt: list[int] = field(default_factory=list)

    def meta(self, chunk_id: str) -> dict:
        """What a stage stores beside the text; a repair is recorded, never silent."""
        meta: dict = {"chunk_id": chunk_id}
        if self.joined:
            meta["joined_paragraphs"] = self.joined
        if self.rebuilt:
            meta["rebuilt_paragraphs"] = self.rebuilt
        return meta


class ReplyChecker:
    """Ask for a stage's paragraphs, check the reply, ask once more if rejected.

    Shared by translate and reflect so the two cannot drift apart. Every call
    made is added to the stats, including rejected ones: they were paid for.
    A reply that fails twice is written to `failed/` and the error re-raised.

    The repairs live here, not in `parse_fragment`: post-processing checks its patches
    with that function, and a patch with two elements must stay rejected.
    `CheckedReply.text` holds the repaired reply, so that every later reader of the
    cached row sees one element per marker.
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
        source: list[str],
        stats: RetryStats,
    ) -> CheckedReply:
        """`source` holds the XHTML paragraphs asked about, one per expected marker."""
        result = self._paid(complete(system, user), stats)
        try:
            return self._checked(result.text, source, chunk_id, result)
        except ValueError as e:
            retry = self._retry(complete, system, user, len(source), e, stats)
            if retry is None:
                self._report(chunk_id, result, e)
                raise
            try:
                reply = self._checked(retry.text, source, chunk_id, result, retry)
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

    def _checked(
        self,
        text: str,
        source: list[str],
        chunk_id: str,
        result: CompletionResult,
        retry: CompletionResult | None = None,
    ) -> CheckedReply:
        text, rebuilt = rebuild_over_split_reply(text, source)
        text, joined = join_split_paragraphs(text, source)
        elements = parse_paragraphs(text, len(source))
        check_alignment(source, split_paragraphs(text, len(source)))
        if rebuilt:
            logger.warning(
                "%s chunk %s: %s came back split over two markers and rebuilt into one element.",
                self.stage,
                chunk_id,
                _numbered(rebuilt),
            )
        if joined:
            logger.warning(
                "%s chunk %s: %s came back split in two and joined into one element.",
                self.stage,
                chunk_id,
                _numbered(joined),
            )
        return CheckedReply(
            text=text,
            elements=elements,
            input_tokens=result.input_tokens + (retry.input_tokens if retry else 0),
            output_tokens=result.output_tokens + (retry.output_tokens if retry else 0),
            joined=joined,
            rebuilt=rebuilt,
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
