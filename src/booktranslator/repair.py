"""Targeted repair of translated chunks from the judge's issue list.

The judge quotes the fragment it objects to and gives a correction:
`grammar: p.11 «Рой мёртв» → «Рой мертва»`. Half of those, in the mechanical
categories, are exact single substitutions.

This module turns such an issue into a **candidate** — paragraph, old text,
new text — and applies nothing on its own. A dry run over Bear Head showed why:
of 32 candidates, two were grammatically convincing and destroyed the meaning.
«ЧелОС нужна ты» ("HumOS needs you") became «ЧелОС нужна тебе» ("you need
HumOS"), reversing it; «Но которую меня?», deliberately awkward because the
original says "But which me?", was flattened to «Но какую?». Neither can be
told from a real fix without reading the source text, which this module never
sees.

So the split is: here we decide what is *mechanically* applicable — the quote
occurs exactly once, in the paragraph the judge named — and a model that has
the original decides whether each candidate should be applied at all.

Design: docs/design/2026-08-03-judge-repair-stage-and-reasoning-on-gateway.md §5
"""

from __future__ import annotations

import json
import logging
import re
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from . import model_json
from .cache import Cache
from .models import SeriesGlossary
from .prompts import Prompt, load_prompt, render_prompt
from .provider import OpenRouterProvider
from .replies import split_markers
from .series import render_for_prompt

logger = logging.getLogger(__name__)

LANG_NAMES = {"en": "English", "ru": "Russian", "hu": "Hungarian"}

# Issue shape produced by prompts/judge.md v3:
#   "grammar: p.11 «было тридцать одного года» → «было тридцать один год»"
_CATEGORY_RE = re.compile(r"^\s*([a-z_]+)\s*[:/]", re.IGNORECASE)
_PARAGRAPH_RE = re.compile(r"\bp\.\s*(\d+)", re.IGNORECASE)
# Both arrow forms appear in practice: the judge is asked for → but writes -> too.
_PAIR_RE = re.compile(r"«([^«»]{2,})»\s*(?:→|->)\s*«([^«»]{1,})»")

# Categories the first pass acts on. Kept here as the default; callers may
# narrow or widen it from config without touching this module.
#
# Measured on Bear Head, 36 candidates through the gatekeeper: grammar was
# rejected 3 of 11, accuracy 7 of 20, glossary 4 of 5. Accuracy behaves like
# grammar and carries the most defects, so it is in. Glossary is out: the
# judge's glossary notes tend to be judgement calls (reordering a character's
# name, swapping one coined term for another) rather than substitutions.
# Re-enable it for a
# single run with --categories when a book leans on its glossary.
MECHANICAL_CATEGORIES = ("grammar", "markup", "accuracy")


class Skip(StrEnum):
    """Why an issue was not applied deterministically."""

    NO_CATEGORY = "не удалось определить категорию"
    OUT_OF_SCOPE = "категория вне списка"
    NO_PARAGRAPH = "не указан номер абзаца"
    BAD_PARAGRAPH = "номер абзаца вне диапазона"
    NO_PAIR = "не записано в виде «X» → «Y»"
    QUOTE_ABSENT = "цитаты нет в названном абзаце"
    QUOTE_AMBIGUOUS = "цитата встречается в абзаце больше одного раза"
    NO_OP = "замена ничего не меняет"


@dataclass(frozen=True)
class ParsedIssue:
    """An issue broken into the parts a substitution needs."""

    raw: str
    category: str | None
    paragraph: int | None
    old: str | None
    new: str | None


@dataclass(frozen=True)
class Candidate:
    """A substitution that *could* be made. Whether it should is decided later."""

    issue: ParsedIssue
    paragraph: int  # 1-based, as the judge numbers them
    old: str
    new: str


@dataclass
class Proposal:
    """What the deterministic pass found in one chunk. Nothing is applied yet."""

    candidates: list[Candidate] = field(default_factory=list)
    skipped: list[tuple[ParsedIssue, Skip]] = field(default_factory=list)

    @property
    def deferred(self) -> list[ParsedIssue]:
        """Issues the model pass could still attempt as free-form edits.

        Excludes the ones a model should not retry: a quote the judge invented
        is not made real by asking a larger model about it, and a category out
        of scope was never ours to act on.
        """
        return [issue for issue, _ in self.deferred_with_reasons]

    @property
    def deferred_with_reasons(self) -> list[tuple[ParsedIssue, Skip]]:
        """`deferred`, each with why it was not a substitution."""
        hopeless = {Skip.QUOTE_ABSENT, Skip.OUT_OF_SCOPE, Skip.NO_CATEGORY, Skip.NO_OP}
        return [(issue, why) for issue, why in self.skipped if why not in hopeless]


def parse_issue(raw: str) -> ParsedIssue:
    """Split one judge issue into category, paragraph number and the pair."""
    cat_m = _CATEGORY_RE.match(raw)
    par_m = _PARAGRAPH_RE.search(raw)
    pair_m = _PAIR_RE.search(raw)
    return ParsedIssue(
        raw=raw,
        category=cat_m.group(1).lower() if cat_m else None,
        paragraph=int(par_m.group(1)) if par_m else None,
        old=pair_m.group(1).strip() if pair_m else None,
        new=pair_m.group(2).strip() if pair_m else None,
    )


def _classify(
    issue: ParsedIssue, paragraphs: list[str], categories: tuple[str, ...]
) -> Skip | None:
    """Return why this issue cannot be substituted, or None if it can."""
    if issue.category is None:
        return Skip.NO_CATEGORY
    if issue.category not in categories:
        return Skip.OUT_OF_SCOPE
    if issue.paragraph is None:
        return Skip.NO_PARAGRAPH
    if not 1 <= issue.paragraph <= len(paragraphs):
        return Skip.BAD_PARAGRAPH
    if issue.old is None or issue.new is None:
        return Skip.NO_PAIR
    if issue.old == issue.new:
        return Skip.NO_OP

    occurrences = paragraphs[issue.paragraph - 1].count(issue.old)
    if occurrences == 0:
        # Either the judge invented the quote (6% of issues by measurement) or
        # a later stage already fixed it. Both mean: do not touch it.
        return Skip.QUOTE_ABSENT
    if occurrences > 1:
        # Which one did the judge mean? Guessing risks corrupting the wrong
        # sentence, so this goes to the model pass instead.
        return Skip.QUOTE_AMBIGUOUS
    return None


def propose(
    paragraphs: list[str],
    issues: list[str],
    categories: tuple[str, ...] = MECHANICAL_CATEGORIES,
) -> Proposal:
    """Find the issues that are unambiguous single substitutions.

    Applies nothing: a candidate here means "this edit can be made cleanly",
    never "this edit is correct". Correctness needs the source text, which
    only the model pass has.
    """
    proposal = Proposal()

    for raw in issues:
        issue = parse_issue(raw)
        why = _classify(issue, paragraphs, categories)
        if why is not None:
            proposal.skipped.append((issue, why))
            continue

        # Safe by construction: _classify established exactly one occurrence.
        assert issue.paragraph is not None and issue.old is not None and issue.new is not None
        proposal.candidates.append(
            Candidate(issue=issue, paragraph=issue.paragraph, old=issue.old, new=issue.new)
        )

    return proposal


def apply_candidates(paragraphs: list[str], candidates: list[Candidate]) -> list[str]:
    """Apply the candidates that were accepted, and only those.

    Paragraphs come back in the same order and the same number, as the pipeline
    contract requires. Nothing outside a quoted fragment is touched, and a
    paragraph with no accepted candidate is returned byte-identical.

    A candidate whose quote no longer occurs exactly once is silently dropped:
    an earlier accepted edit in the same paragraph may have disturbed it, and
    guessing at that point would defeat the whole design.
    """
    out = list(paragraphs)
    for c in candidates:
        idx = c.paragraph - 1
        if not 0 <= idx < len(out):
            continue
        if out[idx].count(c.old) != 1:
            continue
        out[idx] = out[idx].replace(c.old, c.new, 1)
    return out


# ---------------------------------------------------------------------------
# The stage: propose, have a model vet each candidate, apply what survives
# ---------------------------------------------------------------------------


@dataclass
class RepairStats:
    chunks_total: int = 0
    chunks_cached: int = 0
    chunks_repaired: int = 0
    chunks_failed: int = 0
    candidates: int = 0
    accepted: int = 0
    rejected: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    # Issues nothing can act on yet: those whose fix belongs in the glossary,
    # and those the deterministic pass could not turn into a substitution.
    glossary_issues: list[tuple[str, str]] = field(default_factory=list)
    # (chunk_id, issue as the judge wrote it, why it was not a substitution)
    unhandled: list[tuple[str, str, str]] = field(default_factory=list)


@dataclass
class RepairResult:
    chunk_id: str
    paragraphs: list[str]
    accepted: list[tuple[Candidate, str]] = field(default_factory=list)
    rejected: list[tuple[Candidate, str]] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(self.accepted)


class Repairer:
    """Applies the judge's mechanical corrections, one vetted candidate at a time."""

    def __init__(
        self,
        provider: OpenRouterProvider,
        prompt_path: Path,
        cache: Cache,
        glossary: SeriesGlossary | None,
        *,
        model: str,
        categories: tuple[str, ...] = MECHANICAL_CATEGORIES,
        source_lang: str = "en",
        target_lang: str = "ru",
    ):
        self.provider = provider
        self.prompt: Prompt = load_prompt(prompt_path)
        self.cache = cache
        self.model = model
        self.categories = categories
        self.source_lang = source_lang
        self.target_lang = target_lang
        self._glossary_block = render_for_prompt(glossary) if glossary else "(no glossary provided)"
        self._lock = threading.Lock()

    # -- response parsing ---------------------------------------------------

    @staticmethod
    def parse_verdicts(text: str) -> dict[int, tuple[bool, str]]:
        """Parse the gatekeeper's JSON array into {n: (accepted, why)}.

        Raises ValueError on anything unparseable: a verdict list we cannot
        read must not be silently treated as "accept everything".
        """
        t = text.strip()
        if t.startswith("```"):
            t = "\n".join(ln for ln in t.splitlines() if not ln.startswith("```")).strip()
        m = re.search(r"\[.*]", t, re.S)
        if m:
            try:
                data = model_json.loads(m.group(0))
            except json.JSONDecodeError as e:
                raise ValueError(f"Repair verdicts are not valid JSON: {e}") from e
        else:
            # With a single proposal, models answer with the lone verdict object
            # instead of a one-entry array. Same verdict, so read it as one.
            try:
                data = model_json.loads(t)
            except json.JSONDecodeError:
                data = None
            if not (isinstance(data, dict) and "n" in data):
                raise ValueError(
                    f"Repair verdicts are not a JSON array. First 200 chars: {t[:200]!r}"
                )
            data = [data]
        if not isinstance(data, list):
            raise ValueError(f"Repair verdicts are not a JSON array. Got: {type(data).__name__}")

        out: dict[int, tuple[bool, str]] = {}
        for entry in data:
            if not isinstance(entry, dict) or "n" not in entry:
                continue
            try:
                n = int(entry["n"])
            except (TypeError, ValueError):
                continue
            verdict = str(entry.get("verdict", "")).strip().lower()
            out[n] = (verdict.startswith("accept"), str(entry.get("why", "")))
        return out

    def _legacy_result(
        self,
        chunk_id: str,
        paragraphs: list[str],
        proposal: Proposal,
        cached,
        stats: RepairStats,
    ) -> RepairResult:
        repaired = split_markers(cached.content) or []
        if len(repaired) != len(paragraphs):
            raise ValueError(
                f"cached repair for {chunk_id} has {len(repaired)} paragraphs, "
                f"expected {len(paragraphs)}"
            )
        accepted = int(cached.meta.get("accepted", 0))
        with self._lock:
            stats.chunks_cached += 1
            stats.candidates += len(proposal.candidates)
            stats.accepted += accepted
            stats.rejected += max(len(proposal.candidates) - accepted, 0)
            if accepted:
                stats.chunks_repaired += 1
        return RepairResult(chunk_id=chunk_id, paragraphs=repaired)

    # -- many chunks --------------------------------------------------------

    def repair_chunks(
        self,
        jobs: list[tuple[str, list[str], list[str], str]],
        stats: RepairStats,
        *,
        parallelism: int = 1,
        on_done: Callable[[int, str, Exception | None], None] | None = None,
    ) -> None:
        """Repair many chunks, `parallelism` at a time.

        Each job is (chunk_id, paragraphs, issues, original_text). A chunk that
        fails is counted and handed to `on_done` with its error, never raised,
        so one bad reply does not stop the rest.
        """

        def run(job: tuple[str, list[str], list[str], str]) -> Exception | None:
            try:
                self.repair_chunk(*job, stats)
                return None
            except Exception as e:  # noqa: BLE001 — reported through on_done
                return e

        with ThreadPoolExecutor(max_workers=max(1, parallelism)) as pool:
            futures = {pool.submit(run, job): job[0] for job in jobs}
            for done, future in enumerate(as_completed(futures), 1):
                error = future.result()
                if error is not None:
                    with self._lock:
                        stats.chunks_failed += 1
                if on_done:
                    on_done(done, futures[future], error)

    # -- one chunk ----------------------------------------------------------

    def repair_chunk(
        self,
        chunk_id: str,
        paragraphs: list[str],
        issues: list[str],
        original_text: str,
        stats: RepairStats,
    ) -> RepairResult | None:
        """Vet and apply the mechanical corrections for one chunk.

        Returns None when there is nothing to do. Paragraph count is preserved
        by construction — only quoted fragments are ever substituted.
        """
        proposal = propose(paragraphs, issues, self.categories)

        with self._lock:
            for issue, why in proposal.deferred_with_reasons:
                stats.unhandled.append((chunk_id, issue.raw, why.value))

        if not proposal.candidates:
            return None

        context = {
            "source_lang": self.source_lang,
            "source_lang_name": LANG_NAMES.get(self.source_lang, self.source_lang),
            "target_lang": self.target_lang,
            "target_lang_name": LANG_NAMES.get(self.target_lang, self.target_lang),
            "glossary_block": self._glossary_block,
            "original_text": original_text,
            "candidates": [
                {
                    "paragraph": c.paragraph,
                    "note": c.issue.raw,
                    "paragraph_text": paragraphs[c.paragraph - 1],
                    "old": c.old,
                    "new": c.new,
                }
                for c in proposal.candidates
            ],
        }
        system, user = render_prompt(self.prompt, context)

        # The issue list is part of the key: re-running the judge must not
        # serve a repair made from its previous verdicts. The key names the
        # model call, and the call's answer — the verdicts — is what is kept
        # under it; the repaired text is derived from them and kept apart.
        cache_key = Cache.make_key("repair", self.model, self.prompt.version, system, user)
        text_key = Cache.make_key("repair_text", cache_key)
        with self._lock:
            cached = self.cache.get(cache_key)

        result = None
        if cached is not None and not cached.meta.get("verdicts"):
            # Written before verdicts were kept: the row holds the repaired
            # text itself, with no record of which candidate was accepted.
            # It is still a paid, vetted result, so it is used as it stands.
            return self._legacy_result(chunk_id, paragraphs, proposal, cached, stats)
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

        # Validate before caching, so an unreadable verdict list is never stored.
        verdicts = self.parse_verdicts(raw_text)

        accepted: list[tuple[Candidate, str]] = []
        rejected: list[tuple[Candidate, str]] = []
        for i, c in enumerate(proposal.candidates, 1):
            ok, why = verdicts.get(i, (False, "нет вердикта — считаем отказом"))
            (accepted if ok else rejected).append((c, why))

        out_paragraphs = apply_candidates(paragraphs, [c for c, _ in accepted])
        if len(out_paragraphs) != len(paragraphs):  # pragma: no cover - defensive
            raise ValueError(
                f"repair changed the paragraph count for {chunk_id}: "
                f"{len(paragraphs)} -> {len(out_paragraphs)}"
            )

        with self._lock:
            if result is not None:
                # Kept even when every candidate was rejected: otherwise the
                # next run pays for the same refusal again.
                self.cache.put(
                    key=cache_key,
                    stage="repair_verdicts",
                    model=self.model,
                    prompt_version=self.prompt.version,
                    content=raw_text,
                    input_tokens=result.input_tokens,
                    output_tokens=result.output_tokens,
                    meta={"chunk_id": chunk_id, "verdicts": True},
                )
            # Only a change is a `repair` stage entry: an unchanged copy would
            # outrank a later re-run of verify at assembly.
            if accepted and self.cache.get(text_key) is None:
                self.cache.put(
                    key=text_key,
                    stage="repair",
                    model=self.model,
                    prompt_version=self.prompt.version,
                    content="\n".join(
                        f"===PARAGRAPH {i}===\n{p}" for i, p in enumerate(out_paragraphs, 1)
                    ),
                    meta={"chunk_id": chunk_id, "accepted": len(accepted)},
                )

        with self._lock:
            stats.candidates += len(proposal.candidates)
            stats.accepted += len(accepted)
            stats.rejected += len(rejected)
            if result is not None:
                stats.input_tokens += result.input_tokens
                stats.output_tokens += result.output_tokens
            if accepted:
                stats.chunks_repaired += 1
        for c, why in rejected:
            logger.info("repair %s p.%d rejected: «%s» — %s", chunk_id, c.paragraph, c.old, why)

        return RepairResult(
            chunk_id=chunk_id, paragraphs=out_paragraphs, accepted=accepted, rejected=rejected
        )
