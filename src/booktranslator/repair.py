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

import re
from dataclasses import dataclass, field
from enum import StrEnum

# Issue shape produced by prompts/judge.md v3:
#   "grammar: p.11 «было тридцать одного года» → «было тридцать один год»"
_CATEGORY_RE = re.compile(r"^\s*([a-z_]+)\s*[:/]", re.IGNORECASE)
_PARAGRAPH_RE = re.compile(r"\bp\.\s*(\d+)", re.IGNORECASE)
# Both arrow forms appear in practice: the judge is asked for → but writes -> too.
_PAIR_RE = re.compile(r"«([^«»]{2,})»\s*(?:→|->)\s*«([^«»]{1,})»")

# Categories the first pass acts on. Kept here as the default; callers may
# narrow or widen it from config without touching this module.
MECHANICAL_CATEGORIES = ("grammar", "glossary", "markup")


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
        hopeless = {Skip.QUOTE_ABSENT, Skip.OUT_OF_SCOPE, Skip.NO_CATEGORY, Skip.NO_OP}
        return [issue for issue, why in self.skipped if why not in hopeless]


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
