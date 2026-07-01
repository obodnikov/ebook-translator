"""Series-level curated glossary management.

A "series" is a set of books sharing characters, places and invented
terminology (e.g. all 9 Rivers of London novels). The goal is to keep
translations of recurring names identical across books.

Data flow:
    book1 extract -> book1 glossary -> human review ->
        promote to series glossary
    book2 extract runs WITH series glossary loaded into the prompt as
        "known terms"; the model returns only deltas (new entries
        specific to book2, plus optional overrides).
    Reviewer merges book2's new entries back into the series glossary.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .models import (
    Glossary,
    GlossaryEntry,
    SeriesGlossary,
    SeriesGlossaryEntry,
)

# ---------------------------------------------------------------------------
# Workdir for a series
# ---------------------------------------------------------------------------


@dataclass
class SeriesWorkDir:
    """Directory layout for a series: work/<slug>-series/ ."""

    root: Path
    slug: str

    @classmethod
    def for_series(cls, base: Path, slug: str) -> SeriesWorkDir:
        root = base / f"{slug}-series"
        return cls(root=root, slug=slug)

    @property
    def glossary_path(self) -> Path:
        return self.root / "series.glossary.json"

    def ensure(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    def exists(self) -> bool:
        return self.glossary_path.is_file()


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------


def load_series_glossary(path: Path) -> SeriesGlossary:
    return SeriesGlossary.model_validate_json(path.read_text(encoding="utf-8"))


def save_series_glossary(glossary: SeriesGlossary, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        glossary.model_dump_json(indent=2, exclude_none=False),
        encoding="utf-8",
    )


def init_series(
    base: Path,
    slug: str,
    title: str,
    author: str,
    source_lang: str = "en",
    target_lang: str = "ru",
) -> SeriesWorkDir:
    """Create an empty series at work/<slug>-series/."""
    wd = SeriesWorkDir.for_series(base, slug)
    if wd.exists():
        raise FileExistsError(f"Series {slug!r} already exists at {wd.glossary_path}")
    wd.ensure()
    glossary = SeriesGlossary(
        series_slug=slug,
        title=title,
        author=author,
        source_lang=source_lang,
        target_lang=target_lang,
    )
    save_series_glossary(glossary, wd.glossary_path)
    return wd


# ---------------------------------------------------------------------------
# Promote: merge book glossary entries into series glossary
# ---------------------------------------------------------------------------


@dataclass
class PromoteReport:
    added: list[str]
    skipped_existing: list[str]
    skipped_unapproved: list[str]
    conflicts: list[tuple[str, str, str]]  # (original, series_value, book_value)


def _entry_to_series(book_entry: GlossaryEntry, origin_book: str) -> SeriesGlossaryEntry:
    return SeriesGlossaryEntry(
        original=book_entry.original,
        translation=book_entry.translation,
        type=book_entry.type,
        gender=book_entry.gender,
        plural=book_entry.plural,
        notes=book_entry.notes,
        origin_book=origin_book,
    )


def promote(
    series: SeriesGlossary,
    book_glossary: Glossary,
    *,
    require_approved: bool = True,
) -> tuple[SeriesGlossary, PromoteReport]:
    """Merge book_glossary entries into series.

    Rules:
      - If require_approved, only entries with approved_by_human=True are
        considered.
      - New entries (original not in series) are added.
      - Entries whose original is already in series are skipped; if their
        translation differs, a conflict is reported (human decides).
    """
    index = series.index_by_original()
    report = PromoteReport(added=[], skipped_existing=[], skipped_unapproved=[], conflicts=[])

    for e in book_glossary.entries:
        if require_approved and not e.approved_by_human:
            report.skipped_unapproved.append(e.original)
            continue

        existing = index.get(e.original)
        if existing is None:
            new_entry = _entry_to_series(e, origin_book=book_glossary.book)
            series.entries.append(new_entry)
            index[new_entry.original] = new_entry
            report.added.append(e.original)
        else:
            if existing.translation != e.translation:
                report.conflicts.append((e.original, existing.translation, e.translation))
            else:
                report.skipped_existing.append(e.original)

    # Keep entries sorted for stable diffs: by type, then by original.
    type_order = {"person": 0, "place": 1, "concept": 2, "term": 3, "other": 4}
    series.entries.sort(key=lambda x: (type_order.get(x.type, 99), x.original.lower()))
    return series, report


# ---------------------------------------------------------------------------
# Prompt helpers: compact rendering for LLM input
# ---------------------------------------------------------------------------


def render_for_prompt(series: SeriesGlossary) -> str:
    """Compact one-line-per-entry rendering for injection into prompts.

    Format:
        original | translation | type[, gender] | notes
    """
    if not series.entries:
        return "(no known terms yet)"

    lines: list[str] = []
    for e in series.entries:
        meta = e.type
        if e.type == "person" and e.gender:
            meta = f"{e.type}, {e.gender}"
        line = f"{e.original} | {e.translation} | {meta}"
        if e.notes:
            # Keep notes short; strip newlines so lines stay single-line.
            short = " ".join(e.notes.split())
            if len(short) > 120:
                short = short[:117] + "..."
            line += f" | {short}"
        lines.append(line)
    return "\n".join(lines)
