"""Command-line entry point.

Commands:
    btrans glossary extract EPUB [--series SLUG]
    btrans glossary promote BOOK_WORKDIR --series SLUG
    btrans series init SLUG --title "..." --author "..."
    btrans series show SLUG
    btrans translate EPUB [--series SLUG]
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import click
import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from .cache import STAGE_WATERFALL, Cache
from .chunker import chunk_book
from .config import load_config
from .epub_io import read_book, read_book_structured, write_translated_epub
from .glossary import extract_glossary, save_glossary
from .models import Glossary, SeriesGlossary, SeriesGlossaryEntry, Stage
from .pipeline_helpers import (
    ChunkerConfigMismatchError,
    build_reflect_input,
    collect_chunk_originals,
    collect_preferred_translations,
    collect_stage_translations,
    collect_waterfall_paragraphs,
    create_image_provider,
    create_stage_provider,
    normalize_judge_map,
    rehydrate_book_from_waterfall,
    save_chunker_params,
    verify_chunker_params,
)
from .series import (
    SeriesWorkDir,
    init_series,
    load_series_glossary,
    promote,
    save_series_glossary,
)
from .state import WorkDir
from .translator import Translator

load_dotenv()

app = typer.Typer(
    help="Translate EPUB books through OpenRouter LLMs.",
    no_args_is_help=True,
)
glossary_app = typer.Typer(help="Glossary extraction and curation.")
series_app = typer.Typer(help="Series-level curated glossary management.")
cover_app = typer.Typer(help="Cover image replacement and translation.")
app.add_typer(glossary_app, name="glossary")
app.add_typer(series_app, name="series")
app.add_typer(cover_app, name="cover")

console = Console()

DEFAULT_WORK_DIR = Path("work")
DEFAULT_CONFIG = Path("configs/default.yaml")
GLOSSARY_PROMPT = Path("prompts/glossary_extract.md")
TRANSLATE_PROMPT = Path("prompts/translate.md")


# ---------------------------------------------------------------------------
# glossary extract
# ---------------------------------------------------------------------------


@glossary_app.command("extract")
def glossary_extract(
    epub: Path = typer.Argument(..., exists=True, dir_okay=False, help="Source EPUB."),
    series: str | None = typer.Option(
        None,
        "--series",
        "-s",
        help="Series slug: load known terms from its curated glossary.",
    ),
    config_path: Path = typer.Option(
        DEFAULT_CONFIG,
        "--config",
        "-c",
        help="YAML config file.",
    ),
    work_dir: Path = typer.Option(
        DEFAULT_WORK_DIR,
        "--work",
        "-w",
        help="Base directory for artifacts.",
    ),
    model: str | None = typer.Option(
        None,
        "--model",
        "-m",
        help="Override the glossary model (e.g. anthropic/claude-sonnet-4.6).",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Ignore cache and call the LLM again.",
    ),
) -> None:
    """Extract the glossary of proper names and invented terms from an EPUB."""
    from .series import render_for_prompt

    cfg = load_config(config_path if config_path.exists() else None)

    console.print(f"[bold]Reading EPUB:[/bold] {epub}")
    book = read_book(epub)

    console.print(
        f"[bold]Book:[/bold]    {book.meta.title}\n"
        f"[bold]Author:[/bold]  {book.meta.author}\n"
        f"[bold]Chapters:[/bold] {book.meta.chapters}\n"
        f"[bold]Words:[/bold]   ~{book.meta.word_count:,}"
    )

    known_terms: str | None = None
    if series:
        series_wd = SeriesWorkDir.for_series(work_dir, series)
        if not series_wd.exists():
            console.print(
                f"[red]Series {series!r} not found at {series_wd.glossary_path}.[/red]\n"
                f"[dim]Create it with: btrans series init {series} "
                f"--title ... --author ...[/dim]"
            )
            raise typer.Exit(code=1)
        series_glossary = load_series_glossary(series_wd.glossary_path)
        known_terms = render_for_prompt(series_glossary)
        console.print(
            f"[bold]Series:[/bold]  {series} ({len(series_glossary.entries)} known terms)"
        )

    wd = WorkDir.for_book(work_dir, book.meta.title)
    state = wd.load_state()
    state.current_stage = Stage.EXTRACT
    wd.save_state(state)
    console.print(f"[bold]Workdir:[/bold] {wd.root}")

    cache = Cache(wd.cache_path)
    if force:
        cache.conn.execute("DELETE FROM cache WHERE stage = 'glossary'")
        cache.conn.commit()

    provider = create_stage_provider(cfg, "glossary")
    chosen_model = model or cfg.models.glossary
    console.print(f"[bold]Model:[/bold]   {chosen_model}")
    console.print("[dim]Sending full book to the LLM...[/dim]")

    try:
        glossary_obj, result, raw_text = extract_glossary(
            book=book,
            provider=provider,
            prompt_path=GLOSSARY_PROMPT,
            cache=cache,
            model=chosen_model,
            source_lang=cfg.source_lang,
            target_lang=cfg.target_lang,
            known_terms=known_terms,
        )
    except Exception as e:
        wd.raw_glossary_path.write_text(str(e), encoding="utf-8")
        console.print(f"[red]Extraction failed:[/red] {e}")
        raise typer.Exit(code=1) from e

    wd.raw_glossary_path.write_text(raw_text, encoding="utf-8")
    save_glossary(glossary_obj, wd.glossary_path)
    wd.mark_stage(state, Stage.GLOSSARY)

    _print_glossary_summary(glossary_obj, result)
    console.print(
        f"\n[green]Glossary written to:[/green] {wd.glossary_path}\n"
        f"[dim]Review it, edit if needed, then set "
        f"[bold]approved_by_human: true[/bold] on checked entries.[/dim]"
    )


def _print_glossary_summary(glossary_obj: Glossary, result) -> None:
    table = Table(title="Glossary summary", show_header=True)
    table.add_column("Type", style="cyan")
    table.add_column("Count", justify="right")
    table.add_column("Overrides", justify="right")
    by_type: dict[str, int] = {}
    overrides_by_type: dict[str, int] = {}
    for e in glossary_obj.entries:
        by_type[e.type] = by_type.get(e.type, 0) + 1
        if (e.notes or "").startswith("[override]"):
            overrides_by_type[e.type] = overrides_by_type.get(e.type, 0) + 1
    for t in ("person", "place", "concept", "term", "other"):
        if t in by_type:
            table.add_row(t, str(by_type[t]), str(overrides_by_type.get(t, 0)))
    table.add_row(
        "[bold]total[/bold]",
        f"[bold]{len(glossary_obj.entries)}[/bold]",
        f"[bold]{sum(overrides_by_type.values())}[/bold]",
    )
    console.print(table)

    if result is not None:
        console.print(f"[dim]Tokens: {result.input_tokens} in, {result.output_tokens} out[/dim]")
    else:
        console.print("[dim]Served from cache (no tokens used)[/dim]")


# ---------------------------------------------------------------------------
# glossary promote
# ---------------------------------------------------------------------------


@glossary_app.command("promote")
def glossary_promote(
    book_work_dir: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=False,
        help="Book workdir (e.g. work/rivers-of-london).",
    ),
    series: str = typer.Option(
        ...,
        "--series",
        "-s",
        help="Series slug to promote into.",
    ),
    work_root: Path = typer.Option(
        DEFAULT_WORK_DIR,
        "--work",
        "-w",
        help="Base directory for artifacts.",
    ),
    all_entries: bool = typer.Option(
        False,
        "--all",
        help="Promote ALL entries, not only approved_by_human ones.",
    ),
) -> None:
    """Merge approved book-glossary entries into the series glossary."""
    glossary_path = book_work_dir / "glossary.json"
    if not glossary_path.is_file():
        console.print(f"[red]No glossary.json in {book_work_dir}[/red]")
        raise typer.Exit(code=1)
    book_glossary = Glossary.model_validate(json.loads(glossary_path.read_text(encoding="utf-8")))

    series_wd = SeriesWorkDir.for_series(work_root, series)
    if not series_wd.exists():
        console.print(
            f"[red]Series {series!r} not found.[/red]\n"
            f"[dim]Create it: btrans series init {series} "
            f"--title ... --author ...[/dim]"
        )
        raise typer.Exit(code=1)
    series_glossary = load_series_glossary(series_wd.glossary_path)

    updated_series, report = promote(
        series_glossary,
        book_glossary,
        require_approved=not all_entries,
    )
    save_series_glossary(updated_series, series_wd.glossary_path)

    console.print(
        f"[bold]Promoted[/bold] {len(report.added)} new entries into "
        f"[cyan]{series}[/cyan] (now {len(updated_series.entries)} total).\n"
        f"[dim]Skipped existing: {len(report.skipped_existing)}. "
        f"Skipped unapproved: {len(report.skipped_unapproved)}. "
        f"Conflicts: {len(report.conflicts)}.[/dim]"
    )

    if report.conflicts:
        console.print(
            "\n[yellow]Conflicts[/yellow] (series value kept, book value shown for review):"
        )
        for original, series_val, book_val in report.conflicts[:20]:
            console.print(f"  {original!r}: series={series_val!r} book={book_val!r}")
        if len(report.conflicts) > 20:
            console.print(f"  ... and {len(report.conflicts) - 20} more.")


# ---------------------------------------------------------------------------
# series init
# ---------------------------------------------------------------------------


@series_app.command("init")
def series_init(
    slug: str = typer.Argument(..., help="Series slug, e.g. rivers-of-london."),
    title: str = typer.Option(..., "--title", "-t", help="Series title."),
    author: str = typer.Option(..., "--author", "-a", help="Series author."),
    work_root: Path = typer.Option(
        DEFAULT_WORK_DIR,
        "--work",
        "-w",
        help="Base directory for artifacts.",
    ),
) -> None:
    """Create an empty series glossary."""
    cfg = load_config(DEFAULT_CONFIG if DEFAULT_CONFIG.exists() else None)
    try:
        wd = init_series(
            base=work_root,
            slug=slug,
            title=title,
            author=author,
            source_lang=cfg.source_lang,
            target_lang=cfg.target_lang,
        )
    except FileExistsError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=1) from e

    console.print(
        f"[green]Initialised series[/green] [bold]{slug}[/bold] "
        f"at {wd.glossary_path}\n"
        f"[dim]Title:  {title}\n"
        f"Author: {author}\n"
        f"Now run: btrans glossary promote work/<book-slug> --series {slug}[/dim]"
    )


# ---------------------------------------------------------------------------
# series show
# ---------------------------------------------------------------------------


@series_app.command("show")
def series_show(
    slug: str = typer.Argument(...),
    work_root: Path = typer.Option(DEFAULT_WORK_DIR, "--work", "-w"),
    limit: int = typer.Option(20, "--limit", "-n", help="Entries per type."),
) -> None:
    """Show a summary of a series glossary."""
    wd = SeriesWorkDir.for_series(work_root, slug)
    if not wd.exists():
        console.print(f"[red]Series {slug!r} not found.[/red]")
        raise typer.Exit(code=1)
    g = load_series_glossary(wd.glossary_path)

    console.print(
        f"[bold]{g.title}[/bold] — {g.author}\n"
        f"[dim]{g.source_lang} -> {g.target_lang}, "
        f"{len(g.entries)} entries total[/dim]\n"
    )
    for t in ("person", "place", "concept", "term", "other"):
        rows = [e for e in g.entries if e.type == t]
        if not rows:
            continue
        table = Table(title=f"{t} ({len(rows)})", show_header=True, show_lines=False)
        table.add_column("Original", style="cyan")
        table.add_column("Translation")
        table.add_column("G", width=2)
        table.add_column("Origin", style="dim")
        for e in rows[:limit]:
            table.add_row(
                e.original,
                e.translation,
                e.gender or "",
                e.origin_book or "",
            )
        if len(rows) > limit:
            table.caption = f"... {len(rows) - limit} more"
        console.print(table)


# ---------------------------------------------------------------------------
# translate
# ---------------------------------------------------------------------------


@app.command("translate")
def translate(
    epub: Path = typer.Argument(..., exists=True, dir_okay=False, help="Source EPUB."),
    series: str | None = typer.Option(
        None,
        "--series",
        "-s",
        help="Series slug: use its curated glossary for consistency.",
    ),
    glossary_path: Path | None = typer.Option(
        None,
        "--glossary",
        "-g",
        exists=True,
        dir_okay=False,
        help="Path to a single-book glossary.json. Mutually exclusive with --series.",
    ),
    config_path: Path = typer.Option(
        DEFAULT_CONFIG,
        "--config",
        "-c",
        help="YAML config file.",
    ),
    work_dir: Path = typer.Option(
        DEFAULT_WORK_DIR,
        "--work",
        "-w",
        help="Base directory for artifacts.",
    ),
    model: str | None = typer.Option(
        None,
        "--model",
        "-m",
        help="Override the translate model.",
    ),
    out: Path | None = typer.Option(
        None,
        "--out",
        "-o",
        help="Output EPUB path (default: <source>-ru.epub).",
    ),
    limit_chunks: int | None = typer.Option(
        None,
        "--limit-chunks",
        help="Translate only the first N chunks (for quick tests).",
    ),
    parallelism: int | None = typer.Option(
        None,
        "--parallelism",
        "-j",
        help="Number of chunks to translate concurrently (overrides config.translate.parallelism).",
    ),
    no_judge: bool = typer.Option(
        False,
        "--no-judge",
        help="Skip judge + reflect passes entirely.",
    ),
    no_reflect: bool = typer.Option(
        False,
        "--no-reflect",
        help="Run judge (for stats) but skip reflect.",
    ),
    reflect_threshold: int | None = typer.Option(
        None,
        "--reflect-threshold",
        help="Override reflection.trigger_score (reflect chunks scoring ≤ N).",
    ),
    reflect_all: bool = typer.Option(
        False,
        "--reflect-all",
        help="Reflect all chunks regardless of score.",
    ),
    no_proofread: bool = typer.Option(
        False,
        "--no-proofread",
        help="Skip the proofread pass.",
    ),
    no_style: bool = typer.Option(
        False,
        "--no-style",
        help="Skip the style pass.",
    ),
    no_verify: bool = typer.Option(
        False,
        "--no-verify",
        help="Skip the verify pass.",
    ),
    notes: bool | None = typer.Option(
        None,
        "--notes/--no-notes",
        help="Inject reader footnotes from glossary (overrides config.reader_notes.enabled).",
    ),
    note_types: str | None = typer.Option(
        None,
        "--note-types",
        help="Comma-separated glossary types to annotate (e.g. concept,term,place).",
    ),
    cover: bool = typer.Option(
        False,
        "--cover/--no-cover",
        help="Translate the cover image using the image provider (opt-in, costs money).",
    ),
    cover_title: str | None = typer.Option(
        None,
        "--cover-title",
        help="Exact translated title for the cover (optional).",
    ),
    cover_author: str | None = typer.Option(
        None,
        "--cover-author",
        help="Author name for the cover (optional).",
    ),
    cover_model: str | None = typer.Option(
        None,
        "--cover-model",
        help="Override the image model for cover translation (default: cfg.models.cover).",
    ),
    cover_target_lang: str | None = typer.Option(
        None,
        "--cover-target-lang",
        help=(
            "Override target language name for cover prompt "
            "(default: auto-derived from target_lang). "
            "Required when target_lang is not a recognised ISO 639-1 code."
        ),
    ),
    cover_aspect_ratio: str = typer.Option(
        "2:3",
        "--cover-aspect-ratio",
        help="Cover image aspect ratio. Supported: 1:1, 2:3, 3:2, 3:4, 4:3, 9:16, 16:9.",
        click_type=click.Choice(["1:1", "2:3", "3:2", "3:4", "4:3", "9:16", "16:9"]),
    ),
    cover_image_size: str = typer.Option(
        "1K",
        "--cover-image-size",
        help="Cover image resolution: 1K, 2K, or 4K.",
        click_type=click.Choice(["1K", "2K", "4K"]),
    ),
) -> None:
    """Translate an EPUB into the target language."""
    cfg = load_config(config_path if config_path.exists() else None)

    if reflect_threshold is not None and not 1 <= reflect_threshold <= 5:
        console.print("[red]--reflect-threshold must be between 1 and 5.[/red]")
        raise typer.Exit(code=1)

    # Validate --cover preconditions up front (fail-fast): a bad cover flag
    # must not cost the user a full paid translation run before it's rejected.
    # (--cover-aspect-ratio / --cover-image-size are validated at parse time via
    # click.Choice.) Resolved values are reused by the cover block after write.
    cover_model_resolved: str | None = None
    cover_lang_resolved: str | None = None
    if cover:
        cover_model_resolved = cover_model or cfg.models.cover
        if not cover_model_resolved:
            console.print(
                "[red]--cover requires a model.[/red]\n"
                "[dim]Set models.cover in config or pass --cover-model.[/dim]"
            )
            raise typer.Exit(code=1)
        cover_lang_resolved = cover_target_lang or cfg.resolved_target_lang_name()
        if not cover_lang_resolved:
            console.print(
                "[red]--cover requires a target language name.[/red]\n"
                "[dim]Set target_lang_name in config, or pass --cover-target-lang "
                "(e.g. --cover-target-lang Russian).[/dim]"
            )
            raise typer.Exit(code=1)

    console.print(f"[bold]Reading EPUB:[/bold] {epub}")
    book = read_book_structured(epub)
    console.print(
        f"[bold]Book:[/bold]     {book.meta.title}\n"
        f"[bold]Author:[/bold]   {book.meta.author}\n"
        f"[bold]Chapters:[/bold] {book.meta.chapters}\n"
        f"[bold]Words:[/bold]    ~{book.meta.word_count:,}"
    )

    glossary: SeriesGlossary | None = None  # always initialized; conditionally populated below
    if series and glossary_path:
        console.print(
            "[red]--series and --glossary are mutually exclusive.[/red]\n"
            "[dim]Use --series SLUG for books in a curated series, "
            "--glossary PATH for a stand-alone book.[/dim]"
        )
        raise typer.Exit(code=1)

    if series:
        swd = SeriesWorkDir.for_series(work_dir, series)
        if not swd.exists():
            console.print(f"[red]Series {series!r} not found.[/red]")
            raise typer.Exit(code=1)
        glossary = load_series_glossary(swd.glossary_path)
        console.print(f"[bold]Series:[/bold]   {series} ({len(glossary.entries)} terms)")
    elif glossary_path:
        # Load a book-level Glossary and adapt it to the SeriesGlossary
        # shape the Translator expects. We take ALL entries (the operator
        # chose this file explicitly; approved_by_human is advisory).
        book_glossary = Glossary.model_validate(
            json.loads(glossary_path.read_text(encoding="utf-8"))
        )
        glossary = SeriesGlossary(
            series_slug=f"ad-hoc:{book.meta.title}",
            title=book.meta.title,
            author=book.meta.author,
            source_lang=cfg.source_lang,
            target_lang=cfg.target_lang,
            entries=[
                SeriesGlossaryEntry(
                    original=e.original,
                    translation=e.translation,
                    type=e.type,
                    gender=e.gender,
                    plural=e.plural,
                    notes=e.notes,
                    origin_book=book_glossary.book,
                )
                for e in book_glossary.entries
            ],
        )
        console.print(
            f"[bold]Glossary:[/bold] {len(glossary.entries)} terms "
            f"from {glossary_path} [dim](ad-hoc, not tied to a series)[/dim]"
        )
    else:
        console.print(
            "[yellow]No --series or --glossary given; translating without a glossary.[/yellow]"
        )

    wd = WorkDir.for_book(work_dir, book.meta.title)
    state = wd.load_state()
    state.current_stage = Stage.TRANSLATE
    wd.save_state(state)
    console.print(f"[bold]Workdir:[/bold]  {wd.root}")

    chunk_set = chunk_book(
        book,
        target_words=cfg.chunker.target_words,
        overlap_paragraphs=cfg.chunker.overlap_paragraphs,
    )
    if limit_chunks is not None:
        chunk_set.chunks = chunk_set.chunks[:limit_chunks]
    console.print(
        f"[bold]Chunks:[/bold]   {len(chunk_set.chunks)} "
        f"(target {cfg.chunker.target_words} words, overlap "
        f"{cfg.chunker.overlap_paragraphs} paragraphs)"
    )

    # The translator splices each translated paragraph into the live tree
    # (translator.py:_splice_fragments), so `chunk_set` stops holding the
    # source text the moment translation starts. Judge, reflect and verify all
    # need the real original to compare against, so they read it from a second
    # copy of the book that nothing writes to. Re-reading costs ~0.03s.
    source_chunk_set = chunk_book(
        read_book_structured(epub),
        target_words=cfg.chunker.target_words,
        overlap_paragraphs=cfg.chunker.overlap_paragraphs,
    )
    if limit_chunks is not None:
        source_chunk_set.chunks = source_chunk_set.chunks[:limit_chunks]

    effective_parallelism = parallelism if parallelism is not None else cfg.translate.parallelism
    if effective_parallelism > 1:
        console.print(f"[bold]Parallelism:[/bold] {effective_parallelism} chunks")

    cache = Cache(wd.cache_path)
    # Persist chunker params so judge/reflect can verify consistency.
    # Fails fast if existing params differ (prevents mixed-cache state).
    try:
        save_chunker_params(cache, cfg.chunker.target_words, cfg.chunker.overlap_paragraphs)
    except ChunkerConfigMismatchError as e:
        console.print(f"[red]Error:[/red] {e}")
        cache.close()
        raise typer.Exit(code=1) from e
    provider = create_stage_provider(cfg, "translate")
    chosen_model = model or cfg.models.translate
    console.print(f"[bold]Model:[/bold]    {chosen_model}")
    console.print("[dim]Translating chunks...[/dim]\n")

    translator = Translator(
        provider=provider,
        prompt_path=TRANSLATE_PROMPT,
        cache=cache,
        glossary=glossary,
        model=chosen_model,
        source_lang=cfg.source_lang,
        target_lang=cfg.target_lang,
    )

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("translating", total=len(chunk_set.chunks))

        def on_progress(done: int, total: int, chunk, stats) -> None:
            progress.update(
                task,
                completed=done,
                description=(
                    f"{chunk.id} | cached {stats.chunks_cached} "
                    f"new {stats.chunks_translated} "
                    f"failed {stats.chunks_failed}"
                ),
            )

        stats = translator.translate_book(
            chunk_set,
            on_progress=on_progress,
            parallelism=parallelism if parallelism is not None else cfg.translate.parallelism,
        )

    if stats.chunks_failed:
        console.print(
            f"[yellow]Warning:[/yellow] {stats.chunks_failed} chunks failed; "
            "the output EPUB will have those paragraphs in the original language. "
            "Re-run the same command to retry failed chunks (cache keeps successes)."
        )

    # ------------------------------------------------------------------
    # Judge pass
    # ------------------------------------------------------------------
    judge_stats = None
    if not no_judge:
        from .judge import Judge

        JUDGE_PROMPT = Path("prompts/judge.md")

        # Collect original and translated texts
        translate_ids = cache.get_all_chunk_ids_for_stage("translate")
        if translate_ids:
            chunk_originals = collect_chunk_originals(source_chunk_set, translate_ids)
            chunk_translations = collect_stage_translations(cache, translate_ids, stage="translate")

            # Only judge chunks that have both original and translation
            judgeable_ids = set(chunk_originals.keys()) & set(chunk_translations.keys())
            if judgeable_ids:
                judge_model = cfg.models.judge
                console.print(
                    f"\n[dim]Judging {len(judgeable_ids)} chunks with {judge_model}...[/dim]\n"
                )

                judge = Judge(
                    provider=create_stage_provider(cfg, "judge"),
                    prompt_path=JUDGE_PROMPT,
                    cache=cache,
                    glossary=glossary,
                    model=judge_model,
                    source_lang=cfg.source_lang,
                    target_lang=cfg.target_lang,
                )

                judge_parallelism = min(parallelism or cfg.translate.parallelism, 8) or 4

                with Progress(
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    MofNCompleteColumn(),
                    TimeElapsedColumn(),
                    console=console,
                ) as progress:
                    jtask = progress.add_task("judging", total=len(judgeable_ids))

                    def on_judge_progress(done, total, cid, jstats):
                        progress.update(
                            jtask,
                            completed=done,
                            description=(
                                f"judge | cached {jstats.chunks_cached} "
                                f"new {jstats.chunks_judged} "
                                f"failed {jstats.chunks_failed}"
                            ),
                        )

                    judge_stats = judge.judge_chunks(
                        {k: v for k, v in chunk_originals.items() if k in judgeable_ids},
                        {k: v for k, v in chunk_translations.items() if k in judgeable_ids},
                        on_progress=on_judge_progress,
                        parallelism=judge_parallelism,
                    )

                # Print score distribution
                if judge_stats.results:
                    dist: dict[int, int] = {}
                    for r in judge_stats.results:
                        dist[r.score] = dist.get(r.score, 0) + 1
                    dist_str = "  ".join(
                        f"★{k}: {v}" for k, v in sorted(dist.items(), reverse=True)
                    )
                    console.print(f"[bold]Judge scores:[/bold] {dist_str}")

                    threshold = (
                        reflect_threshold
                        if reflect_threshold is not None
                        else cfg.reflection.trigger_score
                    )
                    needs_reflect = [r for r in judge_stats.results if r.score <= threshold]
                    console.print(
                        f"  Chunks needing reflection: {len(needs_reflect)} (score ≤ {threshold})"
                    )

    # ------------------------------------------------------------------
    # Reflect pass
    # ------------------------------------------------------------------
    reflect_stats = None
    if not no_judge and not no_reflect and judge_stats and judge_stats.results:
        from .reflect import Reflector

        REFLECT_PROMPT = Path("prompts/reflect.md")

        threshold = (
            reflect_threshold if reflect_threshold is not None else cfg.reflection.trigger_score
        )

        if reflect_all:
            chunks_to_reflect_results = judge_stats.results
        else:
            chunks_to_reflect_results = [r for r in judge_stats.results if r.score <= threshold]

        if chunks_to_reflect_results:
            reflect_model = cfg.models.reflect
            console.print(
                f"\n[dim]Reflecting on "
                f"{len(chunks_to_reflect_results)} chunks "
                f"with {reflect_model}...[/dim]\n"
            )

            reflector = Reflector(
                provider=create_stage_provider(cfg, "reflect"),
                reflect_prompt_path=REFLECT_PROMPT,
                translate_prompt_path=TRANSLATE_PROMPT,
                cache=cache,
                glossary=glossary,
                model=reflect_model,
                source_lang=cfg.source_lang,
                target_lang=cfg.target_lang,
            )

            # Build reflect input using shared helper (O(n), not O(n²))
            judge_by_id = normalize_judge_map(
                [
                    {"chunk_id": r.chunk_id, "score": r.score, "issues": r.issues}
                    for r in chunks_to_reflect_results
                ]
            )
            chunks_to_reflect_data = build_reflect_input(
                source_chunk_set,
                cache,
                {r.chunk_id for r in chunks_to_reflect_results},
                judge_by_id,
            )

            if chunks_to_reflect_data:
                reflect_parallelism = min(parallelism or cfg.translate.parallelism, 4) or 2

                with Progress(
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    MofNCompleteColumn(),
                    TimeElapsedColumn(),
                    console=console,
                ) as progress:
                    rtask = progress.add_task(
                        "reflecting",
                        total=len(chunks_to_reflect_data),
                    )

                    def on_reflect_progress(done, total, cid, rstats):
                        progress.update(
                            rtask,
                            completed=done,
                            description=(
                                f"reflect | "
                                f"improved {rstats.chunks_improved} "
                                f"unchanged {rstats.chunks_unchanged} "
                                f"failed {rstats.chunks_failed}"
                            ),
                        )

                    reflect_stats = reflector.reflect_chunks(
                        chunks_to_reflect_data,
                        on_progress=on_reflect_progress,
                        parallelism=reflect_parallelism,
                    )

                console.print(
                    f"  Improved: "
                    f"{reflect_stats.chunks_improved}/"
                    f"{reflect_stats.chunks_total}  |  "
                    f"No change: "
                    f"{reflect_stats.chunks_unchanged}/"
                    f"{reflect_stats.chunks_total}"
                )

    # ------------------------------------------------------------------
    # Post-processing passes: proofread, style, verify
    # ------------------------------------------------------------------
    from .postprocess import PostProcessor

    # Build paragraph count map for waterfall parsing
    paragraph_counts = {chunk.id: len(chunk.paragraph_indexes) for chunk in chunk_set.chunks}

    # Current chunk IDs from the chunk_set — anchors all postprocess
    # operations to the current book/chunker config, preventing stale
    # cache entries from being processed.
    current_chunk_ids = [chunk.id for chunk in chunk_set.chunks]

    # Intersect with what's actually in cache (translated)
    cached_translate_ids = set(cache.get_all_chunk_ids_for_stage("translate"))
    valid_chunk_ids = [cid for cid in current_chunk_ids if cid in cached_translate_ids]

    pp_stats_list: list[tuple[str, object]] = []

    # Determine which stages to run based on config and CLI flags
    stages_to_run: list[tuple[str, Path, str]] = []
    if cfg.stages.proofread and not no_proofread:
        stages_to_run.append(("proofread", Path("prompts/proofread.md"), cfg.models.proofread))
    if cfg.stages.style and not no_style:
        stages_to_run.append(("style", Path("prompts/style.md"), cfg.models.style))
    if cfg.stages.verify and not no_verify:
        stages_to_run.append(("verify", Path("prompts/verify.md"), cfg.models.verify))

    for pp_stage, pp_prompt_path, pp_model in stages_to_run:
        # Collect waterfall input for this stage using scoped IDs
        waterfall_paragraphs = collect_waterfall_paragraphs(
            cache, valid_chunk_ids, pp_stage, paragraph_counts
        )

        if not waterfall_paragraphs:
            console.print(
                f"\n[yellow]No parseable translations for {pp_stage} pass; skipping.[/yellow]"
            )
            continue

        # For verify, also need originals
        chunks_data_pp = []
        originals_for_verify: dict[str, str] | None = None
        if pp_stage == "verify":
            originals_for_verify = collect_chunk_originals(
                source_chunk_set, list(waterfall_paragraphs.keys())
            )

        for cid, paragraphs in sorted(waterfall_paragraphs.items()):
            entry: dict = {
                "chunk_id": cid,
                "translated_paragraphs": paragraphs,
            }
            if pp_stage == "verify" and originals_for_verify:
                entry["original_text"] = originals_for_verify.get(cid, "")
            chunks_data_pp.append(entry)

        console.print(
            f"\n[dim]Running {pp_stage} on {len(chunks_data_pp)} chunks with {pp_model}...[/dim]\n"
        )

        processor = PostProcessor(
            provider=create_stage_provider(cfg, pp_stage),
            prompt_path=pp_prompt_path,
            cache=cache,
            glossary=glossary,
            stage=pp_stage,
            model=pp_model,
            source_lang=cfg.source_lang,
            target_lang=cfg.target_lang,
        )

        pp_parallelism = min(parallelism or cfg.translate.parallelism, 8) or 4

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            pptask = progress.add_task(pp_stage, total=len(chunks_data_pp))

            def _make_pp_progress_cb(task_id, stage_name):
                def on_pp_progress(done, total, cid, pstats):
                    progress.update(
                        task_id,
                        completed=done,
                        description=(
                            f"{stage_name} | cached {pstats.chunks_cached} "
                            f"new {pstats.chunks_processed} "
                            f"failed {pstats.chunks_failed}"
                        ),
                    )

                return on_pp_progress

            pp_stats = processor.process_chunks(
                chunks_data_pp,
                on_progress=_make_pp_progress_cb(pptask, pp_stage),
                parallelism=pp_parallelism,
            )

        pp_stats_list.append((pp_stage, pp_stats))
        console.print(
            f"  Changed: {pp_stats.chunks_changed}/{pp_stats.chunks_total}  |  "
            f"Unchanged: {pp_stats.chunks_unchanged}/{pp_stats.chunks_total}"
        )

    out_path = out or _default_output_path(epub, cfg.target_lang)

    # Rehydrate in-memory trees from the final waterfall stage.
    # The Translator only updates trees with translate-stage content;
    # reflect/proofread/style/verify write to cache but not to the tree.
    # This step ensures the EPUB contains the latest post-processed text.
    from .pipeline_helpers import rehydrate_book_from_waterfall

    if pp_stats_list or reflect_stats:
        rehydrated, _ = rehydrate_book_from_waterfall(cache, chunk_set)
        if rehydrated:
            console.print(f"[dim]Rehydrated {rehydrated} chunks from post-processing.[/dim]")

    # --- Reader notes injection (after rehydrate, before write) ---
    from .reader_notes import GlossaryLoadError, inject_notes_for_cli

    # Explicit intent = user passed reader-notes-specific flags (--notes or --note-types).
    # --series/--glossary are for translation consistency and do NOT imply notes intent.
    _notes_explicit = notes is True or note_types is not None
    try:
        inject_notes_for_cli(
            console=console,
            chapters=book.chapters,
            cfg=cfg,
            notes_flag=notes,
            note_types=note_types,
            series=series,
            glossary_path=glossary_path,
            work_dir=work_dir,
            book_title=book.meta.title,
            book_author=book.meta.author,
            preloaded_glossary=glossary,
            explicit=_notes_explicit,
        )
    except GlossaryLoadError as e:
        if _notes_explicit:
            console.print(f"[red]Error loading glossary for reader notes:[/red] {e}")
            raise typer.Exit(code=1) from e
        console.print(
            f"[yellow]Warning: could not load glossary for reader notes (skipping):[/yellow] {e}"
        )

    modified = [ch for ch in book.chapters if ch.paragraphs]
    write_translated_epub(
        source_path=epub,
        dest_path=out_path,
        modified_chapters=modified,
        new_language=cfg.target_lang,
    )

    # --- Cover translation (after write, file→file, best-effort) ---
    # Preconditions (model, target-lang, aspect-ratio) were validated up front.
    if cover:
        import os
        import tempfile

        from .cover import translate_cover

        tmp: Path | None = None
        try:
            _image_provider = create_image_provider(cfg)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_name = tempfile.mkstemp(suffix=".epub", dir=out_path.parent)
            os.close(fd)
            tmp = Path(tmp_name)
            cover_result = translate_cover(
                source_epub=out_path,
                dest_epub=tmp,
                provider=_image_provider,
                model=cover_model_resolved,
                title_translation=cover_title,
                author_name=cover_author,
                target_lang=cover_lang_resolved,
                aspect_ratio=cover_aspect_ratio,
                image_size=cover_image_size,
            )
            os.replace(tmp, out_path)
            console.print(
                f"[green]Cover translated.[/green] {cover_result.mime_type}, "
                f"{len(cover_result.image_bytes):,} bytes"
            )
        except Exception as e:  # noqa: BLE001 — best-effort, EPUB already written
            if tmp is not None:
                tmp.unlink(missing_ok=True)
            console.print(f"[yellow]Cover translation failed (EPUB kept):[/yellow] {e}")

    wd.mark_stage(state, Stage.DONE)
    cache.close()

    # Summary
    summary_parts = [
        f"Chunks: {stats.chunks_translated} new, {stats.chunks_cached} cached, "
        f"{stats.chunks_failed} failed."
    ]

    if judge_stats:
        summary_parts.append(
            f"Judge: {judge_stats.chunks_judged} new, {judge_stats.chunks_cached} cached."
        )
    if reflect_stats:
        summary_parts.append(
            f"Reflect: {reflect_stats.chunks_improved} improved, "
            f"{reflect_stats.chunks_unchanged} unchanged."
        )
    for pp_stage_name, pp_st in pp_stats_list:
        summary_parts.append(
            f"{pp_stage_name.capitalize()}: {pp_st.chunks_changed} changed, "
            f"{pp_st.chunks_unchanged} unchanged."
        )

    console.print(
        f"\n[green]Translated EPUB:[/green] {out_path}\n"
        f"[dim]{' '.join(summary_parts)}\n"
        f"Tokens: {stats.input_tokens:,} in, "
        f"{stats.output_tokens:,} out.[/dim]"
    )

    # Helpful tips
    if not no_judge and judge_stats and judge_stats.results:
        wd_str = str(wd.root)
        console.print(
            "\n[dim]💡 Skip judge+reflect: --no-judge\n"
            f"💡 Review scores: btrans status {wd_str} --scores\n"
            "💡 Revert a reflection: btrans prefer CHUNK translate\n"
            f"💡 Assemble from first pass: btrans assemble {wd_str}"
            " --from translate[/dim]"
        )


def _default_output_path(epub: Path, target_lang: str) -> Path:
    suffix = f"-{target_lang}.epub"
    return epub.with_name(epub.stem + suffix)


# ---------------------------------------------------------------------------
# judge (standalone)
# ---------------------------------------------------------------------------


@app.command("judge")
def judge_cmd(
    epub: Path = typer.Argument(..., exists=True, dir_okay=False, help="Source EPUB."),
    series: str | None = typer.Option(
        None,
        "--series",
        "-s",
        help="Series slug for glossary context.",
    ),
    glossary_path: Path | None = typer.Option(
        None,
        "--glossary",
        "-g",
        exists=True,
        dir_okay=False,
        help="Path to a single-book glossary.json.",
    ),
    config_path: Path = typer.Option(
        DEFAULT_CONFIG,
        "--config",
        "-c",
        help="YAML config file.",
    ),
    work_dir: Path = typer.Option(
        DEFAULT_WORK_DIR,
        "--work",
        "-w",
        help="Base directory for artifacts.",
    ),
    model: str | None = typer.Option(
        None,
        "--model",
        "-m",
        help="Override the judge model.",
    ),
    parallelism: int | None = typer.Option(
        None,
        "--parallelism",
        "-j",
        help="Number of chunks to judge concurrently.",
    ),
    no_cache: bool = typer.Option(
        False,
        "--no-cache",
        help=(
            "Score every chunk again, ignoring cached verdicts. Results are stored "
            "under a fresh run id, so the existing verdicts survive — this is how the "
            "judge's own spread gets measured: score the same text twice and compare."
        ),
    ),
    limit_chunks: int | None = typer.Option(
        None,
        "--limit-chunks",
        help=("Score only the first N chunks. Sampling instead of paying for the whole book."),
    ),
    from_stage: str = typer.Option(
        "translate",
        "--from",
        help=(
            "Which stage to score: translate (default), reflect, proofread, style, "
            "verify, repair — or 'final' for the text the book would be assembled "
            "from, honouring `btrans prefer`."
        ),
    ),
) -> None:
    """Run the judge pass on already-translated chunks.

    Scores each chunk 1-5 and stores results in the cache.
    Use 'btrans status WORKDIR --scores' to view results.
    """
    from .judge import Judge

    cfg = load_config(config_path if config_path.exists() else None)

    console.print(f"[bold]Reading EPUB:[/bold] {epub}")
    book = read_book_structured(epub)
    console.print(f"[bold]Book:[/bold] {book.meta.title}")

    glossary = _resolve_glossary(series, glossary_path, work_dir, cfg, book)

    wd = WorkDir.for_book(work_dir, book.meta.title)
    cache = Cache(wd.cache_path)

    # Get translated chunk IDs
    translate_ids = cache.get_all_chunk_ids_for_stage("translate")
    if not translate_ids:
        console.print("[red]No translated chunks found. Run translate first.[/red]")
        cache.close()
        raise typer.Exit(code=1)

    # Verify chunker config matches what was used during translate
    try:
        verify_chunker_params(cache, cfg.chunker.target_words, cfg.chunker.overlap_paragraphs)
    except ChunkerConfigMismatchError as e:
        console.print(f"[red]Error:[/red] {e}")
        cache.close()
        raise typer.Exit(code=1) from e

    # Build chunk set for originals
    chunk_set = chunk_book(
        book,
        target_words=cfg.chunker.target_words,
        overlap_paragraphs=cfg.chunker.overlap_paragraphs,
    )

    # Collect originals and translations using shared helpers
    chunk_originals = collect_chunk_originals(chunk_set, translate_ids)
    if from_stage == "final":
        # What `btrans assemble` would use, `btrans prefer` overrides included.
        chunk_translations = collect_preferred_translations(cache, translate_ids)
    else:
        stage_ids = cache.get_all_chunk_ids_for_stage(from_stage)
        if not stage_ids:
            console.print(
                f"[red]No chunks cached for stage {from_stage!r}.[/red]\n"
                f"[dim]Known stages: {', '.join(STAGE_WATERFALL)}, or 'final'.[/dim]"
            )
            cache.close()
            raise typer.Exit(code=1)
        chunk_translations = collect_stage_translations(cache, stage_ids, stage=from_stage)

    judgeable_ids = sorted(set(chunk_originals.keys()) & set(chunk_translations.keys()))
    if limit_chunks is not None:
        judgeable_ids = judgeable_ids[:limit_chunks]

    run_id = f"run-{datetime.now(UTC):%Y%m%dT%H%M%SZ}" if no_cache else None
    if run_id:
        console.print(f"[bold]Fresh run:[/bold] {run_id} (cached verdicts left untouched)")
    console.print(f"[bold]Judging stage:[/bold] {from_stage}")
    console.print(f"[bold]Chunks to judge:[/bold] {len(judgeable_ids)}")
    if from_stage != "translate":
        console.print(
            "[yellow]Note:[/yellow] verdicts are tagged with the stage they describe, "
            "but `btrans repair` and `btrans status` read every verdict, so this cache "
            "will now hold more than one per chunk."
        )

    judge_model = model or cfg.models.judge
    console.print(f"[bold]Model:[/bold] {judge_model}")

    JUDGE_PROMPT = Path("prompts/judge.md")
    judge = Judge(
        provider=create_stage_provider(cfg, "judge"),
        prompt_path=JUDGE_PROMPT,
        cache=cache,
        glossary=glossary,
        model=judge_model,
        source_lang=cfg.source_lang,
        target_lang=cfg.target_lang,
        judged_stage=from_stage,
        run_id=run_id,
    )

    if parallelism is not None and parallelism < 1:
        console.print("[red]--parallelism must be ≥ 1.[/red]")
        cache.close()
        raise typer.Exit(code=1)

    judge_parallelism = parallelism or 4

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        jtask = progress.add_task("judging", total=len(judgeable_ids))

        def on_progress(done, total, cid, jstats):
            progress.update(
                jtask,
                completed=done,
                description=(
                    f"judge | cached {jstats.chunks_cached} "
                    f"new {jstats.chunks_judged} "
                    f"failed {jstats.chunks_failed}"
                ),
            )

        judge_stats = judge.judge_chunks(
            {k: v for k, v in chunk_originals.items() if k in judgeable_ids},
            {k: v for k, v in chunk_translations.items() if k in judgeable_ids},
            on_progress=on_progress,
            parallelism=judge_parallelism,
        )

    cache.close()

    # Print results
    if judge_stats.results:
        dist: dict[int, int] = {}
        for r in judge_stats.results:
            dist[r.score] = dist.get(r.score, 0) + 1
        dist_str = "  ".join(f"★{k}: {v}" for k, v in sorted(dist.items(), reverse=True))
        console.print(f"\n[bold]Score distribution:[/bold] {dist_str}")
        console.print(
            f"[dim]Judged: {judge_stats.chunks_judged} new, "
            f"{judge_stats.chunks_cached} cached, "
            f"{judge_stats.chunks_failed} failed. "
            f"Tokens: {judge_stats.input_tokens:,} in, "
            f"{judge_stats.output_tokens:,} out.[/dim]"
        )
    else:
        console.print("[yellow]No chunks were judged.[/yellow]")


# ---------------------------------------------------------------------------
# reflect (standalone)
# ---------------------------------------------------------------------------


@app.command("reflect")
def reflect_cmd(
    epub: Path = typer.Argument(..., exists=True, dir_okay=False, help="Source EPUB."),
    series: str | None = typer.Option(
        None,
        "--series",
        "-s",
        help="Series slug for glossary context.",
    ),
    glossary_path: Path | None = typer.Option(
        None,
        "--glossary",
        "-g",
        exists=True,
        dir_okay=False,
        help="Path to a single-book glossary.json.",
    ),
    config_path: Path = typer.Option(
        DEFAULT_CONFIG,
        "--config",
        "-c",
        help="YAML config file.",
    ),
    work_dir: Path = typer.Option(
        DEFAULT_WORK_DIR,
        "--work",
        "-w",
        help="Base directory for artifacts.",
    ),
    model: str | None = typer.Option(
        None,
        "--model",
        "-m",
        help="Override the reflect model.",
    ),
    threshold: int | None = typer.Option(
        None,
        "--threshold",
        "-t",
        help="Reflect chunks with judge score ≤ N (default from config).",
    ),
    all_chunks: bool = typer.Option(
        False,
        "--all",
        help="Reflect all chunks regardless of score.",
    ),
    parallelism: int | None = typer.Option(
        None,
        "--parallelism",
        "-j",
        help="Number of chunks to reflect concurrently.",
    ),
) -> None:
    """Run the reflect pass on chunks that scored poorly.

    Requires judge scores in cache. Use 'btrans judge' first if needed.
    """
    from .reflect import Reflector

    cfg = load_config(config_path if config_path.exists() else None)

    if threshold is not None and not 1 <= threshold <= 5:
        console.print("[red]--threshold must be between 1 and 5.[/red]")
        raise typer.Exit(code=1)

    console.print(f"[bold]Reading EPUB:[/bold] {epub}")
    book = read_book_structured(epub)
    console.print(f"[bold]Book:[/bold] {book.meta.title}")

    glossary = _resolve_glossary(series, glossary_path, work_dir, cfg, book)

    wd = WorkDir.for_book(work_dir, book.meta.title)
    cache = Cache(wd.cache_path)

    # Get judge scores
    # Reflect re-does the translate stage, so it reads verdicts about
    # that stage — not about a later one, and not from a measurement run.
    judge_scores = cache.get_judge_scores(judged_stage="translate")
    if not judge_scores and not all_chunks:
        console.print(
            "[red]No judge scores found. Run 'btrans judge' first, "
            "or use --all to reflect all translated chunks.[/red]"
        )
        cache.close()
        raise typer.Exit(code=1)

    # Determine which chunks to reflect
    trigger = threshold if threshold is not None else cfg.reflection.trigger_score

    if all_chunks:
        # Reflect all translated chunks regardless of judge scores.
        # judge_by_id may be empty if judge hasn't run — that's fine,
        # build_reflect_input handles missing entries with defaults.
        translate_ids = cache.get_all_chunk_ids_for_stage("translate")
        chunks_to_reflect_ids = set(translate_ids)
        judge_by_id = normalize_judge_map(judge_scores) if judge_scores else {}
    else:
        chunks_to_reflect_ids = set()
        judge_by_id = normalize_judge_map(judge_scores)
        for cid, data in judge_by_id.items():
            if data["score"] <= trigger:
                chunks_to_reflect_ids.add(cid)

    if not chunks_to_reflect_ids:
        console.print(f"[green]All chunks scored above {trigger}. Nothing to reflect.[/green]")
        cache.close()
        return

    console.print(f"[bold]Chunks to reflect:[/bold] {len(chunks_to_reflect_ids)}")

    # Verify chunker config matches what was used during translate
    try:
        verify_chunker_params(cache, cfg.chunker.target_words, cfg.chunker.overlap_paragraphs)
    except ChunkerConfigMismatchError as e:
        console.print(f"[red]Error:[/red] {e}")
        cache.close()
        raise typer.Exit(code=1) from e

    # Build chunk set for originals
    chunk_set = chunk_book(
        book,
        target_words=cfg.chunker.target_words,
        overlap_paragraphs=cfg.chunker.overlap_paragraphs,
    )

    # Build reflect input using shared helper
    chunks_to_reflect_data = build_reflect_input(
        chunk_set, cache, chunks_to_reflect_ids, judge_by_id
    )

    if not chunks_to_reflect_data:
        console.print("[yellow]No chunks with translations to reflect on.[/yellow]")
        cache.close()
        return

    reflect_model = model or cfg.models.reflect
    console.print(f"[bold]Model:[/bold] {reflect_model}")

    REFLECT_PROMPT = Path("prompts/reflect.md")
    reflector = Reflector(
        provider=create_stage_provider(cfg, "reflect"),
        reflect_prompt_path=REFLECT_PROMPT,
        translate_prompt_path=TRANSLATE_PROMPT,
        cache=cache,
        glossary=glossary,
        model=reflect_model,
        source_lang=cfg.source_lang,
        target_lang=cfg.target_lang,
    )

    if parallelism is not None and parallelism < 1:
        console.print("[red]--parallelism must be ≥ 1.[/red]")
        cache.close()
        raise typer.Exit(code=1)

    reflect_parallelism = parallelism or 2

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        rtask = progress.add_task("reflecting", total=len(chunks_to_reflect_data))

        def on_progress(done, total, cid, rstats):
            progress.update(
                rtask,
                completed=done,
                description=(
                    f"reflect | improved {rstats.chunks_improved} "
                    f"unchanged {rstats.chunks_unchanged} "
                    f"failed {rstats.chunks_failed}"
                ),
            )

        reflect_stats = reflector.reflect_chunks(
            chunks_to_reflect_data,
            on_progress=on_progress,
            parallelism=reflect_parallelism,
        )

    cache.close()

    console.print(
        f"\n[bold]Reflect results:[/bold]\n"
        f"  Improved: {reflect_stats.chunks_improved}/{reflect_stats.chunks_total}\n"
        f"  No change: {reflect_stats.chunks_unchanged}/{reflect_stats.chunks_total}\n"
        f"  Failed: {reflect_stats.chunks_failed}/{reflect_stats.chunks_total}\n"
        f"[dim]Tokens: {reflect_stats.input_tokens:,} in, "
        f"{reflect_stats.output_tokens:,} out.[/dim]"
    )


# ---------------------------------------------------------------------------
# proofread (standalone)
# ---------------------------------------------------------------------------


@app.command("proofread")
def proofread_cmd(
    epub: Path = typer.Argument(..., exists=True, dir_okay=False, help="Source EPUB."),
    series: str | None = typer.Option(
        None,
        "--series",
        "-s",
        help="Series slug for glossary context.",
    ),
    glossary_path: Path | None = typer.Option(
        None,
        "--glossary",
        "-g",
        exists=True,
        dir_okay=False,
        help="Path to a single-book glossary.json.",
    ),
    config_path: Path = typer.Option(
        DEFAULT_CONFIG,
        "--config",
        "-c",
        help="YAML config file.",
    ),
    work_dir: Path = typer.Option(
        DEFAULT_WORK_DIR,
        "--work",
        "-w",
        help="Base directory for artifacts.",
    ),
    model: str | None = typer.Option(
        None,
        "--model",
        "-m",
        help="Override the proofread model.",
    ),
    parallelism: int | None = typer.Option(
        None,
        "--parallelism",
        "-j",
        help="Number of chunks to process concurrently.",
    ),
) -> None:
    """Run the proofread pass: fix grammar, punctuation, typos.

    Reads from the latest available stage (waterfall) and stores
    results as stage='proofread'.
    """
    _run_postprocess_cmd(
        stage="proofread",
        prompt_path=Path("prompts/proofread.md"),
        epub=epub,
        series=series,
        glossary_path=glossary_path,
        config_path=config_path,
        work_dir=work_dir,
        model_override=model,
        parallelism=parallelism,
    )


# ---------------------------------------------------------------------------
# style (standalone)
# ---------------------------------------------------------------------------


@app.command("style")
def style_cmd(
    epub: Path = typer.Argument(..., exists=True, dir_okay=False, help="Source EPUB."),
    series: str | None = typer.Option(
        None,
        "--series",
        "-s",
        help="Series slug for glossary context.",
    ),
    glossary_path: Path | None = typer.Option(
        None,
        "--glossary",
        "-g",
        exists=True,
        dir_okay=False,
        help="Path to a single-book glossary.json.",
    ),
    config_path: Path = typer.Option(
        DEFAULT_CONFIG,
        "--config",
        "-c",
        help="YAML config file.",
    ),
    work_dir: Path = typer.Option(
        DEFAULT_WORK_DIR,
        "--work",
        "-w",
        help="Base directory for artifacts.",
    ),
    model: str | None = typer.Option(
        None,
        "--model",
        "-m",
        help="Override the style model.",
    ),
    parallelism: int | None = typer.Option(
        None,
        "--parallelism",
        "-j",
        help="Number of chunks to process concurrently.",
    ),
) -> None:
    """Run the style pass: fix calques, bureaucratic language, improve voice.

    Reads from the latest available stage (waterfall) and stores
    results as stage='style'.
    """
    _run_postprocess_cmd(
        stage="style",
        prompt_path=Path("prompts/style.md"),
        epub=epub,
        series=series,
        glossary_path=glossary_path,
        config_path=config_path,
        work_dir=work_dir,
        model_override=model,
        parallelism=parallelism,
    )


# ---------------------------------------------------------------------------
# verify (standalone)
# ---------------------------------------------------------------------------


@app.command("verify")
def verify_cmd(
    epub: Path = typer.Argument(..., exists=True, dir_okay=False, help="Source EPUB."),
    series: str | None = typer.Option(
        None,
        "--series",
        "-s",
        help="Series slug for glossary context.",
    ),
    glossary_path: Path | None = typer.Option(
        None,
        "--glossary",
        "-g",
        exists=True,
        dir_okay=False,
        help="Path to a single-book glossary.json.",
    ),
    config_path: Path = typer.Option(
        DEFAULT_CONFIG,
        "--config",
        "-c",
        help="YAML config file.",
    ),
    work_dir: Path = typer.Option(
        DEFAULT_WORK_DIR,
        "--work",
        "-w",
        help="Base directory for artifacts.",
    ),
    model: str | None = typer.Option(
        None,
        "--model",
        "-m",
        help="Override the verify model.",
    ),
    parallelism: int | None = typer.Option(
        None,
        "--parallelism",
        "-j",
        help="Number of chunks to process concurrently.",
    ),
) -> None:
    """Run the verify pass: check accuracy against original, fix omissions.

    Reads from the latest available stage (waterfall) and stores
    results as stage='verify'. Requires the original EPUB for comparison.
    """
    _run_postprocess_cmd(
        stage="verify",
        prompt_path=Path("prompts/verify.md"),
        epub=epub,
        series=series,
        glossary_path=glossary_path,
        config_path=config_path,
        work_dir=work_dir,
        model_override=model,
        parallelism=parallelism,
    )


@app.command("repair")
def repair_cmd(
    epub: Path = typer.Argument(..., exists=True, dir_okay=False, help="Source EPUB."),
    series: str | None = typer.Option(
        None, "--series", "-s", help="Series slug for glossary context."
    ),
    glossary_path: Path | None = typer.Option(
        None,
        "--glossary",
        "-g",
        exists=True,
        dir_okay=False,
        help="Path to a single-book glossary.json.",
    ),
    config_path: Path = typer.Option(DEFAULT_CONFIG, "--config", "-c", help="YAML config file."),
    work_dir: Path = typer.Option(
        DEFAULT_WORK_DIR, "--work", "-w", help="Base directory for artifacts."
    ),
    model: str | None = typer.Option(None, "--model", "-m", help="Override the repair model."),
    categories: str | None = typer.Option(
        None,
        "--categories",
        help="Comma-separated judge categories to act on (default: grammar,markup,accuracy).",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show what would be proposed without calling a model or writing anything.",
    ),
) -> None:
    """Apply the judge's mechanical corrections, one vetted fix at a time.

    Each correction the judge quoted is turned into a candidate substitution,
    and a model holding the original decides whether it preserves the meaning.
    Rejected candidates are left alone: a wrong correction is printed in a
    book, a missed one is not.

    Requires a judge pass. Reads the latest available stage and stores results
    as stage='repair'.
    """
    from .pipeline_helpers import (
        collect_chunk_originals,
        collect_waterfall_paragraphs,
        normalize_judge_map,
        verify_chunker_params,
    )
    from .repair import Repairer, RepairStats, propose

    cfg = load_config(config_path if config_path.exists() else None)
    wanted = tuple(
        c.strip() for c in (categories.split(",") if categories else cfg.repair.categories)
    )

    console.print(f"[bold]Reading EPUB:[/bold] {epub}")
    book = read_book_structured(epub)
    console.print(f"[bold]Book:[/bold] {book.meta.title}")

    glossary = _resolve_glossary(series, glossary_path, work_dir, cfg, book)
    wd = WorkDir.for_book(work_dir, book.meta.title)
    cache = Cache(wd.cache_path)

    try:
        # Verdicts about the draft: that is what the run this stage was
        # measured on used, and the quoted fragments still resolve against the
        # waterfall text. Measurement runs are excluded by default.
        judge_map = normalize_judge_map(cache.get_judge_scores(judged_stage="translate"))
        if not judge_map:
            console.print("[red]No judge results found. Run 'btrans judge' first.[/red]")
            raise typer.Exit(code=1)

        try:
            verify_chunker_params(cache, cfg.chunker.target_words, cfg.chunker.overlap_paragraphs)
        except ChunkerConfigMismatchError as e:
            console.print(f"[red]Error:[/red] {e}")
            raise typer.Exit(code=1) from e

        chunk_set = chunk_book(
            book,
            target_words=cfg.chunker.target_words,
            overlap_paragraphs=cfg.chunker.overlap_paragraphs,
        )
        paragraph_counts = {chunk.id: len(chunk.paragraph_indexes) for chunk in chunk_set.chunks}
        chunk_ids = [c.id for c in chunk_set.chunks if c.id in judge_map]
        waterfall = collect_waterfall_paragraphs(cache, chunk_ids, "repair", paragraph_counts)
        if not waterfall:
            console.print("[yellow]No chunks with parseable translations to repair.[/yellow]")
            return

        console.print(f"[bold]Categories:[/bold] {', '.join(wanted)}")

        if dry_run:
            total = sum(
                len(propose(paras, judge_map[cid]["issues"], wanted).candidates)
                for cid, paras in waterfall.items()
            )
            console.print(
                f"[bold]Dry run:[/bold] {total} candidate substitutions across "
                f"{len(waterfall)} chunks. Nothing called, nothing written."
            )
            return

        originals = collect_chunk_originals(chunk_set, list(waterfall))
        chosen_model = model or cfg.models.repair
        console.print(f"[bold]Model:[/bold] {chosen_model}\n")

        repairer = Repairer(
            provider=create_stage_provider(cfg, "repair"),
            prompt_path=Path("prompts/repair.md"),
            cache=cache,
            glossary=glossary,
            model=chosen_model,
            categories=wanted,
            source_lang=cfg.source_lang,
            target_lang=cfg.target_lang,
        )

        stats = RepairStats(chunks_total=len(waterfall))
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("repair", total=len(waterfall))
            for done, (cid, paras) in enumerate(sorted(waterfall.items()), 1):
                try:
                    repairer.repair_chunk(
                        cid, paras, judge_map[cid]["issues"], originals.get(cid, ""), stats
                    )
                except Exception as e:  # noqa: BLE001
                    stats.chunks_failed += 1
                    console.print(f"[yellow]{cid}: {type(e).__name__}: {e}[/yellow]")
                progress.update(
                    task,
                    completed=done,
                    description=(
                        f"repair | accepted {stats.accepted} "
                        f"rejected {stats.rejected} failed {stats.chunks_failed}"
                    ),
                )

        console.print(
            f"\n[bold]Repair results:[/bold]\n"
            f"  Candidates: {stats.candidates}\n"
            f"  Applied:    {stats.accepted}\n"
            f"  Rejected:   {stats.rejected}\n"
            f"  Chunks changed: {stats.chunks_repaired}/{stats.chunks_total}\n"
            f"  Failed: {stats.chunks_failed}\n"
            f"[dim]Tokens: {stats.input_tokens:,} in, {stats.output_tokens:,} out.[/dim]"
        )
        if stats.unhandled:
            console.print(
                f"\n[dim]{len(stats.unhandled)} issues could not be turned into a "
                f"substitution and were left for a human.[/dim]"
            )
    finally:
        cache.close()


# ---------------------------------------------------------------------------
# shared postprocess runner
# ---------------------------------------------------------------------------


def _run_postprocess_cmd(
    stage: str,
    prompt_path: Path,
    epub: Path,
    series: str | None,
    glossary_path: Path | None,
    config_path: Path,
    work_dir: Path,
    model_override: str | None,
    parallelism: int | None,
) -> None:
    """Shared implementation for proofread/style/verify standalone commands."""
    from .pipeline_helpers import (
        collect_chunk_originals,
        collect_waterfall_paragraphs,
        verify_chunker_params,
    )
    from .postprocess import PostProcessor

    cfg = load_config(config_path if config_path.exists() else None)

    console.print(f"[bold]Reading EPUB:[/bold] {epub}")
    book = read_book_structured(epub)
    console.print(f"[bold]Book:[/bold] {book.meta.title}")

    glossary = _resolve_glossary(series, glossary_path, work_dir, cfg, book)

    wd = WorkDir.for_book(work_dir, book.meta.title)
    cache = Cache(wd.cache_path)

    try:
        # Verify translated chunks exist
        translate_ids = cache.get_all_chunk_ids_for_stage("translate")
        if not translate_ids:
            console.print("[red]No translated chunks found. Run translate first.[/red]")
            raise typer.Exit(code=1)

        # Verify chunker config
        try:
            verify_chunker_params(cache, cfg.chunker.target_words, cfg.chunker.overlap_paragraphs)
        except ChunkerConfigMismatchError as e:
            console.print(f"[red]Error:[/red] {e}")
            raise typer.Exit(code=1) from e

        # Build chunk set for paragraph counts and originals
        chunk_set = chunk_book(
            book,
            target_words=cfg.chunker.target_words,
            overlap_paragraphs=cfg.chunker.overlap_paragraphs,
        )

        # Build paragraph count map
        paragraph_counts = {chunk.id: len(chunk.paragraph_indexes) for chunk in chunk_set.chunks}

        # Scope to current chunk_set IDs intersected with cache
        current_chunk_ids = [chunk.id for chunk in chunk_set.chunks]
        translate_id_set = set(translate_ids)
        valid_chunk_ids = [cid for cid in current_chunk_ids if cid in translate_id_set]

        # Collect waterfall translations parsed into paragraphs
        waterfall_paragraphs = collect_waterfall_paragraphs(
            cache, valid_chunk_ids, stage, paragraph_counts
        )

        if not waterfall_paragraphs:
            console.print(
                f"[yellow]No chunks with parseable translations for {stage} pass.[/yellow]"
            )
            return

        # For verify stage, also collect originals
        originals_map: dict[str, str] | None = None
        if stage == "verify":
            originals_map_raw = collect_chunk_originals(
                chunk_set, list(waterfall_paragraphs.keys())
            )
            originals_map = originals_map_raw

        # Build chunks_data for PostProcessor
        chunks_data = []
        for cid, paragraphs in sorted(waterfall_paragraphs.items()):
            entry: dict = {
                "chunk_id": cid,
                "translated_paragraphs": paragraphs,
            }
            if stage == "verify" and originals_map:
                entry["original_text"] = originals_map.get(cid, "")
            chunks_data.append(entry)

        console.print(f"[bold]Stage:[/bold] {stage}\n[bold]Chunks:[/bold] {len(chunks_data)}")

        # Resolve model
        model_map = {
            "proofread": cfg.models.proofread,
            "style": cfg.models.style,
            "verify": cfg.models.verify,
        }
        chosen_model = model_override or model_map.get(stage, cfg.models.translate)
        console.print(f"[bold]Model:[/bold] {chosen_model}")

        processor = PostProcessor(
            provider=create_stage_provider(cfg, stage),
            prompt_path=prompt_path,
            cache=cache,
            glossary=glossary,
            stage=stage,
            model=chosen_model,
            source_lang=cfg.source_lang,
            target_lang=cfg.target_lang,
        )

        effective_parallelism = parallelism or 4

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            ptask = progress.add_task(stage, total=len(chunks_data))

            def on_progress(done, total, cid, pstats):
                progress.update(
                    ptask,
                    completed=done,
                    description=(
                        f"{stage} | cached {pstats.chunks_cached} "
                        f"new {pstats.chunks_processed} "
                        f"failed {pstats.chunks_failed}"
                    ),
                )

            pp_stats = processor.process_chunks(
                chunks_data,
                on_progress=on_progress,
                parallelism=effective_parallelism,
            )

        console.print(
            f"\n[bold]{stage.capitalize()} results:[/bold]\n"
            f"  Changed: {pp_stats.chunks_changed}/{pp_stats.chunks_total}\n"
            f"  Unchanged: {pp_stats.chunks_unchanged}/{pp_stats.chunks_total}\n"
            f"  Failed: {pp_stats.chunks_failed}/{pp_stats.chunks_total}\n"
            f"[dim]Tokens: {pp_stats.input_tokens:,} in, "
            f"{pp_stats.output_tokens:,} out.[/dim]"
        )
    finally:
        cache.close()


# ---------------------------------------------------------------------------
# helper: resolve glossary for judge/reflect commands
# ---------------------------------------------------------------------------


def _resolve_glossary(
    series: str | None,
    glossary_path: Path | None,
    work_dir: Path,
    cfg,
    book,
) -> SeriesGlossary | None:
    """Resolve glossary from --series or --glossary flags."""
    if series and glossary_path:
        console.print("[red]--series and --glossary are mutually exclusive.[/red]")
        raise typer.Exit(code=1)

    if series:
        swd = SeriesWorkDir.for_series(work_dir, series)
        if not swd.exists():
            console.print(f"[red]Series {series!r} not found.[/red]")
            raise typer.Exit(code=1)
        glossary = load_series_glossary(swd.glossary_path)
        console.print(f"[bold]Series:[/bold] {series} ({len(glossary.entries)} terms)")
        return glossary

    if glossary_path:
        book_glossary = Glossary.model_validate(
            json.loads(glossary_path.read_text(encoding="utf-8"))
        )
        glossary = SeriesGlossary(
            series_slug=f"ad-hoc:{book.meta.title}",
            title=book.meta.title,
            author=book.meta.author,
            source_lang=cfg.source_lang,
            target_lang=cfg.target_lang,
            entries=[
                SeriesGlossaryEntry(
                    original=e.original,
                    translation=e.translation,
                    type=e.type,
                    gender=e.gender,
                    plural=e.plural,
                    notes=e.notes,
                    origin_book=book_glossary.book,
                )
                for e in book_glossary.entries
            ],
        )
        return glossary

    return None


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------


@app.command("status")
def status_overview(
    work_path: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=False,
        help="Book workdir (e.g. work/broken-homes).",
    ),
    scores: bool = typer.Option(
        False,
        "--scores",
        help="Show judge scores table.",
    ),
    below: int | None = typer.Option(
        None,
        "--below",
        help="Only show chunks with score below N.",
    ),
    diff: str | None = typer.Option(
        None,
        "--diff",
        help="Show all passes for a specific chunk_id.",
    ),
    stages_filter: str | None = typer.Option(
        None,
        "--stages",
        help="Comma-separated stages to compare (use with --diff).",
    ),
    assembly_map: bool = typer.Option(
        False,
        "--assembly-map",
        help="Show which stage each chunk will use at assembly.",
    ),
    grid: bool = typer.Option(
        False,
        "--grid",
        help="Show chunk × stage matrix with changed/unchanged status.",
    ),
) -> None:
    """Inspect the translation state of a book."""

    cache_path = work_path / "cache.sqlite"
    if not cache_path.exists():
        console.print(f"[red]No cache.sqlite in {work_path}[/red]")
        raise typer.Exit(code=1)

    cache = Cache(cache_path)
    try:
        _run_status(cache, work_path, scores, below, diff, stages_filter, assembly_map, grid)
    finally:
        cache.close()


def _run_status(
    cache: Cache,
    work_path: Path,
    scores: bool,
    below: int | None,
    diff: str | None,
    stages_filter: str | None,
    assembly_map: bool,
    grid: bool,
) -> None:
    """Inner logic for status command (cache is managed by caller)."""
    from .status import (
        build_grid,
        build_status_report,
        get_chunk_diff,
        get_scores,
    )

    # --diff mode: show diffs between stages for a specific chunk
    if diff:
        filter_list = stages_filter.split(",") if stages_filter else None
        entries = get_chunk_diff(cache, diff, filter_list)
        if not entries:
            console.print(f"[yellow]No cached entries for chunk {diff!r}[/yellow]")
            return

        # Filter to waterfall stages only (skip judge which is metadata)
        waterfall_entries = [e for e in entries if e["stage"] != "judge"]

        console.print(f"\n[bold]Chunk:[/bold] {diff}")
        console.print(f"[dim]Stages: {', '.join(e['stage'] for e in waterfall_entries)}[/dim]\n")

        if len(waterfall_entries) == 0:
            console.print("[yellow]No waterfall stages found.[/yellow]")
            return

        # Show first stage (translate) as-is, then diffs between consecutive stages
        import difflib

        first = waterfall_entries[0]
        console.print(
            f"[bold cyan]─── {first['stage']} ───[/bold cyan] "
            f"[dim]({first['model']}, {first['created_at']})[/dim]"
        )
        console.print(first["content"])
        console.print()

        for i in range(1, len(waterfall_entries)):
            prev = waterfall_entries[i - 1]
            curr = waterfall_entries[i]

            prev_lines = prev["content"].splitlines(keepends=True)
            curr_lines = curr["content"].splitlines(keepends=True)

            diff_lines = list(
                difflib.unified_diff(
                    prev_lines,
                    curr_lines,
                    fromfile=prev["stage"],
                    tofile=curr["stage"],
                    lineterm="",
                )
            )

            console.print(
                f"[bold cyan]─── {curr['stage']} ───[/bold cyan] "
                f"[dim]({curr['model']}, {curr['created_at']})[/dim]"
            )

            if not diff_lines:
                console.print("[dim]  (no changes)[/dim]")
            else:
                for line in diff_lines:
                    line = line.rstrip("\n")
                    if line.startswith("+++") or line.startswith("---"):
                        console.print(f"[dim]{line}[/dim]")
                    elif line.startswith("@@"):
                        console.print(f"[blue]{line}[/blue]")
                    elif line.startswith("+"):
                        console.print(f"[green]{line}[/green]")
                    elif line.startswith("-"):
                        console.print(f"[red]{line}[/red]")
                    else:
                        console.print(f"[dim]{line}[/dim]")
            console.print()
        return

    # --grid mode: chunk × stage matrix
    if grid:
        grid_rows = build_grid(cache)
        if not grid_rows:
            console.print("[yellow]No translated chunks in cache.[/yellow]")
            return

        # Determine which stages are present
        all_stages_present = set()
        for row in grid_rows:
            for stage, cell in row.cells.items():
                if cell.present:
                    all_stages_present.add(stage)

        # Column order
        from .status import _GRID_STAGES, _GRID_STAGES_WITH_REFLECT

        stages_order = (
            _GRID_STAGES_WITH_REFLECT if "reflect" in all_stages_present else _GRID_STAGES
        )

        table = Table(
            title="Chunk × Stage grid",
            show_header=True,
            show_lines=False,
        )
        table.add_column("Chunk", style="cyan")
        table.add_column("★", justify="center", width=2)
        for stage in stages_order:
            table.add_column(stage[:5], justify="center", width=5)
        table.add_column("Source", style="dim")

        # Summary counters
        stage_changed: dict[str, int] = {s: 0 for s in stages_order}
        stage_unchanged: dict[str, int] = {s: 0 for s in stages_order}

        for row in grid_rows:
            score_str = ""
            if row.judge_score is not None:
                if row.judge_score >= 4:
                    score_str = f"[green]{row.judge_score}[/green]"
                elif row.judge_score == 3:
                    score_str = f"[yellow]{row.judge_score}[/yellow]"
                elif row.judge_score > 0:
                    score_str = f"[red]{row.judge_score}[/red]"
                else:
                    score_str = "[dim]?[/dim]"

            cells_str: list[str] = []
            # Track the effective source for this chunk
            effective_source = "translate"
            for stage in stages_order:
                cell = row.cells.get(stage)
                if cell is None or not cell.present:
                    cells_str.append("[dim]—[/dim]")
                elif cell.changed is None:
                    # translate (base) — always present
                    cells_str.append("[bold]✓[/bold]")
                    effective_source = stage
                elif cell.changed:
                    cells_str.append("[green]Δ[/green]")
                    effective_source = stage
                    stage_changed[stage] += 1
                else:
                    cells_str.append("[dim]=[/dim]")
                    stage_unchanged[stage] += 1

            table.add_row(row.chunk_id, score_str, *cells_str, effective_source)

        console.print(table)

        # Summary line
        total = len(grid_rows)
        summary_parts = []
        for stage in stages_order:
            if stage == "translate":
                continue
            c = stage_changed.get(stage, 0)
            u = stage_unchanged.get(stage, 0)
            if c + u > 0:
                summary_parts.append(f"{stage}: {c}Δ {u}=")
        console.print("\n[dim]Legend: ✓=base  Δ=changed  =[dim]=unchanged  —=not run[/dim]")
        console.print(f"[dim]Totals ({total} chunks): {' | '.join(summary_parts)}[/dim]\n")
        return

    # --scores mode
    if scores:
        score_entries = get_scores(cache)
        if not score_entries:
            console.print("[yellow]No judge scores found. Run judge first.[/yellow]")
            return

        if below is not None:
            score_entries = [s for s in score_entries if s.score < below]

        if not score_entries:
            console.print(f"[green]All chunks scored {below} or above.[/green]")
            return

        table = Table(
            title=f"Judge scores{f' (below {below})' if below is not None else ''}",
            show_header=True,
        )
        table.add_column("Chunk", style="cyan")
        table.add_column("Score", justify="center")
        table.add_column("Issues")
        table.add_column("Reflected", justify="center")

        for entry in score_entries:
            score_style = "green" if entry.score >= 4 else "yellow" if entry.score == 3 else "red"
            table.add_row(
                entry.chunk_id,
                f"[{score_style}]{entry.score}[/{score_style}]",
                "; ".join(entry.issues[:3]) + ("..." if len(entry.issues) > 3 else ""),
                "✓" if entry.reflected else "",
            )

        console.print(table)
        return

    # --assembly-map mode
    if assembly_map:
        translate_ids = cache.get_all_chunk_ids_for_stage("translate")
        legacy_count = cache.count_legacy_rows("translate")

        if translate_ids:
            table = Table(title="Assembly map", show_header=True)
            table.add_column("Chunk", style="cyan")
            table.add_column("Stage", style="bold")
            table.add_column("Reason", style="dim")

            resolved = cache.resolve_stages_bulk(translate_ids)
            prefs = {p.chunk_id: p.preferred_stage for p in cache.list_preferences()}

            for chunk_id in sorted(translate_ids):
                stage = resolved.get(chunk_id, "—")
                # Only mark as 'preference' if the preference was actually applied
                pref_stage = prefs.get(chunk_id)
                if pref_stage and pref_stage == stage:
                    reason = "preference"
                elif pref_stage:
                    reason = "waterfall (preference unavailable)"
                else:
                    reason = "waterfall"
                table.add_row(chunk_id, stage, reason)

            console.print(table)

        if legacy_count > 0:
            console.print(
                f"\n[yellow]Legacy data:[/yellow] ~{legacy_count} additional "
                f"translated row(s) without chunk_id metadata.\n"
                f"[dim]These will use 'translate' stage by default. "
                f"Re-run translation to populate chunk_ids.[/dim]"
            )

        if not translate_ids and legacy_count == 0:
            console.print("[yellow]No translated chunks in cache.[/yellow]")
        return

    # Default: full overview
    report = build_status_report(cache, work_path)

    console.print(f"\n[bold]{work_path.name}[/bold]")
    console.print("━" * 70)
    console.print(f"\nChunks: {report.total_chunks} total\n")

    # Passes table
    table = Table(title="Passes in cache", show_header=True, show_lines=False)
    table.add_column("Stage", style="cyan")
    table.add_column("Progress", justify="right")
    table.add_column("Cost", justify="right")

    for stage_info in report.stages:
        if stage_info.count > 0:
            done = "✓" if stage_info.count >= stage_info.total_chunks else ""
            progress_str = f"{stage_info.count}/{stage_info.total_chunks} {done}"
            cost_str = f"${stage_info.cost_usd:.2f}"
        else:
            progress_str = f"0/{stage_info.total_chunks}"
            cost_str = "—"
        table.add_row(stage_info.stage, progress_str, cost_str)

    console.print(table)

    # Judge scores distribution
    if report.scores:
        dist: dict[int, int] = {}
        for s in report.scores:
            dist[s.score] = dist.get(s.score, 0) + 1
        reflected_count = sum(1 for s in report.scores if s.reflected)

        dist_str = "  ".join(f"★{k}: {v}" for k, v in sorted(dist.items(), reverse=True))
        console.print(f"\n[bold]Judge scores:[/bold] {dist_str}")
        if reflected_count:
            console.print(f"  Reflected: {reflected_count} chunks")

    # Preferences
    if report.preferences:
        pref_strs = [f"{p['chunk_id']} → {p['stage']}" for p in report.preferences[:5]]
        console.print(
            f"\n[bold]Preferences:[/bold] {len(report.preferences)} override(s) "
            f"({', '.join(pref_strs)}{'...' if len(report.preferences) > 5 else ''})"
        )

    # Assembly map summary
    if report.assembly_map:
        map_str = "  |  ".join(
            f"{stage}: {count}"
            for stage, count in sorted(
                report.assembly_map.items(),
                key=lambda x: STAGE_WATERFALL.index(x[0]) if x[0] in STAGE_WATERFALL else 99,
            )
        )
        console.print(f"\n[bold]Assembly source:[/bold] {map_str}")

    console.print(f"\n[bold]Total cost:[/bold] ${report.total_cost_usd:.2f}\n")


# ---------------------------------------------------------------------------
# prefer
# ---------------------------------------------------------------------------


@app.command("prefer")
def prefer_cmd(
    chunk_id: str = typer.Argument(..., help="Chunk ID (e.g. ch03_c02)."),
    stage: str | None = typer.Argument(
        None,
        help="Stage to prefer (translate, reflect, proofread, style, verify).",
    ),
    reason: str | None = typer.Option(
        None,
        "--reason",
        "-r",
        help="Why this preference.",
    ),
    reset: bool = typer.Option(
        False,
        "--reset",
        help="Remove preference, revert to waterfall.",
    ),
    work_path: Path = typer.Option(
        DEFAULT_WORK_DIR,
        "--work",
        "-w",
        help="Base work directory.",
    ),
    book: str | None = typer.Option(
        None,
        "--book",
        "-b",
        help="Book slug (subfolder of work dir). Auto-detected if only one exists.",
    ),
) -> None:
    """Set or reset the preferred stage for a chunk at assembly time.

    Does NOT delete any data from the cache — only marks which version to use.
    """
    cache_path = _resolve_cache_path(work_path, book)
    if not cache_path:
        raise typer.Exit(code=1)

    cache = Cache(cache_path)

    if reset:
        cache.reset_preference(chunk_id)
        console.print(f"[green]Reset preference for {chunk_id} (back to waterfall).[/green]")
        cache.close()
        return

    if not stage:
        console.print("[red]Provide a stage name, or use --reset.[/red]")
        cache.close()
        raise typer.Exit(code=1)

    if stage not in STAGE_WATERFALL:
        console.print(f"[red]Invalid stage {stage!r}. Valid: {', '.join(STAGE_WATERFALL)}[/red]")
        cache.close()
        raise typer.Exit(code=1)

    # Verify the stage exists for this chunk
    chunk_stages = cache.get_chunk_stages(chunk_id)
    available = {s.stage for s in chunk_stages}
    if stage not in available:
        console.print(
            f"[red]Stage {stage!r} not found for chunk {chunk_id}.[/red]\n"
            f"[dim]Available: {', '.join(sorted(available)) or 'none'}[/dim]"
        )
        cache.close()
        raise typer.Exit(code=1)

    cache.set_preference(chunk_id, stage, reason)
    console.print(
        f"[green]Set preference:[/green] {chunk_id} → [bold]{stage}[/bold]"
        + (f" [dim]({reason})[/dim]" if reason else "")
    )
    cache.close()


# ---------------------------------------------------------------------------
# assemble
# ---------------------------------------------------------------------------


@app.command("assemble")
def assemble_cmd(
    work_path: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=False,
        help="Book workdir (e.g. work/broken-homes).",
    ),
    from_stage: str | None = typer.Option(
        None,
        "--from",
        help="Use only this stage for all chunks (ignores waterfall).",
    ),
    out: Path | None = typer.Option(
        None,
        "--out",
        "-o",
        help="Output EPUB path.",
    ),
    epub: Path | None = typer.Option(
        None,
        "--epub",
        "-e",
        exists=True,
        dir_okay=False,
        help="Source EPUB (for structure).",
    ),
    notes: bool | None = typer.Option(
        None,
        "--notes/--no-notes",
        help="Inject reader footnotes from glossary (overrides config).",
    ),
    note_types: str | None = typer.Option(
        None,
        "--note-types",
        help="Comma-separated glossary types to annotate (e.g. concept,term,place).",
    ),
    series: str | None = typer.Option(
        None,
        "--series",
        "-s",
        help="Series slug (for reader notes glossary source).",
    ),
    glossary_path: Path | None = typer.Option(
        None,
        "--glossary",
        "-g",
        exists=True,
        dir_okay=False,
        help="Path to a book-level glossary.json (for reader notes). "
        "Mutually exclusive with --series.",
    ),
    config_path: Path = typer.Option(
        DEFAULT_CONFIG,
        "--config",
        "-c",
        help="YAML config file.",
    ),
    work_dir: Path = typer.Option(
        DEFAULT_WORK_DIR,
        "--work",
        "-w",
        help="Base directory for artifacts (used for series lookup).",
    ),
) -> None:
    """Assemble an EPUB from cached translations using waterfall or --from.

    With --from STAGE: strict mode. All chunks must have the specified stage
    in cache, otherwise the command errors out. Use this for clean A/B
    comparison between passes.

    Without --from: waterfall mode. Each chunk uses the latest available
    stage (verify > style > proofread > reflect > translate), respecting
    any per-chunk preferences set via 'btrans prefer'.

    With --notes: inject reader footnotes from the glossary into the
    assembled EPUB. Provide glossary via --series or --glossary.
    """
    # Fail fast if user explicitly provided a non-default config that doesn't exist
    if config_path != DEFAULT_CONFIG and not config_path.exists():
        console.print(
            f"[red]Config file not found:[/red] {config_path}\n"
            f"[dim]Check the path or omit --config to use defaults.[/dim]"
        )
        raise typer.Exit(code=1)

    if series and glossary_path:
        console.print(
            "[red]--series and --glossary are mutually exclusive.[/red]\n"
            "[dim]Use --series for books in a curated series, "
            "--glossary for a standalone book.[/dim]"
        )
        raise typer.Exit(code=1)

    cfg = load_config(config_path if config_path.exists() else None)

    cache_path = work_path / "cache.sqlite"
    if not cache_path.exists():
        console.print(f"[red]No cache.sqlite in {work_path}[/red]")
        raise typer.Exit(code=1)

    if from_stage and from_stage not in STAGE_WATERFALL:
        console.print(
            f"[red]Invalid stage {from_stage!r}. Valid: {', '.join(STAGE_WATERFALL)}[/red]"
        )
        raise typer.Exit(code=1)

    if not epub:
        console.print(
            "[red]--epub is required for assembly.[/red]\n"
            "[dim]Provide the source EPUB so structure can be preserved.[/dim]"
        )
        raise typer.Exit(code=1)

    cache = Cache(cache_path)
    try:
        # Total unique chunks
        total_chunks = cache.count_chunks_total("translate")
        if total_chunks == 0:
            console.print("[red]No translated chunks in cache.[/red]")
            raise typer.Exit(code=1)

        translate_ids = cache.get_all_chunk_ids_for_stage("translate")
        legacy_count = cache.count_legacy_rows("translate")

        if from_stage:
            # Strict mode: all tracked chunks must have the requested stage
            stage_ids = set(cache.get_all_chunk_ids_for_stage(from_stage))
            tracked_missing = set(translate_ids) - stage_ids

            console.print(
                f"[bold]Assembling from stage:[/bold] {from_stage} (strict)\n"
                f"[bold]Tracked chunks:[/bold] {len(stage_ids)}/{len(translate_ids)}"
            )

            if tracked_missing:
                console.print(
                    f"\n[red]Error:[/red] {len(tracked_missing)} chunk(s) do not "
                    f"have stage {from_stage!r} in cache:\n"
                    f"[dim]  {', '.join(sorted(tracked_missing)[:10])}"
                    f"{'...' if len(tracked_missing) > 10 else ''}[/dim]\n"
                    f"\n[dim]Run the {from_stage} pass first, or use waterfall "
                    f"mode (omit --from) for mixed assembly.[/dim]"
                )
                raise typer.Exit(code=1)

            if legacy_count > 0:
                console.print(
                    f"[yellow]Warning:[/yellow] {legacy_count} legacy chunk(s) "
                    f"without chunk_id cannot be verified for stage {from_stage!r}.\n"
                    f"[dim]Re-run translation to populate chunk_ids.[/dim]"
                )
        else:
            # Waterfall mode: show assembly map
            if translate_ids:
                resolved = cache.resolve_stages_bulk(translate_ids)
                stage_counts: dict[str, int] = {}
                for stage in resolved.values():
                    stage_counts[stage] = stage_counts.get(stage, 0) + 1
                if legacy_count > 0:
                    stage_counts["translate"] = stage_counts.get("translate", 0) + legacy_count
                map_str = "  |  ".join(f"{s}: {c}" for s, c in sorted(stage_counts.items()))
                console.print(f"[bold]Assembly map:[/bold] {map_str}")
            else:
                console.print(
                    f"[bold]Chunks:[/bold] {total_chunks} "
                    f"[dim](legacy data — will use translate stage)[/dim]"
                )

        # --- Full EPUB assembly ---
        console.print(f"\n[dim]Source EPUB: {epub}[/dim]")

        # Read the structured book and chunk it.
        # Use chunker params from cache metadata (set during translation)
        # to ensure chunk IDs match what's in cache.
        book = read_book_structured(epub)

        cached_chunker = cache.get_meta("chunker_params")
        if cached_chunker:
            chunk_target_words = cached_chunker.get("target_words", cfg.chunker.target_words)
            chunk_overlap = cached_chunker.get("overlap_paragraphs", cfg.chunker.overlap_paragraphs)
            if (
                chunk_target_words != cfg.chunker.target_words
                or chunk_overlap != cfg.chunker.overlap_paragraphs
            ):
                console.print(
                    f"[dim]Using chunker params from cache: "
                    f"target_words={chunk_target_words}, "
                    f"overlap={chunk_overlap} "
                    f"(differs from config)[/dim]"
                )
        else:
            chunk_target_words = cfg.chunker.target_words
            chunk_overlap = cfg.chunker.overlap_paragraphs

        chunk_set = chunk_book(
            book,
            target_words=chunk_target_words,
            overlap_paragraphs=chunk_overlap,
        )

        # Rehydrate trees from cache (waterfall or forced stage).
        # In assemble flow the book is loaded fresh from EPUB, so ALL chunks
        # need rehydration — including those resolved to 'translate'.
        rehydrated, failed_ids = rehydrate_book_from_waterfall(
            cache,
            chunk_set,
            forced_stage=from_stage,
            rehydrate_all=True,
        )

        total_expected = len(chunk_set.chunks)
        console.print(f"[dim]Rehydrated {rehydrated}/{total_expected} chunks from cache.[/dim]")

        # Check for incomplete rehydration
        if failed_ids:
            if from_stage:
                console.print(
                    f"\n[red]Error:[/red] {len(failed_ids)} chunk(s) could not be "
                    f"rehydrated from stage {from_stage!r}:\n"
                    f"[dim]  {', '.join(sorted(failed_ids)[:10])}"
                    f"{'...' if len(failed_ids) > 10 else ''}[/dim]\n"
                    f"\n[dim]The output EPUB would contain untranslated content. "
                    f"Aborting.[/dim]"
                )
            else:
                console.print(
                    f"\n[red]Error:[/red] {len(failed_ids)} chunk(s) have no "
                    f"translated content in cache:\n"
                    f"[dim]  {', '.join(sorted(failed_ids)[:10])}"
                    f"{'...' if len(failed_ids) > 10 else ''}[/dim]\n"
                    f"\n[dim]Run 'btrans translate' first to populate the cache, "
                    f"or check that chunker settings match.[/dim]"
                )
            raise typer.Exit(code=1)

        # --- Reader notes injection ---
        from .reader_notes import GlossaryLoadError, inject_notes_for_cli

        # explicit = user passed notes-specific flags; otherwise silent skip on missing glossary
        _assemble_notes_explicit = notes is True or note_types is not None
        try:
            note_stats = inject_notes_for_cli(
                console=console,
                chapters=book.chapters,
                cfg=cfg,
                notes_flag=notes,
                note_types=note_types,
                series=series,
                glossary_path=glossary_path,
                work_dir=work_dir,
                book_title=book.meta.title,
                book_author=book.meta.author,
                preloaded_glossary=None,
                explicit=_assemble_notes_explicit,
            )
        except GlossaryLoadError as e:
            console.print(f"[red]Error loading glossary:[/red] {e}")
            raise typer.Exit(code=1) from e

        # Write the assembled EPUB
        out_path = out or _default_output_path(epub, cfg.target_lang)
        modified = [ch for ch in book.chapters if ch.paragraphs]
        write_translated_epub(
            source_path=epub,
            dest_path=out_path,
            modified_chapters=modified,
            new_language=cfg.target_lang,
        )

        console.print(f"\n[green]Assembled EPUB:[/green] {out_path}")
        if note_stats and note_stats.notes_injected > 0:
            console.print(f"[dim]  Reader notes: {note_stats.notes_injected} footnotes[/dim]")
    finally:
        cache.close()


# ---------------------------------------------------------------------------
# cover extract
# ---------------------------------------------------------------------------


@cover_app.command("extract")
def cover_extract_cmd(
    epub: Path = typer.Argument(..., exists=True, dir_okay=False, help="Source EPUB."),
    out: Path | None = typer.Option(
        None,
        "--out",
        "-o",
        help="Output image path (default: <epub-stem>-cover.<ext>).",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Overwrite output file if it already exists.",
    ),
) -> None:
    """Extract the cover image from an EPUB to a file.

    Saves the cover image in its original format (JPEG, PNG, etc.).

    Examples:
        btrans cover extract book.epub
        btrans cover extract book.epub --out my-cover.jpg
        btrans cover extract book.epub --out my-cover.jpg --force
    """
    from .cover import find_cover_in_epub

    cover = find_cover_in_epub(epub)
    if cover is None:
        console.print("[red]No cover image found in this EPUB.[/red]")
        raise typer.Exit(code=1)

    # Determine output path
    if out is None:
        ext = _mime_to_extension(cover.media_type)
        out = epub.with_stem(f"{epub.stem}-cover").with_suffix(ext)

    # Validate output path
    if out.is_dir():
        console.print(
            f"[red]--out path is a directory: {out}[/red]\n"
            f"[dim]Provide a file path, not a directory.[/dim]"
        )
        raise typer.Exit(code=2)

    if out.exists() and not force:
        console.print(
            f"[red]Output file already exists: {out}[/red]\n[dim]Use --force to overwrite.[/dim]"
        )
        raise typer.Exit(code=1)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(cover.raw_bytes)

    console.print(
        f"[green]Cover extracted.[/green]\n"
        f"  Source: {cover.archive_path} ({cover.media_type})\n"
        f"  Size: {len(cover.raw_bytes):,} bytes\n"
        f"  Saved to: {out}"
    )


def _mime_to_extension(mime: str | None) -> str:
    """Convert image MIME type to file extension.

    Returns a safe default (.jpg) for None, empty, or unknown MIME types.
    Handles MIME strings with parameters (e.g. "image/png; charset=binary").
    """
    if not mime:
        return ".jpg"
    # Strip parameters and normalize
    normalized = mime.split(";", 1)[0].strip().lower()
    _MAP = {
        "image/jpeg": ".jpg",
        "image/jpg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
        "image/gif": ".gif",
        "image/bmp": ".bmp",
        "image/tiff": ".tiff",
        "image/svg+xml": ".svg",
    }
    return _MAP.get(normalized, ".jpg")


# ---------------------------------------------------------------------------
# cover replace
# ---------------------------------------------------------------------------


@cover_app.command("replace")
def cover_replace(
    epub: Path = typer.Argument(..., exists=True, dir_okay=False, help="Source EPUB."),
    image: Path = typer.Option(
        ...,
        "--image",
        "-i",
        exists=True,
        dir_okay=False,
        help="Path to the new cover image (jpg/png/webp).",
    ),
    out: Path | None = typer.Option(
        None,
        "--out",
        "-o",
        help="Output EPUB path (default: <source>-cover.epub).",
    ),
) -> None:
    """Replace the cover image in an EPUB with a custom image file."""
    from .cover import find_cover_in_epub, replace_cover_from_file

    cover = find_cover_in_epub(epub)
    if cover is None:
        console.print("[red]No cover image found in this EPUB.[/red]")
        raise typer.Exit(code=1)

    console.print(
        f"[bold]Found cover:[/bold] {cover.archive_path} "
        f"({cover.media_type}, {len(cover.raw_bytes):,} bytes)"
    )

    dest = out or epub.with_stem(f"{epub.stem}-cover")
    replace_cover_from_file(epub, dest, image)

    console.print(f"[green]Cover replaced.[/green] Output: {dest}")


# ---------------------------------------------------------------------------
# cover translate
# ---------------------------------------------------------------------------


@cover_app.command("translate")
def cover_translate_cmd(
    epub: Path = typer.Argument(..., exists=True, dir_okay=False, help="Source EPUB."),
    title: str | None = typer.Option(
        None,
        "--title",
        "-t",
        help="Translated book title (optional — if omitted, the model translates all text).",
    ),
    author: str | None = typer.Option(
        None,
        "--author",
        "-a",
        help="Author name for the cover (default: keep original).",
    ),
    model: str | None = typer.Option(
        None,
        "--model",
        "-m",
        help="Override the cover model (default from config).",
    ),
    config_path: Path = typer.Option(
        DEFAULT_CONFIG,
        "--config",
        "-c",
        help="YAML config file.",
    ),
    out: Path | None = typer.Option(
        None,
        "--out",
        "-o",
        help="Output EPUB path (default: <source>-cover-translated.epub).",
    ),
    target_lang: str = typer.Option(
        ...,
        "--target-lang",
        help="Target language name for the prompt (e.g. Russian, German).",
    ),
    aspect_ratio: str = typer.Option(
        "2:3",
        "--aspect-ratio",
        help="Output aspect ratio (default: 2:3 for book covers).",
    ),
    image_size: str = typer.Option(
        "1K",
        "--image-size",
        help="Output resolution: 1K, 2K, or 4K.",
    ),
) -> None:
    """Translate the cover image text using an AI image model.

    Extracts the cover from the EPUB, sends it to an image generation model
    (via OpenRouter) with a prompt to replace English text with the target
    language translation, then writes the result into a new EPUB.

    If --title is provided, the model uses that exact translation for the
    book title. If omitted, the model translates all visible text on the
    cover automatically (useful when you don't have a specific translation).

    Examples:
        btrans cover translate book.epub --target-lang Russian
        btrans cover translate book.epub --target-lang Russian --title "Реки Лондона"
        btrans cover translate book.epub --target-lang Russian \\
            --title "Реки Лондона" --author "Бен Ааронович"
        btrans cover translate book.epub --target-lang Russian --model openai/gpt-5.4-image-2
    """
    from .cover import find_cover_in_epub, translate_cover

    # Fail fast if user explicitly provided a config path that doesn't exist.
    # The default (configs/default.yaml) may not exist in all setups — that's
    # fine, we fall back to built-in defaults. But an explicit path is an error.
    if config_path != DEFAULT_CONFIG and not config_path.exists():
        console.print(
            f"[red]Config file not found: {config_path}[/red]\n"
            f"[dim]Remove --config to use built-in defaults, or fix the path.[/dim]"
        )
        raise typer.Exit(code=1)
    cfg = load_config(config_path if config_path.exists() else None)

    cover = find_cover_in_epub(epub)
    if cover is None:
        console.print("[red]No cover image found in this EPUB.[/red]")
        raise typer.Exit(code=1)

    console.print(
        f"[bold]Found cover:[/bold] {cover.archive_path} "
        f"({cover.media_type}, {len(cover.raw_bytes):,} bytes)"
    )

    chosen_model = model or cfg.models.cover

    console.print(
        f"[bold]Model:[/bold]   {chosen_model}\n"
        f"[bold]Title:[/bold]   {title or '(auto-translate by model)'}\n"
        f"[bold]Author:[/bold]  {author or '(keep original)'}\n"
        f"[bold]Language:[/bold] {target_lang}\n"
        f"[bold]Size:[/bold]    {image_size} @ {aspect_ratio}"
    )
    console.print("[dim]Generating translated cover...[/dim]")

    provider = create_image_provider(cfg)
    dest = out or epub.with_stem(f"{epub.stem}-cover-translated")

    try:
        result = translate_cover(
            source_epub=epub,
            dest_epub=dest,
            provider=provider,
            model=chosen_model,
            title_translation=title,
            author_name=author,
            target_lang=target_lang,
            aspect_ratio=aspect_ratio,
            image_size=image_size,
        )
    except Exception as e:
        console.print(f"[red]Cover translation failed:[/red] {e}")
        raise typer.Exit(code=1) from e

    console.print(
        f"[green]Cover translated successfully.[/green]\n"
        f"  Output: {dest}\n"
        f"  Model used: {result.model}\n"
        f"  Image: {result.mime_type}, {len(result.image_bytes):,} bytes"
    )
    if result.text:
        console.print(f"  [dim]Model note: {result.text[:200]}[/dim]")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _resolve_cache_path(work_base: Path, book_slug: str | None) -> Path | None:
    """Find the cache.sqlite for a book, auto-detecting if only one exists."""
    if not work_base.exists() or not work_base.is_dir():
        console.print(f"[red]Work directory not found: {work_base}[/red]")
        return None

    if book_slug:
        path = work_base / book_slug / "cache.sqlite"
        if not path.exists():
            console.print(f"[red]No cache at {path}[/red]")
            return None
        return path

    # Auto-detect: look for subdirs with cache.sqlite
    candidates = [
        d / "cache.sqlite"
        for d in work_base.iterdir()
        if d.is_dir() and (d / "cache.sqlite").exists()
    ]
    if len(candidates) == 1:
        return candidates[0]
    elif len(candidates) == 0:
        console.print(f"[red]No cache.sqlite found in {work_base}/*/[/red]")
        return None
    else:
        names = [c.parent.name for c in candidates]
        console.print(
            f"[red]Multiple books found: {', '.join(names)}[/red]\n"
            f"[dim]Use --book SLUG to specify which one.[/dim]"
        )
        return None


if __name__ == "__main__":
    app()
