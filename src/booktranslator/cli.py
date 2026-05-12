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
from pathlib import Path

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

from .cache import Cache, STAGE_WATERFALL
from .chunker import chunk_book
from .config import load_config
from .epub_io import read_book, read_book_structured, write_translated_epub
from .glossary import extract_glossary, save_glossary
from .models import Glossary, SeriesGlossary, SeriesGlossaryEntry, Stage
from .pipeline_helpers import (
    ChunkerConfigMismatchError,
    build_reflect_input,
    collect_chunk_originals,
    collect_stage_translations,
    collect_waterfall_paragraphs,
    create_provider,
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
        None, "--series", "-s",
        help="Series slug: load known terms from its curated glossary.",
    ),
    config_path: Path = typer.Option(
        DEFAULT_CONFIG, "--config", "-c", help="YAML config file.",
    ),
    work_dir: Path = typer.Option(
        DEFAULT_WORK_DIR, "--work", "-w", help="Base directory for artifacts.",
    ),
    model: str | None = typer.Option(
        None, "--model", "-m",
        help="Override the glossary model (e.g. anthropic/claude-sonnet-4.6).",
    ),
    force: bool = typer.Option(
        False, "--force", help="Ignore cache and call the LLM again.",
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
            f"[bold]Series:[/bold]  {series} "
            f"({len(series_glossary.entries)} known terms)"
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

    provider = create_provider()
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
        console.print(
            f"[dim]Tokens: {result.input_tokens} in, "
            f"{result.output_tokens} out[/dim]"
        )
    else:
        console.print("[dim]Served from cache (no tokens used)[/dim]")


# ---------------------------------------------------------------------------
# glossary promote
# ---------------------------------------------------------------------------


@glossary_app.command("promote")
def glossary_promote(
    book_work_dir: Path = typer.Argument(
        ..., exists=True, file_okay=False,
        help="Book workdir (e.g. work/rivers-of-london).",
    ),
    series: str = typer.Option(
        ..., "--series", "-s",
        help="Series slug to promote into.",
    ),
    work_root: Path = typer.Option(
        DEFAULT_WORK_DIR, "--work", "-w",
        help="Base directory for artifacts.",
    ),
    all_entries: bool = typer.Option(
        False, "--all",
        help="Promote ALL entries, not only approved_by_human ones.",
    ),
) -> None:
    """Merge approved book-glossary entries into the series glossary."""
    glossary_path = book_work_dir / "glossary.json"
    if not glossary_path.is_file():
        console.print(f"[red]No glossary.json in {book_work_dir}[/red]")
        raise typer.Exit(code=1)
    book_glossary = Glossary.model_validate(
        json.loads(glossary_path.read_text(encoding="utf-8"))
    )

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
        series_glossary, book_glossary, require_approved=not all_entries,
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
            "\n[yellow]Conflicts[/yellow] "
            "(series value kept, book value shown for review):"
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
        DEFAULT_WORK_DIR, "--work", "-w",
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
        None, "--series", "-s",
        help="Series slug: use its curated glossary for consistency.",
    ),
    glossary_path: Path | None = typer.Option(
        None, "--glossary", "-g",
        exists=True, dir_okay=False,
        help="Path to a single-book glossary.json. "
             "Mutually exclusive with --series.",
    ),
    config_path: Path = typer.Option(
        DEFAULT_CONFIG, "--config", "-c", help="YAML config file.",
    ),
    work_dir: Path = typer.Option(
        DEFAULT_WORK_DIR, "--work", "-w", help="Base directory for artifacts.",
    ),
    model: str | None = typer.Option(
        None, "--model", "-m",
        help="Override the translate model.",
    ),
    out: Path | None = typer.Option(
        None, "--out", "-o",
        help="Output EPUB path (default: <source>-ru.epub).",
    ),
    limit_chunks: int | None = typer.Option(
        None, "--limit-chunks",
        help="Translate only the first N chunks (for quick tests).",
    ),
    parallelism: int | None = typer.Option(
        None, "--parallelism", "-j",
        help="Number of chunks to translate concurrently "
             "(overrides config.translate.parallelism).",
    ),
    no_judge: bool = typer.Option(
        False, "--no-judge",
        help="Skip judge + reflect passes entirely.",
    ),
    no_reflect: bool = typer.Option(
        False, "--no-reflect",
        help="Run judge (for stats) but skip reflect.",
    ),
    reflect_threshold: int | None = typer.Option(
        None, "--reflect-threshold",
        help="Override reflection.trigger_score (reflect chunks scoring ≤ N).",
    ),
    reflect_all: bool = typer.Option(
        False, "--reflect-all",
        help="Reflect all chunks regardless of score.",
    ),
    no_proofread: bool = typer.Option(
        False, "--no-proofread",
        help="Skip the proofread pass.",
    ),
    no_style: bool = typer.Option(
        False, "--no-style",
        help="Skip the style pass.",
    ),
    no_verify: bool = typer.Option(
        False, "--no-verify",
        help="Skip the verify pass.",
    ),
) -> None:
    """Translate an EPUB into the target language."""
    cfg = load_config(config_path if config_path.exists() else None)

    if reflect_threshold is not None and not 1 <= reflect_threshold <= 5:
        console.print(
            "[red]--reflect-threshold must be between 1 and 5.[/red]"
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

    glossary = None
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
        console.print(
            f"[bold]Series:[/bold]   {series} ({len(glossary.entries)} terms)"
        )
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
            "[yellow]No --series or --glossary given; "
            "translating without a glossary.[/yellow]"
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

    effective_parallelism = parallelism if parallelism is not None else cfg.translate.parallelism
    if effective_parallelism > 1:
        console.print(
            f"[bold]Parallelism:[/bold] {effective_parallelism} chunks"
        )

    cache = Cache(wd.cache_path)
    # Persist chunker params so judge/reflect can verify consistency.
    # Fails fast if existing params differ (prevents mixed-cache state).
    try:
        save_chunker_params(
            cache, cfg.chunker.target_words, cfg.chunker.overlap_paragraphs
        )
    except ChunkerConfigMismatchError as e:
        console.print(f"[red]Error:[/red] {e}")
        cache.close()
        raise typer.Exit(code=1) from e
    provider = create_provider()
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
            chunk_originals = collect_chunk_originals(
                chunk_set, translate_ids
            )
            chunk_translations = collect_stage_translations(
                cache, translate_ids, stage="translate"
            )

            # Only judge chunks that have both original and translation
            judgeable_ids = (
                set(chunk_originals.keys()) & set(chunk_translations.keys())
            )
            if judgeable_ids:
                judge_model = cfg.models.judge
                console.print(
                    f"\n[dim]Judging {len(judgeable_ids)} chunks "
                    f"with {judge_model}...[/dim]\n"
                )

                judge = Judge(
                    provider=provider,
                    prompt_path=JUDGE_PROMPT,
                    cache=cache,
                    glossary=glossary,
                    model=judge_model,
                    source_lang=cfg.source_lang,
                    target_lang=cfg.target_lang,
                )

                judge_parallelism = (
                    min(parallelism or cfg.translate.parallelism, 8) or 4
                )

                with Progress(
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    MofNCompleteColumn(),
                    TimeElapsedColumn(),
                    console=console,
                ) as progress:
                    jtask = progress.add_task(
                        "judging", total=len(judgeable_ids)
                    )

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
                        {k: v for k, v in chunk_originals.items()
                         if k in judgeable_ids},
                        {k: v for k, v in chunk_translations.items()
                         if k in judgeable_ids},
                        on_progress=on_judge_progress,
                        parallelism=judge_parallelism,
                    )

                # Print score distribution
                if judge_stats.results:
                    dist: dict[int, int] = {}
                    for r in judge_stats.results:
                        dist[r.score] = dist.get(r.score, 0) + 1
                    dist_str = "  ".join(
                        f"★{k}: {v}"
                        for k, v in sorted(dist.items(), reverse=True)
                    )
                    console.print(
                        f"[bold]Judge scores:[/bold] {dist_str}"
                    )

                    threshold = (
                        reflect_threshold
                        if reflect_threshold is not None
                        else cfg.reflection.trigger_score
                    )
                    needs_reflect = [
                        r for r in judge_stats.results
                        if r.score <= threshold
                    ]
                    console.print(
                        f"  Chunks needing reflection: "
                        f"{len(needs_reflect)} (score ≤ {threshold})"
                    )

    # ------------------------------------------------------------------
    # Reflect pass
    # ------------------------------------------------------------------
    reflect_stats = None
    if not no_judge and not no_reflect and judge_stats and judge_stats.results:
        from .reflect import Reflector

        REFLECT_PROMPT = Path("prompts/reflect.md")

        threshold = (
            reflect_threshold
            if reflect_threshold is not None
            else cfg.reflection.trigger_score
        )

        if reflect_all:
            chunks_to_reflect_results = judge_stats.results
        else:
            chunks_to_reflect_results = [
                r for r in judge_stats.results if r.score <= threshold
            ]

        if chunks_to_reflect_results:
            reflect_model = cfg.models.reflect
            console.print(
                f"\n[dim]Reflecting on "
                f"{len(chunks_to_reflect_results)} chunks "
                f"with {reflect_model}...[/dim]\n"
            )

            reflector = Reflector(
                provider=provider,
                reflect_prompt_path=REFLECT_PROMPT,
                translate_prompt_path=TRANSLATE_PROMPT,
                cache=cache,
                glossary=glossary,
                model=reflect_model,
                source_lang=cfg.source_lang,
                target_lang=cfg.target_lang,
            )

            # Build reflect input using shared helper (O(n), not O(n²))
            judge_by_id = normalize_judge_map([
                {"chunk_id": r.chunk_id, "score": r.score, "issues": r.issues}
                for r in chunks_to_reflect_results
            ])
            chunks_to_reflect_data = build_reflect_input(
                chunk_set, cache,
                {r.chunk_id for r in chunks_to_reflect_results},
                judge_by_id,
            )

            if chunks_to_reflect_data:
                reflect_parallelism = (
                    min(parallelism or cfg.translate.parallelism, 4) or 2
                )

                with Progress(
                    TextColumn(
                        "[progress.description]{task.description}"
                    ),
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
    paragraph_counts = {
        chunk.id: len(chunk.paragraph_indexes)
        for chunk in chunk_set.chunks
    }

    # Current chunk IDs from the chunk_set — anchors all postprocess
    # operations to the current book/chunker config, preventing stale
    # cache entries from being processed.
    current_chunk_ids = [chunk.id for chunk in chunk_set.chunks]

    # Intersect with what's actually in cache (translated)
    cached_translate_ids = set(cache.get_all_chunk_ids_for_stage("translate"))
    valid_chunk_ids = [
        cid for cid in current_chunk_ids if cid in cached_translate_ids
    ]

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
                chunk_set, list(waterfall_paragraphs.keys())
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
            f"\n[dim]Running {pp_stage} on {len(chunks_data_pp)} chunks "
            f"with {pp_model}...[/dim]\n"
        )

        processor = PostProcessor(
            provider=provider,
            prompt_path=pp_prompt_path,
            cache=cache,
            glossary=glossary,
            stage=pp_stage,
            model=pp_model,
            source_lang=cfg.source_lang,
            target_lang=cfg.target_lang,
        )

        pp_parallelism = min(
            parallelism or cfg.translate.parallelism, 8
        ) or 4

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
        rehydrated = rehydrate_book_from_waterfall(cache, chunk_set)
        if rehydrated:
            console.print(
                f"[dim]Rehydrated {rehydrated} chunks from post-processing.[/dim]"
            )

    modified = [ch for ch in book.chapters if ch.paragraphs]
    write_translated_epub(
        source_path=epub,
        dest_path=out_path,
        modified_chapters=modified,
        new_language=cfg.target_lang,
    )
    wd.mark_stage(state, Stage.DONE)
    cache.close()

    # Summary
    summary_parts = [
        f"Chunks: {stats.chunks_translated} new, {stats.chunks_cached} cached, "
        f"{stats.chunks_failed} failed."
    ]

    if judge_stats:
        summary_parts.append(
            f"Judge: {judge_stats.chunks_judged} new, "
            f"{judge_stats.chunks_cached} cached."
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
        None, "--series", "-s",
        help="Series slug for glossary context.",
    ),
    glossary_path: Path | None = typer.Option(
        None, "--glossary", "-g",
        exists=True, dir_okay=False,
        help="Path to a single-book glossary.json.",
    ),
    config_path: Path = typer.Option(
        DEFAULT_CONFIG, "--config", "-c", help="YAML config file.",
    ),
    work_dir: Path = typer.Option(
        DEFAULT_WORK_DIR, "--work", "-w", help="Base directory for artifacts.",
    ),
    model: str | None = typer.Option(
        None, "--model", "-m",
        help="Override the judge model.",
    ),
    parallelism: int | None = typer.Option(
        None, "--parallelism", "-j",
        help="Number of chunks to judge concurrently.",
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
        console.print(
            "[red]No translated chunks found. Run translate first.[/red]"
        )
        cache.close()
        raise typer.Exit(code=1)

    # Verify chunker config matches what was used during translate
    try:
        verify_chunker_params(
            cache, cfg.chunker.target_words, cfg.chunker.overlap_paragraphs
        )
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
    chunk_translations = collect_stage_translations(
        cache, translate_ids, stage="translate"
    )

    judgeable_ids = sorted(
        set(chunk_originals.keys()) & set(chunk_translations.keys())
    )
    console.print(f"[bold]Chunks to judge:[/bold] {len(judgeable_ids)}")

    judge_model = model or cfg.models.judge
    console.print(f"[bold]Model:[/bold] {judge_model}")

    JUDGE_PROMPT = Path("prompts/judge.md")
    judge = Judge(
        provider=create_provider(),
        prompt_path=JUDGE_PROMPT,
        cache=cache,
        glossary=glossary,
        model=judge_model,
        source_lang=cfg.source_lang,
        target_lang=cfg.target_lang,
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
        None, "--series", "-s",
        help="Series slug for glossary context.",
    ),
    glossary_path: Path | None = typer.Option(
        None, "--glossary", "-g",
        exists=True, dir_okay=False,
        help="Path to a single-book glossary.json.",
    ),
    config_path: Path = typer.Option(
        DEFAULT_CONFIG, "--config", "-c", help="YAML config file.",
    ),
    work_dir: Path = typer.Option(
        DEFAULT_WORK_DIR, "--work", "-w", help="Base directory for artifacts.",
    ),
    model: str | None = typer.Option(
        None, "--model", "-m",
        help="Override the reflect model.",
    ),
    threshold: int | None = typer.Option(
        None, "--threshold", "-t",
        help="Reflect chunks with judge score ≤ N (default from config).",
    ),
    all_chunks: bool = typer.Option(
        False, "--all",
        help="Reflect all chunks regardless of score.",
    ),
    parallelism: int | None = typer.Option(
        None, "--parallelism", "-j",
        help="Number of chunks to reflect concurrently.",
    ),
) -> None:
    """Run the reflect pass on chunks that scored poorly.

    Requires judge scores in cache. Use 'btrans judge' first if needed.
    """
    from .reflect import Reflector

    cfg = load_config(config_path if config_path.exists() else None)

    if threshold is not None and not 1 <= threshold <= 5:
        console.print(
            "[red]--threshold must be between 1 and 5.[/red]"
        )
        raise typer.Exit(code=1)

    console.print(f"[bold]Reading EPUB:[/bold] {epub}")
    book = read_book_structured(epub)
    console.print(f"[bold]Book:[/bold] {book.meta.title}")

    glossary = _resolve_glossary(series, glossary_path, work_dir, cfg, book)

    wd = WorkDir.for_book(work_dir, book.meta.title)
    cache = Cache(wd.cache_path)

    # Get judge scores
    judge_scores = cache.get_judge_scores()
    if not judge_scores and not all_chunks:
        console.print(
            "[red]No judge scores found. Run 'btrans judge' first, "
            "or use --all to reflect all translated chunks.[/red]"
        )
        cache.close()
        raise typer.Exit(code=1)

    # Determine which chunks to reflect
    trigger = (
        threshold if threshold is not None
        else cfg.reflection.trigger_score
    )

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
        console.print(
            f"[green]All chunks scored above {trigger}. "
            f"Nothing to reflect.[/green]"
        )
        cache.close()
        return

    console.print(
        f"[bold]Chunks to reflect:[/bold] {len(chunks_to_reflect_ids)}"
    )

    # Verify chunker config matches what was used during translate
    try:
        verify_chunker_params(
            cache, cfg.chunker.target_words, cfg.chunker.overlap_paragraphs
        )
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
        console.print(
            "[yellow]No chunks with translations to reflect on.[/yellow]"
        )
        cache.close()
        return

    reflect_model = model or cfg.models.reflect
    console.print(f"[bold]Model:[/bold] {reflect_model}")

    REFLECT_PROMPT = Path("prompts/reflect.md")
    reflector = Reflector(
        provider=create_provider(),
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
        None, "--series", "-s",
        help="Series slug for glossary context.",
    ),
    glossary_path: Path | None = typer.Option(
        None, "--glossary", "-g",
        exists=True, dir_okay=False,
        help="Path to a single-book glossary.json.",
    ),
    config_path: Path = typer.Option(
        DEFAULT_CONFIG, "--config", "-c", help="YAML config file.",
    ),
    work_dir: Path = typer.Option(
        DEFAULT_WORK_DIR, "--work", "-w", help="Base directory for artifacts.",
    ),
    model: str | None = typer.Option(
        None, "--model", "-m",
        help="Override the proofread model.",
    ),
    parallelism: int | None = typer.Option(
        None, "--parallelism", "-j",
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
        None, "--series", "-s",
        help="Series slug for glossary context.",
    ),
    glossary_path: Path | None = typer.Option(
        None, "--glossary", "-g",
        exists=True, dir_okay=False,
        help="Path to a single-book glossary.json.",
    ),
    config_path: Path = typer.Option(
        DEFAULT_CONFIG, "--config", "-c", help="YAML config file.",
    ),
    work_dir: Path = typer.Option(
        DEFAULT_WORK_DIR, "--work", "-w", help="Base directory for artifacts.",
    ),
    model: str | None = typer.Option(
        None, "--model", "-m",
        help="Override the style model.",
    ),
    parallelism: int | None = typer.Option(
        None, "--parallelism", "-j",
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
        None, "--series", "-s",
        help="Series slug for glossary context.",
    ),
    glossary_path: Path | None = typer.Option(
        None, "--glossary", "-g",
        exists=True, dir_okay=False,
        help="Path to a single-book glossary.json.",
    ),
    config_path: Path = typer.Option(
        DEFAULT_CONFIG, "--config", "-c", help="YAML config file.",
    ),
    work_dir: Path = typer.Option(
        DEFAULT_WORK_DIR, "--work", "-w", help="Base directory for artifacts.",
    ),
    model: str | None = typer.Option(
        None, "--model", "-m",
        help="Override the verify model.",
    ),
    parallelism: int | None = typer.Option(
        None, "--parallelism", "-j",
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
            console.print(
                "[red]No translated chunks found. Run translate first.[/red]"
            )
            raise typer.Exit(code=1)

        # Verify chunker config
        try:
            verify_chunker_params(
                cache, cfg.chunker.target_words, cfg.chunker.overlap_paragraphs
            )
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
        paragraph_counts = {
            chunk.id: len(chunk.paragraph_indexes)
            for chunk in chunk_set.chunks
        }

        # Scope to current chunk_set IDs intersected with cache
        current_chunk_ids = [chunk.id for chunk in chunk_set.chunks]
        translate_id_set = set(translate_ids)
        valid_chunk_ids = [
            cid for cid in current_chunk_ids if cid in translate_id_set
        ]

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

        console.print(
            f"[bold]Stage:[/bold] {stage}\n"
            f"[bold]Chunks:[/bold] {len(chunks_data)}"
        )

        # Resolve model
        model_map = {
            "proofread": cfg.models.proofread,
            "style": cfg.models.style,
            "verify": cfg.models.verify,
        }
        chosen_model = model_override or model_map.get(stage, cfg.models.translate)
        console.print(f"[bold]Model:[/bold] {chosen_model}")

        processor = PostProcessor(
            provider=create_provider(),
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
        ..., exists=True, file_okay=False,
        help="Book workdir (e.g. work/broken-homes).",
    ),
    scores: bool = typer.Option(
        False, "--scores", help="Show judge scores table.",
    ),
    below: int | None = typer.Option(
        None, "--below", help="Only show chunks with score below N.",
    ),
    diff: str | None = typer.Option(
        None, "--diff", help="Show all passes for a specific chunk_id.",
    ),
    stages_filter: str | None = typer.Option(
        None, "--stages",
        help="Comma-separated stages to compare (use with --diff).",
    ),
    assembly_map: bool = typer.Option(
        False, "--assembly-map",
        help="Show which stage each chunk will use at assembly.",
    ),
    grid: bool = typer.Option(
        False, "--grid",
        help="Show chunk × stage matrix with changed/unchanged status.",
    ),
) -> None:
    """Inspect the translation state of a book."""
    from .status import (
        build_status_report,
        get_chunk_diff,
        get_scores,
    )

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
        console.print(f"[bold cyan]─── {first['stage']} ───[/bold cyan] "
                      f"[dim]({first['model']}, {first['created_at']})[/dim]")
        console.print(first["content"])
        console.print()

        for i in range(1, len(waterfall_entries)):
            prev = waterfall_entries[i - 1]
            curr = waterfall_entries[i]

            prev_lines = prev["content"].splitlines(keepends=True)
            curr_lines = curr["content"].splitlines(keepends=True)

            diff_lines = list(difflib.unified_diff(
                prev_lines, curr_lines,
                fromfile=prev["stage"], tofile=curr["stage"],
                lineterm="",
            ))

            console.print(f"[bold cyan]─── {curr['stage']} ───[/bold cyan] "
                          f"[dim]({curr['model']}, {curr['created_at']})[/dim]")

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
        from .status import _GRID_STAGES_WITH_REFLECT, _GRID_STAGES
        stages_order = _GRID_STAGES_WITH_REFLECT if "reflect" in all_stages_present else _GRID_STAGES

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
        console.print(f"\n[dim]Legend: ✓=base  Δ=changed  =[dim]=unchanged  —=not run[/dim]")
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
            f"{stage}: {count}" for stage, count in
            sorted(report.assembly_map.items(), key=lambda x: STAGE_WATERFALL.index(x[0]) if x[0] in STAGE_WATERFALL else 99)
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
        None, "--reason", "-r", help="Why this preference.",
    ),
    reset: bool = typer.Option(
        False, "--reset", help="Remove preference, revert to waterfall.",
    ),
    work_path: Path = typer.Option(
        DEFAULT_WORK_DIR, "--work", "-w",
        help="Base work directory.",
    ),
    book: str | None = typer.Option(
        None, "--book", "-b",
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
        console.print(
            f"[red]Invalid stage {stage!r}. "
            f"Valid: {', '.join(STAGE_WATERFALL)}[/red]"
        )
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
        ..., exists=True, file_okay=False,
        help="Book workdir (e.g. work/broken-homes).",
    ),
    from_stage: str | None = typer.Option(
        None, "--from",
        help="Use only this stage for all chunks (ignores waterfall).",
    ),
    out: Path | None = typer.Option(
        None, "--out", "-o", help="Output EPUB path.",
    ),
    epub: Path | None = typer.Option(
        None, "--epub", "-e", exists=True, dir_okay=False,
        help="Source EPUB (for structure). Auto-detected from state if possible.",
    ),
) -> None:
    """Assemble an EPUB from cached translations using waterfall or --from.

    With --from STAGE: strict mode. All chunks must have the specified stage
    in cache, otherwise the command errors out. Use this for clean A/B
    comparison between passes.

    Without --from: waterfall mode. Each chunk uses the latest available
    stage (verify > style > proofread > reflect > translate), respecting
    any per-chunk preferences set via 'btrans prefer'.
    """
    cache_path = work_path / "cache.sqlite"
    if not cache_path.exists():
        console.print(f"[red]No cache.sqlite in {work_path}[/red]")
        raise typer.Exit(code=1)

    if from_stage and from_stage not in STAGE_WATERFALL:
        console.print(
            f"[red]Invalid stage {from_stage!r}. "
            f"Valid: {', '.join(STAGE_WATERFALL)}[/red]"
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

        # For now, print what would happen. Full EPUB assembly requires
        # the source EPUB and the structured book — that integration comes
        # when we wire this into the existing write_translated_epub flow.
        if not epub:
            console.print(
                "\n[yellow]Note:[/yellow] Full EPUB assembly requires --epub "
                "(source EPUB for structure).\n"
                "[dim]This command currently shows the assembly plan. "
                "Full assembly will be wired in iteration 4.[/dim]"
            )
            return

        console.print(
            f"\n[dim]Source EPUB: {epub}[/dim]\n"
            f"[dim]Output: {out or '(default)'}[/dim]\n"
            "[yellow]Full assembly integration pending (iteration 4).[/yellow]"
        )
    finally:
        cache.close()


# ---------------------------------------------------------------------------
# cover replace
# ---------------------------------------------------------------------------


@cover_app.command("replace")
def cover_replace(
    epub: Path = typer.Argument(..., exists=True, dir_okay=False, help="Source EPUB."),
    image: Path = typer.Option(
        ..., "--image", "-i", exists=True, dir_okay=False,
        help="Path to the new cover image (jpg/png/webp).",
    ),
    out: Path | None = typer.Option(
        None, "--out", "-o",
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
        None, "--title", "-t",
        help="Translated book title (optional — if omitted, the model translates all text automatically).",
    ),
    author: str | None = typer.Option(
        None, "--author", "-a",
        help="Author name for the cover (default: keep original).",
    ),
    model: str | None = typer.Option(
        None, "--model", "-m",
        help="Override the cover model (default from config).",
    ),
    config_path: Path = typer.Option(
        DEFAULT_CONFIG, "--config", "-c", help="YAML config file.",
    ),
    out: Path | None = typer.Option(
        None, "--out", "-o",
        help="Output EPUB path (default: <source>-cover-translated.epub).",
    ),
    target_lang: str = typer.Option(
        ..., "--target-lang",
        help="Target language name for the prompt (e.g. Russian, German).",
    ),
    aspect_ratio: str = typer.Option(
        "2:3", "--aspect-ratio",
        help="Output aspect ratio (default: 2:3 for book covers).",
    ),
    image_size: str = typer.Option(
        "1K", "--image-size",
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
        btrans cover translate book.epub --target-lang Russian --title "Реки Лондона" --author "Бен Ааронович"
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

    provider = create_provider()
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
