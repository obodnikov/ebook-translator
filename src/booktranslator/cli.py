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

from .cache import Cache
from .chunker import chunk_book
from .config import load_config
from .epub_io import read_book, read_book_structured, write_translated_epub
from .glossary import extract_glossary, save_glossary
from .models import Glossary, Stage
from .provider import OpenRouterProvider
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
app.add_typer(glossary_app, name="glossary")
app.add_typer(series_app, name="series")

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

    provider = OpenRouterProvider()
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
) -> None:
    """Translate an EPUB into the target language."""
    cfg = load_config(config_path if config_path.exists() else None)

    console.print(f"[bold]Reading EPUB:[/bold] {epub}")
    book = read_book_structured(epub)
    console.print(
        f"[bold]Book:[/bold]     {book.meta.title}\n"
        f"[bold]Author:[/bold]   {book.meta.author}\n"
        f"[bold]Chapters:[/bold] {book.meta.chapters}\n"
        f"[bold]Words:[/bold]    ~{book.meta.word_count:,}"
    )

    glossary = None
    if series:
        swd = SeriesWorkDir.for_series(work_dir, series)
        if not swd.exists():
            console.print(f"[red]Series {series!r} not found.[/red]")
            raise typer.Exit(code=1)
        glossary = load_series_glossary(swd.glossary_path)
        console.print(
            f"[bold]Series:[/bold]   {series} ({len(glossary.entries)} terms)"
        )
    else:
        console.print(
            "[yellow]No --series given; translating without a glossary.[/yellow]"
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

    cache = Cache(wd.cache_path)
    provider = OpenRouterProvider()
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

        stats = translator.translate_book(chunk_set, on_progress=on_progress)

    if stats.chunks_failed:
        console.print(
            f"[yellow]Warning:[/yellow] {stats.chunks_failed} chunks failed; "
            "the output EPUB will have those paragraphs in the original language. "
            "Re-run the same command to retry failed chunks (cache keeps successes)."
        )

    out_path = out or _default_output_path(epub, cfg.target_lang)
    modified = [ch for ch in book.chapters if ch.paragraphs]
    write_translated_epub(
        source_path=epub,
        dest_path=out_path,
        modified_chapters=modified,
        new_language=cfg.target_lang,
    )
    wd.mark_stage(state, Stage.DONE)

    console.print(
        f"\n[green]Translated EPUB:[/green] {out_path}\n"
        f"[dim]Chunks: {stats.chunks_translated} new, {stats.chunks_cached} cached, "
        f"{stats.chunks_failed} failed. "
        f"Tokens: {stats.input_tokens:,} in, {stats.output_tokens:,} out.[/dim]"
    )


def _default_output_path(epub: Path, target_lang: str) -> Path:
    suffix = f"-{target_lang}.epub"
    return epub.with_name(epub.stem + suffix)


if __name__ == "__main__":
    app()
