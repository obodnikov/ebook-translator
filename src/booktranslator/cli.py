"""Command-line entry point. Iteration 1: `btrans glossary` only."""

from __future__ import annotations

from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from .cache import Cache
from .config import load_config
from .epub_io import read_book
from .glossary import extract_glossary, save_glossary
from .models import Stage
from .provider import OpenRouterProvider
from .state import WorkDir

load_dotenv()

app = typer.Typer(
    help="Translate EPUB books through OpenRouter LLMs.",
    no_args_is_help=True,
)
console = Console()


DEFAULT_WORK_DIR = Path("work")
DEFAULT_CONFIG = Path("configs/default.yaml")
GLOSSARY_PROMPT = Path("prompts/glossary_extract.md")


@app.command()
def glossary(
    epub: Path = typer.Argument(..., exists=True, dir_okay=False, help="Source EPUB."),
    config_path: Path = typer.Option(
        DEFAULT_CONFIG, "--config", "-c", help="YAML config file."
    ),
    work_dir: Path = typer.Option(
        DEFAULT_WORK_DIR, "--work", "-w", help="Base directory for artifacts."
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
    cfg = load_config(config_path if config_path.exists() else None)

    console.print(f"[bold]Reading EPUB:[/bold] {epub}")
    book = read_book(epub)

    console.print(
        f"[bold]Book:[/bold]    {book.meta.title}\n"
        f"[bold]Author:[/bold]  {book.meta.author}\n"
        f"[bold]Chapters:[/bold] {book.meta.chapters}\n"
        f"[bold]Words:[/bold]   ~{book.meta.word_count:,}"
    )

    wd = WorkDir.for_book(work_dir, book.meta.title)
    state = wd.load_state()
    state.current_stage = Stage.EXTRACT
    wd.save_state(state)

    console.print(f"[bold]Workdir:[/bold] {wd.root}")

    cache = Cache(wd.cache_path)
    if force:
        # Invalidate existing glossary entries by bumping prompt_version
        # indirectly. Simpler: delete any cached entry for this stage.
        cache.conn.execute("DELETE FROM cache WHERE stage = 'glossary'")
        cache.conn.commit()

    provider = OpenRouterProvider()

    chosen_model = model or cfg.models.glossary
    console.print(f"[bold]Model:[/bold] {chosen_model}")
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
        )
    except Exception as e:
        # Persist raw output if we have any, so the user can debug
        # parsing issues without burning tokens again.
        wd.raw_glossary_path.write_text(str(e), encoding="utf-8")
        console.print(f"[red]Extraction failed:[/red] {e}")
        raise typer.Exit(code=1) from e

    # Save raw response (always, for transparency).
    wd.raw_glossary_path.write_text(raw_text, encoding="utf-8")

    save_glossary(glossary_obj, wd.glossary_path)
    wd.mark_stage(state, Stage.GLOSSARY)

    # --- Report ---
    table = Table(title="Glossary summary", show_header=True)
    table.add_column("Type", style="cyan")
    table.add_column("Count", justify="right")
    by_type: dict[str, int] = {}
    for e in glossary_obj.entries:
        by_type[e.type] = by_type.get(e.type, 0) + 1
    for t in ("person", "place", "concept", "term", "other"):
        if t in by_type:
            table.add_row(t, str(by_type[t]))
    table.add_row("[bold]total[/bold]", f"[bold]{len(glossary_obj.entries)}[/bold]")
    console.print(table)

    if result is not None:
        console.print(
            f"[dim]Tokens: {result.input_tokens} in, "
            f"{result.output_tokens} out[/dim]"
        )
    else:
        console.print("[dim]Served from cache (no tokens used)[/dim]")

    console.print(
        f"\n[green]Glossary written to:[/green] {wd.glossary_path}\n"
        f"[dim]Review it, edit if needed, then set "
        f"[bold]approved_by_human: true[/bold] on checked entries.[/dim]"
    )


if __name__ == "__main__":
    app()
