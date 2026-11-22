"""A workdir translated under other paragraph rules is left as it is.

Cached paragraphs are matched to the book by position. When the list of paragraphs
changes, a chunk can keep its count and change its content: on Shroud ch11_c02 both
lists had 23 paragraphs, and assembling put the translation in the wrong places.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from booktranslator.cache import Cache
from booktranslator.cli import app
from booktranslator.epub_io import PARAGRAPH_RULES
from tests.epub_fixtures import write_minimal_epub

runner = CliRunner()

ROW = (
    "===PARAGRAPH 1===\n<p>Это тестовый абзац про вестигиум.</p>\n"
    "===PARAGRAPH 2===\n<p>Ещё абзац с Сиуоллом.</p>"
)
BOOK_COMMANDS = ["translate", "judge", "reflect", "proofread", "style", "verify", "repair"]


@pytest.fixture
def epub(tmp_path: Path) -> Path:
    return write_minimal_epub(tmp_path / "book.epub")


def _workdir(
    tmp_path: Path,
    *stages: str,
    completed: tuple[str, ...] = ("glossary",),
    current: str = "translate",
    rules: int | None = None,
) -> Path:
    root = tmp_path / "work" / "test-book"
    root.mkdir(parents=True)
    cache = Cache(root / "cache.sqlite")
    for stage in stages:
        cache.put(stage, stage, "m", "1", ROW, meta={"chunk_id": "ch01_c01"})
    cache.close()
    state = {"book_slug": "test-book", "current_stage": current, "completed_stages": completed}
    if rules is not None:
        state["paragraph_rules"] = rules
    (root / "state.json").write_text(json.dumps(state), encoding="utf-8")
    return root


def _files(root: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in root.iterdir()}


def _run(tmp_path: Path, epub: Path, command: str):
    if command == "assemble":
        root = tmp_path / "work" / "test-book"
        args = [command, str(root), "--epub", str(epub), "--out", str(tmp_path / "out.epub")]
    else:
        args = [command, str(epub), "--work", str(tmp_path / "work")]
    if command == "translate":
        args += ["--out", str(tmp_path / "out.epub"), "--no-judge", "--no-notes"]
        args += ["--no-proofread", "--no-style", "--no-verify"]
    provider = MagicMock()
    provider.complete.return_value = MagicMock(
        text=ROW, input_tokens=1, output_tokens=1, finish_reason="stop"
    )
    with patch("booktranslator.cli.create_stage_provider", return_value=provider):
        return runner.invoke(app, args), provider


@pytest.mark.parametrize("command", [*BOOK_COMMANDS, "assemble"])
def test_a_finished_book_of_earlier_rules_is_not_touched(tmp_path, epub, command):
    root = _workdir(tmp_path, "translate", completed=("glossary", "done"), current="done")
    before = _files(root)

    result, provider = _run(tmp_path, epub, command)

    assert result.exit_code == 1
    assert "Nothing was changed" in " ".join(result.output.split())
    assert _files(root) == before
    provider.complete.assert_not_called()
    assert not (tmp_path / "out.epub").exists()


def test_a_glossary_run_since_then_does_not_unlock_it(tmp_path, epub):
    root = _workdir(tmp_path, "translate", completed=("glossary", "done"), current="glossary")
    before = _files(root)

    result, _ = _run(tmp_path, epub, "translate")

    assert result.exit_code == 1
    assert _files(root) == before


def test_later_stage_rows_alone_block_translate(tmp_path, epub):
    """A later row outranks a new translation, and would sit on other paragraphs."""
    root = _workdir(tmp_path, "translate", "proofread")
    before = _files(root)

    result, provider = _run(tmp_path, epub, "translate")

    assert result.exit_code == 1
    assert _files(root) == before
    provider.complete.assert_not_called()


def test_an_unfinished_translation_goes_on_and_gets_the_mark(tmp_path, epub):
    root = _workdir(tmp_path, "translate")

    result, _ = _run(tmp_path, epub, "translate")

    assert result.exit_code == 0, result.output
    state = json.loads((root / "state.json").read_text(encoding="utf-8"))
    assert state["paragraph_rules"] == PARAGRAPH_RULES


def test_an_unfinished_translation_is_not_assembled_before_translate(tmp_path, epub):
    _workdir(tmp_path, "translate")

    result, _ = _run(tmp_path, epub, "assemble")

    assert result.exit_code == 1
    assert "btrans translate" in " ".join(result.output.split())
    assert not (tmp_path / "out.epub").exists()


def test_a_book_of_the_current_rules_assembles(tmp_path, epub):
    _workdir(
        tmp_path, "translate", completed=("glossary", "done"), current="done", rules=PARAGRAPH_RULES
    )

    result, _ = _run(tmp_path, epub, "assemble")

    assert result.exit_code == 0, result.output
    assert (tmp_path / "out.epub").exists()


def test_a_book_with_only_a_glossary_starts_under_the_current_rules(tmp_path, epub):
    root = _workdir(tmp_path, current="glossary")

    result, _ = _run(tmp_path, epub, "translate")

    assert result.exit_code == 0, result.output
    assert (
        json.loads((root / "state.json").read_text("utf-8"))["paragraph_rules"] == PARAGRAPH_RULES
    )


def test_a_failed_run_does_not_mark_an_older_translation_as_current(tmp_path, epub):
    """Its old rows are still the newest for that chunk, and could be assembled."""
    root = _workdir(tmp_path, "translate")
    provider = MagicMock()
    provider.complete.side_effect = RuntimeError("gateway is down")
    args = ["translate", str(epub), "--work", str(tmp_path / "work")]

    with patch("booktranslator.cli.create_stage_provider", return_value=provider):
        result = runner.invoke(app, args)

    assert result.exit_code == 1
    assert json.loads((root / "state.json").read_text("utf-8")).get("paragraph_rules") is None
    assert _run(tmp_path, epub, "assemble")[0].exit_code == 1
