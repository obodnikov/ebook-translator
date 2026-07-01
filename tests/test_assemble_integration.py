"""Integration tests for the btrans assemble command.

Covers:
- Full EPUB assembly from cache (waterfall and --from stage)
- --notes with/without --series
- --config validation (missing file)
- Reader notes injection in assembled output
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest
from typer.testing import CliRunner

from booktranslator.cache import Cache
from booktranslator.cli import app
from booktranslator.models import SeriesGlossary, SeriesGlossaryEntry

runner = CliRunner()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MINIMAL_XHTML = """\
<?xml version="1.0" encoding="utf-8"?>
<html xmlns="http://www.w3.org/1999/xhtml">
<head><title>Test</title></head>
<body>
<p>This is a test paragraph about vestigium.</p>
<p>Another paragraph with DCI Seawoll.</p>
</body>
</html>"""

MINIMAL_OPF = """\
<?xml version="1.0" encoding="utf-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="uid">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:title>Test Book</dc:title>
    <dc:creator>Test Author</dc:creator>
    <dc:language>en</dc:language>
    <dc:identifier id="uid">test-uid-123</dc:identifier>
  </metadata>
  <manifest>
    <item id="ch01" href="ch01.xhtml" media-type="application/xhtml+xml"/>
  </manifest>
  <spine>
    <itemref idref="ch01"/>
  </spine>
</package>"""

MINIMAL_CONTAINER = """\
<?xml version="1.0" encoding="utf-8"?>
<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container" version="1.0">
  <rootfiles>
    <rootfile full-path="content.opf" media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>"""


@pytest.fixture
def minimal_epub(tmp_path: Path) -> Path:
    """Create a minimal valid EPUB file for testing."""
    epub_path = tmp_path / "test-book.epub"
    with zipfile.ZipFile(epub_path, "w") as zf:
        # mimetype must be first and uncompressed
        zf.writestr(
            zipfile.ZipInfo("mimetype"),
            "application/epub+zip",
            compress_type=zipfile.ZIP_STORED,
        )
        zf.writestr("META-INF/container.xml", MINIMAL_CONTAINER)
        zf.writestr("content.opf", MINIMAL_OPF)
        zf.writestr("ch01.xhtml", MINIMAL_XHTML)
    return epub_path


@pytest.fixture
def work_with_cache(tmp_path: Path) -> Path:
    """Create a workdir with a cache containing translated chunks."""
    work_path = tmp_path / "work" / "test-book"
    work_path.mkdir(parents=True)

    cache = Cache(work_path / "cache.sqlite")

    # Store translated content with paragraph markers
    translated = (
        "===PARAGRAPH 1===\n"
        "<p>Это тестовый параграф о вестигиум.</p>\n"
        "===PARAGRAPH 2===\n"
        "<p>Ещё один параграф с DCI Сиуолл.</p>"
    )
    cache.put(
        Cache.make_key("translate", "test-model", "v1", "sys", "user1"),
        "translate",
        "test-model",
        "v1",
        translated,
        input_tokens=100,
        output_tokens=50,
        meta={"chunk_id": "ch01_c01"},
    )

    # Also store a "style" stage version
    styled = (
        "===PARAGRAPH 1===\n"
        "<p>Это тестовый параграф о вестигиуме.</p>\n"
        "===PARAGRAPH 2===\n"
        "<p>Ещё один параграф со старшим инспектором Сиуоллом.</p>"
    )
    cache.put(
        Cache.make_key("style", "test-model", "v1", "sys", "user1"),
        "style",
        "test-model",
        "v1",
        styled,
        input_tokens=100,
        output_tokens=50,
        meta={"chunk_id": "ch01_c01"},
    )

    # Save chunker params
    cache.set_meta("chunker_params", {"target_words": 2000, "overlap_paragraphs": 1})

    cache.close()
    return work_path


@pytest.fixture
def series_glossary(tmp_path: Path) -> Path:
    """Create a series glossary with concept/term entries."""
    series_dir = tmp_path / "work" / "rivers-of-london-series"
    series_dir.mkdir(parents=True)

    glossary = SeriesGlossary(
        series_slug="rivers-of-london",
        title="Rivers of London",
        author="Ben Aaronovitch",
        source_lang="en",
        target_lang="ru",
        entries=[
            SeriesGlossaryEntry(
                original="vestigium",
                translation="вестигиум",
                type="concept",
                notes="Magical trace left by spells or supernatural events.",
                origin_book="rivers-of-london",
            ),
            SeriesGlossaryEntry(
                original="DCI",
                translation="DCI",
                type="term",
                notes="Detective Chief Inspector — senior detective rank.",
                origin_book="rivers-of-london",
            ),
            SeriesGlossaryEntry(
                original="Peter Grant",
                translation="Питер Грант",
                type="person",
                notes="Main character, police constable and wizard apprentice.",
                origin_book="rivers-of-london",
            ),
        ],
    )

    glossary_path = series_dir / "series.glossary.json"
    glossary_path.write_text(glossary.model_dump_json(indent=2), encoding="utf-8")
    return tmp_path / "work"


# ---------------------------------------------------------------------------
# Tests: basic assembly
# ---------------------------------------------------------------------------


class TestAssembleBasic:
    def test_assemble_requires_epub(self, work_with_cache: Path):
        """Assemble without --epub should fail."""
        result = runner.invoke(
            app,
            [
                "assemble",
                str(work_with_cache),
            ],
        )
        assert result.exit_code == 1
        assert "--epub is required" in result.output

    def test_assemble_requires_cache(self, tmp_path: Path, minimal_epub: Path):
        """Assemble with empty workdir (no cache) should fail."""
        empty_work = tmp_path / "empty-work"
        empty_work.mkdir()
        result = runner.invoke(
            app,
            [
                "assemble",
                str(empty_work),
                "--epub",
                str(minimal_epub),
            ],
        )
        assert result.exit_code == 1
        assert "No cache.sqlite" in result.output

    def test_assemble_waterfall_produces_epub(
        self, work_with_cache: Path, minimal_epub: Path, tmp_path: Path
    ):
        """Assemble in waterfall mode produces an EPUB file."""
        out_path = tmp_path / "output.epub"
        result = runner.invoke(
            app,
            [
                "assemble",
                str(work_with_cache),
                "--epub",
                str(minimal_epub),
                "--out",
                str(out_path),
            ],
        )
        assert result.exit_code == 0, result.output
        assert out_path.exists()
        # Verify it's a valid zip
        assert zipfile.is_zipfile(out_path)

    def test_assemble_from_stage_strict(
        self, work_with_cache: Path, minimal_epub: Path, tmp_path: Path
    ):
        """Assemble with --from translate uses only translate stage."""
        out_path = tmp_path / "from-translate.epub"
        result = runner.invoke(
            app,
            [
                "assemble",
                str(work_with_cache),
                "--epub",
                str(minimal_epub),
                "--from",
                "translate",
                "--out",
                str(out_path),
            ],
        )
        assert result.exit_code == 0, result.output
        assert out_path.exists()

    def test_assemble_from_invalid_stage(self, work_with_cache: Path, minimal_epub: Path):
        """Assemble with invalid --from stage should fail."""
        result = runner.invoke(
            app,
            [
                "assemble",
                str(work_with_cache),
                "--epub",
                str(minimal_epub),
                "--from",
                "nonexistent",
            ],
        )
        assert result.exit_code == 1
        assert "Invalid stage" in result.output

    def test_assemble_from_missing_stage(self, work_with_cache: Path, minimal_epub: Path):
        """Assemble with --from for a stage not in cache should fail."""
        result = runner.invoke(
            app,
            [
                "assemble",
                str(work_with_cache),
                "--epub",
                str(minimal_epub),
                "--from",
                "verify",  # not in our test cache
            ],
        )
        assert result.exit_code == 1
        assert "do not have stage" in result.output


# ---------------------------------------------------------------------------
# Tests: config validation
# ---------------------------------------------------------------------------


class TestAssembleConfig:
    def test_nonexistent_explicit_config_fails(self, work_with_cache: Path, minimal_epub: Path):
        """Explicitly provided --config that doesn't exist should fail."""
        result = runner.invoke(
            app,
            [
                "assemble",
                str(work_with_cache),
                "--epub",
                str(minimal_epub),
                "--config",
                "/nonexistent/path/config.yaml",
            ],
        )
        assert result.exit_code == 1
        assert "Config file not found" in result.output

    def test_default_config_missing_uses_defaults(
        self, work_with_cache: Path, minimal_epub: Path, tmp_path: Path
    ):
        """Default config path missing silently uses built-in defaults."""
        out_path = tmp_path / "output.epub"
        # Don't pass --config, let it use the default which may not exist
        # in the test environment — should still work with built-in defaults
        result = runner.invoke(
            app,
            [
                "assemble",
                str(work_with_cache),
                "--epub",
                str(minimal_epub),
                "--out",
                str(out_path),
            ],
        )
        assert result.exit_code == 0, result.output


# ---------------------------------------------------------------------------
# Tests: reader notes
# ---------------------------------------------------------------------------


class TestAssembleReaderNotes:
    def test_notes_without_series_warns(
        self, work_with_cache: Path, minimal_epub: Path, tmp_path: Path
    ):
        """--notes without --series should warn and skip."""
        out_path = tmp_path / "output.epub"
        result = runner.invoke(
            app,
            [
                "assemble",
                str(work_with_cache),
                "--epub",
                str(minimal_epub),
                "--notes",
                "--out",
                str(out_path),
            ],
        )
        assert result.exit_code == 0
        assert "no glossary available" in result.output or "Skipping" in result.output

    def test_notes_with_nonexistent_series_warns(
        self, work_with_cache: Path, minimal_epub: Path, tmp_path: Path
    ):
        """--notes with non-existent series should warn and skip."""
        out_path = tmp_path / "output.epub"
        result = runner.invoke(
            app,
            [
                "assemble",
                str(work_with_cache),
                "--epub",
                str(minimal_epub),
                "--notes",
                "--series",
                "nonexistent",
                "--work",
                str(work_with_cache.parent),
                "--out",
                str(out_path),
            ],
        )
        assert result.exit_code == 0
        assert "not found" in result.output

    def test_notes_injects_footnotes(
        self, work_with_cache: Path, minimal_epub: Path, series_glossary: Path, tmp_path: Path
    ):
        """--notes with valid series injects footnotes into EPUB."""
        out_path = tmp_path / "output.epub"
        result = runner.invoke(
            app,
            [
                "assemble",
                str(work_with_cache),
                "--epub",
                str(minimal_epub),
                "--notes",
                "--series",
                "rivers-of-london",
                "--work",
                str(series_glossary),
                "--out",
                str(out_path),
            ],
        )
        assert result.exit_code == 0, result.output
        assert "footnotes injected" in result.output or "Reader notes:" in result.output

        # Verify the output EPUB contains footnote markup
        with zipfile.ZipFile(out_path, "r") as zf:
            content = zf.read("ch01.xhtml").decode("utf-8")
            assert "noteref" in content or "footnote" in content

    def test_no_notes_flag_skips_injection(
        self, work_with_cache: Path, minimal_epub: Path, series_glossary: Path, tmp_path: Path
    ):
        """--no-notes explicitly disables reader notes."""
        out_path = tmp_path / "output.epub"
        result = runner.invoke(
            app,
            [
                "assemble",
                str(work_with_cache),
                "--epub",
                str(minimal_epub),
                "--no-notes",
                "--series",
                "rivers-of-london",
                "--work",
                str(series_glossary),
                "--out",
                str(out_path),
            ],
        )
        assert result.exit_code == 0, result.output
        # Should NOT have footnotes
        with zipfile.ZipFile(out_path, "r") as zf:
            content = zf.read("ch01.xhtml").decode("utf-8")
            assert "noteref" not in content

    def test_note_types_override(
        self, work_with_cache: Path, minimal_epub: Path, series_glossary: Path, tmp_path: Path
    ):
        """--note-types filters which glossary types get annotated."""
        out_path = tmp_path / "output.epub"
        # Only annotate "concept" — should match "вестигиум" but not "DCI"
        result = runner.invoke(
            app,
            [
                "assemble",
                str(work_with_cache),
                "--epub",
                str(minimal_epub),
                "--notes",
                "--series",
                "rivers-of-london",
                "--work",
                str(series_glossary),
                "--note-types",
                "concept",
                "--out",
                str(out_path),
            ],
        )
        assert result.exit_code == 0, result.output

    def test_notes_with_standalone_glossary(
        self, work_with_cache: Path, minimal_epub: Path, tmp_path: Path
    ):
        """--notes with --glossary (standalone book, no series) works."""
        # Create a book-level glossary.json
        glossary_file = tmp_path / "glossary.json"
        glossary_data = {
            "book": "Test Book",
            "author": "Test Author",
            "source_lang": "en",
            "target_lang": "ru",
            "model": "test",
            "entries": [
                {
                    "original": "vestigium",
                    "translation": "вестигиум",
                    "type": "concept",
                    "notes": "Magical trace left by spells.",
                    "approved_by_human": True,
                },
                {
                    "original": "DCI",
                    "translation": "DCI",
                    "type": "term",
                    "notes": "Detective Chief Inspector.",
                    "approved_by_human": True,
                },
            ],
        }
        glossary_file.write_text(
            json.dumps(glossary_data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        out_path = tmp_path / "standalone-notes.epub"
        result = runner.invoke(
            app,
            [
                "assemble",
                str(work_with_cache),
                "--epub",
                str(minimal_epub),
                "--notes",
                "--glossary",
                str(glossary_file),
                "--out",
                str(out_path),
            ],
        )
        assert result.exit_code == 0, result.output
        assert "footnotes injected" in result.output or "Reader notes:" in result.output

        # Verify footnotes in output
        with zipfile.ZipFile(out_path, "r") as zf:
            content = zf.read("ch01.xhtml").decode("utf-8")
            assert "noteref" in content or "footnote" in content

    def test_series_and_glossary_mutually_exclusive(
        self, work_with_cache: Path, minimal_epub: Path, tmp_path: Path
    ):
        """--series and --glossary together should fail."""
        glossary_file = tmp_path / "glossary.json"
        glossary_file.write_text(
            '{"book":"x","author":"x","source_lang":"en","target_lang":"ru","model":"x","entries":[]}'
        )

        result = runner.invoke(
            app,
            [
                "assemble",
                str(work_with_cache),
                "--epub",
                str(minimal_epub),
                "--notes",
                "--series",
                "rivers-of-london",
                "--glossary",
                str(glossary_file),
            ],
        )
        assert result.exit_code == 1
        assert "mutually exclusive" in result.output

    def test_malformed_glossary_json_fails_cleanly(
        self, work_with_cache: Path, minimal_epub: Path, tmp_path: Path
    ):
        """Malformed glossary JSON should produce a clean error, not a stack trace."""
        bad_file = tmp_path / "bad-glossary.json"
        bad_file.write_text("not valid json {{{", encoding="utf-8")

        result = runner.invoke(
            app,
            [
                "assemble",
                str(work_with_cache),
                "--epub",
                str(minimal_epub),
                "--notes",
                "--glossary",
                str(bad_file),
            ],
        )
        assert result.exit_code == 1
        assert "Error loading glossary" in result.output

    def test_invalid_glossary_schema_fails_cleanly(
        self, work_with_cache: Path, minimal_epub: Path, tmp_path: Path
    ):
        """Glossary with invalid schema should produce a clean error."""
        bad_file = tmp_path / "bad-schema.json"
        bad_file.write_text('{"wrong_field": true}', encoding="utf-8")

        result = runner.invoke(
            app,
            [
                "assemble",
                str(work_with_cache),
                "--epub",
                str(minimal_epub),
                "--notes",
                "--glossary",
                str(bad_file),
            ],
        )
        assert result.exit_code == 1
        assert "Error loading glossary" in result.output
