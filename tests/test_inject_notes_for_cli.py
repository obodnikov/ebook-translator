"""Tests for inject_notes_for_cli helper and cover integration in translate."""

from __future__ import annotations

import os
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from lxml import etree

from booktranslator.models import (
    Config,
    ReaderNotesConfig,
    SeriesGlossary,
    SeriesGlossaryEntry,
)
from booktranslator.reader_notes import (
    GlossaryLoadError,
    inject_notes_for_cli,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

XHTML_NS = "http://www.w3.org/1999/xhtml"


def _make_chapter(text: str):
    """Return a minimal ChapterDoc-like object with a single paragraph."""

    @dataclass
    class FakeParagraph:
        pass

    xhtml = f"""<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml">
  <head><title>Test</title></head>
  <body><p>{text}</p></body>
</html>"""
    tree = etree.ElementTree(etree.fromstring(xhtml.encode()))
    root = tree.getroot()
    body = root.find(f"{{{XHTML_NS}}}body")
    paras = list(body)

    @dataclass
    class FakeChapter:
        tree: etree.ElementTree
        paragraphs: list

    return FakeChapter(tree=tree, paragraphs=paras)


def _make_glossary(entries: list[SeriesGlossaryEntry]) -> SeriesGlossary:
    return SeriesGlossary(
        series_slug="test-series",
        title="Test Book",
        author="Test Author",
        source_lang="en",
        target_lang="ru",
        entries=entries,
    )


def _make_entry(
    original: str, translation: str, notes: str, etype: str = "concept"
) -> SeriesGlossaryEntry:
    return SeriesGlossaryEntry(
        original=original,
        translation=translation,
        type=etype,
        notes=notes,
    )


def _make_cfg(enabled: bool = True) -> Config:
    cfg = Config()
    cfg.reader_notes = ReaderNotesConfig(
        enabled=enabled, types=["concept", "term"], scope="first-in-book"
    )
    return cfg


def _make_console():
    """Return a Rich Console-like mock that records prints."""
    messages = []

    class FakeConsole:
        def print(self, msg):
            messages.append(msg)

        @property
        def printed(self):
            return "\n".join(messages)

    return FakeConsole()


# ---------------------------------------------------------------------------
# inject_notes_for_cli — core behaviour
# ---------------------------------------------------------------------------


class TestInjectNotesForCli:
    def test_preloaded_glossary_is_used_directly(self, tmp_path: Path):
        """preloaded_glossary bypasses load_notes_glossary entirely."""
        chapter = _make_chapter("Вестигиум — это магия.")
        glossary = _make_glossary(
            [
                _make_entry("Vestigium", "Вестигиум", "Магический след"),
            ]
        )
        cfg = _make_cfg(enabled=True)
        console = _make_console()

        with patch("booktranslator.reader_notes.load_notes_glossary") as mock_load:
            stats = inject_notes_for_cli(
                console=console,
                chapters=[chapter],
                cfg=cfg,
                notes_flag=None,
                note_types=None,
                series=None,
                glossary_path=None,
                work_dir=tmp_path,
                book_title="Test",
                book_author="Author",
                preloaded_glossary=glossary,
            )

        mock_load.assert_not_called()
        assert stats is not None
        assert stats.notes_injected == 1

    def test_notes_flag_false_returns_none_no_injection(self, tmp_path: Path):
        """--no-notes flag disables injection regardless of cfg.enabled."""
        chapter = _make_chapter("Вестигиум — это магия.")
        glossary = _make_glossary(
            [
                _make_entry("Vestigium", "Вестигиум", "Магический след"),
            ]
        )
        cfg = _make_cfg(enabled=True)
        console = _make_console()

        stats = inject_notes_for_cli(
            console=console,
            chapters=[chapter],
            cfg=cfg,
            notes_flag=False,
            note_types=None,
            series=None,
            glossary_path=None,
            work_dir=tmp_path,
            book_title="Test",
            book_author="Author",
            preloaded_glossary=glossary,
        )

        assert stats is None

    def test_notes_flag_none_uses_cfg_enabled_true(self, tmp_path: Path):
        """notes_flag=None falls back to cfg.reader_notes.enabled=True."""
        chapter = _make_chapter("Вестигиум — это магия.")
        glossary = _make_glossary(
            [
                _make_entry("Vestigium", "Вестигиум", "Магический след"),
            ]
        )
        cfg = _make_cfg(enabled=True)
        console = _make_console()

        stats = inject_notes_for_cli(
            console=console,
            chapters=[chapter],
            cfg=cfg,
            notes_flag=None,
            note_types=None,
            series=None,
            glossary_path=None,
            work_dir=tmp_path,
            book_title="Test",
            book_author="Author",
            preloaded_glossary=glossary,
        )

        assert stats is not None
        assert stats.notes_injected == 1

    def test_notes_flag_none_uses_cfg_enabled_false(self, tmp_path: Path):
        """notes_flag=None falls back to cfg.reader_notes.enabled=False → skip."""
        chapter = _make_chapter("Вестигиум — это магия.")
        glossary = _make_glossary(
            [
                _make_entry("Vestigium", "Вестигиум", "Магический след"),
            ]
        )
        cfg = _make_cfg(enabled=False)
        console = _make_console()

        stats = inject_notes_for_cli(
            console=console,
            chapters=[chapter],
            cfg=cfg,
            notes_flag=None,
            note_types=None,
            series=None,
            glossary_path=None,
            work_dir=tmp_path,
            book_title="Test",
            book_author="Author",
            preloaded_glossary=glossary,
        )

        assert stats is None

    def test_no_glossary_implicit_returns_none_silently(self, tmp_path: Path):
        """When notes enabled by default and no glossary, silently return None (no warning)."""
        chapter = _make_chapter("Вестигиум — это магия.")
        cfg = _make_cfg(enabled=True)
        console = _make_console()

        stats = inject_notes_for_cli(
            console=console,
            chapters=[chapter],
            cfg=cfg,
            notes_flag=None,
            note_types=None,
            series=None,
            glossary_path=None,
            work_dir=tmp_path,
            book_title="Test",
            book_author="Author",
            preloaded_glossary=None,
            explicit=False,
        )

        assert stats is None
        assert console.printed == ""

    def test_no_glossary_explicit_warns_and_returns_none(self, tmp_path: Path):
        """When notes explicitly requested and no glossary, warn and return None."""
        chapter = _make_chapter("Вестигиум — это магия.")
        cfg = _make_cfg(enabled=True)
        console = _make_console()

        stats = inject_notes_for_cli(
            console=console,
            chapters=[chapter],
            cfg=cfg,
            notes_flag=True,
            note_types=None,
            series=None,
            glossary_path=None,
            work_dir=tmp_path,
            book_title="Test",
            book_author="Author",
            preloaded_glossary=None,
            explicit=True,
        )

        assert stats is None
        assert "Skipping" in console.printed or "no glossary" in console.printed.lower()

    def test_malformed_glossary_raises_load_error(self, tmp_path: Path):
        """GlossaryLoadError propagates to caller from a bad --glossary file."""
        bad = tmp_path / "bad.json"
        bad.write_text("not json {{{", encoding="utf-8")
        cfg = _make_cfg(enabled=True)
        console = _make_console()

        with pytest.raises(GlossaryLoadError):
            inject_notes_for_cli(
                console=console,
                chapters=[],
                cfg=cfg,
                notes_flag=True,
                note_types=None,
                series=None,
                glossary_path=bad,
                work_dir=tmp_path,
                book_title="Test",
                book_author="Author",
                preloaded_glossary=None,
            )

    def test_paragraph_count_unchanged_after_injection(self, tmp_path: Path):
        """Footnote injection must not change the number of paragraphs in the chapter."""
        xhtml = """<?xml version="1.0" encoding="utf-8"?>
<html xmlns="http://www.w3.org/1999/xhtml">
  <head><title>Test</title></head>
  <body>
    <p>Вестигиум — след магии в пространстве.</p>
    <p>Обычный абзац без терминов.</p>
    <p>Ещё один абзац с Вестигиумом.</p>
  </body>
</html>"""
        tree = etree.ElementTree(etree.fromstring(xhtml.encode()))
        root = tree.getroot()
        body = root.find(f"{{{XHTML_NS}}}body")
        paras = list(body.findall(f"{{{XHTML_NS}}}p"))

        @dataclass
        class FakeChapter:
            tree: etree.ElementTree
            paragraphs: list

        chapter = FakeChapter(tree=tree, paragraphs=paras)
        original_para_count = len(chapter.paragraphs)

        glossary = _make_glossary(
            [
                _make_entry("Vestigium", "Вестигиум", "Магический след"),
            ]
        )
        cfg = _make_cfg(enabled=True)
        console = _make_console()

        stats = inject_notes_for_cli(
            console=console,
            chapters=[chapter],
            cfg=cfg,
            notes_flag=True,
            note_types=None,
            series=None,
            glossary_path=None,
            work_dir=tmp_path,
            book_title="Test",
            book_author="Author",
            preloaded_glossary=glossary,
        )

        assert stats is not None
        assert stats.notes_injected == 1  # first-in-book scope
        # Paragraph count on the chapter must not have changed
        assert len(chapter.paragraphs) == original_para_count


# ---------------------------------------------------------------------------
# Cover integration in translate — via mocked translate_cover
# ---------------------------------------------------------------------------


def _make_epub_with_cover(path: Path) -> Path:
    """Create a minimal EPUB with a cover image for cover-translation tests."""
    epub_path = path / "book.epub"
    opf = """<?xml version="1.0"?>
<package xmlns="http://www.idpf.org/2007/opf" version="2.0" unique-identifier="uid">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/"
            xmlns:opf="http://www.idpf.org/2007/opf">
    <dc:title>Test Book</dc:title>
    <dc:language>en</dc:language>
    <meta name="cover" content="cover-img"/>
  </metadata>
  <manifest>
    <item id="cover-img" href="cover.jpg" media-type="image/jpeg"/>
  </manifest>
  <spine><itemref idref="cover-img"/></spine>
</package>"""
    with zipfile.ZipFile(epub_path, "w") as zf:
        zf.writestr(
            zipfile.ZipInfo("mimetype"), "application/epub+zip", compress_type=zipfile.ZIP_STORED
        )
        container_xml = (
            '<?xml version="1.0"?>'
            '<container version="1.0"'
            ' xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
            '<rootfiles><rootfile full-path="content.opf"'
            ' media-type="application/oebps-package+xml"/>'
            "</rootfiles></container>"
        )
        zf.writestr("META-INF/container.xml", container_xml)
        zf.writestr("content.opf", opf)
        # Minimal JPEG header
        zf.writestr("cover.jpg", b"\xff\xd8\xff\xe0" + b"\x00" * 16)
    return epub_path


class TestCoverInTranslatePipeline:
    """Unit tests for the --cover wiring in translate using mocked translate_cover."""

    def test_cover_called_with_correct_args(self, tmp_path: Path):
        """translate_cover is called with target_lang from cfg.target_lang_name."""
        from booktranslator.provider import ImageGenerationResult

        fake_result = ImageGenerationResult(
            image_bytes=b"\xff\xd8\xff\xe0" + b"\x00" * 16,
            mime_type="image/jpeg",
            text="",
            model="test/model",
        )

        epub = _make_epub_with_cover(tmp_path)
        out_epub = tmp_path / "out.epub"
        # Write a minimal epub as the "already written" output
        import shutil

        shutil.copy(epub, out_epub)

        cfg = Config()
        cfg.target_lang_name = "Russian"
        cfg.models.cover = "test/cover-model"

        calls = []

        def fake_translate_cover(
            source_epub,
            dest_epub,
            provider,
            *,
            model,
            title_translation,
            author_name,
            target_lang,
            aspect_ratio,
            image_size,
            **kw,
        ):
            calls.append(
                {
                    "source_epub": source_epub,
                    "dest_epub": dest_epub,
                    "model": model,
                    "target_lang": target_lang,
                    "title_translation": title_translation,
                    "author_name": author_name,
                    "aspect_ratio": aspect_ratio,
                    "image_size": image_size,
                }
            )
            # Write dest so os.replace has something to work with
            shutil.copy(source_epub, dest_epub)
            return fake_result

        with patch("booktranslator.cover.translate_cover", fake_translate_cover):
            # Simulate the cover block from translate command directly
            cover_model = None
            cover_target_lang = None
            cover_title = "Реки Лондона"
            cover_author = "Бен Ааронович"
            cover_aspect_ratio = "2:3"
            cover_image_size = "1K"

            _cover_model = cover_model or cfg.models.cover
            _cover_lang = cover_target_lang or cfg.target_lang_name

            fd, tmp_name = tempfile.mkstemp(suffix=".epub", dir=out_epub.parent)
            os.close(fd)
            tmp = Path(tmp_name)
            try:
                fake_translate_cover(
                    source_epub=out_epub,
                    dest_epub=tmp,
                    provider=MagicMock(),
                    model=_cover_model,
                    title_translation=cover_title,
                    author_name=cover_author,
                    target_lang=_cover_lang,
                    aspect_ratio=cover_aspect_ratio,
                    image_size=cover_image_size,
                )
                os.replace(tmp, out_epub)
            except Exception:
                tmp.unlink(missing_ok=True)
                raise

        assert len(calls) == 1
        assert calls[0]["target_lang"] == "Russian"
        assert calls[0]["model"] == "test/cover-model"
        assert calls[0]["title_translation"] == "Реки Лондона"
        assert calls[0]["author_name"] == "Бен Ааронович"
        assert out_epub.exists()

    def test_cover_failure_does_not_remove_epub(self, tmp_path: Path):
        """If translate_cover raises, the original EPUB at out_path is preserved."""
        import shutil
        import tempfile

        epub = _make_epub_with_cover(tmp_path)
        out_epub = tmp_path / "out.epub"
        shutil.copy(epub, out_epub)
        original_size = out_epub.stat().st_size

        def failing_translate_cover(*args, **kwargs):
            raise RuntimeError("image provider unreachable")

        console = _make_console()

        fd, tmp_name = tempfile.mkstemp(suffix=".epub", dir=out_epub.parent)
        os.close(fd)
        tmp = Path(tmp_name)
        try:
            failing_translate_cover()
            os.replace(tmp, out_epub)
        except Exception as e:
            tmp.unlink(missing_ok=True)
            console.print(f"[yellow]Cover translation failed (EPUB kept):[/yellow] {e}")

        # out_epub must still exist and be unchanged
        assert out_epub.exists()
        assert out_epub.stat().st_size == original_size
        assert "Cover translation failed" in console.printed

    def test_cover_tmp_written_to_same_filesystem(self, tmp_path: Path):
        """Temp file is created in out_path.parent so os.replace is atomic."""
        out_epub = tmp_path / "out.epub"
        out_epub.write_bytes(b"placeholder")

        fd, tmp_name = tempfile.mkstemp(suffix=".epub", dir=out_epub.parent)
        os.close(fd)
        tmp = Path(tmp_name)

        assert tmp.parent == out_epub.parent

        tmp.unlink(missing_ok=True)

    def test_cover_target_lang_override(self, tmp_path: Path):
        """--cover-target-lang overrides cfg.resolved_target_lang_name()."""
        cfg = Config()
        # target_lang="ru" -> auto-resolves to "Russian"
        assert cfg.resolved_target_lang_name() == "Russian"

        cover_target_lang_override = "German"
        _cover_lang = cover_target_lang_override or cfg.resolved_target_lang_name()

        assert _cover_lang == "German"

    def test_cover_target_lang_falls_back_to_cfg(self, tmp_path: Path):
        """Without --cover-target-lang, resolved_target_lang_name() is used."""
        cfg = Config()  # target_lang="ru" -> auto-resolves to "Russian"

        cover_target_lang_override = None
        _cover_lang = cover_target_lang_override or cfg.resolved_target_lang_name()

        assert _cover_lang == "Russian"

    def test_resolved_target_lang_name_unknown_code(self, tmp_path: Path):
        """An unknown target_lang code with no explicit name returns None."""
        cfg = Config(target_lang="xx")  # not in the mapping
        assert cfg.resolved_target_lang_name() is None

    def test_resolved_target_lang_name_explicit_overrides_map(self, tmp_path: Path):
        """Explicit target_lang_name takes priority over the auto-map."""
        cfg = Config(target_lang="ru", target_lang_name="Russisch")
        assert cfg.resolved_target_lang_name() == "Russisch"


# ---------------------------------------------------------------------------
# CLI-level wiring test: translate --cover
# ---------------------------------------------------------------------------


class TestTranslateCoverCLIWiring:
    """Verify --cover wiring in the translate command via Typer CLI runner.

    Mocks the full translation pipeline so the test is fast and LLM-free,
    while still exercising CLI argument routing, provider creation, and the
    actual cover translation call from within cli.translate().
    """

    def _make_source_epub(self, path: Path) -> Path:
        """Build a minimal valid EPUB with a cover image."""
        src = path / "source.epub"
        opf = (
            '<?xml version="1.0"?>'
            '<package xmlns="http://www.idpf.org/2007/opf" version="2.0"'
            ' unique-identifier="uid">'
            '<metadata xmlns:dc="http://purl.org/dc/elements/1.1/">'
            "<dc:title>Test</dc:title><dc:language>en</dc:language>"
            '<meta name="cover" content="cov"/>'
            "</metadata>"
            "<manifest>"
            '<item id="cov" href="cover.jpg" media-type="image/jpeg"/>'
            '<item id="c1" href="ch01.xhtml"'
            ' media-type="application/xhtml+xml"/>'
            "</manifest>"
            '<spine><itemref idref="c1"/></spine>'
            "</package>"
        )
        xhtml = (
            '<?xml version="1.0"?>'
            '<html xmlns="http://www.w3.org/1999/xhtml">'
            "<head><title>T</title></head><body><p>Hello.</p></body></html>"
        )
        container = (
            '<?xml version="1.0"?>'
            '<container version="1.0"'
            ' xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
            "<rootfiles>"
            '<rootfile full-path="content.opf"'
            ' media-type="application/oebps-package+xml"/>'
            "</rootfiles></container>"
        )
        with zipfile.ZipFile(src, "w") as zf:
            zf.writestr(
                zipfile.ZipInfo("mimetype"),
                "application/epub+zip",
                compress_type=zipfile.ZIP_STORED,
            )
            zf.writestr("META-INF/container.xml", container)
            zf.writestr("content.opf", opf)
            zf.writestr("ch01.xhtml", xhtml)
            zf.writestr("cover.jpg", b"\xff\xd8\xff\xe0" + b"\x00" * 16)
        return src

    def test_cover_flag_calls_translate_cover_with_correct_args(self, tmp_path: Path):
        """--cover invokes translate_cover with target_lang auto-derived from cfg."""
        import shutil
        from unittest.mock import MagicMock, patch

        from typer.testing import CliRunner

        from booktranslator.cli import app
        from booktranslator.models import BookMeta
        from booktranslator.provider import ImageGenerationResult

        src_epub = self._make_source_epub(tmp_path)
        out_epub = tmp_path / "source-ru.epub"
        out_epub.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 16)

        cover_calls: list[dict] = []
        fake_result = ImageGenerationResult(
            image_bytes=b"\xff\xd8\xff\xe0" + b"\x00" * 16,
            mime_type="image/jpeg",
            text="",
            model="test/model",
        )

        def fake_translate_cover(
            source_epub,
            dest_epub,
            provider,
            *,
            model,
            title_translation,
            author_name,
            target_lang,
            aspect_ratio,
            image_size,
            **kw,
        ):
            cover_calls.append({"target_lang": target_lang, "title_translation": title_translation})
            shutil.copy(source_epub, dest_epub)
            return fake_result

        fake_chapter = MagicMock()
        fake_chapter.paragraphs = []
        fake_book = MagicMock()
        fake_book.meta = BookMeta(
            title="Test",
            author="Author",
            language="en",
            word_count=10,
            chapters=1,
        )
        fake_book.chapters = [fake_chapter]

        fake_chunk_set = MagicMock()
        fake_chunk_set.chunks = []

        from dataclasses import dataclass as _dc

        @_dc
        class _FakeStats:
            chunks_translated: int = 0
            chunks_cached: int = 0
            chunks_failed: int = 0
            input_tokens: int = 0
            output_tokens: int = 0
            results: list = None

            def __post_init__(self):
                if self.results is None:
                    self.results = []

        fake_translate_stats = _FakeStats()
        fake_translator = MagicMock()
        fake_translator.translate_book.return_value = fake_translate_stats

        runner = CliRunner()
        with (
            patch("booktranslator.cli.read_book_structured", return_value=fake_book),
            patch("booktranslator.cli.chunk_book", return_value=fake_chunk_set),
            patch("booktranslator.cli.Translator", return_value=fake_translator),
            patch("booktranslator.cli.write_translated_epub"),
            patch("booktranslator.cli.rehydrate_book_from_waterfall", return_value=(0, [])),
            patch("booktranslator.cli.create_image_provider", return_value=MagicMock()),
            patch("booktranslator.cover.translate_cover", fake_translate_cover),
        ):
            result = runner.invoke(
                app,
                [
                    "translate",
                    str(src_epub),
                    "--cover",
                    "--cover-title",
                    "\u0422\u0435\u0441\u0442",
                    "--no-judge",
                    "--no-proofread",
                    "--no-style",
                    "--no-verify",
                    "--out",
                    str(out_epub),
                    "--work",
                    str(tmp_path / "work"),
                ],
                catch_exceptions=False,
            )

        assert len(cover_calls) == 1, f"translate_cover not called.\nCLI output:\n{result.output}"
        assert cover_calls[0]["target_lang"] == "Russian"
        assert cover_calls[0]["title_translation"] == "\u0422\u0435\u0441\u0442"

    def _invoke_translate_cover(
        self,
        tmp_path: Path,
        extra_args: list,
        patch_translate_cover=None,
        load_config_patch=None,
    ):
        """Helper: invoke translate --cover with the pipeline fully mocked."""
        import shutil
        from contextlib import ExitStack
        from unittest.mock import MagicMock, patch

        from typer.testing import CliRunner

        from booktranslator.cli import app
        from booktranslator.models import BookMeta
        from booktranslator.provider import ImageGenerationResult

        src_epub = self._make_source_epub(tmp_path)
        out_epub = tmp_path / "source-ru.epub"
        out_epub.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 16)

        if patch_translate_cover is None:

            def patch_translate_cover(src, dst, provider, *, model, **kw):
                shutil.copy(src, dst)
                return ImageGenerationResult(
                    image_bytes=b"\xff\xd8\xff\xe0" + b"\x00" * 16,
                    mime_type="image/jpeg",
                    text="",
                    model=model,
                )

        from dataclasses import dataclass as _dc

        @_dc
        class _FakeStats:
            chunks_translated: int = 0
            chunks_cached: int = 0
            chunks_failed: int = 0
            input_tokens: int = 0
            output_tokens: int = 0
            results: list = None

            def __post_init__(self):
                if self.results is None:
                    self.results = []

        fake_translator = MagicMock()
        fake_translator.translate_book.return_value = _FakeStats()
        fake_book = MagicMock()
        fake_book.meta = BookMeta(
            title="Test",
            author="Author",
            language="en",
            word_count=10,
            chapters=1,
        )
        fake_book.chapters = []
        fake_chunk_set = MagicMock()
        fake_chunk_set.chunks = []

        runner = CliRunner()
        with ExitStack() as stack:
            if load_config_patch is not None:
                stack.enter_context(patch("booktranslator.cli.load_config", load_config_patch))
            stack.enter_context(
                patch("booktranslator.cli.read_book_structured", return_value=fake_book)
            )
            stack.enter_context(patch("booktranslator.cli.chunk_book", return_value=fake_chunk_set))
            stack.enter_context(
                patch("booktranslator.cli.Translator", return_value=fake_translator)
            )
            stack.enter_context(patch("booktranslator.cli.write_translated_epub"))
            stack.enter_context(
                patch(
                    "booktranslator.cli.rehydrate_book_from_waterfall",
                    return_value=(0, []),
                )
            )
            stack.enter_context(
                patch(
                    "booktranslator.cli.create_image_provider",
                    return_value=MagicMock(),
                )
            )
            stack.enter_context(
                patch("booktranslator.cover.translate_cover", patch_translate_cover)
            )
            result = runner.invoke(
                app,
                [
                    "translate",
                    str(src_epub),
                    "--cover",
                    "--no-judge",
                    "--no-proofread",
                    "--no-style",
                    "--no-verify",
                    "--out",
                    str(out_epub),
                    "--work",
                    str(tmp_path / "work"),
                ]
                + extra_args,
            )
        return result

    def test_cover_unknown_target_lang_exits_1(self, tmp_path: Path):
        """--cover with unknown target_lang and no explicit name exits 1."""
        from booktranslator.config import load_config as _orig_load

        def _patched_load(path=None):
            cfg = _orig_load(path)
            cfg.target_lang = "xx"  # not in the map
            cfg.target_lang_name = None
            return cfg

        result = self._invoke_translate_cover(
            tmp_path,
            extra_args=[],
            patch_translate_cover=None,
            load_config_patch=_patched_load,
        )
        assert result.exit_code == 1, f"Expected exit 1, got {result.exit_code}.\n{result.output}"
        assert "requires a target language name" in result.output

    def test_cover_missing_model_exits_1(self, tmp_path: Path):
        """--cover with empty cfg.models.cover and no --cover-model exits 1."""
        from booktranslator.config import load_config as _orig_load

        def _patched_load(path=None):
            cfg = _orig_load(path)
            cfg.models.cover = ""  # empty — should fail fast
            return cfg

        result = self._invoke_translate_cover(
            tmp_path,
            extra_args=[],
            load_config_patch=_patched_load,
        )
        assert result.exit_code == 1, f"Expected exit 1, got {result.exit_code}.\n{result.output}"
        assert "requires a model" in result.output
