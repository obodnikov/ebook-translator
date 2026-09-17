"""Tests for reading the EPUB package version, and what it decides.

`read_package_version` is the only thing that chooses between popup
footnotes (EPUB3) and endnotes (EPUB2). A silent fallback on an EPUB3 book
would quietly produce the wrong markup, so every fallback path is covered
here, together with the dispatch it feeds.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

from lxml import etree

from booktranslator import epub_io
from booktranslator.epub_io import read_book, read_book_structured, read_package_version
from booktranslator.models import ReaderNotesConfig, SeriesGlossary, SeriesGlossaryEntry
from booktranslator.reader_notes import inject_reader_notes, is_epub3

XHTML_NS = "http://www.w3.org/1999/xhtml"

_CONTAINER = """<?xml version="1.0" encoding="utf-8"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles>
    <rootfile full-path="{opf}" media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>"""

_CHAPTER = f"""<?xml version="1.0" encoding="utf-8"?>
<html xmlns="{XHTML_NS}">
<head><title>Test</title></head>
<body>
<p>A trace of vestigium in the room.</p>
</body>
</html>"""


def _opf(version_attr: str) -> str:
    """A package document, with `version_attr` spliced into <package>."""
    return f"""<?xml version="1.0" encoding="utf-8"?>
<package xmlns="http://www.idpf.org/2007/opf"{version_attr} unique-identifier="uid">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:title>Test Book</dc:title>
    <dc:creator>Test Author</dc:creator>
    <dc:language>en</dc:language>
    <dc:identifier id="uid">uid-1</dc:identifier>
  </metadata>
  <manifest>
    <item id="ch01" href="ch01.xhtml" media-type="application/xhtml+xml"/>
  </manifest>
  <spine><itemref idref="ch01"/></spine>
</package>"""


def _write_epub(
    path: Path,
    *,
    version_attr: str = ' version="2.0"',
    container: str | None = None,
    opf_path: str = "content.opf",
) -> Path:
    """Write a tiny but valid EPUB, with room to break one piece at a time."""
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(
            zipfile.ZipInfo("mimetype"),
            "application/epub+zip",
            compress_type=zipfile.ZIP_STORED,
        )
        if container is not None:
            zf.writestr("META-INF/container.xml", container)
        zf.writestr(opf_path, _opf(version_attr))
        prefix = opf_path.rsplit("/", 1)[0] + "/" if "/" in opf_path else ""
        zf.writestr(f"{prefix}ch01.xhtml", _CHAPTER)
    return path


def _epub(path: Path, **kwargs) -> Path:
    opf_path = kwargs.get("opf_path", "content.opf")
    kwargs.setdefault("container", _CONTAINER.format(opf=opf_path))
    return _write_epub(path, **kwargs)


class TestReadPackageVersion:
    def test_epub2(self, tmp_path: Path):
        book = _epub(tmp_path / "b.epub", version_attr=' version="2.0"')
        assert read_package_version(book) == "2.0"
        assert is_epub3(read_package_version(book)) is False

    def test_epub3(self, tmp_path: Path):
        book = _epub(tmp_path / "b.epub", version_attr=' version="3.0"')
        assert read_package_version(book) == "3.0"
        assert is_epub3(read_package_version(book)) is True

    def test_epub3_minor_release(self, tmp_path: Path):
        book = _epub(tmp_path / "b.epub", version_attr=' version="3.2"')
        assert read_package_version(book) == "3.2"
        assert is_epub3(read_package_version(book)) is True

    def test_version_is_trimmed(self, tmp_path: Path):
        book = _epub(tmp_path / "b.epub", version_attr=' version="  3.0  "')
        assert read_package_version(book) == "3.0"

    def test_package_in_a_subdirectory(self, tmp_path: Path):
        book = _epub(
            tmp_path / "b.epub", version_attr=' version="3.0"', opf_path="OEBPS/content.opf"
        )
        assert read_package_version(book) == "3.0"


class TestFallsBackToEpub2:
    """Every unreadable case lands on the version every reader understands."""

    def test_no_version_attribute(self, tmp_path: Path):
        book = _epub(tmp_path / "b.epub", version_attr="")
        assert read_package_version(book) == "2.0"

    def test_empty_version_attribute(self, tmp_path: Path):
        book = _epub(tmp_path / "b.epub", version_attr=' version="   "')
        assert read_package_version(book) == "2.0"

    def test_no_container(self, tmp_path: Path):
        book = _write_epub(tmp_path / "b.epub", version_attr=' version="3.0"', container=None)
        assert read_package_version(book) == "2.0"

    def test_container_without_rootfile(self, tmp_path: Path):
        container = """<?xml version="1.0" encoding="utf-8"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles/>
</container>"""
        book = _write_epub(tmp_path / "b.epub", version_attr=' version="3.0"', container=container)
        assert read_package_version(book) == "2.0"

    def test_rootfile_without_a_path(self, tmp_path: Path):
        container = """<?xml version="1.0" encoding="utf-8"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles><rootfile media-type="application/oebps-package+xml"/></rootfiles>
</container>"""
        book = _write_epub(tmp_path / "b.epub", version_attr=' version="3.0"', container=container)
        assert read_package_version(book) == "2.0"

    def test_rootfile_points_at_nothing(self, tmp_path: Path):
        book = _write_epub(
            tmp_path / "b.epub",
            version_attr=' version="3.0"',
            container=_CONTAINER.format(opf="missing.opf"),
        )
        assert read_package_version(book) == "2.0"

    def test_unparsable_container(self, tmp_path: Path):
        book = _write_epub(
            tmp_path / "b.epub", version_attr=' version="3.0"', container="<container"
        )
        assert read_package_version(book) == "2.0"

    def test_not_a_zip(self, tmp_path: Path):
        broken = tmp_path / "broken.epub"
        broken.write_bytes(b"this is not a zip archive")
        assert read_package_version(broken) == "2.0"

    def test_truncated_zip(self, tmp_path: Path):
        book = _epub(tmp_path / "b.epub", version_attr=' version="3.0"')
        data = book.read_bytes()
        truncated = tmp_path / "cut.epub"
        truncated.write_bytes(data[: len(data) // 2])
        assert read_package_version(truncated) == "2.0"

    def test_missing_file(self, tmp_path: Path):
        assert read_package_version(tmp_path / "nothing-here.epub") == "2.0"


class TestVersionReachesBookMeta:
    """Also a regression guard: both readers open a zip to get the version."""

    def test_read_book(self, tmp_path: Path):
        book = _epub(tmp_path / "b.epub", version_attr=' version="3.0"')
        assert read_book(book).meta.epub_version == "3.0"

    def test_read_book_structured(self, tmp_path: Path):
        book = _epub(tmp_path / "b.epub", version_attr=' version="3.0"')
        assert read_book_structured(book).meta.epub_version == "3.0"

    def test_epub2_book(self, tmp_path: Path):
        book = _epub(tmp_path / "b.epub", version_attr=' version="2.0"')
        assert read_book_structured(book).meta.epub_version == "2.0"


class TestMarkupFollowsTheDetectedVersion:
    """The whole point of reading the version: end to end, book to markup."""

    @staticmethod
    def _glossary() -> SeriesGlossary:
        return SeriesGlossary(
            series_slug="test",
            title="Test",
            author="Test",
            source_lang="en",
            target_lang="ru",
            entries=[
                SeriesGlossaryEntry(
                    original="vestigium",
                    translation="vestigium",
                    type="concept",
                    notes="Magical trace left by spells.",
                )
            ],
        )

    def _inject(self, path: Path) -> str:
        book = read_book_structured(path)
        stats = inject_reader_notes(
            chapters=book.chapters,
            glossary=self._glossary(),
            config=ReaderNotesConfig(enabled=True, types=["concept"], scope="first-in-book"),
            epub_version=book.meta.epub_version,
        )
        assert stats.notes_injected == 1
        return etree.tostring(book.chapters[0].tree, encoding="unicode")

    def test_epub3_book_gets_an_aside(self, tmp_path: Path):
        out = self._inject(_epub(tmp_path / "b.epub", version_attr=' version="3.0"'))
        assert "<aside" in out
        assert 'class="reader-notes"' not in out
        assert 'epub:type="noteref"' in out

    def test_epub2_book_gets_endnotes(self, tmp_path: Path):
        out = self._inject(_epub(tmp_path / "b.epub", version_attr=' version="2.0"'))
        assert "<aside" not in out
        assert 'class="reader-notes"' in out
        assert 'epub:type="noteref"' in out

    def test_a_book_with_no_declared_version_gets_endnotes(self, tmp_path: Path):
        """The fallback has to be the safe one, not the popup markup."""
        out = self._inject(_epub(tmp_path / "b.epub", version_attr=""))
        assert "<aside" not in out
        assert 'class="reader-notes"' in out


class TestPackagePathQuirks:
    """The path in container.xml does not always match the stored member."""

    def test_leading_slash_in_full_path(self, tmp_path: Path):
        book = _write_epub(
            tmp_path / "b.epub",
            version_attr=' version="3.0"',
            container=_CONTAINER.format(opf="/content.opf"),
        )
        assert read_package_version(book) == "3.0"

    def test_case_differs_from_the_stored_member(self, tmp_path: Path):
        book = _write_epub(
            tmp_path / "b.epub",
            version_attr=' version="3.0"',
            container=_CONTAINER.format(opf="Content.OPF"),
        )
        assert read_package_version(book) == "3.0"

    def test_leading_slash_on_a_nested_package(self, tmp_path: Path):
        book = _write_epub(
            tmp_path / "b.epub",
            version_attr=' version="3.0"',
            container=_CONTAINER.format(opf="/OEBPS/content.opf"),
            opf_path="OEBPS/content.opf",
        )
        assert read_package_version(book) == "3.0"


class TestFallbackIsReported:
    """A fallback must be visible, not passed off as a declared 2.0."""

    def test_unreadable_version_logs_a_warning(self, tmp_path: Path, caplog):
        book = _epub(tmp_path / "b.epub", version_attr="")
        with caplog.at_level("WARNING", logger="booktranslator.epub_io"):
            assert read_package_version(book) == "2.0"
        assert "Assuming EPUB 2.0" in caplog.text
        assert "b.epub" in caplog.text

    def test_broken_archive_logs_a_warning(self, tmp_path: Path, caplog):
        broken = tmp_path / "broken.epub"
        broken.write_bytes(b"not a zip")
        with caplog.at_level("WARNING", logger="booktranslator.epub_io"):
            assert read_package_version(broken) == "2.0"
        assert "Assuming EPUB 2.0" in caplog.text

    def test_a_declared_version_says_nothing(self, tmp_path: Path, caplog):
        book = _epub(tmp_path / "b.epub", version_attr=' version="2.0"')
        with caplog.at_level("WARNING", logger="booktranslator.epub_io"):
            assert read_package_version(book) == "2.0"
        assert caplog.text == ""


class TestVersionComesFromTheParsedBook:
    """No second pass over the archive when ebooklib already knows."""

    def test_parsed_version_is_used(self, tmp_path: Path, monkeypatch):
        book = _epub(tmp_path / "b.epub", version_attr=' version="3.0"')

        def _should_not_run(_path):
            raise AssertionError("the archive should not be read a second time")

        monkeypatch.setattr("booktranslator.epub_io.read_package_version", _should_not_run)
        assert read_book_structured(book).meta.epub_version == "3.0"

    def test_archive_sniff_covers_a_blank_parsed_version(self, tmp_path: Path, monkeypatch):
        book = _epub(tmp_path / "b.epub", version_attr=' version="3.0"')
        monkeypatch.setattr("booktranslator.epub_io.read_package_version", lambda _p: "3.2")

        real_read = epub_io.epub.read_epub

        def _blank_version(path, *args, **kwargs):
            parsed = real_read(path, *args, **kwargs)
            parsed.version = ""
            return parsed

        monkeypatch.setattr("booktranslator.epub_io.epub.read_epub", _blank_version)
        assert read_book_structured(book).meta.epub_version == "3.2"
