#!/usr/bin/env python3
"""
Split a merged EPUB (e.g. an epubmerge anthology) into individual books.

Strategy:
    1. Parse META-INF/container.xml to find the root OPF.
    2. Parse OPF manifest/spine and NCX to build a list of books.
       Each top-level navPoint in NCX = one book.
    3. Each book's files live under a single directory prefix
       (e.g. "1/1/", "2/3/", "3/"), inferred from the book's entry src.
    4. To extract a book: copy its files (stripping the prefix), write a
       new OPF/NCX with only that book's manifest/spine/navigation, and
       package as a fresh EPUB.

Usage:
    python tools/split_epub.py INPUT.epub --list
    python tools/split_epub.py INPUT.epub --book N [--out DIR]
"""

from __future__ import annotations

import argparse
import re
import sys
import unicodedata
import uuid
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from urllib.parse import unquote, urlparse

from lxml import etree

# ---------------------------------------------------------------------------
# XML namespaces used across EPUB 2.0 / NCX / container specs.
# ---------------------------------------------------------------------------

NS = {
    "opf": "http://www.idpf.org/2007/opf",
    "dc": "http://purl.org/dc/elements/1.1/",
    "ncx": "http://www.daisy.org/z3986/2005/ncx/",
    "container": "urn:oasis:names:tc:opendocument:xmlns:container",
    "xhtml": "http://www.w3.org/1999/xhtml",
}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class ManifestItem:
    item_id: str
    href: str  # href relative to the OPF file (= root of the archive here)
    media_type: str


@dataclass
class Book:
    index: int  # 1-based position in the anthology
    title: str
    prefix: str  # e.g. "1/1/", "3/"
    entry_href: str  # href (no fragment) of the book's TOC entry
    nav_subtree: etree._Element  # the top-level <navPoint> for this book
    manifest_items: list[ManifestItem]  # items whose href starts with prefix
    spine_idrefs: list[str]  # idrefs for this book, in reading order


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def href_to_path(href: str) -> str:
    """Strip URL fragment and decode percent-escapes from a manifest/toc href."""
    parsed = urlparse(href)
    return unquote(parsed.path)


def book_prefix_from_href(href: str) -> str:
    """
    Derive the directory prefix for a book from its entry href.

    Examples:
        '1/1/OEBPS/cover.xml'   -> '1/1/'
        '1/3/titlepage.xhtml'   -> '1/3/'
        '3/titlepage.xhtml'     -> '3/'
        '3/OEBPS/xhtml/01.xhtml'-> '3/'
    Heuristic: take path components up to (but not including) the first one
    that looks like an EPUB-internal folder ('OEBPS', 'xhtml', 'text')
    or a file (contains '.').
    """
    path = href_to_path(href)
    parts = path.split("/")
    known_subdirs = {"OEBPS", "xhtml", "text", "images", "css"}
    collected: list[str] = []
    for part in parts:
        if not part:
            continue
        if "." in part or part in known_subdirs:
            break
        collected.append(part)
    if not collected:
        raise ValueError(f"Cannot derive book prefix from href: {href!r}")
    return "/".join(collected) + "/"


def slugify(text: str) -> str:
    """Lowercase ASCII slug, dashes between tokens, alnum only."""
    normalized = unicodedata.normalize("NFKD", text)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    ascii_text = ascii_text.lower()
    ascii_text = re.sub(r"[^a-z0-9]+", "-", ascii_text)
    return ascii_text.strip("-")


def strip_prefix(href: str, prefix: str) -> str:
    """Remove the book prefix from a manifest/toc href. Fragment preserved."""
    path, _, frag = href.partition("#")
    if not path.startswith(prefix):
        raise ValueError(f"Expected href {href!r} to start with prefix {prefix!r}")
    stripped = path[len(prefix) :]
    return stripped + (f"#{frag}" if frag else "")


# ---------------------------------------------------------------------------
# Parsing the merged anthology
# ---------------------------------------------------------------------------


class Anthology:
    """A parsed merged EPUB, exposing the list of books inside."""

    def __init__(self, epub_path: Path):
        self.epub_path = epub_path
        self.zip = zipfile.ZipFile(epub_path, "r")
        self.opf_path = self._find_opf_path()
        self.opf_tree = self._parse(self.opf_path)
        self.ncx_path, self.ncx_tree = self._find_and_parse_ncx()

        self.manifest: dict[str, ManifestItem] = self._parse_manifest()
        self.spine_idrefs: list[str] = self._parse_spine()
        self.books: list[Book] = self._identify_books()

    # --- low-level helpers ---

    def _parse(self, archive_path: str) -> etree._ElementTree:
        with self.zip.open(archive_path) as fh:
            return etree.parse(fh)

    def _find_opf_path(self) -> str:
        with self.zip.open("META-INF/container.xml") as fh:
            tree = etree.parse(fh)
        rootfile = tree.find(".//container:rootfiles/container:rootfile", NS)
        if rootfile is None:
            raise RuntimeError("container.xml has no rootfile entry")
        return rootfile.get("full-path")

    def _find_and_parse_ncx(self) -> tuple[str, etree._ElementTree]:
        # NCX is referenced from manifest as media-type
        # application/x-dtbncx+xml. It may live next to the OPF.
        manifest_el = self.opf_tree.find(".//opf:manifest", NS)
        for item in manifest_el.findall("opf:item", NS):
            if item.get("media-type") == "application/x-dtbncx+xml":
                ncx_href = item.get("href")
                ncx_path = self._resolve_from_opf(ncx_href)
                return ncx_path, self._parse(ncx_path)
        raise RuntimeError("No NCX file found in manifest")

    def _resolve_from_opf(self, href: str) -> str:
        """Resolve an href (from manifest) into an archive path."""
        opf_dir = self.opf_path.rsplit("/", 1)[0] if "/" in self.opf_path else ""
        if not opf_dir:
            return href
        return f"{opf_dir}/{href}"

    # --- OPF parsing ---

    def _parse_manifest(self) -> dict[str, ManifestItem]:
        items: dict[str, ManifestItem] = {}
        for item in self.opf_tree.findall(".//opf:manifest/opf:item", NS):
            items[item.get("id")] = ManifestItem(
                item_id=item.get("id"),
                href=item.get("href"),
                media_type=item.get("media-type"),
            )
        return items

    def _parse_spine(self) -> list[str]:
        return [
            itemref.get("idref")
            for itemref in self.opf_tree.findall(".//opf:spine/opf:itemref", NS)
        ]

    # --- NCX parsing: find books ---

    def _identify_books(self) -> list[Book]:
        nav_map = self.ncx_tree.find(".//ncx:navMap", NS)
        if nav_map is None:
            raise RuntimeError("NCX has no navMap")

        top_points = nav_map.findall("ncx:navPoint", NS)
        if not top_points:
            raise RuntimeError("navMap has no top-level navPoints")

        books: list[Book] = []
        for idx, np in enumerate(top_points, start=1):
            title_el = np.find("ncx:navLabel/ncx:text", NS)
            title = (title_el.text or "").strip() if title_el is not None else f"Book {idx}"

            content_el = np.find("ncx:content", NS)
            if content_el is None:
                raise RuntimeError(f"navPoint for book {idx!r} has no content src")
            entry_src = content_el.get("src")
            prefix = book_prefix_from_href(entry_src)

            manifest_items = [
                item
                for item in self.manifest.values()
                if href_to_path(item.href).startswith(prefix)
            ]
            manifest_ids = {item.item_id for item in manifest_items}
            spine_idrefs = [idref for idref in self.spine_idrefs if idref in manifest_ids]

            books.append(
                Book(
                    index=idx,
                    title=title,
                    prefix=prefix,
                    entry_href=href_to_path(entry_src),
                    nav_subtree=np,
                    manifest_items=manifest_items,
                    spine_idrefs=spine_idrefs,
                )
            )
        return books

    # --- Anthology-level metadata ---

    def creator(self) -> str:
        el = self.opf_tree.find(".//dc:creator", NS)
        if el is not None and el.text:
            return el.text.strip()
        return "Unknown"

    def language(self) -> str:
        el = self.opf_tree.find(".//dc:language", NS)
        if el is not None and el.text:
            return el.text.strip()
        return "en"


# ---------------------------------------------------------------------------
# Building a new EPUB for a single book
# ---------------------------------------------------------------------------


class BookWriter:
    """Writes a single Book as a standalone EPUB file."""

    def __init__(self, anthology: Anthology, book: Book):
        self.anthology = anthology
        self.book = book

    def write(self, out_path: Path) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Write to a temp buffer first, then to disk, to avoid partial files.
        buffer = BytesIO()
        with zipfile.ZipFile(buffer, "w") as zf:
            # mimetype must be first and uncompressed (EPUB spec).
            zf.writestr(
                zipfile.ZipInfo("mimetype"),
                "application/epub+zip",
                compress_type=zipfile.ZIP_STORED,
            )
            zf.writestr(
                "META-INF/container.xml",
                self._make_container_xml(),
                compress_type=zipfile.ZIP_DEFLATED,
            )
            zf.writestr(
                "content.opf",
                self._make_opf(),
                compress_type=zipfile.ZIP_DEFLATED,
            )
            zf.writestr(
                "toc.ncx",
                self._make_ncx(),
                compress_type=zipfile.ZIP_DEFLATED,
            )
            self._copy_book_files(zf)

        out_path.write_bytes(buffer.getvalue())

    # --- builders ---

    def _make_container_xml(self) -> bytes:
        root = etree.Element(
            "{{{}}}container".format(NS["container"]),
            nsmap={None: NS["container"]},
            version="1.0",
        )
        rootfiles = etree.SubElement(root, "{{{}}}rootfiles".format(NS["container"]))
        etree.SubElement(
            rootfiles,
            "{{{}}}rootfile".format(NS["container"]),
            attrib={
                "full-path": "content.opf",
                "media-type": "application/oebps-package+xml",
            },
        )
        return etree.tostring(root, xml_declaration=True, encoding="utf-8", standalone=True)

    def _make_opf(self) -> bytes:
        package = etree.Element(
            "{{{}}}package".format(NS["opf"]),
            nsmap={None: NS["opf"]},
            attrib={"version": "2.0", "unique-identifier": "bookid"},
        )
        metadata = etree.SubElement(
            package,
            "{{{}}}metadata".format(NS["opf"]),
            nsmap={"dc": NS["dc"], "opf": NS["opf"]},
        )

        identifier = etree.SubElement(
            metadata,
            "{{{}}}identifier".format(NS["dc"]),
            attrib={"id": "bookid"},
        )
        identifier.set("{{{}}}scheme".format(NS["opf"]), "UUID")
        identifier.text = f"urn:uuid:{uuid.uuid4()}"

        title = etree.SubElement(metadata, "{{{}}}title".format(NS["dc"]))
        title.text = self.book.title

        creator = etree.SubElement(metadata, "{{{}}}creator".format(NS["dc"]))
        creator.set("{{{}}}role".format(NS["opf"]), "aut")
        creator.text = self.anthology.creator()

        language = etree.SubElement(metadata, "{{{}}}language".format(NS["dc"]))
        language.text = self.anthology.language()

        # Point to NCX (id "ncx") so EPUB2 readers recognise the TOC.
        # The spine references it by id below.
        manifest_el = etree.SubElement(package, "{{{}}}manifest".format(NS["opf"]))
        etree.SubElement(
            manifest_el,
            "{{{}}}item".format(NS["opf"]),
            attrib={
                "id": "ncx",
                "href": "toc.ncx",
                "media-type": "application/x-dtbncx+xml",
            },
        )
        for item in self.book.manifest_items:
            etree.SubElement(
                manifest_el,
                "{{{}}}item".format(NS["opf"]),
                attrib={
                    "id": item.item_id,
                    "href": strip_prefix(item.href, self.book.prefix),
                    "media-type": item.media_type,
                },
            )

        spine_el = etree.SubElement(package, "{{{}}}spine".format(NS["opf"]), attrib={"toc": "ncx"})
        for idref in self.book.spine_idrefs:
            etree.SubElement(
                spine_el,
                "{{{}}}itemref".format(NS["opf"]),
                attrib={"idref": idref, "linear": "yes"},
            )

        return etree.tostring(package, xml_declaration=True, encoding="utf-8", pretty_print=True)

    def _make_ncx(self) -> bytes:
        ncx = etree.Element(
            "{{{}}}ncx".format(NS["ncx"]),
            nsmap={None: NS["ncx"]},
            attrib={"version": "2005-1"},
        )

        head = etree.SubElement(ncx, "{{{}}}head".format(NS["ncx"]))
        for name, value in (
            ("dtb:uid", f"urn:uuid:{uuid.uuid4()}"),
            ("dtb:depth", "2"),
            ("dtb:totalPageCount", "0"),
            ("dtb:maxPageNumber", "0"),
        ):
            etree.SubElement(
                head,
                "{{{}}}meta".format(NS["ncx"]),
                attrib={"name": name, "content": value},
            )

        doc_title = etree.SubElement(ncx, "{{{}}}docTitle".format(NS["ncx"]))
        etree.SubElement(doc_title, "{{{}}}text".format(NS["ncx"])).text = self.book.title

        nav_map = etree.SubElement(ncx, "{{{}}}navMap".format(NS["ncx"]))
        play_order = [1]  # mutable counter across the nested walker

        def walk(src_point: etree._Element, target_parent: etree._Element) -> None:
            """Copy a navPoint, rewriting src to strip the book prefix."""
            label_text_el = src_point.find("ncx:navLabel/ncx:text", NS)
            content_el = src_point.find("ncx:content", NS)
            if label_text_el is None or content_el is None:
                return
            label = (label_text_el.text or "").strip()
            src = content_el.get("src")
            try:
                new_src = strip_prefix(src, self.book.prefix)
            except ValueError:
                # Reference to something outside the book: skip entry.
                return

            new_point = etree.SubElement(
                target_parent,
                "{{{}}}navPoint".format(NS["ncx"]),
                attrib={
                    "id": f"navpt_{play_order[0]}",
                    "playOrder": str(play_order[0]),
                },
            )
            play_order[0] += 1
            new_label = etree.SubElement(new_point, "{{{}}}navLabel".format(NS["ncx"]))
            etree.SubElement(new_label, "{{{}}}text".format(NS["ncx"])).text = label
            etree.SubElement(
                new_point,
                "{{{}}}content".format(NS["ncx"]),
                attrib={"src": new_src},
            )
            for child in src_point.findall("ncx:navPoint", NS):
                walk(child, new_point)

        # Skip the book's own wrapping navPoint (its label = book title) and
        # promote its children to top level. If it has no children, keep it.
        children = self.book.nav_subtree.findall("ncx:navPoint", NS)
        if children:
            for child in children:
                walk(child, nav_map)
        else:
            walk(self.book.nav_subtree, nav_map)

        return etree.tostring(ncx, xml_declaration=True, encoding="utf-8", pretty_print=True)

    def _copy_book_files(self, zf: zipfile.ZipFile) -> None:
        src_zip = self.anthology.zip
        prefix = self.book.prefix
        # Copy every archive member whose path falls under the book's prefix.
        for name in src_zip.namelist():
            if name.endswith("/"):
                continue
            if not name.startswith(prefix):
                continue
            new_name = name[len(prefix) :]
            with src_zip.open(name) as fh:
                zf.writestr(new_name, fh.read(), compress_type=zipfile.ZIP_DEFLATED)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def cmd_list(anthology: Anthology) -> None:
    print(f"Anthology: {anthology.epub_path}")
    print(f"Creator:   {anthology.creator()}")
    print(f"Books:     {len(anthology.books)}")
    print()
    print(f"{'#':>3}  {'Files':>5}  {'Spine':>5}  {'Prefix':<8}  Title")
    print(f"{'-' * 3}  {'-' * 5}  {'-' * 5}  {'-' * 8}  {'-' * 40}")
    for book in anthology.books:
        print(
            f"{book.index:>3}  "
            f"{len(book.manifest_items):>5}  "
            f"{len(book.spine_idrefs):>5}  "
            f"{book.prefix:<8}  "
            f"{book.title}"
        )


def cmd_extract(anthology: Anthology, index: int, out_dir: Path) -> Path:
    matches = [b for b in anthology.books if b.index == index]
    if not matches:
        raise SystemExit(f"No book with index {index}. Valid: 1..{len(anthology.books)}")
    book = matches[0]
    author_slug = slugify(anthology.creator())
    title_slug = slugify(book.title)
    filename = f"{title_slug}-{author_slug}.epub"
    out_path = out_dir / filename

    writer = BookWriter(anthology, book)
    writer.write(out_path)

    print(f"Book:      {book.title}")
    print(f"Prefix:    {book.prefix}")
    print(f"Files:     {len(book.manifest_items)} manifest, {len(book.spine_idrefs)} in spine")
    print(f"Output:    {out_path}")
    print(f"Size:      {out_path.stat().st_size:,} bytes")
    return out_path


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Split a merged EPUB anthology into individual books.",
    )
    parser.add_argument("epub", type=Path, help="Path to the merged EPUB")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--list",
        action="store_true",
        help="List books contained in the anthology and exit.",
    )
    group.add_argument(
        "--book",
        type=int,
        metavar="N",
        help="Extract book number N (see --list for numbering).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("books/extracted"),
        help="Output directory (default: books/extracted).",
    )
    args = parser.parse_args(argv)

    if not args.epub.is_file():
        parser.error(f"File not found: {args.epub}")

    anthology = Anthology(args.epub)

    if args.list:
        cmd_list(anthology)
    else:
        cmd_extract(anthology, args.book, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
