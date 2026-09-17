"""Cover image extraction, replacement, and AI translation for EPUB files.

Two modes:
  1. Replace — swap the cover image with a user-provided file.
  2. Translate — use an AI image model to translate text on the cover.

Both modes work by locating the cover image inside the EPUB zip archive,
then either replacing it with new bytes or sending it to an image model
for text translation.
"""

from __future__ import annotations

import re
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from lxml import etree

from .provider import ImageGenerationResult, OpenRouterProvider

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class CoverInfo:
    """Metadata about the cover image found inside an EPUB."""

    archive_path: str  # path inside the zip (e.g. "OEBPS/images/cover.jpg")
    media_type: str  # MIME type (e.g. "image/jpeg")
    raw_bytes: bytes  # original image bytes
    manifest_href: str = ""  # raw href from OPF manifest (e.g. "../images/cover.jpg")


# ---------------------------------------------------------------------------
# EPUB cover detection
# ---------------------------------------------------------------------------

# XML namespaces used in OPF files.
_NS = {
    "opf": "http://www.idpf.org/2007/opf",
    "dc": "http://purl.org/dc/elements/1.1/",
}


def find_cover_in_epub(epub_path: Path) -> CoverInfo | None:
    """Locate the cover image inside an EPUB file.

    Strategy (tries in order):
      1. EPUB2: <meta name="cover" content="ITEM_ID"/> in OPF → manifest item.
      2. EPUB3: manifest item with properties="cover-image".
      3. Heuristic: first <item> with media-type image/* whose id or href
         contains "cover".

    Returns None if no cover image can be identified.
    """
    with zipfile.ZipFile(epub_path, "r") as zf:
        opf_path = _find_opf_path(zf)
        if opf_path is None:
            return None

        opf_bytes = zf.read(opf_path)
        opf_tree = etree.fromstring(opf_bytes)

        # Resolve paths relative to OPF location
        opf_dir = "/".join(opf_path.split("/")[:-1])

        # Strategy 1: EPUB2 meta cover
        cover_id = None
        for meta in opf_tree.findall(".//opf:meta[@name='cover']", _NS):
            cover_id = meta.get("content")
            break

        manifest_items = opf_tree.findall(".//opf:manifest/opf:item", _NS)

        if cover_id:
            for item in manifest_items:
                if item.get("id") == cover_id:
                    return _extract_cover_item(zf, item, opf_dir)

        # Strategy 2: EPUB3 properties="cover-image"
        for item in manifest_items:
            props = item.get("properties", "")
            if "cover-image" in props:
                return _extract_cover_item(zf, item, opf_dir)

        # Strategy 3: heuristic — first image item with "cover" in id/href
        for item in manifest_items:
            media_type = item.get("media-type", "")
            if not media_type.startswith("image/"):
                continue
            item_id = (item.get("id") or "").lower()
            href = (item.get("href") or "").lower()
            if "cover" in item_id or "cover" in href:
                return _extract_cover_item(zf, item, opf_dir)

    return None


def _find_opf_path(zf: zipfile.ZipFile) -> str | None:
    """Find the OPF file path from META-INF/container.xml."""
    try:
        container_bytes = zf.read("META-INF/container.xml")
    except KeyError:
        return None

    container = etree.fromstring(container_bytes)
    ns = {"c": "urn:oasis:names:tc:opendocument:xmlns:container"}
    rootfile = container.find(".//c:rootfile", ns)
    if rootfile is not None:
        return rootfile.get("full-path")
    return None


def _extract_cover_item(
    zf: zipfile.ZipFile,
    item: etree._Element,
    opf_dir: str,
) -> CoverInfo | None:
    """Read the cover image bytes from the zip given a manifest <item>."""
    href = item.get("href", "")
    media_type = item.get("media-type", "image/jpeg")

    # Resolve relative to the OPF directory, so "../images/cover.jpg" lands
    # where it should.
    archive_path = _normalize_zip_path(opf_dir, href)

    # Try the normalized path, then the raw href as fallback
    archive_names = set(zf.namelist())
    for candidate in (archive_path, href):
        if candidate in archive_names:
            raw = zf.read(candidate)
            return CoverInfo(
                archive_path=candidate,
                media_type=media_type,
                raw_bytes=raw,
                manifest_href=href,
            )

    return None


# ---------------------------------------------------------------------------
# Package repair: cover declaration, declared page size, title and author
# ---------------------------------------------------------------------------

# Manifest media types that can hold a page displaying the cover.
_PAGE_MEDIA_TYPES = frozenset({"application/xhtml+xml", "text/html"})

_XLINK_NS = "http://www.w3.org/1999/xlink"


@dataclass
class CoverFixes:
    """What a cover write changed in the package, besides the image bytes.

    Each field stays None when that particular thing was already in order,
    so callers can report only what they actually touched.
    """

    media_type: str | None = None  # manifest media-type rewritten to this
    cover_meta: str | None = None  # <meta name="cover"> now names this item id
    cover_property: str | None = None  # properties="cover-image" added to this id
    guide_href: str | None = None  # <guide> cover reference added, pointing here
    page_path: str | None = None  # the page that displays the cover, if found
    page_size: tuple[int, int] | None = None  # size that page now declares
    title: str | None = None  # dc:title set to this
    author: str | None = None  # dc:creator set to this

    def any_change(self) -> bool:
        """True when the package was actually rewritten."""
        return any(
            value is not None
            for value in (
                self.media_type,
                self.cover_meta,
                self.cover_property,
                self.guide_href,
                self.page_size,
                self.title,
                self.author,
            )
        )


def _parse_xml(data: bytes) -> etree._Element | None:
    """Parse a package or page file, tolerating the usual small defects.

    OceanofPDF and calibre builds vary; recover mode keeps a stray entity
    from costing us the whole file. Returns None when nothing usable is left.
    """
    parser = etree.XMLParser(recover=True, resolve_entities=False)
    try:
        return etree.fromstring(data, parser)
    except etree.XMLSyntaxError:
        return None


def _normalize_zip_path(base_dir: str, href: str) -> str:
    """Resolve an href written inside the book to a path in the zip archive.

    Drops any fragment or query, strips a leading slash (some EPUBs write
    absolute-looking hrefs), then collapses "." and ".." segments.
    """
    clean = href.split("#", 1)[0].split("?", 1)[0].lstrip("/")
    if not clean:
        return ""
    resolved = PurePosixPath(base_dir, clean) if base_dir else PurePosixPath(clean)
    parts: list[str] = []
    for part in resolved.parts:
        if part == "..":
            if parts:
                parts.pop()
        elif part != ".":
            parts.append(part)
    return "/".join(parts)


# ---------------------------------------------------------------------------
# Image size, read straight from the file header
# ---------------------------------------------------------------------------


def _image_dimensions(data: bytes) -> tuple[int, int] | None:
    """Read pixel width and height from an image file header.

    Covers the formats `_detect_image_mime_from_bytes` recognizes, except
    TIFF, whose size needs a full directory walk and which no image model we
    use returns. Returns None when the size cannot be read — the caller then
    leaves the markup alone rather than write a size that may be wrong.
    """
    mime = _detect_image_mime_from_bytes(data)
    if mime == "image/jpeg":
        return _jpeg_dimensions(data)
    if mime == "image/png":
        # The IHDR chunk always comes first: 8 signature bytes, 4 length,
        # 4 chunk type, then width and height as big-endian 32-bit numbers.
        if len(data) < 24:
            return None
        return _checked_size(int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big"))
    if mime == "image/gif":
        if len(data) < 10:
            return None
        return _checked_size(
            int.from_bytes(data[6:8], "little"), int.from_bytes(data[8:10], "little")
        )
    if mime == "image/bmp":
        if len(data) < 26:
            return None
        # Height is signed: negative means the rows are stored top-down.
        return _checked_size(
            abs(int.from_bytes(data[18:22], "little", signed=True)),
            abs(int.from_bytes(data[22:26], "little", signed=True)),
        )
    if mime == "image/webp":
        return _webp_dimensions(data)
    return None


def _checked_size(width: int, height: int) -> tuple[int, int] | None:
    """Accept a size only if both sides are positive."""
    return (width, height) if width > 0 and height > 0 else None


def _jpeg_dimensions(data: bytes) -> tuple[int, int] | None:
    """Walk JPEG marker segments until a start-of-frame carries the size."""
    position = 2  # skip the start-of-image marker
    end = len(data)
    while position + 3 < end:
        if data[position] != 0xFF:
            position += 1
            continue
        marker = data[position + 1]
        # Padding, restart markers and the standalone ones carry no length.
        if marker == 0xFF:
            position += 1
            continue
        if marker in (0x01, 0xD8) or 0xD0 <= marker <= 0xD7:
            position += 2
            continue
        if position + 4 > end:
            return None
        segment_length = int.from_bytes(data[position + 2 : position + 4], "big")
        # Start-of-frame markers hold the size. 0xC4, 0xC8 and 0xCC share the
        # range but describe Huffman and arithmetic coding tables instead.
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            if position + 9 > end:
                return None
            height = int.from_bytes(data[position + 5 : position + 7], "big")
            width = int.from_bytes(data[position + 7 : position + 9], "big")
            return _checked_size(width, height)
        if segment_length < 2:
            return None
        position += 2 + segment_length
    return None


def _webp_dimensions(data: bytes) -> tuple[int, int] | None:
    """Read the canvas size from a WebP file, in any of its three flavours."""
    if len(data) < 30:
        return None
    chunk = data[12:16]
    if chunk == b"VP8 ":
        # Lossy: 3-byte frame tag and 3-byte sync code, then 14-bit sizes.
        return _checked_size(
            int.from_bytes(data[26:28], "little") & 0x3FFF,
            int.from_bytes(data[28:30], "little") & 0x3FFF,
        )
    if chunk == b"VP8L":
        # Lossless: one signature byte, then width-1 and height-1 as 14 bits.
        bits = int.from_bytes(data[21:25], "little")
        return _checked_size((bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1)
    if chunk == b"VP8X":
        # Extended: 4 flag bytes, then canvas width-1 and height-1 as 24 bits.
        return _checked_size(
            int.from_bytes(data[24:27], "little") + 1,
            int.from_bytes(data[27:30], "little") + 1,
        )
    return None


# ---------------------------------------------------------------------------
# The page that displays the cover
# ---------------------------------------------------------------------------


def _image_href(element: etree._Element) -> str | None:
    """The image an <image> or <img> element points at, if any."""
    if not isinstance(element.tag, str):
        return None
    name = etree.QName(element).localname
    if name == "image":
        return element.get(f"{{{_XLINK_NS}}}href") or element.get("href")
    if name == "img":
        return element.get("src")
    return None


def _cover_image_nodes(
    doc: etree._Element, page_dir: str, cover_archive_path: str
) -> list[etree._Element]:
    """Elements in a page that display the given cover image."""
    nodes: list[etree._Element] = []
    for element in doc.iter():
        href = _image_href(element)
        if href and _normalize_zip_path(page_dir, href) == cover_archive_path:
            nodes.append(element)
    return nodes


def _find_cover_page(
    zf: zipfile.ZipFile,
    opf_tree: etree._Element,
    opf_dir: str,
    cover: CoverInfo,
) -> tuple[str, str] | None:
    """Locate the page that displays the cover image.

    Returns the page's path inside the zip together with its href relative
    to the package file (which is what a <guide> reference needs), or None
    when no page shows the cover.
    """
    names = set(zf.namelist())
    for item in opf_tree.findall(".//opf:manifest/opf:item", _NS):
        if item.get("media-type", "") not in _PAGE_MEDIA_TYPES:
            continue
        href = item.get("href", "")
        path = _normalize_zip_path(opf_dir, href)
        if path not in names:
            continue
        doc = _parse_xml(zf.read(path))
        if doc is None:
            continue
        page_dir = "/".join(path.split("/")[:-1])
        if _cover_image_nodes(doc, page_dir, cover.archive_path):
            return path, href
    return None


def _resize_cover_page(
    page_bytes: bytes,
    page_dir: str,
    cover_archive_path: str,
    width: int,
    height: int,
) -> bytes | None:
    """Rewrite the cover size that the cover page declares.

    A calibre-built cover page wraps the image in an <svg> whose viewBox and
    <image> carry the original file's pixel size. Leave those behind after
    the cover is replaced and the reader fits the new image into the old box:
    with different proportions that means blank strips around the cover.

    Returns the new page bytes, or None when nothing needed changing.
    """
    doc = _parse_xml(page_bytes)
    if doc is None:
        return None
    nodes = _cover_image_nodes(doc, page_dir, cover_archive_path)
    if not nodes:
        return None

    changed = False
    for node in nodes:
        changed |= _set_pixel_size(node, width, height)
        svg = _enclosing_svg(node)
        if svg is not None:
            changed |= _set_view_box(svg, width, height)
            changed |= _set_pixel_size(svg, width, height)
    if not changed:
        return None
    return etree.tostring(doc.getroottree(), xml_declaration=True, encoding="utf-8")


def _enclosing_svg(element: etree._Element) -> etree._Element | None:
    """The nearest <svg> the element sits in, if any."""
    parent = element.getparent()
    while parent is not None:
        if isinstance(parent.tag, str) and etree.QName(parent).localname == "svg":
            return parent
        parent = parent.getparent()
    return None


def _set_pixel_size(element: etree._Element, width: int, height: int) -> bool:
    """Set width and height, but only where they are already plain numbers.

    A percentage or a value with units (100%, 12em) is a layout choice the
    book made; only a bare pixel count describes the image itself.
    """
    changed = False
    for name, value in (("width", width), ("height", height)):
        current = (element.get(name) or "").strip()
        if not current.isdigit() or current == str(value):
            continue
        element.set(name, str(value))
        changed = True
    return changed


def _set_view_box(svg: etree._Element, width: int, height: int) -> bool:
    """Replace the size half of an svg viewBox, keeping its origin."""
    current = (svg.get("viewBox") or "").strip()
    if not current:
        return False
    numbers = re.split(r"[\s,]+", current)
    if len(numbers) != 4:
        return False
    updated = [numbers[0], numbers[1], str(width), str(height)]
    if updated == numbers:
        return False
    svg.set("viewBox", " ".join(updated))
    return True


# ---------------------------------------------------------------------------
# Package file rewriting
# ---------------------------------------------------------------------------


def _append_child(parent: etree._Element, tag: str) -> etree._Element:
    """Append a child in the package namespace, keeping the indentation.

    The element is created with the package namespace as its default, so
    lxml writes it without a prefix. That matters: a package whose metadata
    element itself carries a prefix would otherwise get an <opf:meta>, and
    some readers look for a literal <meta name="cover">.
    """
    siblings = list(parent)
    separator = siblings[0].tail if len(siblings) >= 2 else None
    closing = siblings[-1].tail if siblings else None

    child = etree.SubElement(parent, tag, nsmap={None: _NS["opf"]})
    if siblings:
        siblings[-1].tail = separator or closing
        child.tail = closing
    return child


def _apply_opf_fixes(
    opf_bytes: bytes,
    cover: CoverInfo,
    *,
    new_media_type: str | None,
    cover_page_href: str | None,
    title: str | None,
    author: str | None,
    fixes: CoverFixes,
) -> bytes | None:
    """Rewrite the package file: cover declaration, media type, title, author.

    Records what it changed in `fixes`. Returns the new bytes, or None when
    the package was already in order.

    Raises:
        RuntimeError: If the package file cannot be read, or the cover's
            manifest item is missing while a new media type has to be
            written — writing the image alone would leave the package
            describing the wrong format.
    """
    tree = _parse_xml(opf_bytes)
    if tree is None:
        raise RuntimeError("Cannot read the package file (OPF) to update the cover.")

    cover_item: etree._Element | None = None
    for item in tree.findall(".//opf:manifest/opf:item", _NS):
        if item.get("href", "") == cover.manifest_href:
            cover_item = item
            break

    if cover_item is None and new_media_type:
        raise RuntimeError(
            f"Failed to update OPF manifest: no item with "
            f"href='{cover.manifest_href}' found. "
            f"Cannot safely update media-type for the cover image."
        )

    changed = False
    if cover_item is not None and new_media_type and cover_item.get("media-type") != new_media_type:
        cover_item.set("media-type", new_media_type)
        fixes.media_type = new_media_type
        changed = True

    metadata = tree.find(".//opf:metadata", _NS)
    epub3 = (tree.get("version") or "").startswith("3")

    # Without a cover declaration a reader cannot tell which image is the
    # cover, and draws a placeholder from the title and author instead.
    # EPUB3 declares it on the manifest item, EPUB2 in metadata and guide.
    if cover_item is not None:
        if epub3:
            changed |= _declare_cover_epub3(cover_item, fixes)
        else:
            changed |= _declare_cover_epub2(metadata, cover_item, fixes)
            changed |= _add_guide_reference(tree, cover_page_href, fixes)

    if metadata is not None:
        changed |= _set_metadata_text(metadata, "dc:title", title, fixes, "title")
        changed |= _set_metadata_text(metadata, "dc:creator", author, fixes, "author")

    if not changed:
        return None
    return etree.tostring(tree.getroottree(), xml_declaration=True, encoding="utf-8")


def _declare_cover_epub3(cover_item: etree._Element, fixes: CoverFixes) -> bool:
    """Mark the manifest item as the cover image, EPUB3 style."""
    properties = (cover_item.get("properties") or "").split()
    if "cover-image" in properties:
        return False
    properties.append("cover-image")
    cover_item.set("properties", " ".join(properties))
    fixes.cover_property = cover_item.get("id") or ""
    return True


def _declare_cover_epub2(
    metadata: etree._Element | None,
    cover_item: etree._Element,
    fixes: CoverFixes,
) -> bool:
    """Point <meta name="cover"> at the cover's manifest item, EPUB2 style."""
    cover_id = cover_item.get("id")
    if metadata is None or not cover_id:
        return False

    existing = None
    for element in metadata.findall("opf:meta", _NS):
        if element.get("name") == "cover":
            existing = element
            break

    if existing is None:
        meta = _append_child(metadata, f"{{{_NS['opf']}}}meta")
        meta.set("name", "cover")
        meta.set("content", cover_id)
    elif existing.get("content") != cover_id:
        existing.set("content", cover_id)
    else:
        return False

    fixes.cover_meta = cover_id
    return True


def _add_guide_reference(
    tree: etree._Element, cover_page_href: str | None, fixes: CoverFixes
) -> bool:
    """Add the EPUB2 guide reference that names the cover page."""
    if not cover_page_href:
        return False
    guide = tree.find("opf:guide", _NS)
    if guide is not None and guide.find("opf:reference[@type='cover']", _NS) is not None:
        return False
    if guide is None:
        guide = _append_child(tree, f"{{{_NS['opf']}}}guide")
    reference = _append_child(guide, f"{{{_NS['opf']}}}reference")
    reference.set("type", "cover")
    reference.set("title", "Cover")
    reference.set("href", cover_page_href)
    fixes.guide_href = cover_page_href
    return True


def _set_metadata_text(
    metadata: etree._Element,
    path: str,
    value: str | None,
    fixes: CoverFixes,
    field: str,
) -> bool:
    """Set the text of a Dublin Core metadata element, if asked and needed."""
    if not value:
        return False
    element = metadata.find(path, _NS)
    if element is None or (element.text or "") == value:
        return False
    element.text = value
    setattr(fixes, field, value)
    return True


# ---------------------------------------------------------------------------
# Writing the output EPUB
# ---------------------------------------------------------------------------


def _write_cover_epub(
    source_epub: Path,
    dest_epub: Path,
    cover: CoverInfo,
    new_image: bytes | None,
    *,
    new_media_type: str | None = None,
    title: str | None = None,
    author: str | None = None,
) -> CoverFixes:
    """Write a new EPUB with the cover image and its declaration in order.

    Besides swapping the image bytes this repairs what a reader needs in
    order to show the cover at all: the cover declaration in the package
    file, the guide reference naming the cover page, and the pixel size that
    page declares. The source file is never modified.

    Args:
        source_epub: Path to the original EPUB.
        dest_epub: Path for the output EPUB. Must differ from source_epub.
        cover: The cover found in source_epub.
        new_image: Replacement image bytes, or None to keep the current
            cover image and only repair its declaration.
        new_media_type: MIME type to write into the manifest, when the new
            image is a different format than the old one.
        title: Value for dc:title, or None to leave it alone.
        author: Value for dc:creator, or None to leave it alone.

    Returns:
        What was changed besides the image bytes.

    Raises:
        ValueError: If source and destination paths are the same.
        RuntimeError: If the package file is missing or unreadable, or the
            cover's manifest item is gone while the media type must change.
    """
    # Safety: prevent overwriting the source file
    if source_epub.resolve() == dest_epub.resolve():
        raise ValueError(
            "Source and destination EPUB paths are the same. "
            "Use a different --out path to avoid overwriting the original."
        )

    image_bytes = new_image if new_image is not None else cover.raw_bytes
    fixes = CoverFixes()
    replacements: dict[str, bytes] = {}

    with zipfile.ZipFile(source_epub, "r") as zf:
        opf_path = _find_opf_path(zf)
        if opf_path is None:
            raise RuntimeError("Cannot update the cover: no package file (OPF) found in the EPUB.")
        opf_bytes = zf.read(opf_path)
        opf_tree = _parse_xml(opf_bytes)
        if opf_tree is None:
            raise RuntimeError("Cannot read the package file (OPF) to update the cover.")
        opf_dir = "/".join(opf_path.split("/")[:-1])

        page = _find_cover_page(zf, opf_tree, opf_dir, cover)
        cover_page_href: str | None = None
        if page is not None:
            page_path, cover_page_href = page
            fixes.page_path = page_path
            size = _image_dimensions(image_bytes)
            if size is not None:
                page_dir = "/".join(page_path.split("/")[:-1])
                resized = _resize_cover_page(
                    zf.read(page_path), page_dir, cover.archive_path, size[0], size[1]
                )
                if resized is not None:
                    replacements[page_path] = resized
                    fixes.page_size = size

        updated_opf = _apply_opf_fixes(
            opf_bytes,
            cover,
            new_media_type=new_media_type,
            cover_page_href=cover_page_href,
            title=title,
            author=author,
            fixes=fixes,
        )
        if updated_opf is not None:
            replacements[opf_path] = updated_opf

    if new_image is not None:
        replacements[cover.archive_path] = new_image

    dest_epub.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(source_epub, "r") as src, zipfile.ZipFile(dest_epub, "w") as dst:
        # mimetype first, uncompressed (EPUB spec)
        dst.writestr(
            zipfile.ZipInfo("mimetype"),
            "application/epub+zip",
            compress_type=zipfile.ZIP_STORED,
        )
        for info in src.infolist():
            if info.filename == "mimetype":
                continue
            if info.filename in replacements:
                dst.writestr(
                    info.filename,
                    replacements[info.filename],
                    compress_type=zipfile.ZIP_DEFLATED,
                )
            else:
                with src.open(info) as fh:
                    dst.writestr(info, fh.read())

    return fixes


# ---------------------------------------------------------------------------
# Cover replacement
# ---------------------------------------------------------------------------


def replace_cover(
    source_epub: Path,
    dest_epub: Path,
    new_image: bytes,
    new_media_type: str | None = None,
) -> CoverInfo:
    """Replace the cover image in an EPUB with new image bytes.

    Creates a new EPUB at dest_epub with the cover image swapped, its
    declaration repaired if the source lacked one, and the size declared by
    the cover page brought in line with the new image. The original file is
    not modified.

    Args:
        source_epub: Path to the original EPUB.
        dest_epub: Path for the output EPUB. Must differ from source_epub.
        new_image: Raw bytes of the new cover image.
        new_media_type: MIME type of the new image (auto-detected if None).
            Must match the original cover's media type, or be compatible
            (e.g. both are image/jpeg). If formats differ, raises ValueError.

    Returns:
        CoverInfo of the original cover that was replaced.

    Raises:
        ValueError: If no cover image is found in the EPUB, if the
            new image format doesn't match the original, or if
            source and destination paths are the same.
    """
    # Safety: prevent overwriting the source file
    if source_epub.resolve() == dest_epub.resolve():
        raise ValueError(
            "Source and destination EPUB paths are the same. "
            "Use a different --out path to avoid overwriting the original."
        )
    cover = find_cover_in_epub(source_epub)
    if cover is None:
        raise ValueError(f"No cover image found in {source_epub.name}. Cannot replace cover.")

    # Validate media type compatibility.
    # If new_media_type is provided and differs from the original, reject it.
    # This prevents OPF manifest declaring image/jpeg while bytes are PNG.
    if new_media_type and new_media_type != cover.media_type:
        # Allow compatible pairs (e.g. image/jpg vs image/jpeg)
        normalized_orig = _normalize_image_mime(cover.media_type)
        normalized_new = _normalize_image_mime(new_media_type)
        if normalized_orig != normalized_new:
            raise ValueError(
                f"Cover format mismatch: original is {cover.media_type}, "
                f"but replacement is {new_media_type}. "
                f"Use the same format as the original, or convert your image "
                f"to {cover.media_type} before replacing. "
                f"(Use `btrans cover translate` if you want the format changed too.)"
            )

    _write_cover_epub(source_epub, dest_epub, cover, new_image)
    return cover


def _normalize_image_mime(mime: str) -> str:
    """Normalize image MIME types for comparison."""
    mime = mime.lower().strip()
    # Common aliases
    if mime in ("image/jpg", "image/jpeg"):
        return "image/jpeg"
    return mime


def _detect_image_mime_from_bytes(data: bytes) -> str | None:
    """Detect image MIME type from file magic bytes.

    Returns the MIME type string or None if not a recognized image format.
    Supports JPEG, PNG, WebP, GIF, BMP, TIFF.
    """
    if not data or len(data) < 4:
        return None

    # JPEG: starts with FF D8 FF
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    # PNG: starts with 89 50 4E 47 0D 0A 1A 0A
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    # WebP: starts with RIFF....WEBP
    if data[:4] == b"RIFF" and len(data) >= 12 and data[8:12] == b"WEBP":
        return "image/webp"
    # GIF: starts with GIF87a or GIF89a
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    # BMP: starts with BM
    if data[:2] == b"BM":
        return "image/bmp"
    # TIFF: starts with II (little-endian) or MM (big-endian)
    if data[:4] in (b"II\x2a\x00", b"MM\x00\x2a"):
        return "image/tiff"

    return None


def replace_cover_from_file(
    source_epub: Path,
    dest_epub: Path,
    image_path: Path,
) -> CoverInfo:
    """Replace cover using an image file path.

    Detects MIME type from actual file bytes (not extension) to prevent
    writing mismatched content into the EPUB.

    Raises:
        FileNotFoundError: If image file doesn't exist.
        ValueError: If file is not a recognized image format, or if
            the detected format doesn't match the original cover format.
    """
    if not image_path.is_file():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    image_bytes = image_path.read_bytes()

    # Detect MIME from actual bytes, not filename extension
    detected_mime = _detect_image_mime_from_bytes(image_bytes)
    if detected_mime is None:
        raise ValueError(
            f"Cannot determine image format of {image_path.name}. "
            f"File does not appear to be a valid image (JPEG, PNG, WebP, GIF, BMP, TIFF)."
        )

    return replace_cover(source_epub, dest_epub, image_bytes, detected_mime)


# ---------------------------------------------------------------------------
# Repairing a book that already has its cover
# ---------------------------------------------------------------------------


def repair_cover(
    source_epub: Path,
    dest_epub: Path,
    *,
    title: str | None = None,
    author: str | None = None,
) -> tuple[CoverInfo, CoverFixes]:
    """Fix the cover declaration of a finished book without touching the image.

    Books built from a source that never declared its cover show up in a
    library as a generated placeholder, and a cover page still declaring the
    previous image's size leaves blank strips around the cover. This repairs
    both, and sets dc:title / dc:creator when they are given, so a reader
    that falls back to metadata shows the translated name.

    Args:
        source_epub: Path to the book to repair.
        dest_epub: Path for the output EPUB. Must differ from source_epub.
        title: Value for dc:title, or None to leave it alone.
        author: Value for dc:creator, or None to leave it alone.

    Returns:
        The cover that was found, and what the repair changed.

    Raises:
        ValueError: If no cover image is found, or if source and
            destination paths are the same.
        RuntimeError: If the package file is missing or unreadable.
    """
    cover = find_cover_in_epub(source_epub)
    if cover is None:
        raise ValueError(f"No cover image found in {source_epub.name}. Nothing to repair.")

    fixes = _write_cover_epub(source_epub, dest_epub, cover, None, title=title, author=author)
    return cover, fixes


# ---------------------------------------------------------------------------
# AI cover translation
# ---------------------------------------------------------------------------


def translate_cover(
    source_epub: Path,
    dest_epub: Path,
    provider: OpenRouterProvider,
    *,
    model: str,
    title_translation: str | None = None,
    author_name: str | None = None,
    target_lang: str = "Russian",
    prompt_template: str | None = None,
    aspect_ratio: str = "2:3",
    image_size: str = "1K",
) -> ImageGenerationResult:
    """Translate the cover image text using an AI image model.

    Extracts the cover from the EPUB, sends it to the image model with
    a prompt to replace text, then writes the result into a new EPUB with
    the cover declaration and the cover page's declared size in order.

    Args:
        source_epub: Path to the original EPUB.
        dest_epub: Path for the output EPUB.
        provider: OpenRouterProvider instance.
        model: Image model ID (e.g. google/gemini-3.1-flash-image-preview).
        title_translation: Translated book title (optional — if None, the
            model translates all text automatically). Also written into
            dc:title, so a reader falling back to metadata shows it.
        author_name: Author name (transliterated or original). Also written
            into dc:creator when given.
        target_lang: Target language name for the prompt.
        prompt_template: Custom prompt (uses default if None).
        aspect_ratio: Output aspect ratio.
        image_size: Output resolution.

    Returns:
        ImageGenerationResult from the model.

    Raises:
        ValueError: If no cover image is found.
        RuntimeError: If image generation fails.
    """
    cover = find_cover_in_epub(source_epub)
    if cover is None:
        raise ValueError(f"No cover image found in {source_epub.name}. Cannot translate cover.")

    # Build the prompt
    if prompt_template is None:
        prompt = _build_default_prompt(
            title_translation=title_translation,
            author_name=author_name,
            target_lang=target_lang,
        )
    else:
        prompt = prompt_template.format(
            title=title_translation,
            author=author_name or "",
            target_lang=target_lang,
        )

    # Call the image model
    result = provider.generate_image(
        model=model,
        prompt=prompt,
        input_image=cover.raw_bytes,
        input_mime_type=cover.media_type,
        aspect_ratio=aspect_ratio,
        image_size=image_size,
    )

    # Validate provider output before writing into EPUB
    if not result.image_bytes:
        raise RuntimeError(
            f"Image model returned empty image data. "
            f"Model: {result.model}, MIME: {result.mime_type}"
        )
    if not result.mime_type or not result.mime_type.startswith("image/"):
        raise RuntimeError(
            f"Image model returned invalid MIME type: {result.mime_type!r}. "
            f"Expected image/* format."
        )
    # Verify the bytes are actually a valid image
    detected_mime = _detect_image_mime_from_bytes(result.image_bytes)
    if detected_mime is None:
        raise RuntimeError(
            f"Image model returned bytes that are not a recognized image format. "
            f"Declared MIME: {result.mime_type}, first bytes: {result.image_bytes[:16]!r}"
        )

    # Write the generated image into a new EPUB. The image model may return
    # a different format than the original, so the manifest media-type has
    # to follow; the detected MIME (from the bytes) is the authoritative one.
    # A translated title or author also goes into the metadata, so a reader
    # that falls back to it shows the translated name instead of the original.
    generated_mime = detected_mime
    if _normalize_image_mime(generated_mime) == _normalize_image_mime(cover.media_type):
        generated_mime = None  # same format — leave the manifest alone

    _write_cover_epub(
        source_epub,
        dest_epub,
        cover,
        result.image_bytes,
        new_media_type=generated_mime,
        title=title_translation,
        author=author_name,
    )

    return result


def _build_default_prompt(
    title_translation: str | None,
    author_name: str | None,
    target_lang: str,
) -> str:
    """Build the default prompt for cover translation.

    If title_translation is provided, the model uses that exact text.
    If None, the model translates all text on the cover automatically.
    """
    if title_translation:
        # Explicit title provided — instruct model to use it
        title_line = f"\nTranslated title to use: '{title_translation}'"
    else:
        # No title — model translates everything automatically
        title_line = (
            "\nTranslate the book title yourself — choose a natural, "
            "idiomatic translation that sounds like a real published book title."
        )

    author_line = ""
    if author_name:
        author_line = f"\nAuthor name on cover: '{author_name}'"

    return (
        f"This is a book cover image. Your task is to translate ALL text "
        f"on this cover into {target_lang}.{title_line}{author_line}\n\n"
        f"IMPORTANT RULES:\n"
        f"- Replace ALL English text with the {target_lang} translation\n"
        f"- Keep the EXACT same visual style, colors, layout, and artwork\n"
        f"- Keep the same font style and weight (bold, italic, etc.)\n"
        f"- Keep the same text positioning and size proportions\n"
        f"- Do NOT change any non-text elements (images, patterns, colors)\n"
        f"- The result should look like a professionally designed "
        f"{target_lang}-language book cover\n"
        f"- Preserve the original aspect ratio and composition\n"
        f"- If the translated text is longer than the original, slightly "
        f"reduce font size to fit — do NOT overflow or crop text\n"
        f"- Subtitle, series name, and any other text should also be "
        f"translated into {target_lang}"
    )
