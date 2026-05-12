"""Cover image extraction, replacement, and AI translation for EPUB files.

Two modes:
  1. Replace — swap the cover image with a user-provided file.
  2. Translate — use an AI image model to translate text on the cover.

Both modes work by locating the cover image inside the EPUB zip archive,
then either replacing it with new bytes or sending it to an image model
for text translation.
"""

from __future__ import annotations

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

    # Resolve relative to OPF directory using proper POSIX path normalization.
    # This handles cases like "../images/cover.jpg" correctly.
    # Strip leading slash from href (some EPUBs use absolute-looking paths)
    clean_href = href.lstrip("/")

    if opf_dir:
        resolved = PurePosixPath(opf_dir, clean_href)
    else:
        resolved = PurePosixPath(clean_href)
    # Normalize: collapse ".." and "." segments
    parts: list[str] = []
    for part in resolved.parts:
        if part == "..":
            if parts:
                parts.pop()
        elif part != ".":
            parts.append(part)
    archive_path = "/".join(parts)

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
# Cover replacement
# ---------------------------------------------------------------------------


def replace_cover(
    source_epub: Path,
    dest_epub: Path,
    new_image: bytes,
    new_media_type: str | None = None,
) -> CoverInfo:
    """Replace the cover image in an EPUB with new image bytes.

    Creates a new EPUB at dest_epub with the cover image swapped.
    The original file is not modified.

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
        raise ValueError(
            f"No cover image found in {source_epub.name}. "
            "Cannot replace cover."
        )

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
                f"(OPF manifest rewriting is not yet implemented.)"
            )

    dest_epub.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(source_epub, "r") as src:
        with zipfile.ZipFile(dest_epub, "w") as dst:
            # mimetype first, uncompressed (EPUB spec)
            dst.writestr(
                zipfile.ZipInfo("mimetype"),
                "application/epub+zip",
                compress_type=zipfile.ZIP_STORED,
            )
            for info in src.infolist():
                if info.filename == "mimetype":
                    continue
                if info.filename == cover.archive_path:
                    # Replace cover image
                    dst.writestr(
                        info.filename,
                        new_image,
                        compress_type=zipfile.ZIP_DEFLATED,
                    )
                else:
                    with src.open(info) as fh:
                        dst.writestr(info, fh.read())

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
    a prompt to replace text, then writes the result into a new EPUB.

    Args:
        source_epub: Path to the original EPUB.
        dest_epub: Path for the output EPUB.
        provider: OpenRouterProvider instance.
        model: Image model ID (e.g. google/gemini-3.1-flash-image-preview).
        title_translation: Translated book title (optional — if None, the
            model translates all text automatically).
        author_name: Author name (transliterated or original).
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
        raise ValueError(
            f"No cover image found in {source_epub.name}. "
            "Cannot translate cover."
        )

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

    # Replace the cover in the EPUB with the generated image.
    # The AI model may return a different format than the original.
    # We handle this by updating the OPF manifest to match the new format.
    # Use the detected MIME (from bytes) as the authoritative format.
    generated_mime = detected_mime
    original_mime = cover.media_type

    if _normalize_image_mime(generated_mime) != _normalize_image_mime(original_mime):
        # Format differs — need to update OPF manifest media-type
        _replace_cover_with_manifest_update(
            source_epub, dest_epub, cover, result.image_bytes, generated_mime
        )
    else:
        # Same format — simple replacement, no manifest change needed
        replace_cover(source_epub, dest_epub, result.image_bytes, None)

    return result


def _replace_cover_with_manifest_update(
    source_epub: Path,
    dest_epub: Path,
    cover: CoverInfo,
    new_image: bytes,
    new_media_type: str,
) -> None:
    """Replace cover image AND update OPF manifest media-type to match.

    Used when the AI model returns a different image format than the original.
    Updates the <item media-type="..."> in the OPF to reflect the new format.

    Raises:
        ValueError: If source and destination are the same path.
        RuntimeError: If the OPF manifest item cannot be found/updated.
    """
    # Safety: prevent overwriting the source file
    if source_epub.resolve() == dest_epub.resolve():
        raise ValueError(
            "Source and destination EPUB paths are the same. "
            "Use a different --out path to avoid overwriting the original."
        )

    dest_epub.parent.mkdir(parents=True, exist_ok=True)

    # Find the OPF path
    with zipfile.ZipFile(source_epub, "r") as zf:
        opf_path = _find_opf_path(zf)

    if opf_path is None:
        raise RuntimeError(
            "Cannot update OPF manifest: no OPF file found in EPUB. "
            "The cover image format differs from the original but "
            "manifest cannot be updated."
        )

    with zipfile.ZipFile(source_epub, "r") as src:
        with zipfile.ZipFile(dest_epub, "w") as dst:
            # mimetype first, uncompressed (EPUB spec)
            dst.writestr(
                zipfile.ZipInfo("mimetype"),
                "application/epub+zip",
                compress_type=zipfile.ZIP_STORED,
            )
            for info in src.infolist():
                if info.filename == "mimetype":
                    continue
                if info.filename == cover.archive_path:
                    # Replace cover image bytes
                    dst.writestr(
                        info.filename,
                        new_image,
                        compress_type=zipfile.ZIP_DEFLATED,
                    )
                elif info.filename == opf_path:
                    # Update OPF manifest media-type for the cover item.
                    # Uses manifest_href for exact matching.
                    opf_bytes = src.read(info.filename)
                    updated_opf = _update_opf_cover_media_type(
                        opf_bytes, cover.manifest_href, new_media_type
                    )
                    dst.writestr(
                        info.filename,
                        updated_opf,
                        compress_type=zipfile.ZIP_DEFLATED,
                    )
                else:
                    with src.open(info) as fh:
                        dst.writestr(info, fh.read())


def _update_opf_cover_media_type(
    opf_bytes: bytes,
    cover_manifest_href: str,
    new_media_type: str,
) -> bytes:
    """Update the media-type attribute of the cover item in OPF XML.

    Matches the manifest item by exact href comparison with the stored
    manifest_href from CoverInfo. Raises RuntimeError if no matching
    item is found (to avoid silently producing invalid EPUBs).

    Args:
        opf_bytes: Raw OPF XML bytes.
        cover_manifest_href: The exact href attribute from the cover's
            manifest item (as stored in CoverInfo.manifest_href).
        new_media_type: The new MIME type to set.

    Returns:
        Updated OPF XML bytes.

    Raises:
        RuntimeError: If no manifest item with matching href is found.
    """
    tree = etree.fromstring(opf_bytes)

    manifest_items = tree.findall(".//{http://www.idpf.org/2007/opf}item")
    updated = False
    for item in manifest_items:
        href = item.get("href", "")
        if href == cover_manifest_href:
            item.set("media-type", new_media_type)
            updated = True
            break

    if not updated:
        raise RuntimeError(
            f"Failed to update OPF manifest: no item with "
            f"href='{cover_manifest_href}' found. "
            f"Cannot safely update media-type for the cover image."
        )

    return etree.tostring(tree, xml_declaration=True, encoding="utf-8")


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
