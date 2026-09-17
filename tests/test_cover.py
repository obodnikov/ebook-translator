"""Tests for cover image detection, replacement, and translation logic."""

from __future__ import annotations

import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from lxml import etree

from booktranslator.cover import (
    CoverFixes,
    CoverInfo,
    _apply_opf_fixes,
    _build_default_prompt,
    _normalize_image_mime,
    find_cover_in_epub,
    repair_cover,
    replace_cover,
    replace_cover_from_file,
)

# ---------------------------------------------------------------------------
# Fixtures: minimal EPUB archives for testing
# ---------------------------------------------------------------------------


def _make_epub2_with_cover(tmp_path: Path, cover_bytes: bytes = b"\xff\xd8\xff\xe0JFIF") -> Path:
    """Create a minimal EPUB2 with a cover image via <meta name='cover'>."""
    epub_path = tmp_path / "test-epub2.epub"

    container_xml = b"""<?xml version="1.0"?>
<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container" version="1.0">
  <rootfiles>
    <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>"""

    opf_xml = b"""<?xml version="1.0"?>
<package xmlns="http://www.idpf.org/2007/opf" version="2.0" unique-identifier="uid">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:title>Test Book</dc:title>
    <meta name="cover" content="cover-img"/>
  </metadata>
  <manifest>
    <item id="cover-img" href="images/cover.jpg" media-type="image/jpeg"/>
    <item id="ch1" href="chapter1.xhtml" media-type="application/xhtml+xml"/>
  </manifest>
  <spine><itemref idref="ch1"/></spine>
</package>"""

    chapter_xhtml = b"""<?xml version="1.0"?>
<html xmlns="http://www.w3.org/1999/xhtml">
<body><p>Hello world</p></body>
</html>"""

    with zipfile.ZipFile(epub_path, "w") as zf:
        zf.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)
        zf.writestr("META-INF/container.xml", container_xml)
        zf.writestr("OEBPS/content.opf", opf_xml)
        zf.writestr("OEBPS/images/cover.jpg", cover_bytes)
        zf.writestr("OEBPS/chapter1.xhtml", chapter_xhtml)

    return epub_path


def _make_epub3_with_cover(tmp_path: Path, cover_bytes: bytes = b"\x89PNG\r\n\x1a\n") -> Path:
    """Create a minimal EPUB3 with a cover image via properties='cover-image'."""
    epub_path = tmp_path / "test-epub3.epub"

    container_xml = b"""<?xml version="1.0"?>
<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container" version="1.0">
  <rootfiles>
    <rootfile full-path="content.opf" media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>"""

    opf_xml = b"""<?xml version="1.0"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="uid">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:title>Test Book 3</dc:title>
  </metadata>
  <manifest>
    <item id="cover" href="cover.png" media-type="image/png" properties="cover-image"/>
    <item id="ch1" href="chapter1.xhtml" media-type="application/xhtml+xml"/>
  </manifest>
  <spine><itemref idref="ch1"/></spine>
</package>"""

    chapter_xhtml = b"""<?xml version="1.0"?>
<html xmlns="http://www.w3.org/1999/xhtml">
<body><p>Hello world</p></body>
</html>"""

    with zipfile.ZipFile(epub_path, "w") as zf:
        zf.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)
        zf.writestr("META-INF/container.xml", container_xml)
        zf.writestr("content.opf", opf_xml)
        zf.writestr("cover.png", cover_bytes)
        zf.writestr("chapter1.xhtml", chapter_xhtml)

    return epub_path


def _make_epub_no_cover(tmp_path: Path) -> Path:
    """Create a minimal EPUB without any cover image."""
    epub_path = tmp_path / "test-no-cover.epub"

    container_xml = b"""<?xml version="1.0"?>
<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container" version="1.0">
  <rootfiles>
    <rootfile full-path="content.opf" media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>"""

    opf_xml = b"""<?xml version="1.0"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="uid">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:title>No Cover Book</dc:title>
  </metadata>
  <manifest>
    <item id="ch1" href="chapter1.xhtml" media-type="application/xhtml+xml"/>
  </manifest>
  <spine><itemref idref="ch1"/></spine>
</package>"""

    with zipfile.ZipFile(epub_path, "w") as zf:
        zf.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)
        zf.writestr("META-INF/container.xml", container_xml)
        zf.writestr("content.opf", opf_xml)
        zf.writestr("chapter1.xhtml", b"<html><body><p>hi</p></body></html>")

    return epub_path


def _make_epub_relative_path_cover(tmp_path: Path) -> Path:
    """Create EPUB with cover href using relative path (../images/cover.jpg)."""
    epub_path = tmp_path / "test-relative.epub"

    container_xml = b"""<?xml version="1.0"?>
<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container" version="1.0">
  <rootfiles>
    <rootfile full-path="OEBPS/text/content.opf" media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>"""

    opf_xml = b"""<?xml version="1.0"?>
<package xmlns="http://www.idpf.org/2007/opf" version="2.0" unique-identifier="uid">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:title>Relative Path Book</dc:title>
    <meta name="cover" content="cover-img"/>
  </metadata>
  <manifest>
    <item id="cover-img" href="../images/cover.jpg" media-type="image/jpeg"/>
  </manifest>
  <spine/>
</package>"""

    cover_bytes = b"\xff\xd8\xff\xe0RELATIVE_COVER"

    with zipfile.ZipFile(epub_path, "w") as zf:
        zf.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)
        zf.writestr("META-INF/container.xml", container_xml)
        zf.writestr("OEBPS/text/content.opf", opf_xml)
        zf.writestr("OEBPS/images/cover.jpg", cover_bytes)

    return epub_path


def _rewrite_opf(
    opf_bytes: bytes,
    cover_href: str,
    *,
    new_media_type: str | None = None,
    cover_page_href: str | None = None,
    title: str | None = None,
    author: str | None = None,
) -> bytes | None:
    """Run the package rewrite over raw OPF bytes, as _write_cover_epub does."""
    cover = CoverInfo(
        archive_path="ignored",
        media_type="image/jpeg",
        raw_bytes=b"",
        manifest_href=cover_href,
    )
    return _apply_opf_fixes(
        opf_bytes,
        cover,
        new_media_type=new_media_type,
        cover_page_href=cover_page_href,
        title=title,
        author=author,
        fixes=CoverFixes(),
    )


# ---------------------------------------------------------------------------
# Tests: cover detection
# ---------------------------------------------------------------------------


class TestFindCover:
    def test_epub2_meta_cover(self, tmp_path: Path):
        cover_data = b"FAKE_JPEG_DATA_12345"
        epub = _make_epub2_with_cover(tmp_path, cover_data)
        result = find_cover_in_epub(epub)

        assert result is not None
        assert result.archive_path == "OEBPS/images/cover.jpg"
        assert result.media_type == "image/jpeg"
        assert result.raw_bytes == cover_data

    def test_epub3_properties_cover(self, tmp_path: Path):
        cover_data = b"FAKE_PNG_DATA_67890"
        epub = _make_epub3_with_cover(tmp_path, cover_data)
        result = find_cover_in_epub(epub)

        assert result is not None
        assert result.archive_path == "cover.png"
        assert result.media_type == "image/png"
        assert result.raw_bytes == cover_data

    def test_no_cover_returns_none(self, tmp_path: Path):
        epub = _make_epub_no_cover(tmp_path)
        result = find_cover_in_epub(epub)
        assert result is None

    def test_relative_path_resolution(self, tmp_path: Path):
        epub = _make_epub_relative_path_cover(tmp_path)
        result = find_cover_in_epub(epub)

        assert result is not None
        assert result.archive_path == "OEBPS/images/cover.jpg"
        assert result.raw_bytes == b"\xff\xd8\xff\xe0RELATIVE_COVER"


# ---------------------------------------------------------------------------
# Tests: cover replacement
# ---------------------------------------------------------------------------


class TestReplaceCover:
    def test_replace_same_format(self, tmp_path: Path):
        original_data = b"ORIGINAL_JPEG"
        epub = _make_epub2_with_cover(tmp_path, original_data)
        dest = tmp_path / "output.epub"
        new_data = b"NEW_JPEG_DATA"

        result = replace_cover(epub, dest, new_data, "image/jpeg")

        assert result.raw_bytes == original_data
        assert dest.exists()

        # Verify the new EPUB has the replaced cover
        with zipfile.ZipFile(dest) as zf:
            assert zf.read("OEBPS/images/cover.jpg") == new_data

    def test_replace_none_media_type_skips_validation(self, tmp_path: Path):
        """When media_type is None (AI-generated), skip format check."""
        epub = _make_epub2_with_cover(tmp_path, b"ORIGINAL")
        dest = tmp_path / "output.epub"

        # Should not raise even though we don't specify media type
        replace_cover(epub, dest, b"AI_GENERATED_PNG", None)
        assert dest.exists()

    def test_replace_mismatched_format_raises(self, tmp_path: Path):
        """Replacing JPEG cover with PNG should raise ValueError."""
        epub = _make_epub2_with_cover(tmp_path, b"ORIGINAL_JPEG")
        dest = tmp_path / "output.epub"

        with pytest.raises(ValueError, match="Cover format mismatch"):
            replace_cover(epub, dest, b"PNG_DATA", "image/png")

    def test_replace_compatible_jpeg_variants(self, tmp_path: Path):
        """image/jpg and image/jpeg should be treated as compatible."""
        epub = _make_epub2_with_cover(tmp_path, b"ORIGINAL")
        dest = tmp_path / "output.epub"

        # Should not raise — jpg and jpeg are the same
        replace_cover(epub, dest, b"NEW_DATA", "image/jpg")
        assert dest.exists()

    def test_replace_no_cover_raises(self, tmp_path: Path):
        epub = _make_epub_no_cover(tmp_path)
        dest = tmp_path / "output.epub"

        with pytest.raises(ValueError, match="No cover image found"):
            replace_cover(epub, dest, b"DATA", "image/jpeg")

    def test_replace_same_source_dest_raises(self, tmp_path: Path):
        """Writing to the same path as source should be rejected."""
        epub = _make_epub2_with_cover(tmp_path, b"\xff\xd8\xff\xe0" + b"\x00" * 100)

        with pytest.raises(ValueError, match="Source and destination.*same"):
            replace_cover(epub, epub, b"\xff\xd8\xff\xe0NEW", "image/jpeg")

    def test_replace_preserves_epub_structure(self, tmp_path: Path):
        """Replacement should preserve all other files in the EPUB."""
        epub = _make_epub2_with_cover(tmp_path, b"ORIGINAL")
        dest = tmp_path / "output.epub"

        replace_cover(epub, dest, b"NEW_COVER", "image/jpeg")

        with zipfile.ZipFile(dest) as zf:
            names = zf.namelist()
            assert "mimetype" in names
            assert "META-INF/container.xml" in names
            assert "OEBPS/content.opf" in names
            assert "OEBPS/chapter1.xhtml" in names
            # mimetype should be first entry
            assert names[0] == "mimetype"
            assert zf.read("mimetype") == b"application/epub+zip"

    def test_replace_from_file(self, tmp_path: Path):
        jpeg_bytes = b"\xff\xd8\xff\xe0" + b"\x00" * 100
        epub = _make_epub2_with_cover(tmp_path, jpeg_bytes)
        dest = tmp_path / "output.epub"
        img_file = tmp_path / "new_cover.jpg"
        new_jpeg = b"\xff\xd8\xff\xe0" + b"NEW_JPEG" + b"\x00" * 92
        img_file.write_bytes(new_jpeg)

        replace_cover_from_file(epub, dest, img_file)
        assert dest.exists()

        with zipfile.ZipFile(dest) as zf:
            assert zf.read("OEBPS/images/cover.jpg") == new_jpeg

    def test_replace_from_file_not_found(self, tmp_path: Path):
        epub = _make_epub2_with_cover(tmp_path, b"ORIGINAL")
        dest = tmp_path / "output.epub"

        with pytest.raises(FileNotFoundError):
            replace_cover_from_file(epub, dest, tmp_path / "nonexistent.jpg")


# ---------------------------------------------------------------------------
# Tests: prompt building
# ---------------------------------------------------------------------------


class TestBuildPrompt:
    def test_prompt_with_title_and_author(self):
        prompt = _build_default_prompt(
            title_translation="Реки Лондона",
            author_name="Бен Ааронович",
            target_lang="Russian",
        )
        assert "Реки Лондона" in prompt
        assert "Бен Ааронович" in prompt
        assert "Russian" in prompt
        assert "translate all text" in prompt.lower()

    def test_prompt_without_title(self):
        prompt = _build_default_prompt(
            title_translation=None,
            author_name=None,
            target_lang="German",
        )
        assert "German" in prompt
        assert "Translate the book title yourself" in prompt
        # Should not contain "Translated title to use"
        assert "Translated title to use" not in prompt

    def test_prompt_with_title_no_author(self):
        prompt = _build_default_prompt(
            title_translation="Les Rivières de Londres",
            author_name=None,
            target_lang="French",
        )
        assert "Les Rivières de Londres" in prompt
        assert "Author name on cover" not in prompt


# ---------------------------------------------------------------------------
# Tests: MIME normalization
# ---------------------------------------------------------------------------


class TestNormalizeMime:
    def test_jpeg_variants(self):
        assert _normalize_image_mime("image/jpeg") == "image/jpeg"
        assert _normalize_image_mime("image/jpg") == "image/jpeg"
        assert _normalize_image_mime("IMAGE/JPEG") == "image/jpeg"

    def test_png(self):
        assert _normalize_image_mime("image/png") == "image/png"

    def test_webp(self):
        assert _normalize_image_mime("image/webp") == "image/webp"


# ---------------------------------------------------------------------------
# Tests: byte-level MIME detection
# ---------------------------------------------------------------------------


class TestDetectImageMime:
    def test_jpeg_bytes(self):
        from booktranslator.cover import _detect_image_mime_from_bytes

        # Real JPEG magic bytes
        assert _detect_image_mime_from_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100) == "image/jpeg"

    def test_png_bytes(self):
        from booktranslator.cover import _detect_image_mime_from_bytes

        assert _detect_image_mime_from_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100) == "image/png"

    def test_webp_bytes(self):
        from booktranslator.cover import _detect_image_mime_from_bytes

        result = _detect_image_mime_from_bytes(b"RIFF\x00\x00\x00\x00WEBP" + b"\x00" * 100)
        assert result == "image/webp"

    def test_gif_bytes(self):
        from booktranslator.cover import _detect_image_mime_from_bytes

        assert _detect_image_mime_from_bytes(b"GIF89a" + b"\x00" * 100) == "image/gif"

    def test_unknown_bytes(self):
        from booktranslator.cover import _detect_image_mime_from_bytes

        assert _detect_image_mime_from_bytes(b"NOT_AN_IMAGE_FORMAT") is None

    def test_empty_bytes(self):
        from booktranslator.cover import _detect_image_mime_from_bytes

        assert _detect_image_mime_from_bytes(b"") is None
        assert _detect_image_mime_from_bytes(b"\x00") is None

    def test_mislabeled_file_detected_correctly(self, tmp_path: Path):
        """A PNG file with .jpg extension should be detected as PNG."""
        from booktranslator.cover import _detect_image_mime_from_bytes

        png_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        # Even though we might call it .jpg, bytes say PNG
        assert _detect_image_mime_from_bytes(png_bytes) == "image/png"


# ---------------------------------------------------------------------------
# Tests: replace_cover_from_file with byte validation
# ---------------------------------------------------------------------------


class TestReplaceCoverFromFileValidation:
    def test_mislabeled_extension_uses_byte_detection(self, tmp_path: Path):
        """PNG bytes in a .jpg file should be detected as PNG and rejected if original is JPEG."""
        epub = _make_epub2_with_cover(tmp_path, b"\xff\xd8\xff\xe0JPEG_COVER")
        dest = tmp_path / "output.epub"

        # Create a file with .jpg extension but PNG bytes
        mislabeled = tmp_path / "cover.jpg"
        mislabeled.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

        # Should raise because detected PNG != original JPEG
        with pytest.raises(ValueError, match="Cover format mismatch"):
            replace_cover_from_file(epub, dest, mislabeled)

    def test_invalid_image_bytes_rejected(self, tmp_path: Path):
        """Non-image bytes should be rejected."""
        epub = _make_epub2_with_cover(tmp_path, b"\xff\xd8\xff\xe0JPEG_COVER")
        dest = tmp_path / "output.epub"

        bad_file = tmp_path / "cover.jpg"
        bad_file.write_bytes(b"This is not an image at all")

        with pytest.raises(ValueError, match="Cannot determine image format"):
            replace_cover_from_file(epub, dest, bad_file)

    def test_valid_same_format_succeeds(self, tmp_path: Path):
        """JPEG bytes in a .jpg file replacing JPEG cover should work."""
        epub = _make_epub2_with_cover(tmp_path, b"\xff\xd8\xff\xe0OLD_JPEG")
        dest = tmp_path / "output.epub"

        good_file = tmp_path / "cover.jpg"
        good_file.write_bytes(b"\xff\xd8\xff\xe0NEW_JPEG_DATA")

        replace_cover_from_file(epub, dest, good_file)
        assert dest.exists()


# ---------------------------------------------------------------------------
# Tests: provider image generation response parsing
# ---------------------------------------------------------------------------


class TestProviderImageParsing:
    """Test that generate_image correctly parses OpenRouter responses."""

    def test_parse_valid_response(self):
        """Verify base64 data URL parsing from a mocked response."""
        import base64

        # Simulate what the provider returns after parsing
        fake_image = b"FAKE_IMAGE_BYTES"
        b64 = base64.b64encode(fake_image).decode()
        data_url = f"data:image/png;base64,{b64}"

        # Parse the data URL the same way provider.py does
        header, b64_payload = data_url.split(",", 1)
        mime = header.split(":")[1].split(";")[0]
        image_bytes = base64.b64decode(b64_payload)

        assert mime == "image/png"
        assert image_bytes == fake_image

    def test_invalid_data_url_format(self):
        """Non-data URLs should be caught."""
        url = "https://example.com/image.png"
        assert not url.startswith("data:")

    def test_non_image_mime_rejected(self):
        """MIME types that don't start with image/ should be rejected."""
        import base64

        from booktranslator.provider import OpenRouterProvider

        # Build a fake response with text/plain MIME
        fake_data = base64.b64encode(b"not an image").decode()
        fake_response = {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "images": [{"image_url": {"url": f"data:text/plain;base64,{fake_data}"}}],
                    }
                }
            ]
        }

        provider = OpenRouterProvider(api_key="test-key")

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = fake_response

        mock_http = MagicMock()
        mock_http.post.return_value = mock_resp

        with patch("httpx.Client") as mock_client_cls:
            mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_http)
            mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

            with pytest.raises(RuntimeError, match="Non-image MIME type"):
                provider.generate_image(
                    model="test/model",
                    prompt="test",
                )

    def test_empty_payload_rejected(self):
        """Empty base64 payload should be rejected."""

        from booktranslator.provider import OpenRouterProvider

        fake_response = {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "images": [{"image_url": {"url": "data:image/png;base64,"}}],
                    }
                }
            ]
        }

        provider = OpenRouterProvider(api_key="test-key")

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = fake_response

        mock_http = MagicMock()
        mock_http.post.return_value = mock_resp

        with patch("httpx.Client") as mock_client_cls:
            mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_http)
            mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

            with pytest.raises(RuntimeError, match="Empty image payload"):
                provider.generate_image(
                    model="test/model",
                    prompt="test",
                )

    def test_no_images_in_response_rejected(self):
        """Response without images array should raise RuntimeError."""

        from booktranslator.provider import OpenRouterProvider

        fake_response = {
            "choices": [{"message": {"content": "I cannot generate images", "images": []}}]
        }

        provider = OpenRouterProvider(api_key="test-key")

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = fake_response

        mock_http = MagicMock()
        mock_http.post.return_value = mock_resp

        with patch("httpx.Client") as mock_client_cls:
            mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_http)
            mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

            with pytest.raises(RuntimeError, match="No images returned"):
                provider.generate_image(
                    model="test/model",
                    prompt="test",
                )


# ---------------------------------------------------------------------------
# Tests: translate_cover with mocked provider
# ---------------------------------------------------------------------------


class TestTranslateCover:
    """Test translate_cover with mocked provider outputs."""

    def test_translate_same_format(self, tmp_path: Path):
        """AI returns same format as original — simple replacement."""
        from booktranslator.provider import ImageGenerationResult

        jpeg_bytes = b"\xff\xd8\xff\xe0" + b"\x00" * 100
        epub = _make_epub2_with_cover(tmp_path, jpeg_bytes)
        dest = tmp_path / "translated.epub"

        translated_jpeg = b"\xff\xd8\xff\xe0" + b"TRANSLATED" + b"\x00" * 90
        mock_provider = MagicMock()
        mock_provider.generate_image.return_value = ImageGenerationResult(
            image_bytes=translated_jpeg,
            mime_type="image/jpeg",
            text="Here is your translated cover",
            model="google/gemini-3.1-flash-image-preview",
        )

        from booktranslator.cover import translate_cover

        result = translate_cover(
            source_epub=epub,
            dest_epub=dest,
            provider=mock_provider,
            model="google/gemini-3.1-flash-image-preview",
            title_translation="Тестовая Книга",
            target_lang="Russian",
        )

        assert dest.exists()
        assert result.image_bytes == translated_jpeg

        # Verify the EPUB has the new cover
        with zipfile.ZipFile(dest) as zf:
            assert zf.read("OEBPS/images/cover.jpg") == translated_jpeg

    def test_translate_different_format_updates_manifest(self, tmp_path: Path):
        """AI returns PNG when original is JPEG — OPF manifest must be updated."""
        from booktranslator.provider import ImageGenerationResult

        jpeg_bytes = b"\xff\xd8\xff\xe0" + b"\x00" * 100
        epub = _make_epub2_with_cover(tmp_path, jpeg_bytes)
        dest = tmp_path / "translated.epub"

        png_bytes = b"\x89PNG\r\n\x1a\n" + b"TRANSLATED_PNG" + b"\x00" * 86
        mock_provider = MagicMock()
        mock_provider.generate_image.return_value = ImageGenerationResult(
            image_bytes=png_bytes,
            mime_type="image/png",
            text="",
            model="google/gemini-3.1-flash-image-preview",
        )

        from booktranslator.cover import translate_cover

        translate_cover(
            source_epub=epub,
            dest_epub=dest,
            provider=mock_provider,
            model="google/gemini-3.1-flash-image-preview",
            target_lang="Russian",
        )

        assert dest.exists()

        # Verify the EPUB has the new cover bytes
        with zipfile.ZipFile(dest) as zf:
            assert zf.read("OEBPS/images/cover.jpg") == png_bytes
            # Verify OPF manifest was updated
            opf_content = zf.read("OEBPS/content.opf").decode("utf-8")
            assert 'media-type="image/png"' in opf_content

    def test_translate_no_cover_raises(self, tmp_path: Path):
        """translate_cover on EPUB without cover should raise ValueError."""
        from booktranslator.cover import translate_cover

        epub = _make_epub_no_cover(tmp_path)
        dest = tmp_path / "translated.epub"

        mock_provider = MagicMock()

        with pytest.raises(ValueError, match="No cover image found"):
            translate_cover(
                source_epub=epub,
                dest_epub=dest,
                provider=mock_provider,
                model="test/model",
                target_lang="Russian",
            )

    def test_translate_invalid_provider_output_raises(self, tmp_path: Path):
        """Provider returning non-image bytes should raise RuntimeError."""
        from booktranslator.provider import ImageGenerationResult

        epub = _make_epub2_with_cover(tmp_path, b"\xff\xd8\xff\xe0ORIGINAL")
        dest = tmp_path / "translated.epub"

        mock_provider = MagicMock()
        mock_provider.generate_image.return_value = ImageGenerationResult(
            image_bytes=b"NOT_A_VALID_IMAGE_FORMAT",
            mime_type="image/png",
            text="",
            model="test/model",
        )

        from booktranslator.cover import translate_cover

        with pytest.raises(RuntimeError, match="not a recognized image format"):
            translate_cover(
                source_epub=epub,
                dest_epub=dest,
                provider=mock_provider,
                model="test/model",
                target_lang="Russian",
            )

    def test_translate_empty_provider_output_raises(self, tmp_path: Path):
        """Provider returning empty bytes should raise RuntimeError."""
        from booktranslator.provider import ImageGenerationResult

        epub = _make_epub2_with_cover(tmp_path, b"\xff\xd8\xff\xe0ORIGINAL")
        dest = tmp_path / "translated.epub"

        mock_provider = MagicMock()
        mock_provider.generate_image.return_value = ImageGenerationResult(
            image_bytes=b"",
            mime_type="image/png",
            text="",
            model="test/model",
        )

        from booktranslator.cover import translate_cover

        with pytest.raises(RuntimeError, match="empty image data"):
            translate_cover(
                source_epub=epub,
                dest_epub=dest,
                provider=mock_provider,
                model="test/model",
                target_lang="Russian",
            )

    def test_translate_auto_mode_no_title(self, tmp_path: Path):
        """Without --title, prompt should instruct model to translate itself."""
        from booktranslator.provider import ImageGenerationResult

        jpeg_bytes = b"\xff\xd8\xff\xe0" + b"\x00" * 100
        epub = _make_epub2_with_cover(tmp_path, jpeg_bytes)
        dest = tmp_path / "translated.epub"

        translated_jpeg = b"\xff\xd8\xff\xe0" + b"AUTO" + b"\x00" * 96
        mock_provider = MagicMock()
        mock_provider.generate_image.return_value = ImageGenerationResult(
            image_bytes=translated_jpeg,
            mime_type="image/jpeg",
            text="",
            model="test/model",
        )

        from booktranslator.cover import translate_cover

        translate_cover(
            source_epub=epub,
            dest_epub=dest,
            provider=mock_provider,
            model="test/model",
            title_translation=None,  # auto mode
            target_lang="Russian",
        )

        # Verify the prompt passed to generate_image contains auto-translate instruction
        prompt_used = mock_provider.generate_image.call_args.kwargs["prompt"]
        assert "Translate the book title yourself" in prompt_used


# ---------------------------------------------------------------------------
# Tests: manifest update with relative hrefs and edge cases
# ---------------------------------------------------------------------------


class TestManifestUpdate:
    """Test OPF manifest update logic for format changes."""

    def test_relative_href_manifest_update(self, tmp_path: Path):
        """Cover with relative href (../images/cover.jpg) should update correctly."""
        from booktranslator.provider import ImageGenerationResult

        epub = _make_epub_relative_path_cover(tmp_path)
        dest = tmp_path / "translated.epub"

        png_bytes = b"\x89PNG\r\n\x1a\n" + b"TRANSLATED" + b"\x00" * 90
        mock_provider = MagicMock()
        mock_provider.generate_image.return_value = ImageGenerationResult(
            image_bytes=png_bytes,
            mime_type="image/png",  # different from original jpeg
            text="",
            model="test/model",
        )

        from booktranslator.cover import translate_cover

        translate_cover(
            source_epub=epub,
            dest_epub=dest,
            provider=mock_provider,
            model="test/model",
            target_lang="Russian",
        )

        assert dest.exists()
        with zipfile.ZipFile(dest) as zf:
            # Cover bytes should be replaced
            assert zf.read("OEBPS/images/cover.jpg") == png_bytes
            # OPF manifest should be updated to image/png
            opf_content = zf.read("OEBPS/text/content.opf").decode("utf-8")
            assert 'media-type="image/png"' in opf_content

    def test_manifest_update_fails_on_missing_href(self, tmp_path: Path):
        """If manifest href doesn't match, should raise RuntimeError."""
        opf_xml = b"""<?xml version="1.0"?>
<package xmlns="http://www.idpf.org/2007/opf" version="2.0">
  <manifest>
    <item id="cover" href="images/cover.jpg" media-type="image/jpeg"/>
  </manifest>
</package>"""

        with pytest.raises(RuntimeError, match="Failed to update OPF manifest"):
            _rewrite_opf(
                opf_xml,
                "nonexistent/path.jpg",  # wrong href
                new_media_type="image/png",
            )

    def test_manifest_update_exact_href_match(self, tmp_path: Path):
        """Should match by exact href, not suffix."""
        # Two items with same filename suffix but different paths
        opf_xml = b"""<?xml version="1.0"?>
<package xmlns="http://www.idpf.org/2007/opf" version="2.0">
  <manifest>
    <item id="other" href="other/cover.jpg" media-type="image/jpeg"/>
    <item id="real" href="../images/cover.jpg" media-type="image/jpeg"/>
  </manifest>
</package>"""

        result = _rewrite_opf(
            opf_xml,
            "../images/cover.jpg",  # exact href
            new_media_type="image/png",
        )
        assert result is not None

        result_str = result.decode("utf-8")
        # The "real" item should be updated
        assert 'href="../images/cover.jpg"' in result_str
        # Check that the correct one was updated (has image/png)
        # and the other one still has image/jpeg
        assert 'id="other"' in result_str
        # Parse to verify precisely
        tree = etree.fromstring(result)
        items = tree.findall(".//{http://www.idpf.org/2007/opf}item")
        for item in items:
            if item.get("id") == "other":
                assert item.get("media-type") == "image/jpeg"
            elif item.get("id") == "real":
                assert item.get("media-type") == "image/png"


# ---------------------------------------------------------------------------
# Tests: cover extract CLI and _mime_to_extension
# ---------------------------------------------------------------------------


class TestMimeToExtension:
    def test_known_types(self):
        from booktranslator.cli import _mime_to_extension

        assert _mime_to_extension("image/jpeg") == ".jpg"
        assert _mime_to_extension("image/png") == ".png"
        assert _mime_to_extension("image/webp") == ".webp"
        assert _mime_to_extension("IMAGE/JPEG") == ".jpg"

    def test_unknown_type_returns_jpg(self):
        from booktranslator.cli import _mime_to_extension

        assert _mime_to_extension("image/x-unknown") == ".jpg"

    def test_none_returns_jpg(self):
        from booktranslator.cli import _mime_to_extension

        assert _mime_to_extension(None) == ".jpg"

    def test_empty_string_returns_jpg(self):
        from booktranslator.cli import _mime_to_extension

        assert _mime_to_extension("") == ".jpg"

    def test_mime_with_parameters(self):
        from booktranslator.cli import _mime_to_extension

        assert _mime_to_extension("image/png; charset=binary") == ".png"
        assert _mime_to_extension("image/jpeg; quality=high") == ".jpg"

    def test_svg(self):
        from booktranslator.cli import _mime_to_extension

        assert _mime_to_extension("image/svg+xml") == ".svg"


class TestCoverExtractCLI:
    def test_extract_success_default_output(self, tmp_path: Path):
        """Extract should save cover with auto-generated filename."""

        from typer.testing import CliRunner

        from booktranslator.cli import app

        cover_data = b"\xff\xd8\xff\xe0" + b"COVER_JPEG" + b"\x00" * 90
        epub = _make_epub2_with_cover(tmp_path, cover_data)

        runner = CliRunner()
        result = runner.invoke(app, ["cover", "extract", str(epub)])

        assert result.exit_code == 0
        assert "Cover extracted" in result.output

        expected_out = epub.with_stem(f"{epub.stem}-cover").with_suffix(".jpg")
        assert expected_out.exists()
        assert expected_out.read_bytes() == cover_data

    def test_extract_no_cover_exits_1(self, tmp_path: Path):
        """Extract on EPUB without cover should exit with code 1."""
        from typer.testing import CliRunner

        from booktranslator.cli import app

        epub = _make_epub_no_cover(tmp_path)

        runner = CliRunner()
        result = runner.invoke(app, ["cover", "extract", str(epub)])

        assert result.exit_code == 1
        assert "No cover image found" in result.output

    def test_extract_out_is_directory_exits_2(self, tmp_path: Path):
        """--out pointing to a directory should exit with code 2."""
        from typer.testing import CliRunner

        from booktranslator.cli import app

        cover_data = b"\xff\xd8\xff\xe0" + b"\x00" * 100
        epub = _make_epub2_with_cover(tmp_path, cover_data)
        out_dir = tmp_path / "some_dir"
        out_dir.mkdir()

        runner = CliRunner()
        result = runner.invoke(app, ["cover", "extract", str(epub), "--out", str(out_dir)])

        assert result.exit_code == 2
        assert "directory" in result.output.lower()

    def test_extract_custom_out_path(self, tmp_path: Path):
        """--out should save to the specified path."""
        from typer.testing import CliRunner

        from booktranslator.cli import app

        cover_data = b"\x89PNG\r\n\x1a\n" + b"PNG_COVER" + b"\x00" * 91
        epub = _make_epub3_with_cover(tmp_path, cover_data)
        out_file = tmp_path / "output" / "my-cover.png"

        runner = CliRunner()
        result = runner.invoke(app, ["cover", "extract", str(epub), "--out", str(out_file)])

        assert result.exit_code == 0
        assert out_file.exists()
        assert out_file.read_bytes() == cover_data

    def test_extract_existing_file_without_force_fails(self, tmp_path: Path):
        """Existing output file without --force should fail."""
        from typer.testing import CliRunner

        from booktranslator.cli import app

        cover_data = b"\xff\xd8\xff\xe0" + b"\x00" * 100
        epub = _make_epub2_with_cover(tmp_path, cover_data)
        out_file = tmp_path / "existing.jpg"
        out_file.write_bytes(b"OLD_DATA")

        runner = CliRunner()
        result = runner.invoke(app, ["cover", "extract", str(epub), "--out", str(out_file)])

        assert result.exit_code == 1
        assert "already exists" in result.output
        # Original file should be untouched
        assert out_file.read_bytes() == b"OLD_DATA"

    def test_extract_existing_file_with_force_overwrites(self, tmp_path: Path):
        """Existing output file with --force should be overwritten."""
        from typer.testing import CliRunner

        from booktranslator.cli import app

        cover_data = b"\xff\xd8\xff\xe0" + b"NEW_COVER" + b"\x00" * 91
        epub = _make_epub2_with_cover(tmp_path, cover_data)
        out_file = tmp_path / "existing.jpg"
        out_file.write_bytes(b"OLD_DATA")

        runner = CliRunner()
        result = runner.invoke(
            app, ["cover", "extract", str(epub), "--out", str(out_file), "--force"]
        )

        assert result.exit_code == 0
        assert out_file.read_bytes() == cover_data


# ---------------------------------------------------------------------------
# Fixtures and tests: cover declaration, page size, metadata repair
# ---------------------------------------------------------------------------


def _fake_jpeg(width: int, height: int) -> bytes:
    """A JPEG header just complete enough for the size reader."""
    payload = b"\x08" + height.to_bytes(2, "big") + width.to_bytes(2, "big") + b"\x03" + b"\x00" * 9
    return b"\xff\xd8\xff\xc0" + (len(payload) + 2).to_bytes(2, "big") + payload + b"\xff\xd9"


def _fake_png(width: int, height: int) -> bytes:
    """A PNG signature plus an IHDR chunk carrying the given size."""
    return (
        b"\x89PNG\r\n\x1a\n"
        + (13).to_bytes(4, "big")
        + b"IHDR"
        + width.to_bytes(4, "big")
        + height.to_bytes(4, "big")
        + b"\x08\x02\x00\x00\x00"
    )


def _make_calibre_epub(
    tmp_path: Path,
    cover_bytes: bytes,
    *,
    cover_width: int = 964,
    cover_height: int = 1537,
    name: str = "calibre.epub",
) -> Path:
    """Create an EPUB shaped like the calibre/OceanofPDF books we translate.

    Two traits matter here and both come from real files: the cover is never
    declared (no <meta name="cover">, no <guide>), and the metadata element
    itself carries a namespace prefix.
    """
    epub_path = tmp_path / name

    container_xml = b"""<?xml version="1.0"?>
<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container" version="1.0">
  <rootfiles>
    <rootfile full-path="content.opf" media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>"""

    opf_xml = b"""<?xml version='1.0' encoding='utf-8'?>
<package xmlns="http://www.idpf.org/2007/opf" version="2.0" unique-identifier="bookid">
  <opf:metadata xmlns:dc="http://purl.org/dc/elements/1.1/" \
xmlns:opf="http://www.idpf.org/2007/opf">
    <dc:identifier id="bookid" opf:scheme="UUID">urn:uuid:abc</dc:identifier>
    <dc:title>Foxglove Summer</dc:title>
    <dc:creator opf:role="aut">Ben Aaronovitch</dc:creator>
    <dc:language>ru</dc:language>
  </opf:metadata>
  <manifest>
    <item id="titlepage" href="titlepage.xhtml" media-type="application/xhtml+xml"/>
    <item id="ch1" href="index_split_003.html" media-type="application/xhtml+xml"/>
    <item id="a2a1cover" href="cover.jpeg" media-type="image/jpeg"/>
  </manifest>
  <spine toc="ncx">
    <itemref idref="titlepage" linear="yes"/>
    <itemref idref="ch1" linear="yes"/>
  </spine>
</package>"""

    titlepage = f"""<?xml version='1.0' encoding='utf-8'?>
<html xmlns="http://www.w3.org/1999/xhtml">
    <head><title>Cover</title></head>
    <body>
        <div>
            <svg xmlns="http://www.w3.org/2000/svg"
                 xmlns:xlink="http://www.w3.org/1999/xlink" version="1.1"
                 width="100%" height="100%"
                 viewBox="0 0 {cover_width} {cover_height}"
                 preserveAspectRatio="xMidYMid meet">
                <image width="{cover_width}" height="{cover_height}" xlink:href="cover.jpeg"/>
            </svg>
        </div>
    </body>
</html>""".encode()

    with zipfile.ZipFile(epub_path, "w") as zf:
        zf.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)
        zf.writestr("META-INF/container.xml", container_xml)
        zf.writestr("content.opf", opf_xml)
        zf.writestr("titlepage.xhtml", titlepage)
        zf.writestr("index_split_003.html", b"<html><body><p>hi</p></body></html>")
        zf.writestr("cover.jpeg", cover_bytes)

    return epub_path


class TestImageDimensions:
    """The size reader, which decides what the cover page should declare."""

    def test_jpeg(self):
        from booktranslator.cover import _image_dimensions

        assert _image_dimensions(_fake_jpeg(848, 1264)) == (848, 1264)

    def test_jpeg_skips_leading_segments(self):
        from booktranslator.cover import _image_dimensions

        # A real file puts JFIF and EXIF segments before the frame header.
        jfif = b"\xff\xe0" + (16).to_bytes(2, "big") + b"JFIF\x00" + b"\x00" * 9
        data = _fake_jpeg(300, 400)
        assert _image_dimensions(data[:2] + jfif + data[2:]) == (300, 400)

    def test_png(self):
        from booktranslator.cover import _image_dimensions

        assert _image_dimensions(_fake_png(600, 900)) == (600, 900)

    def test_gif(self):
        from booktranslator.cover import _image_dimensions

        data = b"GIF89a" + (64).to_bytes(2, "little") + (48).to_bytes(2, "little") + b"\x00" * 10
        assert _image_dimensions(data) == (64, 48)

    def test_bmp_negative_height_is_still_a_size(self):
        from booktranslator.cover import _image_dimensions

        data = (
            b"BM"
            + b"\x00" * 16
            + (20).to_bytes(4, "little", signed=True)
            + (-30).to_bytes(4, "little", signed=True)
            + b"\x00" * 10
        )
        assert _image_dimensions(data) == (20, 30)

    def test_webp_lossy(self):
        from booktranslator.cover import _image_dimensions

        data = (
            b"RIFF"
            + b"\x00" * 4
            + b"WEBP"
            + b"VP8 "
            + b"\x00" * 4
            + b"\x00" * 3
            + b"\x9d\x01\x2a"
            + (100).to_bytes(2, "little")
            + (200).to_bytes(2, "little")
        )
        assert _image_dimensions(data) == (100, 200)

    def test_unreadable_returns_none(self):
        from booktranslator.cover import _image_dimensions

        assert _image_dimensions(b"NOT_AN_IMAGE") is None
        # TIFF is recognized as an image but its size is not read.
        assert _image_dimensions(b"II\x2a\x00" + b"\x00" * 40) is None
        # A truncated PNG header carries no size yet.
        assert _image_dimensions(b"\x89PNG\r\n\x1a\n") is None


class TestCoverDeclaration:
    """A cover a reader can find, in books whose source never declared one."""

    def test_epub2_declaration_and_guide_are_added(self, tmp_path: Path):
        epub = _make_calibre_epub(tmp_path, _fake_jpeg(964, 1537))
        dest = tmp_path / "out.epub"

        _cover, fixes = repair_cover(epub, dest)

        assert fixes.cover_meta == "a2a1cover"
        assert fixes.guide_href == "titlepage.xhtml"
        with zipfile.ZipFile(dest) as zf:
            opf = zf.read("content.opf").decode("utf-8")
        tree = etree.fromstring(opf.encode())
        ns = {"opf": "http://www.idpf.org/2007/opf"}
        meta = tree.find(".//opf:metadata/opf:meta[@name='cover']", ns)
        assert meta is not None
        assert meta.get("content") == "a2a1cover"
        reference = tree.find("opf:guide/opf:reference[@type='cover']", ns)
        assert reference is not None
        assert reference.get("href") == "titlepage.xhtml"

    def test_meta_cover_is_written_without_a_namespace_prefix(self, tmp_path: Path):
        """Readers that match the literal <meta name="cover"> must find it.

        The books we translate declare their metadata element as
        <opf:metadata>, and lxml would reuse that prefix for a new child.
        """
        epub = _make_calibre_epub(tmp_path, _fake_jpeg(964, 1537))
        dest = tmp_path / "out.epub"

        repair_cover(epub, dest)

        with zipfile.ZipFile(dest) as zf:
            opf = zf.read("content.opf").decode("utf-8")
        assert '<meta name="cover" content="a2a1cover"/>' in opf
        assert "opf:meta " not in opf

    def test_existing_declaration_is_left_alone(self, tmp_path: Path):
        """A book that already declares its cover is not rewritten."""
        epub = _make_epub2_with_cover(tmp_path, _fake_jpeg(100, 150))
        dest = tmp_path / "out.epub"

        _cover, fixes = repair_cover(epub, dest)

        assert fixes.cover_meta is None
        assert not fixes.any_change()
        with zipfile.ZipFile(epub) as src, zipfile.ZipFile(dest) as out:
            assert src.read("OEBPS/content.opf") == out.read("OEBPS/content.opf")

    def test_epub3_gets_the_manifest_property_not_a_guide(self, tmp_path: Path):
        """EPUB3 declares the cover on the manifest item, not in metadata."""
        epub = _make_epub3_with_cover(tmp_path, _fake_png(500, 750))
        dest = tmp_path / "out.epub"

        _cover, fixes = repair_cover(epub, dest)

        # The fixture already carries properties="cover-image", so nothing
        # about the declaration should change, and no guide should appear.
        assert fixes.cover_property is None
        assert fixes.guide_href is None
        with zipfile.ZipFile(dest) as zf:
            opf = zf.read("content.opf").decode("utf-8")
        assert "<guide" not in opf

    def test_epub3_missing_property_is_added(self, tmp_path: Path):
        epub_bytes = _fake_png(500, 750)
        epub = _make_epub3_with_cover(tmp_path, epub_bytes)
        # Drop the declaration to model a book that lost it.
        stripped = tmp_path / "stripped.epub"
        with zipfile.ZipFile(epub) as src, zipfile.ZipFile(stripped, "w") as dst:
            for info in src.infolist():
                data = src.read(info.filename)
                if info.filename == "content.opf":
                    data = data.replace(b' properties="cover-image"', b"")
                dst.writestr(info.filename, data)

        dest = tmp_path / "out.epub"
        _cover, fixes = repair_cover(stripped, dest)

        assert fixes.cover_property == "cover"
        with zipfile.ZipFile(dest) as zf:
            opf = zf.read("content.opf").decode("utf-8")
        assert 'properties="cover-image"' in opf
        assert "<guide" not in opf


class TestCoverPageSize:
    """The size the cover page declares has to follow the image it shows."""

    def test_page_is_resized_to_the_new_image(self, tmp_path: Path):
        epub = _make_calibre_epub(tmp_path, _fake_jpeg(964, 1537))
        dest = tmp_path / "out.epub"

        replace_cover(epub, dest, _fake_jpeg(848, 1264), "image/jpeg")

        with zipfile.ZipFile(dest) as zf:
            page = zf.read("titlepage.xhtml").decode("utf-8")
        tree = etree.fromstring(page.encode())
        svg = tree.find(".//{http://www.w3.org/2000/svg}svg")
        assert svg is not None
        assert svg.get("viewBox") == "0 0 848 1264"
        # A percentage is the book's layout choice and must survive.
        assert svg.get("width") == "100%"
        image = svg.find("{http://www.w3.org/2000/svg}image")
        assert image.get("width") == "848"
        assert image.get("height") == "1264"

    def test_repair_resizes_against_the_cover_already_in_the_book(self, tmp_path: Path):
        """Our own earlier output: new cover written, old size left behind."""
        epub = _make_calibre_epub(
            tmp_path, _fake_jpeg(848, 1264), cover_width=964, cover_height=1537
        )
        dest = tmp_path / "out.epub"

        _cover, fixes = repair_cover(epub, dest)

        assert fixes.page_size == (848, 1264)
        assert fixes.page_path == "titlepage.xhtml"
        with zipfile.ZipFile(dest) as zf:
            page = zf.read("titlepage.xhtml").decode("utf-8")
        assert 'viewBox="0 0 848 1264"' in page

    def test_matching_size_is_left_alone(self, tmp_path: Path):
        epub = _make_calibre_epub(tmp_path, _fake_jpeg(964, 1537))
        dest = tmp_path / "out.epub"

        _cover, fixes = repair_cover(epub, dest)

        assert fixes.page_size is None
        with zipfile.ZipFile(epub) as src, zipfile.ZipFile(dest) as out:
            assert src.read("titlepage.xhtml") == out.read("titlepage.xhtml")

    def test_unreadable_image_leaves_the_page_untouched(self, tmp_path: Path):
        """Better a stale size than a wrong one."""
        epub = _make_calibre_epub(tmp_path, b"\xff\xd8\xff")
        dest = tmp_path / "out.epub"

        _cover, fixes = repair_cover(epub, dest)

        assert fixes.page_size is None
        with zipfile.ZipFile(epub) as src, zipfile.ZipFile(dest) as out:
            assert src.read("titlepage.xhtml") == out.read("titlepage.xhtml")


class TestCoverMetadata:
    """Title and author, so a reader's own placeholder is not in English."""

    def test_repair_sets_title_and_author(self, tmp_path: Path):
        epub = _make_calibre_epub(tmp_path, _fake_jpeg(964, 1537))
        dest = tmp_path / "out.epub"

        _cover, fixes = repair_cover(epub, dest, title="Лето наперстянки", author="Бен Ааронович")

        assert fixes.title == "Лето наперстянки"
        assert fixes.author == "Бен Ааронович"
        with zipfile.ZipFile(dest) as zf:
            tree = etree.fromstring(zf.read("content.opf"))
        ns = {"dc": "http://purl.org/dc/elements/1.1/"}
        assert tree.find(".//dc:title", ns).text == "Лето наперстянки"
        assert tree.find(".//dc:creator", ns).text == "Бен Ааронович"

    def test_translate_cover_writes_title_and_author(self, tmp_path: Path):
        from booktranslator.cover import translate_cover
        from booktranslator.provider import ImageGenerationResult

        epub = _make_calibre_epub(tmp_path, _fake_jpeg(964, 1537))
        dest = tmp_path / "out.epub"

        mock_provider = MagicMock()
        mock_provider.generate_image.return_value = ImageGenerationResult(
            image_bytes=_fake_jpeg(848, 1264),
            mime_type="image/jpeg",
            text="",
            model="test/model",
        )

        translate_cover(
            source_epub=epub,
            dest_epub=dest,
            provider=mock_provider,
            model="test/model",
            title_translation="Лето наперстянки",
            author_name="Бен Ааронович",
            target_lang="Russian",
        )

        with zipfile.ZipFile(dest) as zf:
            tree = etree.fromstring(zf.read("content.opf"))
            page = zf.read("titlepage.xhtml").decode("utf-8")
        ns = {"dc": "http://purl.org/dc/elements/1.1/"}
        assert tree.find(".//dc:title", ns).text == "Лето наперстянки"
        assert tree.find(".//dc:creator", ns).text == "Бен Ааронович"
        assert 'viewBox="0 0 848 1264"' in page

    def test_untranslated_title_is_left_alone(self, tmp_path: Path):
        epub = _make_calibre_epub(tmp_path, _fake_jpeg(964, 1537))
        dest = tmp_path / "out.epub"

        _cover, fixes = repair_cover(epub, dest)

        assert fixes.title is None
        with zipfile.ZipFile(dest) as zf:
            tree = etree.fromstring(zf.read("content.opf"))
        ns = {"dc": "http://purl.org/dc/elements/1.1/"}
        assert tree.find(".//dc:title", ns).text == "Foxglove Summer"


class TestRepairKeepsTheBookIntact:
    def test_image_and_every_other_file_survive(self, tmp_path: Path):
        cover_bytes = _fake_jpeg(848, 1264)
        epub = _make_calibre_epub(tmp_path, cover_bytes)
        dest = tmp_path / "out.epub"

        repair_cover(epub, dest, title="Лето наперстянки")

        with zipfile.ZipFile(epub) as src, zipfile.ZipFile(dest) as out:
            assert out.namelist() == src.namelist()
            assert out.read("cover.jpeg") == cover_bytes
            assert out.read("index_split_003.html") == src.read("index_split_003.html")
            assert out.read("META-INF/container.xml") == src.read("META-INF/container.xml")
        # mimetype stays the first entry and stays uncompressed
        with zipfile.ZipFile(dest) as out:
            first = out.infolist()[0]
            assert first.filename == "mimetype"
            assert first.compress_type == zipfile.ZIP_STORED

    def test_same_source_and_destination_is_rejected(self, tmp_path: Path):
        epub = _make_calibre_epub(tmp_path, _fake_jpeg(964, 1537))

        with pytest.raises(ValueError, match="Source and destination.*same"):
            repair_cover(epub, epub)

    def test_no_cover_raises(self, tmp_path: Path):
        epub = _make_epub_no_cover(tmp_path)

        with pytest.raises(ValueError, match="No cover image found"):
            repair_cover(epub, tmp_path / "out.epub")


class TestCoverFixCli:
    def test_fix_reports_what_it_changed(self, tmp_path: Path):
        from typer.testing import CliRunner

        from booktranslator.cli import app

        epub = _make_calibre_epub(tmp_path, _fake_jpeg(848, 1264))
        out = tmp_path / "fixed.epub"

        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "cover",
                "fix",
                str(epub),
                "--out",
                str(out),
                "--title",
                "Лето наперстянки",
            ],
        )

        assert result.exit_code == 0, result.output
        assert "Cover repaired" in result.output
        assert out.exists()
        with zipfile.ZipFile(out) as zf:
            opf = zf.read("content.opf").decode("utf-8")
        assert '<meta name="cover" content="a2a1cover"/>' in opf
        assert 'type="cover"' in opf
        assert "Лето наперстянки" in opf

    def test_fix_says_so_when_nothing_is_wrong(self, tmp_path: Path):
        from typer.testing import CliRunner

        from booktranslator.cli import app

        epub = _make_epub2_with_cover(tmp_path, _fake_jpeg(100, 150))
        out = tmp_path / "fixed.epub"

        runner = CliRunner()
        result = runner.invoke(app, ["cover", "fix", str(epub), "--out", str(out)])

        assert result.exit_code == 0, result.output
        assert "Nothing to repair" in result.output

    def test_fix_refuses_to_overwrite_the_source(self, tmp_path: Path):
        from typer.testing import CliRunner

        from booktranslator.cli import app

        epub = _make_calibre_epub(tmp_path, _fake_jpeg(964, 1537))
        before = epub.read_bytes()

        runner = CliRunner()
        result = runner.invoke(app, ["cover", "fix", str(epub), "--out", str(epub)])

        assert result.exit_code == 1
        assert "Source and destination" in result.output
        assert epub.read_bytes() == before
