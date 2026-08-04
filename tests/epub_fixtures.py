"""A minimal valid EPUB, shared by the tests that need a real book on disk.

Small enough to chunk into one chunk, with two paragraphs carrying words that
are easy to assert on: "vestigium" and "Seawoll".
"""

from __future__ import annotations

import zipfile
from pathlib import Path

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
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles>
    <rootfile full-path="content.opf" media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>"""


def write_minimal_epub(path: Path) -> Path:
    """Write the minimal EPUB to `path` and return it."""
    with zipfile.ZipFile(path, "w") as zf:
        # mimetype must come first and stay uncompressed
        zf.writestr(
            zipfile.ZipInfo("mimetype"),
            "application/epub+zip",
            compress_type=zipfile.ZIP_STORED,
        )
        zf.writestr("META-INF/container.xml", MINIMAL_CONTAINER)
        zf.writestr("content.opf", MINIMAL_OPF)
        zf.writestr("ch01.xhtml", MINIMAL_XHTML)
    return path
