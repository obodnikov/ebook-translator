# AI rules — EPUB I/O, chunking, cover, assembly (Python)

Scope: `src/booktranslator/epub_io.py`, `chunker.py`, `cover.py`, `reader_notes.py`, and the
assemble path in `postprocess.py` / `cli.py` (`assemble`, `cover extract|replace|translate|fix`).
This layer turns an EPUB into translatable units and rebuilds it byte-faithfully. See
[ARCHITECTURE.md §3 (steps 1–2, 7, 11), §4–5](ARCHITECTURE.md); this file is the coding contract.

## Language & build

- Python 3.12+ with full type hints; exported functions have explicit return types. `ruff` clean
  (line length 100). No mypy gate, but write as if there were one.
- EPUB parsing is `ebooklib` for the container/manifest, `lxml` for XHTML trees. Do **not** write a
  custom EPUB or XML parser (anti-scope, ARCHITECTURE §12).

## Structure preservation is the prime contract

The whole point of the tool is that the translated book is the *same* book in Russian. Treat the
source EPUB as immutable reference data:

- **Round-trip identity.** Reading an EPUB and writing it back with no translations applied must
  reproduce assets, CSS, images, fonts, cover, nav/TOC, and spine order unchanged. Only text
  nodes, a small set of metadata fields (`dc:language`, optionally `dc:title` and `dc:creator`),
  and the cover declaration may ever change — see "Cover" below for what the cover write repairs
  and why.
- **Edit text nodes in place, by xpath.** Locate each paragraph by its xpath in the original
  `lxml` tree and replace only its text content. **Preserve inline tags** (`<em>`, `<i>`, `<a>`,
  `<strong>`, …) and their order — never flatten them into plain text and never re-serialize a
  whole document from scratch when a node-level edit suffices.
- **Never reorder or rename** spine items, manifest entries, or asset paths. New bytes (translated
  cover) go into a *new* output EPUB; the source file is never mutated in place.

## XML / entity hygiene

- The translation round-trips through JSON and Jinja2, so XML special characters (`&`, `<`, `>`)
  and entities are a known failure mode (see `docs/chats/fixing-xml-entity-escaping-*`). When you
  reinsert translated text into XHTML, escape exactly once — no double-escaping (`&amp;amp;`) and
  no raw `&` that breaks XML well-formedness.
- Parse in a strict mode and keep the original tree around for reinsertion; do not "clean up" or
  reformat XHTML the author wrote.

## Chunking contract (`chunker.py`)

- The unit of translation is a **chunk** (~2000 words of paragraphs), not a chapter. Chapters are
  only a logical grouping / id namespace (`ch03_c02`).
- Each paragraph carries its `xpath` so assembly can find it again. `reassemble(chunks)` must equal
  `book.chapters` by paragraph text and xpath — keep this invariant; it is what makes assembly
  deterministic.
- Overlap paragraphs (prev/next) are **context only** — they are fed to the model but never
  re-translated or written back. Don't let overlap text leak into the output.

## Cover (`cover.py`)

- Four independent operations, all non-destructive (always produce a new EPUB, never edit the
  source): **extract** (cover bytes → file), **replace** (user file → cover), **translate** (AI
  image model re-renders the cover text in the target language), **fix** (repair the declaration
  of a book already built, image untouched, no model call).
- Cover detection follows the documented priority: EPUB2 `<meta name="cover">` → EPUB3
  `properties="cover-image"` → heuristic (first `image/*` with "cover" in id/href). Keep these
  strategies in order; don't silently pick the first image.
- **Every cover write leaves the cover findable.** Writing the image is not enough: a reader that
  cannot tell which image is the cover draws its own placeholder from the title and author, so
  every write goes through `_write_cover_epub`, which also declares the cover when the source
  never did — `<meta name="cover">` plus a `<guide>` reference for EPUB2, `properties="cover-image"`
  for EPUB3 — and never both. Don't add a cover write path that skips it.
- **The declared page size follows the image.** The cover page's `viewBox` and `<image>` size
  describe the file that was there before; left alone, a new cover with different proportions gets
  letterboxed. Read the size from the image header (`_image_dimensions`) and rewrite it. When the
  size cannot be read, leave the markup alone — a stale size beats a wrong one. Touch only
  attributes that are already a bare pixel count; a percentage or a unit is the book's own layout.
- **New package elements must serialize without a namespace prefix.** These books declare
  `<opf:metadata>`, and lxml reuses that prefix for a new child — readers that match a literal
  `<meta name="cover">` would then miss it. Create elements through `_append_child`, which binds
  the package namespace as the default.
- Metadata this layer may write, beyond `dc:language`: `dc:title` and `dc:creator`, and only when
  the caller passes a translation for them. Everything else in the package is reference data.
- Cover translation is a *best-effort adaptation*, not pixel-perfect — set expectations in
  user-facing text, not by trying to match fonts exactly. The image model call goes through the
  **image** provider (see [AI_PROVIDER.md](AI_PROVIDER.md)), never the text provider.
- No image library. Sizes come from the file header, for the formats
  `_detect_image_mime_from_bytes` already recognizes; don't add Pillow for this.

## Reader notes (`reader_notes.py`)

- **The footnote markup follows the source book's package version**
  (`BookMeta.epub_version`). Popup footnotes are an EPUB3 mechanism: there the note
  goes in `<aside epub:type="footnote">`. EPUB2 has no such mechanism and does not allow
  `<aside>` — a strict reader may drop the element and the note with it — so an EPUB2 book
  gets visible endnotes at the end of the chapter, built from `<div>`. Don't collapse the two
  paths into one, and don't hide the EPUB2 block: a hidden note with no popup leads nowhere.
- **Create `epub:type` with the prefix bound** (`nsmap={"epub": EPUB_NS}`). Without it lxml
  invents `ns0:type` for the books whose `<html>` declares no `xmlns:epub` — namespace-correct
  but invisible to a reader matching the literal `epub:type`. The same trap as the cover's
  `<meta name="cover">`; the test fixture must include a chapter with no `xmlns:epub`, or the
  bug goes unnoticed.
- Every marker carries an id and every note a link back to it. A reader without popups
  navigates, and without a way back the reader is stranded at the end of the chapter.
- Note styles go into the chapter's own `<head>`, never into the book's stylesheet — that file
  is the book's asset and round-trip identity covers it.
- Notes are numbered in reading order and the numbering rises through the book. Matching walks
  the glossary per paragraph, so the numbers are corrected per chapter after injection. That
  correction is all-or-nothing: plan the whole chapter first and change nothing unless every
  marker resolved to a note, or a marker that could not be accounted for ends up pointing at an
  id the renumbering has moved away.
- **Version detection fails loudly.** `read_package_version` falls back to "2.0" because that is
  what every reader understands, but it logs the fallback: an EPUB3 book mistaken for EPUB2 would
  silently get the wrong markup. Take the version from the book ebooklib already parsed
  (`_version_from_book`) and keep the archive sniff as the fallback.
- Don't add `<p>` inside an EPUB2 note: `_find_paragraphs` would take it for prose if the
  output is ever read back in.

## What stays out of this layer

- No LLM orchestration, cache, or stage logic (that's [AI_PIPELINE.md](AI_PIPELINE.md)).
- No provider/HTTP/prompt concerns (that's [AI_PROVIDER.md](AI_PROVIDER.md)).
- No PDF typesetting, no DRM removal, no bespoke parser — all anti-scope (ARCHITECTURE §12).
