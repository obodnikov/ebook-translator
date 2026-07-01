# AI rules — EPUB I/O, chunking, cover, assembly (Python)

Scope: `src/booktranslator/epub_io.py`, `chunker.py`, `cover.py`, and the assemble path in
`postprocess.py` / `cli.py` (`assemble`, `cover extract|replace|translate`). This layer turns an
EPUB into translatable units and rebuilds it byte-faithfully. See
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
  nodes and a small set of metadata fields (`dc:language`, optionally `dc:title`) may ever change.
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

- Three independent operations, all non-destructive (always produce a new EPUB, never edit the
  source): **extract** (cover bytes → file), **replace** (user file → cover), **translate** (AI
  image model re-renders the cover text in the target language).
- Cover detection follows the documented priority: EPUB2 `<meta name="cover">` → EPUB3
  `properties="cover-image"` → heuristic (first `image/*` with "cover" in id/href). Keep these
  strategies in order; don't silently pick the first image.
- Cover translation is a *best-effort adaptation*, not pixel-perfect — set expectations in
  user-facing text, not by trying to match fonts exactly. The image model call goes through the
  **image** provider (see [AI_PROVIDER.md](AI_PROVIDER.md)), never the text provider.

## What stays out of this layer

- No LLM orchestration, cache, or stage logic (that's [AI_PIPELINE.md](AI_PIPELINE.md)).
- No provider/HTTP/prompt concerns (that's [AI_PROVIDER.md](AI_PROVIDER.md)).
- No PDF typesetting, no DRM removal, no bespoke parser — all anti-scope (ARCHITECTURE §12).
