# book-translator

A CLI tool that translates EPUB books from English to Russian using LLMs via OpenRouter
(`btrans`). It preserves the source layout one-to-one (same XHTML, tags, CSS, images, cover,
nav/TOC), uses a per-book glossary for name/term consistency, runs a multi-stage proofreading
waterfall, and supports pause/resume so a mid-run failure never restarts from scratch.

> Status: **implemented and in use.** The pipeline (extract → chunk → glossary → translate →
> judge → reflect → assemble preview → proofread → style → verify → assemble final), the
> SQLite cache, pause/resume, the split text/image providers, the cover-translation feature,
> reader-notes, and series glossaries are built and covered by tests under `tests/`.
> [ARCHITECTURE.md](ARCHITECTURE.md) is the authoritative design source;
> [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) tracks the phased build.
>
> Build & test: `pip install -e ".[dev]"` (into `.venv`), `pytest`, `ruff check .`,
> `ruff format --check .`.

## Read before making changes

1. [ARCHITECTURE.md](ARCHITECTURE.md) — system structure, pipeline, modules, data formats,
   stability zones, anti-scope and roadmap. The single source of truth for *what* and *why*.
2. The relevant `AI_*.md` file(s) for the code you are touching — coding rules (see below).
3. [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) — phased plan and settled decisions.
4. `docs/chats/` — prior implementation conversations that produced the design.

## Coding rules live in `AI_*.md` (do not duplicate them here or in ARCHITECTURE.md)

| File | Scope |
| --- | --- |
| [AI_EPUB.md](AI_EPUB.md) | EPUB I/O, chunking, cover, assembly (`epub_io.py`, `chunker.py`, `cover.py`, assemble path): structure preservation, round-trip identity, xpath/inline-tag contracts. |
| [AI_PIPELINE.md](AI_PIPELINE.md) | Pipeline state machine and stages (`pipeline_helpers.py`, `state.py`, `cache.py`, `translator.py`, `judge.py`, `reflect.py`, `postprocess.py`): pause/resume, the paragraph-count contract, cache keys, prompt versioning, the stage waterfall. |
| [AI_PROVIDER.md](AI_PROVIDER.md) | Providers, prompts, config (`provider.py`, `prompts.py`, `config.py`, `models.py`): the split text/image routing, retries, Jinja2 + frontmatter prompts, secrets, cost logging. |

`ARCHITECTURE.md` and the `AI_*.md` files must not redefine or duplicate each other's content:
`ARCHITECTURE.md` is rationale and contracts; `AI_*.md` files are the coding rules that enforce
them.

## Working agreement

- **Confirm before acting.** Never create, edit, or delete files, run state-changing commands,
  or write to external systems without explicit user approval. First explain the situation,
  propose specifics (which files, what changes, what commands), then wait for a clear "yes."
  Read-only work (reading, searching, analyzing, answering) needs no confirmation.
  Exception: if the user says "just do it" / "go ahead," proceed directly.
- **Don't commit unprompted.** Run `git add` / `git commit` / `git push` only when the user
  explicitly asks — never as an unrequested side-effect of another task.
- **Never burn money or corrupt a book to make a check pass.** The LLM calls cost real money and
  the output is someone's book. Never weaken a contract (paragraph-count match, round-trip
  identity, glossary consistency) or loosen/skip a test to turn a red check green. If a chunk
  genuinely cannot be translated within contract, surface it (logged skip + state flag), don't
  paper over it.
- **Stop and ask** if anything is unclear or contradictory.

## Documentation language

User-facing documentation (README, guides, config comments) is written in **Russian** per the
rules in [ARCHITECTURE.md §13](ARCHITECTURE.md). Code-level agent rules — this file and the
`AI_*.md` files — are kept in **English**. Command examples, file names, and code stay verbatim.

## Tooling

- **Python 3.12+**, packaged with setuptools (`src/` layout, entry point `btrans`). Work inside
  the project virtualenv (`.venv`); install with `pip install -e ".[dev]"`.
- Quality gates: **ruff** (lint + format, line length 100, rules `E,F,W,I,UP,B,SIM`) and
  **pytest**. There is **no mypy gate** — the package ships `py.typed` and uses type hints, but
  static typing is not enforced in CI; don't add a mypy step unless asked.
- **pre-commit** runs ruff (`--fix`) + ruff-format + gitleaks on every commit. GitLab CI runs
  the shared gitleaks template (`.gitlab-ci.yml`).
- Secrets (`OPENROUTER_API_KEY`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`) live only in `.env`
  (gitignored); `.env.example` is the committed template. Never log keys or full book text.
- No Node, Docker, or backend services are part of this project.
