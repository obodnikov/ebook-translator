# AI rules — Pipeline state machine & stages (Python)

Scope: `src/booktranslator/pipeline_helpers.py`, `state.py`, `cache.py`, `translator.py`,
`judge.py`, `reflect.py`, `postprocess.py` (proofread / style / verify), and the stage commands in
`cli.py`. This layer drives chunks through the stages, caches results, and supports pause/resume.
See [ARCHITECTURE.md §3, §5, §9](ARCHITECTURE.md); this file is the coding contract.

## Language & build

- Python 3.12+, full type hints, `ruff` clean (line length 100). No mypy gate.
- Pipeline work must be **idempotent and resumable**: re-running a stage on an already-processed
  chunk should hit the cache, not re-bill the model.

## The paragraph-count contract is non-negotiable

- Every stage that rewrites text (translate, reflect, proofread, style, verify) must return
  **exactly N paragraphs for N input paragraphs**. A mismatch is a contract violation, not a
  warning to ignore — retry with a more explicit prompt, and if it still fails, fall back per the
  waterfall rather than writing a corrupted chunk. See `test_postprocess.py::
  test_handles_paragraph_count_mismatch` and `collect_waterfall_paragraphs` in
  `pipeline_helpers.py` for the established behaviour; preserve it.
- Never "fix" a count mismatch by padding, dropping, or merging paragraphs silently. The fallback
  is: use the latest stage output that *does* satisfy the count (waterfall), and flag the chunk.

## Cache (`cache.py`) — keys and the waterfall

- The cache is SQLite keyed by `sha256(chunk_text + model + prompt_version + glossary_hash +
  stage)`. All five inputs are part of the identity — changing any of them is a new entry, so
  experimenting with a model or prompt never corrupts existing results.
- Stages form a **waterfall**: later stages (verify > style > proofread > reflect > translate)
  take priority, but only when their output is valid (paragraph count holds). When reading "the
  current best text" for a chunk, resolve through the waterfall / `preferred_stage`, don't assume
  the last-written row is correct.
- Invalidation is explicit: bump `prompt_version`, change the model, or invalidate a specific
  chunk/stage. Do not clear the whole cache as a shortcut — it throws away paid work.
- Never write a result to the cache before it has passed validation (count + parse).

## Prompt versioning drives correctness *and* cost

- Each prompt file carries `version: N` in its frontmatter, and that version is part of the cache
  key. If you change a prompt's behaviour, **bump its version** so stale cached outputs are
  invalidated. Forgetting this silently serves old translations — treat a prompt edit without a
  version bump as a bug.

## State machine, pause/resume (`state.py`, `pipeline_helpers.py`)

- `state.json` in `work/<book-slug>/` holds `current_stage` + `completed_stages`. `--resume`
  reads it and continues; the two pause points (after glossary, after translate-preview) write a
  review file, send a Telegram notification, and exit 0 — the process does **not** block waiting
  for input.
- On resume from a pause, validate the artifact the human was supposed to edit
  (`glossary.json` exists & parses; `translated-preview.epub` exists) before advancing. Don't
  blindly jump stages.
- Respect the cost hard limit: if accumulated cost exceeds `cost.hard_limit_usd`, abort cleanly
  with state saved and a notification — never keep spending past the ceiling.

## Judge / reflect (`judge.py`, `reflect.py`)

- Judge is a cheap scoring pass (1–5). Reflect (Andrew Ng's method) runs only on low-scoring
  chunks or under `--reflect-all`; it must re-validate the paragraph-count contract on its output
  before it can win the waterfall. A reflect pass that returns identical text is a no-op — don't
  re-bill or mark it as an improvement (see `test_reflect.py::test_unchanged_when_same_output`).

## Errors & resilience

- Classify failures (ARCHITECTURE §9): transient (429/5xx) → retry with backoff; permanent (400 /
  invalid JSON) → log, skip the chunk, flag it in state, continue; budget exceeded → abort;
  malformed EPUB → fail fast at extract. Don't turn a permanent error into an infinite retry.

## What stays out of this layer

- No EPUB tree manipulation or assembly (that's [AI_EPUB.md](AI_EPUB.md)).
- No raw provider/HTTP, prompt rendering, or config parsing (that's
  [AI_PROVIDER.md](AI_PROVIDER.md)).
