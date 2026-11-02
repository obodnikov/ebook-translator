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
  The one repair of a mismatch, `replies.rebuild_over_split_reply`, is described under "Errors &
  resilience"; don't widen it.

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
- Never write a result to the cache before it has passed validation (count + parse + alignment).
- A prompt must not depend on the order chunks finish in, or a second run pays again for chunks
  it already has. Keep overlap as source text (AI_EPUB.md); the first Tainted Cup run showed 5–7
  healthy chunks re-bought on a `-j 8` re-run and 48 on `-j 1` before that was fixed.

## Prompt versioning drives correctness *and* cost

- Each prompt file carries `version: N` in its frontmatter, and that version is part of the cache
  key. If you change a prompt's behaviour, **bump its version** so stale cached outputs are
  invalidated. Forgetting this silently serves old translations — treat a prompt edit without a
  version bump as a bug.
- A version bump doesn't remove the rows of the old version. A `reflect` row outranks a new
  `translate` row in the waterfall, so on a book in progress the later stages would go on reading
  the old text. Tell the user to run `btrans cache clear <book> --stage translate` first.
- The translate prompt's "Direct speech" section is load-bearing, examples included. Version 4 only
  forbade splitting a paragraph, which left the model no way to write speech after narration: 16 of
  132 first replies were rejected (four of the five saved ones for splitting), and half the chunks
  of The Tainted Cup opened speech with guillemets. With the examples, the ten hardest chunks of
  that book had no split and every speech paragraph opened with an em dash. The same version also
  needed the ellipsis line: without it, three of the ten switched to three dots. Measure again on
  real chunks when editing it.

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

- `reflection.trigger_score` is calibrated to one judge: another model or level moves the scores,
  so re-measure the threshold when either changes. Compare final text after all stages, not text
  right after reflect: later stages fix much of the same.
- Judge is a cheap scoring pass (1–5). Reflect (Andrew Ng's method) runs only on low-scoring
  chunks or under `--reflect-all`; it must re-validate the paragraph-count contract on its output
  before it can win the waterfall. A reflect pass that returns identical text is a no-op — don't
  re-bill or mark it as an improvement (see `test_reflect.py::test_unchanged_when_same_output`).

## Errors & resilience

- When the server names a wait (`Retry-After`, or OpenRouter's 402 `in_flight_budget_exhausted`
  body), the provider retry waits that long (capped at `MAX_RETRY_AFTER_SECONDS`) instead of the
  exponential backoff — five quick attempts inside a 120 s wait lose the chunk for nothing.
- Classify failures (ARCHITECTURE §9): transient (429/5xx) → retry with backoff; permanent (400 /
  invalid JSON) → log, skip the chunk, flag it in state, continue; budget exceeded → abort;
  malformed EPUB → fail fast at extract. Don't turn a permanent error into an infinite retry.
- **Read every JSON reply through `model_json.loads`, never a bare `json.loads`.** It strips code
  fences, tolerates commentary after the value, and escapes the straight `"` a model leaves inside
  a string when it quotes the source text — the break that `raw_decode` cannot repair. Repairs are
  validated by parsing, so a valid reply is never rewritten. Don't grow a fourth private parser in
  a stage; extend that module instead, and keep the stage's own salvage (truncation, preamble)
  around it.
- Invalid JSON stays a permanent error, with two exceptions, both capped at **one** extra attempt
  on a **fresh** reply — a cached row that no longer parses must surface as a failure, never as a
  reason to spend money:
  - the judge sends an unreadable verdict back to the model to be re-emitted as JSON
    (`prompts/json_fix.md`, a short call carrying only the broken reply);
  - a post-processing stage whose reply ignored the delta format asks the same question again with
    the format restated (`_FORMAT_RETRY_NOTE`, appended in code so no prompt version changes).
    Never a "repair" of that reply: when a model answers with analysis instead of patches there is
    no JSON to repair, and turning prose into patches would let it invent text nobody vetted.
    These are full-price calls, so a run gets a budget (`PostProcessor._retry_budget`) — a model
    that has stopped following the format must not silently double the cost of the stage.
- A translate or reflect reply is checked in full before it is cached: exactly N markers, every
  paragraph one well-formed XHTML element (`replies.parse_paragraphs`), **and** every paragraph
  lined up with its source (`replies.check_alignment`). The count alone is not enough — a
  paragraph missing `</p>`, or the model's own "wait, I mis-numbered" pasted between markers, both
  pass it. So does a reply with a hole in it: a model that split a paragraph and noticed the count
  was off keeps the count right by dropping a line of speech or the end of the chunk, and that
  reply is N well-formed elements. So does a paragraph printed twice: one marker holding that
  paragraph and the next, then the next again under its own marker. The judge cannot be the
  safeguard: it runs after the row is cached, and it missed one of the eight such chunks in The
  Tainted Cup. The alignment limits are measured (see the comment in `replies.py`), so measure again
  on real caches before moving them.
  A rejected **fresh** reply is asked for once more with the reason appended
  (`replies.RETRY_NOTE`, in code so no prompt version changes), naming every bad paragraph, within
  the run's `replies.retry_budget`. A cached reply that fails is reported with the
  `btrans cache clear` command that removes it — never re-bought. The alignment check applies to
  fresh replies only.
- A reply has two repairs, both for a paragraph the model split, and nothing else is repaired.
  They exist because the text is all there and a re-ask costs a whole chunk; the re-ask is also the
  worse answer, since a model told "exactly N" keeps splitting and stops at marker N. Keep them
  narrow, because the same shapes can hide other defects:
  - `replies.join_split_paragraphs`: same-looking sibling elements under **one** marker become one
    element. The joined text must fit the source paragraph and repeat nothing of the neighbouring
    markers, because two elements under one marker can also be the next source paragraph in the
    wrong place (The Tainted Cup ch52_c02), and joining that prints it twice.
  - `replies.rebuild_over_split_reply`: too many markers, numbered 1 to M, where each extra marker
    is narration followed by speech from one source paragraph that opens that way, and each part
    is about as long as the same part of the source. Comparing only the whole paragraph rebuilt just
    52% of 375 known-answer trials exactly, and 78 came out wrong and still passed every later
    check. Numbering that restarts means two answers in one reply. Too few markers, or anything
    else, stays rejected.

  Neither is silent: each is logged and recorded in the row's meta (`joined_paragraphs`,
  `rebuilt_paragraphs`), and the result goes through every check above. They live in `ReplyChecker`,
  not in `parse_fragment`, because post-processing patches are checked with that function and a
  patch with two elements must stay rejected. The cached row holds the repaired text, so that later
  readers see one element per marker.
- A failed translate chunk stops `btrans translate` once the other chunks are done: no later stage,
  no EPUB, no `done` in `state.json`, the chunk ids in `last_error`, exit code 1. "Skip the chunk and
  continue" (ARCHITECTURE §9) means continue with the other chunks of the stage. The later stages
  cost money and only polish what is there, and going on ends in a part-English book marked as
  done. `--allow-failed-chunks` goes on when the user decides to; `last_error` still says what failed.
- `replies.parse_fragment` forgives two slips that lose no text and that a re-ask does not reliably
  cure: a bare `&`, and an `epub:` attribute whose `xmlns:epub` declaration the model left off
  (The Tainted Cup ch36_c04 failed both asks over that alone). The cached row keeps the model's
  reply, so every reader of a fragment must go through this function. Keep this to slips of markup:
  a fragment that holds two elements, or anything but one element, stays rejected.
- A chunk goes into the live tree whole or not at all: parse every paragraph first, then splice.
- The waterfall that feeds proofread / style / verify / repair skips a stage whose paragraphs do
  not parse, the same way it skips a count mismatch. Those passes send patches for changed
  paragraphs only, so they cannot be relied on to close a broken tag — don't pay them to try.
- Each patch of those passes is checked on its own in `PostProcessor._apply_patches`: it must be
  one well-formed element, and must not carry a sentence (40+ characters) of a different
  paragraph that its own paragraph lacks — the sign of paragraphs mixed up, which XHTML checks
  cannot see. A failing patch is dropped and its paragraph kept; the chunk's other patches
  stand. Don't widen this to rejecting the chunk: that throws away every good edit with it.
- Repair caches the gatekeeper's verdicts (`repair_verdicts`, outside the waterfall, cleared with
  `repair`) under the call's key, rejections included, and the repaired text as a separate
  `repair` row — only when something was accepted, since an unchanged copy would outrank a later
  verify. A cached row must hold what its key names: the key is the model call, so the row is its
  answer, not text derived from it.
- A reply that fails to parse is never cached, so it is gone when the run ends. Write it to
  `work/<book>/failed/<stage>-<chunk>.txt` before raising, and log the failure with
  `finish_reason` and the length — 200 characters in an error message is not enough to tell a
  truncated reply from a model that ignored the format.

## What stays out of this layer

- No EPUB tree manipulation or assembly (that's [AI_EPUB.md](AI_EPUB.md)).
- No raw provider/HTTP, prompt rendering, or config parsing (that's
  [AI_PROVIDER.md](AI_PROVIDER.md)).
