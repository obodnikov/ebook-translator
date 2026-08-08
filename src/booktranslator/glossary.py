"""Glossary extraction, parsing, and persistence."""

from __future__ import annotations

import json
import re
import secrets
from pathlib import Path

from pydantic import ValidationError

from .cache import Cache
from .epub_io import ExtractedBook
from .models import Glossary, GlossaryEntry
from .prompts import load_prompt, render_prompt
from .provider import CompletionResult, EmptyCompletionError, OpenRouterProvider

LANG_NAMES = {
    "en": "English",
    "ru": "Russian",
    "hu": "Hungarian",
}


class TruncationError(ValueError):
    """Raised when the model response indicates the book text was truncated.

    A truncated glossary would silently cover only part of the book, which
    violates the "never corrupt a book" contract. Callers must surface this
    loudly and must NOT write the glossary to disk.
    """

    pass


def _strip_code_fences(text: str) -> str:
    """Remove ``` or ```json fences from a model's output, if any."""
    m = re.search(r"```(?:json)?\s*(\{.*\}|\[.*\])\s*```", text, re.DOTALL)
    if m:
        return m.group(1)
    return text.strip()


def _parse_llm_response(raw: str) -> tuple[str | None, str | None, list[GlossaryEntry]]:
    """Parse the LLM's JSON output into (bookstart, bookend, entries).

    Returns the bookstart value, bookend value (both None if absent/list
    response), and the list of validated GlossaryEntry objects.

    Tolerates the extra `override` field used for series-aware extraction:
    it is folded into `notes` (prefixed with "[override] ") because the
    book-level `GlossaryEntry` schema doesn't carry it. The promote step
    decides what to do with overrides.
    """
    cleaned = _strip_code_fences(raw)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Model response is not valid JSON: {e}. First 200 chars: {cleaned[:200]!r}"
        ) from e

    if isinstance(data, list):
        bookstart = None
        bookend = None
        entries_raw = data
    elif isinstance(data, dict) and "entries" in data:
        bookstart = data.get("bookstart")  # may be None if model omitted it
        bookend = data.get("bookend")  # may be None if model omitted it
        entries_raw = data["entries"]
    else:
        raise ValueError(
            "Expected JSON with an 'entries' list or a top-level list, "
            f"got keys: {list(data) if isinstance(data, dict) else type(data)}"
        )

    entries: list[GlossaryEntry] = []
    errors: list[str] = []
    for i, item in enumerate(entries_raw):
        if not isinstance(item, dict):
            errors.append(f"  entry {i}: not an object ({item!r})")
            continue
        item = dict(item)
        is_override = bool(item.pop("override", False))
        if is_override:
            existing_notes = item.get("notes") or ""
            prefix = "[override] "
            item["notes"] = prefix + existing_notes if existing_notes else prefix.strip()
        try:
            entries.append(GlossaryEntry.model_validate(item))
        except ValidationError as e:
            errors.append(f"  entry {i}: {e.errors()[0]['msg']} ({item!r})")

    if errors and not entries:
        raise ValueError(
            "No valid glossary entries could be parsed. Issues:\n" + "\n".join(errors[:10])
        )
    return bookstart, bookend, entries


def extract_glossary(
    book: ExtractedBook,
    provider: OpenRouterProvider,
    prompt_path: Path,
    cache: Cache,
    model: str | None = None,
    source_lang: str = "en",
    target_lang: str = "ru",
    known_terms: str | None = None,
) -> tuple[Glossary, CompletionResult | None, str]:
    """Run the glossary extraction pass on the whole book.

    Returns a tuple of (Glossary, CompletionResult or None if cache hit,
    raw LLM text for debugging).

    If `known_terms` is given (typically the series glossary rendered as
    a compact table), it is injected into the prompt so the model can
    skip already-known names and only return new or overriding entries.

    Raises TruncationError if the bookstart/bookend nonces are missing or wrong,
    meaning the provider silently truncated the book text. The cache is
    NOT written in that case.
    """
    prompt = load_prompt(prompt_path)
    chosen_model = model or prompt.model
    if not chosen_model:
        raise ValueError("No model specified (neither CLI/config nor prompt frontmatter)")

    # Build stable context (WITHOUT nonce) for the cache key. The cache key
    # must be deterministic across runs so that cache hits work correctly.
    base_context = {
        "title": book.meta.title,
        "author": book.meta.author,
        "source_lang": source_lang,
        "source_lang_name": LANG_NAMES.get(source_lang, source_lang),
        "target_lang": target_lang,
        "target_lang_name": LANG_NAMES.get(target_lang, target_lang),
        "book_text": book.full_text(),
        "known_terms": known_terms or "",
    }
    system, user = render_prompt(prompt, base_context)

    cache_key = Cache.make_key(
        "glossary_v2",  # v2: bookend guard introduced; invalidates pre-guard cache entries
        chosen_model,
        prompt.version,
        system,
        user,
    )
    cached = cache.get(cache_key)
    if cached is not None:
        # Cache hit: re-validate guard meta to reject poisoned/pre-guard entries.
        cached_meta = cached.meta or {}
        stored_bookstart = cached_meta.get("bookstart")
        stored_bookend = cached_meta.get("bookend")
        if stored_bookstart is None or stored_bookend is None:
            # Old cache entry without guard meta — invalidate and recompute.
            cache.conn.execute("DELETE FROM cache WHERE key = ?", (cache_key,))
            cache.conn.commit()
            cached = None

    if cached is None:
        # Live call path: inject bookstart+bookend nonces.
        # Two nonces: one prepended (bookstart) and one appended (bookend).
        # A provider that truncates from the beginning keeps the tail
        # (bookend survives) but loses bookstart; a provider that truncates
        # from the end loses bookend. Both cases are caught.
        # The nonce values are NOT revealed in the prompt instruction —
        # the model can only know them by reading the full text.
        bookstart = secrets.token_hex(4)
        bookend = secrets.token_hex(4)
        # Reuse already-built book_text from base_context to avoid calling
        # full_text() twice (it may be expensive for large EPUBs).
        book_text_with_nonces = (
            f"[[BOOKSTART::{bookstart}]]\n\n"
            + base_context["book_text"]
            + f"\n\n[[BOOKEND::{bookend}]]"
        )
        nonce_context = {**base_context, "book_text": book_text_with_nonces}
        # Re-render BOTH system and user with nonce_context for the live call.
        # Do not assume book_text only appears in the user template — future
        # prompt edits might reference it in the system section too.
        system_nonce, user_nonce = render_prompt(prompt, nonce_context)

        result = provider.complete(
            model=chosen_model,
            system=system_nonce,
            user=user_nonce,
            temperature=prompt.temperature,
            max_tokens=prompt.max_tokens,
            reasoning_effort=prompt.reasoning_effort,
        )
        raw_text = result.text

        # Belt and braces. provider.complete() already raises on an empty
        # response, but a stubbed or third-party provider might not, and an
        # empty glossary poisons every stage of the book that follows. Without
        # this the failure would surface as "not valid JSON", which sends the
        # reader looking in the wrong place.
        if not raw_text.strip():
            finish_info = (
                f" (finish_reason={result.finish_reason!r})" if result.finish_reason else ""
            )
            raise EmptyCompletionError(
                f"Model returned empty content{finish_info}. "
                "Reasoning shares the response budget with the answer — lower "
                "reasoning_effort in the prompt, or shrink the request."
            )

        # Parse and validate bookend BEFORE writing to cache.
        returned_bookstart, returned_bookend, entries = _parse_llm_response(raw_text)

        # Normalize: strip whitespace in case model adds surrounding spaces.
        returned_bookstart = (returned_bookstart or "").strip()
        returned_bookend = (returned_bookend or "").strip()

        # Bookstart guard: truncation from the beginning loses the start marker.
        if returned_bookstart != bookstart:
            raise TruncationError(
                f"Книга, похоже, обрезана провайдером (начало): ожидался "
                f"bookstart={bookstart!r}, получено {returned_bookstart!r}. "
                "Глоссарий по неполному тексту не сохраняем. "
                "Проверьте окно контекста провайдера стадии glossary."
            )
        # Bookend guard: truncation from the end loses the end marker.
        if returned_bookend != bookend:
            raise TruncationError(
                f"Книга, похоже, обрезана провайдером (конец): ожидался "
                f"bookend={bookend!r}, получено {returned_bookend!r}. "
                "Глоссарий по неполному тексту не сохраняем. "
                "Проверьте окно контекста провайдера стадии glossary."
            )

        # Cache AFTER successful validation — never cache a truncated/empty response.
        # Store bookstart/bookend in meta so cache hits can be re-validated.
        cache.put(
            key=cache_key,
            stage="glossary",
            model=chosen_model,
            prompt_version=prompt.version,
            content=raw_text,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            meta={"bookstart": bookstart, "bookend": bookend},
        )

        glossary = Glossary(
            book=book.meta.title,
            author=book.meta.author,
            source_lang=source_lang,
            target_lang=target_lang,
            model=chosen_model,
            entries=entries,
        )
        return glossary, result, raw_text

    # Cache hit path: parse without bookend validation (already verified on write).
    result = None
    raw_text = cached.content
    _, _, entries = _parse_llm_response(raw_text)

    glossary = Glossary(
        book=book.meta.title,
        author=book.meta.author,
        source_lang=source_lang,
        target_lang=target_lang,
        model=chosen_model,
        entries=entries,
    )
    return glossary, result, raw_text


def save_glossary(glossary: Glossary, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        glossary.model_dump_json(indent=2, exclude_none=False),
        encoding="utf-8",
    )
