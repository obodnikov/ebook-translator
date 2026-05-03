"""Glossary extraction, parsing, and persistence."""

from __future__ import annotations

import json
import re
from pathlib import Path

from pydantic import ValidationError

from .cache import Cache
from .epub_io import ExtractedBook
from .models import Glossary, GlossaryEntry
from .prompts import Prompt, load_prompt, render_prompt
from .provider import CompletionResult, OpenRouterProvider


LANG_NAMES = {
    "en": "English",
    "ru": "Russian",
    "hu": "Hungarian",
}


def _strip_code_fences(text: str) -> str:
    """Remove ``` or ```json fences from a model's output, if any."""
    m = re.search(r"```(?:json)?\s*(\{.*\}|\[.*\])\s*```", text, re.DOTALL)
    if m:
        return m.group(1)
    return text.strip()


def _parse_llm_response(raw: str) -> list[GlossaryEntry]:
    """Parse the LLM's JSON output into a list of validated entries."""
    cleaned = _strip_code_fences(raw)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Model response is not valid JSON: {e}. "
            f"First 200 chars: {cleaned[:200]!r}"
        ) from e

    if isinstance(data, list):
        entries_raw = data
    elif isinstance(data, dict) and "entries" in data:
        entries_raw = data["entries"]
    else:
        raise ValueError(
            "Expected JSON with an 'entries' list or a top-level list, "
            f"got keys: {list(data) if isinstance(data, dict) else type(data)}"
        )

    entries: list[GlossaryEntry] = []
    errors: list[str] = []
    for i, item in enumerate(entries_raw):
        try:
            entries.append(GlossaryEntry.model_validate(item))
        except ValidationError as e:
            errors.append(f"  entry {i}: {e.errors()[0]['msg']} ({item!r})")

    if errors and not entries:
        raise ValueError(
            "No valid glossary entries could be parsed. Issues:\n"
            + "\n".join(errors[:10])
        )
    return entries


def extract_glossary(
    book: ExtractedBook,
    provider: OpenRouterProvider,
    prompt_path: Path,
    cache: Cache,
    model: str | None = None,
    source_lang: str = "en",
    target_lang: str = "ru",
) -> tuple[Glossary, CompletionResult | None, str]:
    """Run the glossary extraction pass on the whole book.

    Returns a tuple of (Glossary, CompletionResult or None if cache hit,
    raw LLM text for debugging).
    """
    prompt = load_prompt(prompt_path)
    chosen_model = model or prompt.model
    if not chosen_model:
        raise ValueError("No model specified (neither CLI/config nor prompt frontmatter)")

    context = {
        "title": book.meta.title,
        "author": book.meta.author,
        "source_lang": source_lang,
        "source_lang_name": LANG_NAMES.get(source_lang, source_lang),
        "target_lang": target_lang,
        "target_lang_name": LANG_NAMES.get(target_lang, target_lang),
        "book_text": book.full_text(),
    }
    system, user = render_prompt(prompt, context)

    cache_key = Cache.make_key(
        "glossary",
        chosen_model,
        prompt.version,
        system,
        user,
    )
    cached = cache.get(cache_key)
    if cached is not None:
        raw_text = cached.content
        result: CompletionResult | None = None
    else:
        result = provider.complete(
            model=chosen_model,
            system=system,
            user=user,
            temperature=prompt.temperature,
            max_tokens=prompt.max_tokens,
        )
        raw_text = result.text
        cache.put(
            key=cache_key,
            stage="glossary",
            model=chosen_model,
            prompt_version=prompt.version,
            content=raw_text,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
        )

    entries = _parse_llm_response(raw_text)

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
