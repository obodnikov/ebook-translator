"""Read JSON out of a model's reply, tolerating the ways models break it.

Every stage that asks for JSON gets it back wrapped in code fences, trailed
by commentary, or — most often — with an unescaped straight quote inside a
string, because the prompt asks the model to quote the source text and the
source text contains ``"``::

    {"issues": ["accuracy: p.10 «and "No," Ada said» — ..."]}

``json.loads`` reads the quote before ``No`` as the end of the string and
then demands a comma. ``raw_decode`` cannot save this one: the object is
broken in the middle, not merely followed by extra text. So this module
walks the text and escapes the quotes that cannot be a string terminator,
then parses again.

Every repair is validated by parsing: a candidate that does not parse is
discarded, so a reply that was already valid can never be corrupted here.

The repair is deliberately narrow — stray quotes, nothing else. A reply
broken some other way still raises, and the caller decides what to do next
(the judge asks the model to re-emit it; see ``Judge._refetch_valid_json``).
"""

from __future__ import annotations

import json
from typing import Any

__all__ = ["loads", "strip_code_fences"]

# After a string ends, JSON allows only these: a member separator, the colon
# of a key, or the close of the enclosing array or object.
_STRUCTURAL = ":]},"
# In the shapes our prompts ask for — an array of strings, an array of
# objects, an object with string values — a comma after a string is followed
# by the next string, object, or array. Nothing else.
_AFTER_COMMA = '"{['


def strip_code_fences(text: str) -> str:
    """Remove a leading ```/```json line and its closing fence."""
    text = text.strip()
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _closes_string(text: str, pos: int, *, strict: bool) -> bool:
    """Could the quote just before `pos` be the end of a JSON string?

    `strict` also requires that a comma be followed by the start of another
    value, which is what separates a real separator from a comma inside a
    quoted sentence.
    """
    i = pos
    while i < len(text) and text[i].isspace():
        i += 1
    if i >= len(text):
        # Nothing follows: the reply is truncated. Treat the quote as the
        # terminator and let the parse attempt decide.
        return True
    char = text[i]
    if char not in _STRUCTURAL:
        return False
    if char != "," or not strict:
        return True
    i += 1
    while i < len(text) and text[i].isspace():
        i += 1
    return i < len(text) and text[i] in _AFTER_COMMA


def _escape_stray_quotes(text: str, *, strict: bool) -> str:
    """Escape the double quotes that appear inside a string value."""
    out: list[str] = []
    in_string = False
    i = 0
    while i < len(text):
        char = text[i]
        if not in_string:
            out.append(char)
            in_string = char == '"'
            i += 1
        elif char == "\\":
            # Keep an existing escape sequence intact, whatever it escapes.
            out.append(text[i : i + 2])
            i += 2
        elif char == '"':
            if _closes_string(text, i + 1, strict=strict):
                out.append(char)
                in_string = False
            else:
                out.append('\\"')
            i += 1
        else:
            out.append(char)
            i += 1
    return "".join(out)


def _parse(text: str) -> Any:
    """Parse `text`, allowing commentary after the value. Raises on failure."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # The model appended an explanation after a complete value.
        value, _ = json.JSONDecoder().raw_decode(text)
        return value


def loads(text: str) -> Any:
    """Parse a model's reply as JSON, repairing stray quotes if needed.

    Raises `json.JSONDecodeError` — reporting the failure of the unrepaired
    text, which is the one worth showing a human — if nothing parses.
    """
    stripped = strip_code_fences(text)
    first_error: json.JSONDecodeError | None = None
    seen: set[str] = set()

    for candidate in (
        stripped,
        _escape_stray_quotes(stripped, strict=True),
        _escape_stray_quotes(stripped, strict=False),
    ):
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            return _parse(candidate)
        except (json.JSONDecodeError, ValueError) as e:
            if first_error is None and isinstance(e, json.JSONDecodeError):
                first_error = e

    if first_error is not None:
        raise first_error
    raise json.JSONDecodeError("No JSON value in reply", stripped, 0)
