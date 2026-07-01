"""Load markdown-formatted prompts with YAML frontmatter and Jinja2 templating.

Prompt file layout:

    ---
    version: 1
    model: anthropic/claude-sonnet-4.6
    temperature: 0.2
    ---
    # System
    ...system template...

    # User
    ...user template...

Both sections use Jinja2 placeholders. The module returns the rendered
system/user strings ready to pass as chat messages.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import frontmatter
from jinja2 import Environment, StrictUndefined

_SECTION_RE = re.compile(r"^# (System|User)\s*$", re.MULTILINE)


@dataclass
class Prompt:
    name: str
    version: str
    model: str | None
    temperature: float
    max_tokens: int | None
    reasoning_effort: str | None  # none|low|medium|high; None => don't send the param
    system_tmpl: str
    user_tmpl: str


def _split_sections(body: str) -> tuple[str, str]:
    """Split prompt body into 'System' and 'User' sections by `# System` / `# User`."""
    matches = list(_SECTION_RE.finditer(body))
    if not matches:
        # No explicit sections: treat the whole body as user.
        return "", body.strip()

    sections: dict[str, str] = {}
    for i, m in enumerate(matches):
        name = m.group(1).lower()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        sections[name] = body[start:end].strip()

    return sections.get("system", ""), sections.get("user", "")


def load_prompt(path: Path) -> Prompt:
    """Parse a prompt markdown file into a `Prompt`."""
    post = frontmatter.load(path)
    fm = post.metadata
    system_tmpl, user_tmpl = _split_sections(post.content)

    return Prompt(
        name=path.stem,
        version=str(fm.get("version", "1")),
        model=fm.get("model"),
        temperature=float(fm.get("temperature", 0.3)),
        max_tokens=fm.get("max_tokens"),
        reasoning_effort=fm.get("reasoning_effort"),
        system_tmpl=system_tmpl,
        user_tmpl=user_tmpl,
    )


_env = Environment(
    undefined=StrictUndefined,
    trim_blocks=True,
    lstrip_blocks=True,
    keep_trailing_newline=False,
)


def render_prompt(prompt: Prompt, context: dict[str, Any]) -> tuple[str, str]:
    """Render system and user templates against the given context."""
    system = _env.from_string(prompt.system_tmpl).render(**context) if prompt.system_tmpl else ""
    user = _env.from_string(prompt.user_tmpl).render(**context) if prompt.user_tmpl else ""
    return system, user
