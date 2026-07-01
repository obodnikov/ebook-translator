"""Per-book working directory and pipeline state tracking."""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from .models import PipelineState, Stage


def slugify(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii").lower()
    ascii_text = re.sub(r"[^a-z0-9]+", "-", ascii_text)
    return ascii_text.strip("-") or "book"


@dataclass
class WorkDir:
    """Paths for artifacts of one book's translation run."""

    root: Path
    slug: str

    @classmethod
    def for_book(cls, base: Path, title: str) -> WorkDir:
        slug = slugify(title)
        root = base / slug
        root.mkdir(parents=True, exist_ok=True)
        return cls(root=root, slug=slug)

    @property
    def state_path(self) -> Path:
        return self.root / "state.json"

    @property
    def glossary_path(self) -> Path:
        return self.root / "glossary.json"

    @property
    def raw_glossary_path(self) -> Path:
        """Raw LLM response for glossary, useful for debugging bad outputs."""
        return self.root / "glossary-raw.txt"

    @property
    def cache_path(self) -> Path:
        return self.root / "cache.sqlite"

    @property
    def log_path(self) -> Path:
        return self.root / "log.jsonl"

    # --- state I/O ---

    def load_state(self) -> PipelineState:
        if not self.state_path.exists():
            return PipelineState(book_slug=self.slug)
        data = json.loads(self.state_path.read_text(encoding="utf-8"))
        return PipelineState.model_validate(data)

    def save_state(self, state: PipelineState) -> None:
        self.state_path.write_text(state.model_dump_json(indent=2), encoding="utf-8")

    def mark_stage(self, state: PipelineState, stage: Stage) -> None:
        state.current_stage = stage
        if stage not in state.completed_stages and stage not in (
            Stage.PAUSE_1,
            Stage.PAUSE_2,
        ):
            state.completed_stages.append(stage)
        self.save_state(state)
