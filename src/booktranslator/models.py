"""Pydantic models shared across the pipeline."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class ModelsConfig(BaseModel):
    glossary: str = "anthropic/claude-sonnet-4.6"
    translate: str = "anthropic/claude-sonnet-4.6"
    judge: str = "anthropic/claude-haiku-4.5"
    reflect: str = "anthropic/claude-sonnet-4.6"
    proofread: str = "anthropic/claude-haiku-4.5"
    style: str = "anthropic/claude-sonnet-4.6"
    verify: str = "anthropic/claude-sonnet-4.6"
    cover: str = "google/gemini-3.1-flash-image-preview"


class ChunkerConfig(BaseModel):
    target_words: int = 2000
    overlap_paragraphs: int = 1


class TranslateConfig(BaseModel):
    """Runtime knobs for the translate stage."""

    parallelism: int = 1  # How many chunks to translate concurrently.


class ReflectionConfig(BaseModel):
    trigger_score: int = 3
    extended_thinking: bool = True


class PausesConfig(BaseModel):
    after_glossary: bool = True
    after_translate: bool = True


class StagesConfig(BaseModel):
    proofread: bool = True
    style: bool = True
    verify: bool = True


class RetryConfig(BaseModel):
    max_attempts: int = 5
    backoff_base: float = 2.0


class NotificationsConfig(BaseModel):
    enabled: bool = True
    provider: Literal["telegram", "none"] = "telegram"


class ReaderNotesConfig(BaseModel):
    enabled: bool = False
    types: list[str] = Field(default_factory=lambda: ["concept", "term"])
    scope: Literal["first-in-chapter", "first-in-book", "all"] = "first-in-book"


class CostConfig(BaseModel):
    hard_limit_usd: float = 50.0


class Config(BaseModel):
    source_lang: str = "en"
    target_lang: str = "ru"
    chunker: ChunkerConfig = Field(default_factory=ChunkerConfig)
    translate: TranslateConfig = Field(default_factory=TranslateConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    reflection: ReflectionConfig = Field(default_factory=ReflectionConfig)
    pauses: PausesConfig = Field(default_factory=PausesConfig)
    stages: StagesConfig = Field(default_factory=StagesConfig)
    retry: RetryConfig = Field(default_factory=RetryConfig)
    notifications: NotificationsConfig = Field(default_factory=NotificationsConfig)
    reader_notes: ReaderNotesConfig = Field(default_factory=ReaderNotesConfig)
    cost: CostConfig = Field(default_factory=CostConfig)


# ---------------------------------------------------------------------------
# Glossary
# ---------------------------------------------------------------------------


class GlossaryEntry(BaseModel):
    original: str
    translation: str
    type: Literal["person", "place", "concept", "term", "other"]
    gender: Literal["m", "f", "n", "unknown"] | None = None
    plural: str | None = None
    notes: str | None = None
    approved_by_human: bool = False


class Glossary(BaseModel):
    book: str
    author: str
    source_lang: str
    target_lang: str
    model: str
    entries: list[GlossaryEntry] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Series glossary (canonical, curated across multiple books)
# ---------------------------------------------------------------------------


class SeriesGlossaryEntry(BaseModel):
    """Curated entry used across all books in a series.

    Simplified compared to `GlossaryEntry`: no `approved_by_human` (by
    construction everything in a series glossary is approved). Tracks
    which book the entry originally came from, for auditing.
    """

    original: str
    translation: str
    type: Literal["person", "place", "concept", "term", "other"]
    gender: Literal["m", "f", "n", "unknown"] | None = None
    plural: str | None = None
    notes: str | None = None
    origin_book: str | None = None


class SeriesGlossary(BaseModel):
    series_slug: str
    title: str
    author: str
    source_lang: str
    target_lang: str
    entries: list[SeriesGlossaryEntry] = Field(default_factory=list)

    def index_by_original(self) -> dict[str, SeriesGlossaryEntry]:
        return {e.original: e for e in self.entries}


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class Stage(str, Enum):
    INIT = "init"
    EXTRACT = "extract"
    GLOSSARY = "glossary"
    PAUSE_1 = "pause_1"
    TRANSLATE = "translate"
    PAUSE_2 = "pause_2"
    DONE = "done"


class PipelineState(BaseModel):
    book_slug: str
    current_stage: Stage = Stage.INIT
    completed_stages: list[Stage] = Field(default_factory=list)
    total_cost_usd: float = 0.0
    last_error: str | None = None


# ---------------------------------------------------------------------------
# Extracted book metadata
# ---------------------------------------------------------------------------


class BookMeta(BaseModel):
    title: str
    author: str
    language: str
    word_count: int
    chapters: int
