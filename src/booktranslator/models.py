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


class ChunkerConfig(BaseModel):
    target_words: int = 2000
    overlap_paragraphs: int = 1


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


class CostConfig(BaseModel):
    hard_limit_usd: float = 50.0


class Config(BaseModel):
    source_lang: str = "en"
    target_lang: str = "ru"
    chunker: ChunkerConfig = Field(default_factory=ChunkerConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    reflection: ReflectionConfig = Field(default_factory=ReflectionConfig)
    pauses: PausesConfig = Field(default_factory=PausesConfig)
    stages: StagesConfig = Field(default_factory=StagesConfig)
    retry: RetryConfig = Field(default_factory=RetryConfig)
    notifications: NotificationsConfig = Field(default_factory=NotificationsConfig)
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
