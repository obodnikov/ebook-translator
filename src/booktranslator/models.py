"""Pydantic models shared across the pipeline."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class ProviderConfig(BaseModel):
    """Configuration for a single OpenAI-compatible provider endpoint."""

    base_url: str = "https://openrouter.ai/api/v1"
    api_key_env: str = "OPENROUTER_API_KEY"
    extra_headers: dict[str, str] = Field(default_factory=dict)
    # Largest response this endpoint can return, in bytes. Reasoning and answer
    # share it. None means no known ceiling (OpenRouter). kiro-gateway caps
    # around 23 000 and gives no warning of its own — see
    # docs/design/2026-08-03-judge-repair-stage-and-reasoning-on-gateway.md §4.
    max_response_bytes: int | None = None


class ProvidersConfig(BaseModel):
    """Independent provider endpoints. `text` is the default for all text
    stages; any stage may override it via its own optional field."""

    text: ProviderConfig = Field(default_factory=ProviderConfig)
    image: ProviderConfig = Field(default_factory=ProviderConfig)

    # Optional per-stage overrides. Unset -> fall back to `text`.
    glossary: ProviderConfig | None = None
    translate: ProviderConfig | None = None
    judge: ProviderConfig | None = None
    reflect: ProviderConfig | None = None
    proofread: ProviderConfig | None = None
    style: ProviderConfig | None = None
    verify: ProviderConfig | None = None
    repair: ProviderConfig | None = None


class ModelsConfig(BaseModel):
    glossary: str = "anthropic/claude-sonnet-4.6"
    translate: str = "anthropic/claude-sonnet-4.6"
    judge: str = "anthropic/claude-haiku-4.5"
    reflect: str = "anthropic/claude-sonnet-4.6"
    proofread: str = "anthropic/claude-haiku-4.5"
    style: str = "anthropic/claude-sonnet-4.6"
    verify: str = "anthropic/claude-sonnet-4.6"
    repair: str = "anthropic/claude-sonnet-4.6"
    cover: str = "google/gemini-3.1-flash-image-preview"


class ChunkerConfig(BaseModel):
    target_words: int = 2000
    overlap_paragraphs: int = 1


class TranslateConfig(BaseModel):
    """Runtime knobs for the translate stage."""

    parallelism: int = 1  # How many chunks to translate concurrently.


class ReflectionConfig(BaseModel):
    trigger_score: int = 3


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
    enabled: bool = True
    types: list[str] = Field(default_factory=lambda: ["concept", "term"])
    scope: Literal["first-in-chapter", "first-in-book", "all"] = "first-in-book"


class CostConfig(BaseModel):
    hard_limit_usd: float = 50.0


# Mapping from common ISO 639-1 codes to human-readable names for cover prompts.
# Extend as needed; absent codes require explicit target_lang_name in config.
_LANG_NAME_MAP: dict[str, str] = {
    "ru": "Russian",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "it": "Italian",
    "pt": "Portuguese",
    "nl": "Dutch",
    "pl": "Polish",
    "uk": "Ukrainian",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "ar": "Arabic",
    "tr": "Turkish",
    "sv": "Swedish",
    "no": "Norwegian",
    "da": "Danish",
    "fi": "Finnish",
    "cs": "Czech",
    "hu": "Hungarian",
}


class RepairConfig(BaseModel):
    """Targeted repair of the judge's findings.

    `categories` lists which judge issue categories the stage acts on. The
    default holds those where a correction is a substitution rather than a
    judgement call. Register and naturalness were measured to be beyond
    automatic repair and stay out.

    Accuracy was excluded on the assumption that it was "too mixed"; measuring
    it disproved that, and the same run disqualified glossary instead — see
    docs/design/2026-08-03-judge-repair-stage-and-reasoning-on-gateway.md §5
    for the counts.
    """

    enabled: bool = True
    categories: list[str] = Field(default_factory=lambda: ["grammar", "markup", "accuracy"])


class Config(BaseModel):
    source_lang: str = "en"
    target_lang: str = "ru"
    # Human-readable language name for cover-translation prompts.
    # Auto-derived from target_lang if not set explicitly.
    target_lang_name: str | None = None
    providers: ProvidersConfig = Field(default_factory=ProvidersConfig)
    chunker: ChunkerConfig = Field(default_factory=ChunkerConfig)
    translate: TranslateConfig = Field(default_factory=TranslateConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    reflection: ReflectionConfig = Field(default_factory=ReflectionConfig)
    pauses: PausesConfig = Field(default_factory=PausesConfig)
    stages: StagesConfig = Field(default_factory=StagesConfig)
    retry: RetryConfig = Field(default_factory=RetryConfig)
    notifications: NotificationsConfig = Field(default_factory=NotificationsConfig)
    reader_notes: ReaderNotesConfig = Field(default_factory=ReaderNotesConfig)
    repair: RepairConfig = Field(default_factory=RepairConfig)
    cost: CostConfig = Field(default_factory=CostConfig)

    def resolved_target_lang_name(self) -> str | None:
        """Return the effective human-readable target language name.

        Returns target_lang_name if explicitly set, otherwise looks up
        target_lang in the built-in ISO code map using the base language
        code (e.g. 'ru-RU' -> 'ru'). Returns None if the code is not in
        the map — callers must then require explicit input.
        """
        if self.target_lang_name:
            return self.target_lang_name
        # Normalize: lowercase, strip region subtag (e.g. 'ru-RU' -> 'ru')
        base = self.target_lang.lower().replace("_", "-").split("-", 1)[0]
        return _LANG_NAME_MAP.get(base)


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


class Stage(StrEnum):
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
