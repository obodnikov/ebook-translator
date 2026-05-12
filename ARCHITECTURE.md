# 🏗 Архитектура: book-translator

> CLI-инструмент для перевода EPUB-книг с английского на русский с помощью LLM через OpenRouter. Сохраняет исходную вёрстку, использует глоссарий для консистентности имён/терминов, поддерживает многоэтапную вычитку и возобновление после сбоя.

---

## Оглавление

1. [Цель и границы](#1-цель-и-границы)
2. [Модель использования](#2-модель-использования)
3. [Pipeline](#3-pipeline)
4. [Структура репозитория](#4-структура-репозитория)
5. [Ключевые модули и контракты](#5-ключевые-модули-и-контракты)
6. [Форматы данных](#6-форматы-данных)
7. [Промпты](#7-промпты)
8. [Модели и пресеты](#8-модели-и-пресеты)
9. [Ошибки, кэш, возобновление](#9-ошибки-кэш-возобновление)
10. [Оценка стоимости](#10-оценка-стоимости)
11. [Безопасность](#11-безопасность)
12. [Anti-scope и roadmap](#12-anti-scope-и-roadmap)
13. [Правила написания документации](#13-правила-написания-документации)

---

## 1. Цель и границы

### Цель

Перевести EPUB ~400–500 страниц (≈120–180K слов) с английского на русский так, чтобы:

- Вёрстка сохранилась один-в-один: те же XHTML-файлы, теги, CSS, картинки, cover, nav, TOC.
- Имена, места, неологизмы переводились консистентно на протяжении всей книги.
- Перевод был "хорошим фанатским": читаемый, без калек, с сохранением авторского стиля.
- Процесс был повторяемым (28 романов Чайковски — значит 28 одинаковых прогонов).
- Сбой посреди процесса не требовал начинать сначала.

### В scope

- EN → RU (единственная поддерживаемая пара в v1).
- EPUB 2/3 на входе и выходе.
- OpenRouter как единственный провайдер LLM.
- CLI + YAML-конфиги.
- Полуавтоматический процесс с двумя pause-точками и Telegram-уведомлениями.

### Не в scope

- Вёрстка PDF для типографии.
- GUI.
- Снятие DRM (делается отдельно через Calibre + DeDRM).
- Локальные модели (Ollama).
- Другие языковые пары (HU → RU и прочие — в roadmap).
- Собственный EPUB-парсер (используем `ebooklib` + `lxml`).
- ML-метрики качества (BLEU, COMET) — только LLM-as-judge + выборочная проверка человеком.

---

## 2. Модель использования

### Базовая команда

```bash
btrans translate path/to/book.epub
```

По умолчанию читает `configs/default.yaml`, создаёт рабочую директорию `work/<book-slug>/`, проходит pipeline с двумя паузами.

### Pause-точки

Скрипт **не блокирует** на ожидании ввода. Он:

1. Записывает текущее состояние в `work/<book-slug>/`.
2. Отправляет Telegram-уведомление со ссылкой на файл для ревью.
3. Завершается с exit code 0.

Пользователь возвращается когда угодно, правит нужный файл, запускает `--resume`:

```bash
btrans translate path/to/book.epub --resume
```

Скрипт определяет по состоянию рабочей директории, на каком шаге он стоит, и продолжает.

**Две pause-точки по умолчанию:**

| # | Когда | Что пауза просит сделать | Файл для ревью |
|---|---|---|---|
| 1 | После извлечения глоссария | Проверить имена/термины, при необходимости поправить | `work/<book>/glossary.json` |
| 2 | После перевода, до вычитки | Прочитать 1 главу, убедиться, что всё ок | `work/<book>/translated-preview.epub` |

Паузы можно пропустить флагом `--no-pause`, тогда pipeline идёт end-to-end (полезно для повторных прогонов после отладки глоссария).

### Режимы запуска

```bash
# Только извлечь глоссарий и остановиться
btrans glossary book.epub

# Использовать пресет premium (Opus 4.7 для перевода)
btrans translate book.epub --preset premium

# Перекрыть отдельные настройки
btrans translate book.epub --model-translate anthropic/claude-opus-4.7 --no-reflect

# Продолжить после правки glossary
btrans translate book.epub --resume

# Перегенерировать конкретную главу (сбросить кэш для неё)
btrans translate book.epub --invalidate-chapter ch07

# Dry-run: распарсить и показать стоимость, не делая запросов
btrans estimate book.epub
```

### Уведомления Telegram

Конфигурируются через `.env` (`TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`). События, по которым шлётся уведомление:

- Pause-точка 1: "Глоссарий готов, 247 терминов, жду проверки: file://..."
- Pause-точка 2: "Перевод готов, preview EPUB: file://...".
- Завершение: "Готово. Стоимость $12.40, время 2ч 17м, итоговый файл: ..."
- Ошибка: "Сбой на главе 18. Лог: ..."

### Запуск в фоне

Pipeline долгий (1–3 часа). Типичный запуск:

```bash
nohup btrans translate book.epub > run.log 2>&1 &
```

Скрипт сам закрывается на pause-точках и шлёт уведомление — долгоживущий процесс не нужен.

---

## 3. Pipeline

### Визуальная схема

```
┌────────────┐
│ EPUB input │
└──────┬─────┘
       ▼
┌──────────────────────┐
│ 1. Extract           │  ebooklib → {chapters: [xhtml...], metadata, assets}
│    (парсинг EPUB)    │
└──────┬───────────────┘
       ▼
┌──────────────────────┐
│ 2. Chunk             │  lxml → chunks.json: [{id, chapter, paragraphs, ~2000 words}]
│    (разбивка глав)   │
└──────┬───────────────┘
       ▼
┌──────────────────────┐
│ 3. Glossary extract  │  LLM pass → glossary.json (raw)
│    (pre-pass)        │
└──────┬───────────────┘
       ▼
  ═══════════════  PAUSE #1  ═══════════════
  Telegram: "Проверь glossary.json"
  User: редактирует, запускает --resume
  ══════════════════════════════════════════
       ▼
┌──────────────────────┐
│ 4. Translate         │  Для каждого chunk: LLM(text + glossary + overlap) → ru
│    (chunk-by-chunk)  │  → cache.sqlite
└──────┬───────────────┘
       ▼
┌──────────────────────┐
│ 5. Judge             │  LLM-as-judge: score 1–5 для каждого chunk
│    (quality scoring) │
└──────┬───────────────┘
       ▼
┌──────────────────────┐
│ 6. Reflect (условно) │  Для chunks со score ≤ 3: рефлексия по методу Ына
│                      │  → повторный перевод → cache.sqlite
└──────┬───────────────┘
       ▼
┌──────────────────────┐
│ 7. Assemble preview  │  lxml → вставка переводов в исходный XHTML
│                      │  → translated-preview.epub
└──────┬───────────────┘
       ▼
  ═══════════════  PAUSE #2  ═══════════════
  Telegram: "Посмотри preview EPUB"
  User: читает, запускает --resume
  ══════════════════════════════════════════
       ▼
┌──────────────────────┐
│ 8. Proofread         │  LLM: грамматика, пунктуация, опечатки
└──────┬───────────────┘
       ▼
┌──────────────────────┐
│ 9. Style edit        │  LLM: кальки, канцелярит, естественность
└──────┬───────────────┘
       ▼
┌──────────────────────┐
│ 10. Verify           │  LLM: сверка с оригиналом, пропуски, искажения
│     (с bilingual     │
│     контекстом)      │
└──────┬───────────────┘
       ▼
┌──────────────────────┐
│ 11. Assemble final   │  → book-ru.epub
│     + metadata       │  (язык = ru, title/author обновлены)
└──────┬───────────────┘
       ▼
┌────────────┐
│ EPUB output│
└────────────┘
```

### Детали шагов

#### Шаг 1: Extract

- Вход: путь к EPUB.
- Выход: в-памяти объект `Book` c полями:
  - `metadata` (title, author, lang, identifier)
  - `spine` (порядок XHTML-файлов)
  - `chapters` (dict: file_path → lxml tree)
  - `assets` (CSS, images, fonts — не трогаем, переносим as-is)
  - `nav` (nav.xhtml / toc.ncx)
- Библиотека: `ebooklib` для распаковки, `lxml` для парсинга XHTML.
- **Важно**: парсим в строгом режиме, сохраняем оригинальное дерево для последующей замены текстовых узлов.

#### Шаг 2: Chunk

- Вход: `Book.chapters`.
- Выход: `chunks.json` — плоский список объектов.
- Алгоритм:
  - Идём по главам в порядке spine.
  - Внутри главы собираем параграфы (`<p>`, `<blockquote>`, `<h1..h6>`) с их xpath в исходном XHTML.
  - Копим параграфы до ~2000 слов, закрываем chunk.
  - Сохраняем overlap-информацию: id предыдущего и следующего chunk-а в той же главе (их текст подаётся в промпт как контекст, но не переводится заново).
- Единица перевода — chunk, а не глава. Глава остаётся только как логическая группа.

#### Шаг 3: Glossary extract

- Вход: все chunks целиком (или по большим сегментам, если не влезает в контекст — Claude Sonnet 4.6 имеет 1M, обычно влезает весь роман).
- Промпт: `prompts/glossary_extract.md`.
- Выход: `glossary.json` — список терминов с полями `original`, `translation_suggested`, `type` (person / place / concept / term), `gender` (для персонажей, если определимо), `notes`.
- Модель по умолчанию: `anthropic/claude-sonnet-4.6`.
- **Важно**: LLM инструктирован возвращать ровно JSON по схеме, без вводных слов. Валидация pydantic.

#### Шаг 4: Translate

- Вход: chunk + glossary + overlap (параграф до и после).
- Промпт: `prompts/translate.md`.
- Выход: переведённый текст, структурированный по параграфам (соответствие 1:1 с оригиналом).
- Модель по умолчанию: `anthropic/claude-sonnet-4.6`.
- Кэш-ключ: `sha256(chunk_text + model + prompt_version + glossary_hash)`.
- **Контракт параграфов**: если в оригинале N параграфов, в переводе должно быть ровно N. Проверяется, при несовпадении — retry с более явным промптом.

#### Шаг 5: Judge

- Вход: оригинальный chunk + перевод.
- Промпт: `prompts/judge.md`.
- Выход: `{score: 1-5, issues: [...]}`.
- Модель: `anthropic/claude-haiku-4.5` (дёшево, задача простая).
- Критерии scoring: точность, естественность, консистентность с глоссарием, стиль.

#### Шаг 6: Reflect (условно)

- Триггер: score ≤ 3 ИЛИ флаг `--reflect-all`.
- Метод Эндрю Ына: рефлексия по переводу → список замечаний → повторный перевод с учётом замечаний.
- Два промпта: `prompts/reflect.md` (критика) + reuse `translate.md` с добавленным полем `reflection_notes`.
- Модель: `anthropic/claude-sonnet-4.6` с extended thinking включённым.

#### Шаг 7: Assemble preview

- Вход: все chunks с финальным переводом из кэша.
- Процесс: идём по исходным XHTML-деревьям, находим параграфы по xpath, заменяем их текстовое содержимое на перевод, **сохраняя inline-теги** (`<em>`, `<a>`, `<i>`).
- Выход: `translated-preview.epub` — полностью валидный EPUB для быстрого просмотра в ридере.

#### Шаги 8–10: Proofread / Style / Verify

Каждый — отдельный проход по chunks с своим промптом:

- **Proofread** (`prompts/proofread.md`): грамматика, пунктуация, опечатки. Модель: `claude-haiku-4.5`.
- **Style** (`prompts/style.md`): кальки, канцелярит, авторский голос. Модель: `claude-sonnet-4.6`.
- **Verify** (`prompts/verify.md`): сверка с оригиналом (bilingual context). Модель: `claude-sonnet-4.6`.

Все три опциональны и включаются флагами `--proofread`, `--style`, `--verify` (по умолчанию все три включены, выключить можно `--no-*`).

#### Шаг 11: Assemble final

- Финальная сборка с переводом после вычитки.
- Обновление метаданных: `dc:language = ru`, `dc:title` = переведённое название (из glossary или спросить у LLM отдельным вызовом), `dc:creator` как был (транслитерация опционально).
- Запаковка в `book-ru.epub`.

---

## 4. Структура репозитория

```
book-translator/
├── pyproject.toml              # зависимости, entry point btrans
├── README.md                   # quickstart
├── ARCHITECTURE.md             # этот документ
├── .env.example                # шаблон для OPENROUTER_API_KEY и Telegram
├── .gitignore                  # work/, .env, *.epub (приватные книги)
│
├── configs/
│   ├── default.yaml            # дефолтные модели, chunk size, pause-points
│   ├── premium.yaml            # preset для Чайковски (Opus 4.7)
│   └── budget.yaml             # preset эконом (Haiku 4.5)
│
├── prompts/
│   ├── glossary_extract.md     # pre-pass: извлечение имён/терминов
│   ├── translate.md            # основной перевод chunk
│   ├── reflect.md              # рефлексия по методу Ына
│   ├── judge.md                # scoring перевода 1–5
│   ├── proofread.md            # корректура
│   ├── style.md                # стилистическая правка
│   └── verify.md               # сверка с оригиналом
│
├── src/booktranslator/
│   ├── __init__.py
│   ├── cli.py                  # Typer или Click: btrans translate/glossary/estimate
│   ├── config.py               # загрузка YAML, merge с CLI-флагами, pydantic
│   ├── epub_io.py              # read/write EPUB, сохранение структуры
│   ├── chunker.py              # разбивка глав на chunks по параграфам
│   ├── glossary.py             # extract, merge, validate, apply
│   ├── translator.py           # оркестрация перевода одного chunk
│   ├── reviewer.py             # proofread / style / verify
│   ├── judge.py                # LLM-as-judge для scoring
│   ├── cache.py                # SQLite: chunk_hash → translated_text
│   ├── pipeline.py             # главная машина состояний, pause/resume
│   ├── provider.py             # OpenRouter client (OpenAI-compatible SDK)
│   ├── prompts.py              # загрузка .md с Jinja2, frontmatter parsing
│   ├── notifier.py             # Telegram-уведомления
│   ├── state.py                # работа с work/<book>/ и состоянием pipeline
│   └── models.py               # pydantic-модели: Chunk, GlossaryEntry, Config, ...
│
├── tests/
│   ├── test_chunker.py
│   ├── test_glossary.py
│   ├── test_epub_io.py         # round-trip: read → write → verify идентичность
│   ├── test_prompts.py         # рендер Jinja2, наличие плейсхолдеров
│   └── fixtures/
│       └── sample.epub         # маленькая тестовая книга
│
└── work/                       # gitignored
    └── <book-slug>/
        ├── original.epub
        ├── chunks.json
        ├── glossary.json       # ← человек правит тут
        ├── cache.sqlite
        ├── scores.json
        ├── translated-preview.epub
        ├── book-ru.epub
        ├── state.json          # текущая stage, позволяет resume
        ├── cost-report.json
        └── log.jsonl
```

---

## 5. Ключевые модули и контракты

### `epub_io.py`

```python
class EpubReader:
    def read(self, path: Path) -> Book: ...

class EpubWriter:
    def write(self, book: Book, path: Path) -> None: ...
    # Гарантирует байт-в-байт сохранение assets, CSS, картинок.
    # Изменяет только текстовые узлы внутри XHTML и mime-metadata (lang).
```

**Контракт**: `EpubWriter(book).write(path); EpubReader.read(path) == book` для случая, когда translations пусты (round-trip identity для неизменённой книги).

### `chunker.py`

```python
@dataclass
class Paragraph:
    xpath: str                  # путь в XHTML дереве главы
    text: str                   # плоский текст, inline-теги сохранены отдельно
    inline_markers: list[InlineMarker]  # позиции <em>, <a>, etc для reconstruction

@dataclass
class Chunk:
    id: str                     # ch03_c02
    chapter_file: str           # OEBPS/chapter03.xhtml
    paragraphs: list[Paragraph]
    word_count: int
    prev_overlap: list[Paragraph]  # 1 параграф из предыдущего chunk
    next_overlap: list[Paragraph]

class Chunker:
    def __init__(self, target_words: int = 2000, overlap_paragraphs: int = 1): ...
    def chunk_book(self, book: Book) -> list[Chunk]: ...
```

**Контракт**: `reassemble(chunks) == book.chapters` (по тексту и xpath параграфов).

### `glossary.py`

```python
@dataclass
class GlossaryEntry:
    original: str
    translation: str
    type: Literal["person", "place", "concept", "term", "other"]
    gender: Literal["m", "f", "n", "unknown"] | None
    plural: str | None          # для русских склонений
    notes: str | None
    approved_by_human: bool = False

class Glossary:
    entries: list[GlossaryEntry]

    def to_prompt_section(self) -> str: ...
    def hash(self) -> str: ...
    @classmethod
    def from_json(cls, path: Path) -> "Glossary": ...
    def to_json(self, path: Path) -> None: ...
```

### `translator.py`

```python
class Translator:
    def __init__(self, provider: Provider, prompt: Prompt, glossary: Glossary): ...

    async def translate_chunk(self, chunk: Chunk) -> TranslatedChunk:
        # 1. Build prompt with glossary + overlap context
        # 2. Check cache
        # 3. If miss: call provider
        # 4. Validate paragraph count matches
        # 5. Store in cache
        # 6. Return TranslatedChunk
```

### `pipeline.py`

```python
class Stage(Enum):
    EXTRACT = "extract"
    CHUNK = "chunk"
    GLOSSARY = "glossary"
    PAUSE_1 = "pause_1"
    TRANSLATE = "translate"
    JUDGE = "judge"
    REFLECT = "reflect"
    ASSEMBLE_PREVIEW = "assemble_preview"
    PAUSE_2 = "pause_2"
    PROOFREAD = "proofread"
    STYLE = "style"
    VERIFY = "verify"
    ASSEMBLE_FINAL = "assemble_final"
    DONE = "done"

class Pipeline:
    def run(self, resume: bool = False) -> None:
        # Читает state.json, определяет текущий stage, идёт дальше.
        # На pause-stage: пишет файл для ревью, шлёт Telegram, exit.
```

### `provider.py`

```python
class OpenRouterProvider:
    def __init__(self, api_key: str, app_name: str = "book-translator"): ...

    async def complete(
        self,
        model: str,
        messages: list[Message],
        temperature: float = 0.3,
        max_tokens: int | None = None,
        extended_thinking: bool = False,
    ) -> CompletionResult:
        # Использует openai SDK с base_url=https://openrouter.ai/api/v1
        # Retry с экспоненциальным backoff на 429, 500, 502, 503.
        # Логирует стоимость в work/<book>/log.jsonl.
```

Единственный провайдер. Разные модели — через поле `model` (например `anthropic/claude-sonnet-4.6`, `openai/gpt-5.4-mini`).

### `cache.py`

```python
class Cache:
    def __init__(self, db_path: Path): ...

    def get(self, key: str) -> CachedEntry | None: ...
    def put(self, key: str, value: str, metadata: dict) -> None: ...

    @staticmethod
    def make_key(
        chunk_text: str,
        model: str,
        prompt_version: str,
        glossary_hash: str,
        stage: str,
    ) -> str:
        # sha256 всего вместе
```

SQLite схема:

```sql
CREATE TABLE cache (
    key TEXT PRIMARY KEY,
    chunk_id TEXT,
    stage TEXT,           -- translate, reflect, proofread, style, verify
    model TEXT,
    prompt_version TEXT,
    input_tokens INTEGER,
    output_tokens INTEGER,
    cost_usd REAL,
    created_at TEXT,
    content TEXT
);
CREATE INDEX idx_chunk_stage ON cache(chunk_id, stage);
```

### `notifier.py`

```python
class TelegramNotifier:
    def __init__(self, bot_token: str, chat_id: str): ...
    def send(self, text: str, file_path: Path | None = None) -> None: ...

class NullNotifier:
    # Для тестов и если Telegram не настроен.
    def send(self, *args, **kwargs) -> None: ...
```

---

## 6. Форматы данных

### `configs/default.yaml`

```yaml
source_lang: en
target_lang: ru

chunker:
  target_words: 2000
  overlap_paragraphs: 1

models:
  glossary:   anthropic/claude-sonnet-4.6
  translate:  anthropic/claude-sonnet-4.6
  judge:      anthropic/claude-haiku-4.5
  reflect:    anthropic/claude-sonnet-4.6
  proofread:  anthropic/claude-haiku-4.5
  style:      anthropic/claude-sonnet-4.6
  verify:     anthropic/claude-sonnet-4.6

reflection:
  trigger_score: 3       # reflect если judge <= 3
  extended_thinking: true

pauses:
  after_glossary: true
  after_translate: true

stages:
  proofread: true
  style: true
  verify: true

retry:
  max_attempts: 5
  backoff_base: 2.0      # секунды

notifications:
  enabled: true
  provider: telegram

cost:
  hard_limit_usd: 50     # аборт если превысит
```

### `glossary.json`

```json
{
  "book": "Empire in Black and Gold",
  "author": "Adrian Tchaikovsky",
  "source_lang": "en",
  "target_lang": "ru",
  "generated_at": "2026-05-03T14:12:00Z",
  "model": "anthropic/claude-sonnet-4.6",
  "entries": [
    {
      "original": "Kinden",
      "translation": "киндены",
      "type": "concept",
      "gender": null,
      "plural": "киндены",
      "notes": "Раса/вид в мире Чайковски, люди с чертами насекомых",
      "approved_by_human": false
    },
    {
      "original": "Stenwold Maker",
      "translation": "Стенвольд Мейкер",
      "type": "person",
      "gender": "m",
      "plural": null,
      "notes": "Главный герой, учёный и дипломат",
      "approved_by_human": false
    }
  ]
}
```

Поле `approved_by_human` пользователь выставляет в `true` после ручной проверки. Pipeline проверяет, что все entries проверены (или warning, если нет).

### `chunks.json`

```json
{
  "book_slug": "empire-in-black-and-gold",
  "total_words": 167432,
  "target_words_per_chunk": 2000,
  "chunks": [
    {
      "id": "ch01_c01",
      "chapter_file": "OEBPS/chapter01.xhtml",
      "word_count": 1987,
      "paragraph_xpaths": [
        "/html/body/div[1]/p[1]",
        "/html/body/div[1]/p[2]"
      ],
      "prev_overlap_ids": [],
      "next_overlap_ids": ["ch01_c02"]
    }
  ]
}
```

Сам текст параграфов в `chunks.json` не храним (избыточно). Берём из исходного EPUB по xpath.

### `state.json`

```json
{
  "book_slug": "empire-in-black-and-gold",
  "started_at": "2026-05-03T10:00:00Z",
  "current_stage": "pause_1",
  "completed_stages": ["extract", "chunk", "glossary"],
  "config_path": "configs/default.yaml",
  "total_cost_usd": 2.14,
  "last_error": null
}
```

### `cost-report.json`

Финальный отчёт после DONE:

```json
{
  "total_cost_usd": 12.47,
  "by_stage": {
    "glossary": 0.82,
    "translate": 4.31,
    "judge": 0.11,
    "reflect": 1.22,
    "proofread": 0.95,
    "style": 3.18,
    "verify": 1.88
  },
  "by_model": {
    "anthropic/claude-sonnet-4.6": 11.41,
    "anthropic/claude-haiku-4.5": 1.06
  },
  "tokens": {
    "input": 412330,
    "output": 498221
  },
  "chunks": 84,
  "reflected_chunks": 19,
  "duration_seconds": 8241
}
```

---

## 7. Промпты

### Формат файла

Каждый промпт — markdown с YAML frontmatter. Пример `prompts/translate.md`:

```markdown
---
version: 3
model: anthropic/claude-sonnet-4.6
temperature: 0.3
max_tokens: 8000
---

# System

You are a professional literary translator from {{ source_lang_name }} to {{ target_lang_name }}.
You translate fiction for experienced readers of {{ target_lang_name }} literature.

Your priorities:
1. Accuracy — no omissions, no additions.
2. Natural {{ target_lang_name }} — no calques, no translationese.
3. Author's voice — preserve tone, rhythm, register.
4. Consistency — use the glossary exactly as given.

# User

## Glossary
{{ glossary_json }}

## Overlap context (previous paragraph, do NOT translate)
{{ prev_overlap_text }}

## Text to translate
{{ chunk_text }}

## Overlap context (next paragraph, do NOT translate)
{{ next_overlap_text }}

## Output format
Return only the translation, paragraph-per-paragraph, matching input paragraph count exactly.
Separate paragraphs with a single blank line.
No preamble, no commentary, no markdown.
```

### Плейсхолдеры (Jinja2)

| Плейсхолдер | Заполняется из |
|---|---|
| `{{ source_lang }}` / `{{ source_lang_name }}` | config.source_lang → "en" / "English" |
| `{{ target_lang }}` / `{{ target_lang_name }}` | config.target_lang → "ru" / "Russian" |
| `{{ chunk_text }}` | текст текущего chunk |
| `{{ prev_overlap_text }}` / `{{ next_overlap_text }}` | overlap параграфы |
| `{{ glossary_json }}` | glossary.to_prompt_section() |
| `{{ reflection_notes }}` | для reflect-прохода |
| `{{ style_notes }}` | опциональный раздел из конфига (авторский стиль) |

### Версионирование

`version: N` в frontmatter. Попадает в cache key. Изменили промпт → версию поднимаете вручную → кэш инвалидируется.

### Загрузка в коде

```python
@dataclass
class Prompt:
    name: str
    version: str
    model: str
    temperature: float
    max_tokens: int | None
    system: str          # Jinja2 template
    user: str            # Jinja2 template

def load_prompt(path: Path) -> Prompt: ...
def render_prompt(prompt: Prompt, context: dict) -> tuple[str, str]: ...
```

---

## 8. Модели и пресеты

### Дефолт (`configs/default.yaml`)

| Этап | Модель | Обоснование |
|---|---|---|
| Glossary extract | `anthropic/claude-sonnet-4.6` | 1M контекст, хороший structured output |
| Translate | `anthropic/claude-sonnet-4.6` | Флагман цена/качество для fiction |
| Judge | `anthropic/claude-haiku-4.5` | Дёшево, задача простая |
| Reflect | `anthropic/claude-sonnet-4.6` (extended thinking on) | Глубокая критика |
| Proofread | `anthropic/claude-haiku-4.5` | Грамматика/пунктуация не требует фронтира |
| Style | `anthropic/claude-sonnet-4.6` | Тонкое чувство языка |
| Verify | `anthropic/claude-sonnet-4.6` | Bilingual контекст |

### Preset `premium`

Override для translate и reflect на `anthropic/claude-opus-4.7`. Для авторов, где критичен стиль (Чайковски, Ле Гуин, Мьевиль).

### Preset `budget`

Translate + glossary на `anthropic/claude-haiku-4.5`, остальное на `openai/gpt-5.4-mini`. Для non-fiction и документалки.

### Смена модели без пересборки кэша

Смена модели — это новый cache key, старые переводы остаются в кэше, новые пишутся отдельно. Безопасно экспериментировать.

---

## 9. Ошибки, кэш, возобновление

### Уровни сбоя и реакция

| Уровень | Пример | Реакция |
|---|---|---|
| Transient (HTTP 429/500/502/503) | Rate limit OpenRouter | Retry с backoff до 5 раз |
| Permanent (HTTP 400, invalid JSON) | Промпт сломан, модель вернула не JSON | Log, skip chunk, пометить в state.json, продолжить с следующих |
| Network | Timeout | Retry с backoff, при 5 подряд — pause + уведомление |
| Budget exceeded | Превышен `cost.hard_limit_usd` | Abort, уведомление, state сохранён |
| EPUB malformed | Невалидный XHTML | Fail fast на stage EXTRACT, нет смысла продолжать |

### Кэш

- Ключ: `sha256(chunk_text + model + prompt_version + glossary_hash + stage)`.
- Значение: переведённый/отредактированный текст + метаданные (tokens, cost, timestamp).
- Операции:
  - При старте stage: для каждого chunk спрашиваем кэш, попавшие — пропускаем.
  - Новый результат → в кэш сразу после валидации.
  - Инвалидация: `btrans cache invalidate --chunk ch07 --stage translate` или сменой `prompt_version` / модели.

### Возобновление

`state.json` хранит `current_stage`. Логика `--resume`:

1. Загрузить `state.json`.
2. Если `current_stage == PAUSE_1`, проверить что `glossary.json` существует и валиден, перейти на `TRANSLATE`.
3. Если `current_stage == PAUSE_2`, проверить `translated-preview.epub`, перейти на `PROOFREAD`.
4. Иначе — продолжить с current_stage, полагаясь на кэш для уже обработанных chunks.

Без `--resume` на непустой work/ → spросить у пользователя через CLI: "Found existing work. Resume? (y/n)".

---

## 10. Оценка стоимости

### На 150K слов входа (≈ 200K tokens) и 180K tokens выхода

Расчёт по дефолту (Sonnet 4.6 / Haiku 4.5):

| Этап | Input | Output | Модель | Стоимость |
|---|---|---|---|---|
| Glossary | 200K | 15K | Sonnet 4.6 | $0.83 |
| Translate | 250K (с overlap и glossary) | 180K | Sonnet 4.6 | $3.45 |
| Judge | 430K (orig + trans) | 5K | Haiku 4.5 | $0.46 |
| Reflect (30% chunks) | 130K | 60K | Sonnet 4.6 | $1.29 |
| Proofread | 200K | 200K | Haiku 4.5 | $1.20 |
| Style | 400K | 250K | Sonnet 4.6 | $4.95 |
| Verify | 430K | 30K | Sonnet 4.6 | $1.74 |
| **Итого** | | | | **~$14** |

Плюс маржа OpenRouter ~5% → **~$15**.

Для preset `premium` (Opus 4.7 на translate + reflect): **~$22–25**.

Для preset `budget` (Haiku + gpt-5.4-mini): **~$4–6**.

### Hard limit

`cost.hard_limit_usd` в конфиге. Если накопленная стоимость в `log.jsonl` превысит — abort, Telegram-уведомление. Default 50.

### Команда `btrans estimate book.epub`

Оценивает стоимость без запросов к LLM: парсит EPUB, считает токены, применяет прайсинг из `configs/pricing.yaml`. Возвращает таблицу по этапам. Полезно перед запуском.

---

## 11. Безопасность

### Ключи и секреты

- `OPENROUTER_API_KEY`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` — только через `.env` или env vars.
- `.env` в `.gitignore`.
- `.env.example` — с пустыми значениями, коммитится.
- В `log.jsonl` токены не пишем; в `cost-report.json` только агрегированную стоимость.

### DRM и правовое

- Скрипт работает только с уже расшифрованными EPUB. DRM не снимаем.
- В README явно: "снятие DRM — вне scope, используйте Calibre + DeDRM на свой риск. Используйте инструмент только для личного перевода легально приобретённых книг."
- `.gitignore` исключает `*.epub` на уровне корня, чтобы случайно не закоммитить чужую книгу.

### Данные книги

- OpenRouter и провайдеры получают полный текст книги через API. Политика хранения — смотри [OpenRouter privacy](https://openrouter.ai/privacy) и политики Anthropic/OpenAI.
- Опционально: флаг `--zero-data-retention` (OpenRouter поддерживает) для более строгой политики, если доступен на выбранной модели.

---

## 12. Anti-scope и roadmap

### Anti-scope (v1)

- Печать и вёрстка PDF.
- GUI.
- Многоязычность (HU, DE, FR, JP...).
- Локальные модели.
- Собственный EPUB-парсер.
- Автоматическая постпечатная проверка качества (ML-метрики).
- Параллельный перевод несколькими модерями с голосованием.

### Roadmap

**v1.0** (MVP):
- EN → RU.
- Весь pipeline с двумя паузами.
- Дефолт + premium + budget пресеты.
- Telegram-уведомления.
- SQLite-кэш, resume.

**v1.1**:
- `btrans estimate` для оценки стоимости до запуска.
- `btrans cache` для управления кэшем.
- Параллельная обработка chunks (текущая — последовательная, для простоты).

**v1.2**:
- HU → RU. Требует: language-aware промпты (про отсутствие рода в венгерском), тесты на HU-образце, новый пресет.

**v1.3** (reader notes from glossary):
- Использовать глоссарий как источник подстрочных читательских сносок
  в финальном EPUB. Никаких новых полей в модели — используем
  существующие `type` + `notes` + `translation`.
- Мотивация: русскому читателю британский криминально-мистический роман
  непонятен без контекста (DCI, районы Лондона, отсылки к поп-культуре,
  латинские заклинания). Официальные переводы делают это сносками —
  у нас есть ровно та же информация в `notes` каждой записи.
- Механизм — вариант C (детерминистическая пост-обработка при сборке):
  - Перевод идёт как обычно, без сносок.
  - На этапе `btrans assemble --notes`: фильтруем glossary по
    `type in config.reader_notes.types` AND `notes` не пустой.
  - Для каждой главы (в порядке spine) ищем `entry.translation` в
    тексте (точный substring match, case-insensitive).
  - Первое вхождение (по scope) оборачиваем в EPUB footnote-ref,
    `notes` идёт в aside.
- Почему не вариант A (инжекция в промпт перевода): перевод идёт
  chunk-ами параллельно, модель не знает в каком chunk-е термин
  встречается впервые.
- Почему не вариант B (pymorphy3 + fuzzy match): хрупко и дорого.
- Контроль читательской нагрузки:
  - Конфиг `reader_notes.types: [concept, term]` (дефолт).
  - Конфиг `reader_notes.scope: first-in-chapter | first-in-book | all`.
  - `--notes` / `--no-notes` в CLI.
- Текст сноски = `notes` из glossary (пока на английском).

**v1.4** (перевод notes на target_lang):
- Батчевый перевод `notes` для concept/term записей через Haiku.
- Сохраняется отдельно, не трогая series glossary.
- `btrans assemble --notes` использует переведённые если доступны.

**v2.0**:
- Обобщение на любую пару языков через `source_lang` / `target_lang` в конфиге.
- Набор language-specific промпт-модификаторов.
- Возможно, веб-интерфейс для правки глоссария.

**v2.x** (идеи):
- Интеграция с Calibre-библиотекой (массовый перевод серии).
- Автоматическая валидация EPUB через EPUBCheck.
- Экспорт bilingual EPUB (оригинал + перевод бок-о-бок).

---

## 13. Правила написания документации

Вся пользовательская документация проекта (README, руководства, комментарии в конфигах) пишется на русском языке с соблюдением следующих правил:

### Язык и стиль

- Писать литературным русским языком, понятным широкому читателю — не только разработчику.
- Избегать программистского жаргона и сленга: не «чекнуть», а «проверить»; не «пропатчить», а «исправить»; не «задеплоить», а «развернуть».
- Обращение к читателю — на «вы».
- Тон — спокойный, деловой, без панибратства и без канцелярита.

### Иностранные термины и сокращения

- При первом упоминании иностранного термина или аббревиатуры давать русский аналог, а в скобках — оригинальное сокращение. Формат: **русский аналог (английский акроним)**.
  - Примеры: программа командной строки (CLI), защита от копирования (DRM), ограничение частоты запросов (rate limit), языковая модель (LLM).
- При повторных упоминаниях допустимо использовать только русский вариант или только аббревиатуру, если она уже была расшифрована.
- Устоявшиеся технические имена собственные (EPUB, Python, SQLite, OpenRouter) не переводятся.

### Примеры команд и код

- Примеры команд в терминале, имена файлов, фрагменты кода — оставлять как есть, без перевода.
- Комментарии внутри примеров кода допустимо писать на русском.

### Структура документа

- Заголовки — на русском.
- Нумерованные списки — для последовательных шагов.
- Маркированные списки — для перечислений без порядка.
- Избегать необоснованных сокращений в прозе (не «доки», а «документация»; не «конфиг», а «файл настроек» или «конфигурация»).

---

## Приложение: стэк и зависимости

- **Python 3.12+** (pattern matching, StrEnum, async/await).
- **uv** или **poetry** для управления зависимостями.
- **ebooklib** — парсинг EPUB.
- **lxml** — XHTML tree с сохранением структуры.
- **openai** (Python SDK) — через OpenRouter base_url.
- **pydantic v2** — модели данных, валидация.
- **typer** — CLI.
- **jinja2** — шаблоны промптов.
- **python-frontmatter** — парсинг markdown frontmatter.
- **PyYAML** — конфиги.
- **tenacity** — retry-логика.
- **httpx** — HTTP-клиент для Telegram Bot API.
- **rich** — CLI-вывод, progress bars, таблицы.
- **pytest** — тесты.

---

*Документ составлен 3 мая 2026 года.*
