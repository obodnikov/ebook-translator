# Design: встраивание перевода обложки и читательских сносок в пайплайн `translate`

> Status: **proposed** (план, не реализовано)
> Date: 2026-06-30
> Scope: `cli.py`, `reader_notes.py`, `cover.py` (без изменений API), `models.py`,
> `configs/default.yaml`, README, тесты.
> Related rules: [AI_PIPELINE.md](../../AI_PIPELINE.md), [AI_PROVIDER.md](../../AI_PROVIDER.md),
> [AI_EPUB.md](../../AI_EPUB.md), [ARCHITECTURE.md](../../ARCHITECTURE.md).

---

## 1. Задача

Две финишные операции — **перевод обложки** (`cover translate`) и **читательские сноски
из глоссария** (`assemble --notes`) — сейчас живут отдельными ручными командами и
выполняются после `btrans translate`. Нужно встроить их в основной прогон `translate`,
чтобы полный результат (переведённый текст + сноски + русская обложка) собирался одной
командой, сохраняя возможность управлять каждым шагом флагами.

### Решения, принятые до начала (заказчик)

1. Встраиваем **оба** шага (notes + cover).
2. Reader notes **включены по умолчанию** (`reader_notes.enabled: true`).
3. Cover остаётся **opt-in** (`--cover`, по умолчанию выключен): стоит денег и для сложных
   обложек неидеален.

---

## 2. Текущее состояние (где «вне пайплайна»)

### 2.1 `translate`

`btrans translate` ([cli.py:381](../../src/booktranslator/cli.py)) в конце:
1. rehydrate деревьев из waterfall-кэша (`rehydrate_book_from_waterfall`);
2. `write_translated_epub(source, dest, modified_chapters, new_language)` — **без сносок,
   без обложки**.

Глоссарий уже загружен в переменную `glossary` как `SeriesGlossary` (и для `--series`,
и для `--glossary` — последний адаптируется к форме `SeriesGlossary`,
[cli.py:494-534](../../src/booktranslator/cli.py)).

### 2.2 Reader notes

Реализованы только в `assemble --notes`
([cli.py:2419-2481](../../src/booktranslator/cli.py)):
- `load_notes_glossary(series, glossary_path, work_dir, title, author, src, tgt)` →
  `SeriesGlossary | None`;
- `resolve_notes_config(cfg.reader_notes, note_types_override)` → `ReaderNotesConfig`;
- `inject_reader_notes(chapters, glossary, config)` → `InjectionStats` — детерминированно
  оборачивает первое вхождение `translation` в EPUB-сноску, текст берёт из поля `notes`.
  **LLM не вызывается, денег не стоит.** Модифицирует lxml-деревья in-place; вызывать
  ПОСЛЕ rehydrate и ПЕРЕД `write_translated_epub` ([reader_notes.py:219-242](../../src/booktranslator/reader_notes.py)).

`inject_reader_notes` принимает именно `SeriesGlossary` — ровно то, что у `translate`
уже есть. Интеграция тривиальна по типам.

### 2.3 Cover translate

Отдельная команда `cover translate`
([cli.py:2637](../../src/booktranslator/cli.py)), операция **файл→файл** через
**image-провайдер** (`create_image_provider(cfg)`):

```python
translate_cover(
    source_epub, dest_epub, provider, *,
    model, title_translation=None, author_name=None,
    target_lang="Russian", prompt_template=None,
    aspect_ratio="2:3", image_size="1K",
) -> ImageGenerationResult
```

`target_lang` здесь — **человекочитаемое имя** ("Russian"), а не код "ru".

---

## 3. Принцип: это «финишные» операции, а не стадии waterfall

Ни сноски, ни обложка не являются кэшируемыми стадиями перевода (`Stage` enum,
`STAGE_WATERFALL` в [cache.py](../../src/booktranslator/cache.py)):

- **Сноски** — детерминированная пост-обработка деревьев глав. Нет вызова модели,
  нечего кэшировать, paragraph-count contract не затрагивается (footnote-ref
  оборачивает текст внутри абзаца, число абзацев не меняется; `<aside>` добавляется
  в `<body>`, не в поток абзацев).
- **Обложка** — один image-вызов на книгу, не chunk-stage.

Поэтому **не трогаем** state-машину, кэш-ключи, `Stage` enum и paragraph-count
контракт. Сноски и обложка — это **опции сборки финального EPUB**, встраиваемые вокруг
`write_translated_epub`. Это согласуется с [AI_PIPELINE.md](../../AI_PIPELINE.md)
(«No EPUB tree manipulation or assembly — that's AI_EPUB.md») и
[AI_EPUB.md](../../AI_EPUB.md).

---

## 4. Целевой порядок шагов в `translate`

```
read_book_structured
  → chunk_book
  → translate (chunks)        [LLM, text provider, cached]
  → judge / reflect           [LLM, text provider, cached]
  → proofread / style / verify[LLM, text provider, cached]
  → rehydrate_book_from_waterfall
  → inject_reader_notes        ← НОВОЕ (детерминированно, бесплатно, default ON)
  → write_translated_epub      → out_path
  → translate_cover            ← НОВОЕ (image provider, $, default OFF, --cover)
       source=out_path, dest=tmp → os.replace(tmp, out_path)
```

Сноски — **до** записи (модифицируют деревья). Обложка — **после** записи (операция
файл→файл над готовым EPUB).

---

## 5. Изменения по файлам

### 5.1 `src/booktranslator/models.py`

```python
class ReaderNotesConfig(BaseModel):
    enabled: bool = True          # было False — теперь ON по умолчанию
    types: list[str] = Field(default_factory=lambda: ["concept", "term"])
    scope: Literal["first-in-chapter", "first-in-book", "all"] = "first-in-book"

class Config(BaseModel):
    source_lang: str = "en"
    target_lang: str = "ru"
    target_lang_name: str = "Russian"   # НОВОЕ: имя языка для промпта обложки
    ...
```

`target_lang_name` нужно, потому что `translate_cover` хочет "Russian", а в конфиге
только код "ru". Standalone-команда `cover translate` сохраняет свой обязательный
`--target-lang`; пайплайн берёт значение из `cfg.target_lang_name`
(переопределяется флагом `--cover-target-lang`).

### 5.2 `configs/default.yaml`

```yaml
source_lang: en
target_lang: ru
target_lang_name: Russian      # НОВОЕ

reader_notes:
  enabled: true                # было false
  types: [concept, term]
  scope: first-in-book
```

> ⚠️ **Побочный эффект:** `reader_notes.enabled: true` влияет и на standalone
> `assemble` — он тоже начнёт вставлять сноски без явного `--notes`. Это прямое
> следствие «включить по умолчанию». `--no-notes` по-прежнему отключает.

### 5.3 `src/booktranslator/reader_notes.py` — общий хелпер (DRY)

Сейчас оркестрация сносок (загрузка глоссария → resolve config → inject → печать
статистики) сидит внутри `assemble_cmd`. Вынести в переиспользуемую функцию, чтобы
`translate` и `assemble` звали один путь.

```python
def inject_notes_for_cli(
    *,
    console,
    chapters: list,                       # book.chapters
    cfg: Config,
    notes_flag: bool | None,              # из --notes/--no-notes (None = из конфига)
    note_types: str | None,               # из --note-types
    series: str | None,
    glossary_path: Path | None,
    work_dir: Path,
    book_title: str,
    book_author: str,
    preloaded_glossary: SeriesGlossary | None = None,  # translate передаёт готовый
) -> InjectionStats | None:
    """Resolve config + glossary and inject reader notes into chapter trees.

    Returns InjectionStats if notes were attempted, None if disabled.
    Prints the same warnings/summary that assemble_cmd printed.
    """
```

Логика:
1. `notes_enabled = notes_flag if notes_flag is not None else cfg.reader_notes.enabled`.
   Если выключено → `return None`.
2. Глоссарий: если `preloaded_glossary` передан — использовать его (путь `translate`);
   иначе `load_notes_glossary(...)` (путь `assemble`).
3. Те же предупреждения, что сейчас в `assemble_cmd` (нет глоссария / серия не найдена
   → skip).
4. `resolve_notes_config(cfg.reader_notes, note_types_override=note_types)` →
   `inject_reader_notes(...)` → печать статистики.

`assemble_cmd` заменяет свой блок ([cli.py:2419-2481](../../src/booktranslator/cli.py))
на вызов этого хелпера (передаёт `preloaded_glossary=None`).

### 5.4 `src/booktranslator/cli.py` — команда `translate`

**Новые опции:**

| Флаг | Тип | Default | Назначение |
|------|-----|---------|------------|
| `--notes/--no-notes` | `bool \| None` | `None` (→ конфиг) | вкл/выкл сноски |
| `--note-types` | `str \| None` | `None` | типы для аннотирования (`concept,term,place`) |
| `--cover/--no-cover` | `bool` | `False` | перевести обложку после сборки |
| `--cover-title` | `str \| None` | `None` | точный перевод названия для обложки |
| `--cover-author` | `str \| None` | `None` | имя автора для обложки |
| `--cover-model` | `str \| None` | `None` | override image-модели (→ `cfg.models.cover`) |
| `--cover-target-lang` | `str \| None` | `None` | override `cfg.target_lang_name` |
| `--cover-aspect-ratio` | `str` | `"2:3"` | соотношение сторон |
| `--cover-image-size` | `str` | `"1K"` | разрешение `1K/2K/4K` |

**Сноски** — вставка после rehydrate, перед `write_translated_epub`
([вокруг cli.py:908-919](../../src/booktranslator/cli.py)):

```python
note_stats = inject_notes_for_cli(
    console=console,
    chapters=book.chapters,
    cfg=cfg,
    notes_flag=notes,
    note_types=note_types,
    series=series,
    glossary_path=glossary_path,
    work_dir=work_dir,
    book_title=book.meta.title,
    book_author=book.meta.author,
    preloaded_glossary=glossary,   # уже загружен выше
)

modified = [ch for ch in book.chapters if ch.paragraphs]
write_translated_epub(source_path=epub, dest_path=out_path,
                      modified_chapters=modified, new_language=cfg.target_lang)
```

**Обложка** — после записи EPUB (файл→файл, temp + atomic replace):

```python
if cover:
    from .cover import translate_cover
    import os, tempfile
    image_provider = create_image_provider(cfg)
    cover_lang = cover_target_lang or cfg.target_lang_name
    cover_model = cover_model_opt or cfg.models.cover
    fd, tmp_name = tempfile.mkstemp(suffix=".epub", dir=out_path.parent)
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        result = translate_cover(
            source_epub=out_path, dest_epub=tmp,
            provider=image_provider, model=cover_model,
            title_translation=cover_title, author_name=cover_author,
            target_lang=cover_lang,
            aspect_ratio=cover_aspect_ratio, image_size=cover_image_size,
        )
        os.replace(tmp, out_path)             # atomic, заменяет out_path
        console.print(f"[green]Cover translated.[/green] {result.mime_type}, "
                      f"{len(result.image_bytes):,} bytes")
    except Exception as e:                     # noqa: BLE001 — best-effort
        tmp.unlink(missing_ok=True)
        console.print(f"[yellow]Cover translation failed (EPUB kept):[/yellow] {e}")
```

**Почему temp + `os.replace`, а не `dest=out_path`:** `translate_cover` сначала читает
исходный EPUB (`find_cover_in_epub` + `_replace_cover_with_manifest_update` перечитывает
zip), затем пишет dest. Читать и писать один и тот же путь — небезопасно. Пишем во
временный файл рядом (та же ФС → `os.replace` атомарен), затем заменяем.

**Почему best-effort (try/except, не падаем):** обложка — необязательный финиш и зависит
от внешнего image-провайдера/сети. Сбой не должен обнулять дорогой переведённый текст:
переведённый EPUB на `out_path` уже записан и остаётся.

### 5.5 `src/booktranslator/cli.py` — `assemble`

Заменить inline-блок сносок на `inject_notes_for_cli(..., preloaded_glossary=None)`.
Поведение идентично текущему (с поправкой на новый дефолт `enabled: true`).

### 5.6 Документация

- **README.md** (русский, см. [ARCHITECTURE.md §13](../../ARCHITECTURE.md)):
  - В «Что делает translate под капотом» — развернуть прежнее уточнение «translate не
    добавляет сноски»: теперь сноски **включаются по умолчанию** (отключить `--no-notes`),
    появился `--cover`.
  - В таблицу «Какой провайдер для чего» — `translate --cover` использует **image**.
  - Раздел про cover/notes: упомянуть, что обе операции доступны прямо из `translate`,
    а standalone-команды остаются для точечной работы.
- **ARCHITECTURE.md**: если есть явная фиксация «translate не делает notes/cover» —
  обновить (notes теперь часть финиша translate; cover — opt-in финиш).

---

## 6. Тесты

Текущие CLI-тесты ([test_cli_integration.py](../../tests/test_cli_integration.py)) —
control-flow на хелперах, **полный `translate` end-to-end они не гоняют**, поэтому
ни один тест не ломается от нового дефолта. План:

1. **`inject_notes_for_cli`** — юнит-тесты:
   - `preloaded_glossary` используется (не лезет в `load_notes_glossary`);
   - `notes_flag=False` → `None`, инъекции нет;
   - `notes_flag=None` + `cfg.reader_notes.enabled=true` → инъекция;
   - нет глоссария → warning + skip, EPUB-деревья не тронуты.
2. **assemble** — существующие [test_assemble_integration.py](../../tests/test_assemble_integration.py)
   прогнать после рефакторинга на хелпер (поведение не должно измениться).
3. **cover в translate** — тест с **замоканным** `translate_cover` (без реального
   image-вызова): проверить, что `--cover` зовёт его с `target_lang=cfg.target_lang_name`,
   что результат пишется во временный файл и `os.replace` заменяет `out_path`, и что
   исключение из `translate_cover` не роняет команду (EPUB остаётся).
4. **reader_notes** — паритет: убедиться, что paragraph-count не меняется
   (контракт из [AI_PIPELINE.md](../../AI_PIPELINE.md)) — уже покрыто
   [test_reader_notes.py](../../tests/test_reader_notes.py), добавить кейс «через translate-хелпер».

Прогон: `pytest`, `ruff check .`, `ruff format --check .`.

---

## 7. Краевые случаи и риски

| Случай | Поведение |
|--------|-----------|
| `translate` без `--series`/`--glossary` | сноски: warning + skip (нет глоссария); cover не зависит от глоссария |
| `--cover` без image-ключа / сети | best-effort: warning, переведённый EPUB остаётся |
| `out_path` на другой ФС, чем temp | temp создаётся в `out_path.parent` → та же ФС, `os.replace` атомарен |
| Книга без обложки | `translate_cover` бросает `ValueError` → ловится, warning, EPUB остаётся |
| `--no-notes` при `enabled: true` | сноски отключены явным флагом |
| standalone `assemble` после смены дефолта | теперь вставляет сноски без флага (задокументировать) |
| paragraph-count contract | не затрагивается: сноски не меняют число абзацев |

---

## 8. Объём и порядок реализации

1. `models.py` + `configs/default.yaml` (дефолты, `target_lang_name`).
2. `reader_notes.py`: хелпер `inject_notes_for_cli`.
3. `cli.py`: рефактор `assemble` на хелпер (проверить зелёные тесты).
4. `cli.py`: опции и логика сносок+обложки в `translate`.
5. Тесты (новые + прогон существующих).
6. README / ARCHITECTURE.

Изменения API `cover.py`/`reader_notes.py`-движка не требуются — только новая
CLI-оркестрация и общий хелпер.

---

## 9. Явно вне объёма

- Перевод текста самих сносок на целевой язык (отдельный пункт роадмапа, README
  «Что пока не реализовано», версия 1.4).
- Кэширование/версионирование результата обложки (один вызов на книгу; повторный
  `--cover` = повторный вызов).
- Изменение paragraph-count контракта, cache-ключей, `Stage` enum, waterfall.
- Telegram-уведомления и точки паузы (отдельный отложенный пункт).
</content>
</invoke>
