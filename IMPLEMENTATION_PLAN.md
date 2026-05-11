# 📋 Implementation Plan: book-translator

> Живой документ: что сделано, что в процессе, что на очереди.
> Последнее обновление: 4 мая 2026.

---

## Оглавление

1. [Контекст и цель](#1-контекст-и-цель)
2. [Что уже работает (v0.1, майский чекпоинт)](#2-что-уже-работает-v01-майский-чекпоинт)
3. [Текущее ограничение и немедленная задача](#3-текущее-ограничение-и-немедленная-задача)
4. [Ближайшие итерации](#4-ближайшие-итерации)
5. [Roadmap v1.x и дальше](#5-roadmap-v1x-и-дальше)
6. [Открытые вопросы и риски](#6-открытые-вопросы-и-риски)
7. [Полезные команды](#7-полезные-команды)

---

## 1. Контекст и цель

**Задача.** Перевести серию из 9 романов Ben Aaronovitch "Rivers of London"
(EN → RU) с сохранением вёрстки EPUB, консистентностью имён и мест на
протяжении всей серии, "хорошим фанатским" качеством перевода.

**Stack.**
- Python 3.12+, `uv` / `pip install -e .`
- EPUB I/O: `ebooklib` + `lxml`.
- LLM: OpenAI Python SDK с `base_url=https://openrouter.ai/api/v1`.
- Дефолтные модели: `anthropic/claude-sonnet-4.6` (перевод/глоссарий) и
  `anthropic/claude-haiku-4.5` (proofread/judge).
- Кэш: SQLite на книгу.
- CLI: Typer + Rich.
- Промпты: markdown с YAML frontmatter + Jinja2.

**Важные принципы.**
- OpenRouter — единственный провайдер LLM.
- Глоссарий серии — canonical source для имён/мест/терминов.
- Pause-точки: ручной review глоссария до перевода, preview после.
- Каждое изменение промпта бампает `version:` → ключ кэша.
- `work/` и `books/` в `.gitignore`.

---

## 2. Что уже работает (v0.1, майский чекпоинт)

### 2.1 Архитектура

- `ARCHITECTURE.md` — подробный дизайн pipeline, конфигурации, форматы
  данных, модели по умолчанию, roadmap.
- `pyproject.toml` с editable install, entry point `btrans`.
- `.env.example`, `.gitignore`, `configs/default.yaml`.

### 2.2 Разделение антологий (`tools/split_epub.py`)

Извлечение одной книги из epubmerge-подобного EPUB-сборника.

- Парсит `META-INF/container.xml` → корневой OPF.
- Определяет книги по верхним `navPoint` в `toc.ncx`.
- Высчитывает directory prefix (`1/1/`, `2/3/`, `3/`) для каждой книги.
- Копирует файлы книги, переписывает hrefs, генерирует новый UUID.
- Сохраняет EPUB-совместимый пакет (mimetype первым, uncompressed).

Использование:

```bash
python tools/split_epub.py INPUT.epub --list
python tools/split_epub.py INPUT.epub --book N [--out DIR]
```

**Проверено** на "Rivers of London 1-9 Collection": корректно извлекает
все 9 книг (Rivers of London, Moon Over Soho, Whispers Under Ground,
Broken Homes, Foxglove Summer, The Hanging Tree, Lies Sleeping,
False Value, Amongst Our Weapons).

### 2.3 Извлечение глоссария книги

```
btrans glossary extract EPUB [--series SLUG] [--model M] [--force]
```

- Читает EPUB плоским текстом, передаёт целиком Sonnet 4.6 одним запросом.
- Промпт `prompts/glossary_extract.md` v3: инструкции по типам (person /
  place / concept / term / other), gender для персонажей, правила
  именования агентств/HQ, обработка персонифицированных духов.
- При `--series SLUG` в промпт подаётся compact-рендер known terms
  серии; модель возвращает только новое + optional overrides.
- Ответ парсится в `Glossary` (pydantic), `override: true` становится
  префиксом `[override]` в `notes`.
- Результат: `work/<book-slug>/glossary.json`.
- Кэш в `work/<book-slug>/cache.sqlite` — повторный запуск бесплатен.

**Результаты на тестах:**

| Книга | Слов | Записей в глоссарии | Стоимость |
|---|---|---|---|
| Rivers of London | 102K | 107 | ~$0.69 |
| Broken Homes (с series=106 known) | 92K | 145 (144 new + 1 overlap) | ~$0.62 |

### 2.4 Series glossary (canonical cross-book)

```
btrans series init SLUG --title T --author A
btrans series show SLUG [--limit N]
btrans glossary promote WORKDIR --series SLUG [--all]
```

- `work/<slug>-series/series.glossary.json` — единый файл серии.
- Упрощённая схема (`SeriesGlossaryEntry` без `approved_by_human`, с
  `origin_book` для аудита).
- `promote` берёт только `approved_by_human=true` по умолчанию (флаг
  `--all` для массового импорта).
- Дедупликация по `original`; конфликты (разные `translation`)
  репортятся, series-версия сохраняется.
- Compact-рендер для промпта: `original | translation | type[, gender] | notes`.

**Текущее состояние серии `rivers-of-london`:** 250 терминов
(106 из Rivers of London + 144 из Broken Homes). 89 person, 70 place,
32 concept, 57 term, 2 other.

### 2.5 Перевод EPUB (chapter-level, XHTML-preserving)

```
btrans translate EPUB [--series SLUG] [--limit-chunks N] [-j K] [--out P]
```

- `epub_io.read_book_structured` — читает EPUB, сохраняет lxml-деревья
  всех XHTML-документов в spine + raw bytes всего архива.
- `chunker.chunk_book` — жадно упаковывает параграфы (`<p>`, `<h1..6>`,
  `<blockquote>`) в chunks по 2000 слов с overlap в 1 параграф.
- `prompts/translate.md` — XHTML-фрагменты per paragraph, явные маркеры
  `===PARAGRAPH N===`, блок known-terms, overlap before/after как
  read-only контекст. Правила сохранения inline-тегов.
- `translator.Translator` — оркестрация одного chunk: rendering,
  кэш lookup, LLM call, parsing по маркерам, validation count,
  namespace-aware splicing обратно в lxml-дерево.
- `epub_io.write_translated_epub` — собирает новый EPUB, сохраняя
  byte-for-byte всё кроме изменённых XHTML и опционально `dc:language`
  в OPF.

**Проверено end-to-end на Broken Homes:**
- Smoke-test 2 chunks: 0 failed, 146/146 параграфов переведены,
  inline `<i class="calibre1">` сохранён, `<p class="epi">` тоже.
- 7 chunks в кэше, собран `books/extracted/broken-homes-preview-7chunks-ru.epub`
  (превью: 3 первые главы на русском, остальные ждут).
- Перевод качественный: "Аллен Фраст", "Пис-Поттаж", «Вольво V70»,
  правильные тире, кавычки-ёлочки, литературные обороты.

### 2.6 Конфигурируемый параллелизм (свежий коммит)

```
btrans translate ... -j 4      # 4 параллельных chunk
```

- `config.translate.parallelism` (default 1).
- `translator` переделан на `ThreadPoolExecutor` + `threading.Lock` для
  кэша, stats и splicing.
- SQLite connection открывается с `check_same_thread=False`.
- Ожидание: при 4–8 параллельных chunks full book с 60 chunks ≈
  10–20 минут вместо прошлых ~90 минут.

### 2.7 Стендэлон-сценарий без серии

```
btrans translate EPUB --glossary work/<book-slug>/glossary.json
```

- `--series` и `--glossary` взаимоисключающие.
- При `--glossary` book-level `Glossary` приводится in-memory к
  `SeriesGlossary` (берутся все entries; approved_by_human здесь не
  фильтрует — выбор файла оператором и есть согласие).
- Промпт и кэш одинаковые для обоих путей.
- Без `--series`/`--glossary` — жёлтое предупреждение и перевод без
  глоссария (только для быстрой прикидки).

---

## 3. Текущее ограничение и немедленная задача

### Наблюдаемая проблема

При последовательном прогоне полной книги Broken Homes **каждый chunk
обрабатывался ~90 секунд** вместо ожидаемых ~10 с. Причина — в каждый
запрос уходит весь series glossary (~15–20K токенов input) плюс сам
chunk и overlap, что увеличивает latency Sonnet 4.6.

### Фикс, который уже в коде

Параллелизм `-j K`. Закрывает проблему практически: при `-j 8` общее
время падает в ~8× при тех же input-токенах. Rate limits OpenRouter не
должны быть узким местом на 8 параллельных потоках.

### Следующий шаг (отдельная сессия)

1. Запустить полный перевод Broken Homes с `-j 4` или `-j 8`:
   ```bash
   btrans translate books/extracted/broken-homes-ben-aaronovitch.epub \
     --series rivers-of-london -j 8
   ```
2. Измерить реальное время, зафиксировать стоимость.
3. Проверить собранный EPUB: читается ли, все ли главы на русском,
   inline-форматирование на месте.
4. Если всё ок → запустить для книги 1 "Rivers of London".

**Статус (4 мая 2026).** Полный перевод Broken Homes выполнен
(60/60 chunks, 0 failed после фикса экранирования амперсандов в
LLM-выводе), собран `books/extracted/broken-homes-ben-aaronovitch-ru.epub`.
Оператор вычитывает результат на электронной книге, собирает
замечания для следующих итераций. Следующий плановый переводимый
объект — книга 1 "Rivers of London", на готовой серии из 250 терминов.

---

## 4. Ближайшие итерации

### Итерация 4 — Judge + Reflect + Multi-pass storage

**Цель.** Автоматически находить слабые места перевода и переделывать
их, сохраняя все варианты для сравнения и отката.

#### 4.1 Multi-pass storage (фундамент для итераций 4 и 5)

**Принцип.** Каждый проход (translate, reflect, proofread, style, verify)
хранится в кэше как отдельная запись с `stage`. Ничего не перезаписывается.
При сборке EPUB выбирается нужный вариант по waterfall или явному указанию.

**Новая таблица в SQLite:**

```sql
CREATE TABLE IF NOT EXISTS chunk_preferences (
    chunk_id TEXT PRIMARY KEY,
    preferred_stage TEXT NOT NULL,
    reason TEXT,
    updated_at TEXT DEFAULT (datetime('now'))
);
```

**Логика выбора при сборке EPUB (waterfall с override):**

1. Если есть запись в `chunk_preferences` → берём указанный stage.
2. Иначе — waterfall по умолчанию:
   `verify > style > proofread > reflect > translate`
   (берём самый "поздний" доступный stage).
3. CLI-флаг `--assemble-from STAGE` перекрывает всё:
   собирает EPUB только из указанного stage для всех chunks.

**CLI-команды:**

```bash
# Собрать EPUB из конкретного прохода (игнорируя все последующие)
btrans assemble book.epub --from translate      # "чистый" первый перевод
btrans assemble book.epub --from reflect        # после рефлексии
btrans assemble book.epub --from style          # после стилистики

# Для одного chunk выбрать предпочтительный вариант
btrans prefer ch03_c02 translate --reason "reflect ухудшил диалог"
btrans prefer ch03_c02 style     --reason "style хорошо убрал кальку"

# Посмотреть все проходы для chunk-а
btrans diff ch03_c02
btrans diff ch03_c02 --stages translate,reflect  # сравнить два

# Сбросить предпочтение (вернуться к waterfall)
btrans prefer ch03_c02 --reset
```

**Стоимость хранения:** 60 chunks × 5 stages × ~3KB = ~900KB на книгу.
На 9 книгах серии <10MB. SQLite справляется без проблем.

#### 4.2 Judge

- `prompts/judge.md`: модель (Haiku 4.5) получает original + translation,
  возвращает `{score: 1-5, issues: [...]}`.
- Критерии scoring: точность, естественность, консистентность с
  глоссарием, стиль.
- Результаты сохраняются в кэше (`stage=judge`), не участвуют в
  waterfall сборки (это метаданные, не текст).
- CLI: `btrans judge book.epub [--model M]`.
- Отчёт: `btrans scores show work/<book>/` — таблица
  chunk_id | score | issues.

#### 4.3 Reflect

- Триггер: score ≤ `reflection.trigger_score` (default 3) ИЛИ
  флаг `--reflect-all`.
- Метод Andrew Ng: рефлексия по переводу → список замечаний →
  повторный перевод с учётом замечаний.
- Два промпта: `prompts/reflect.md` (критика) + reuse `translate.md`
  с добавленным полем `reflection_notes`.
- Модель: `anthropic/claude-sonnet-4.6` с extended thinking.
- Результат сохраняется как `stage=reflect` — первый перевод
  (`stage=translate`) остаётся нетронутым.
- CLI: `btrans reflect book.epub [--threshold N] [--all]`.
- Config: `reflection.trigger_score`, `reflection.extended_thinking`.

#### 4.4 UX при запуске

```
$ btrans translate book.epub --series rivers-of-london -j 8

[translate] ████████████████████ 60/60 done (12m 34s, $4.31)

[judge] ████████████████████ 60/60 done (1m 02s, $0.11)
  Score distribution: ★5: 34  ★4: 18  ★3: 6  ★2: 2  ★1: 0
  Chunks needing reflection: 8 (score ≤ 3)

[reflect] ████████████████████ 8/8 done (3m 15s, $0.89)
  Improved: 7/8  |  No change: 1/8

Summary:
  Total cost: $5.31  |  Time: 16m 51s
  Output: books/extracted/broken-homes-ru.epub

  💡 Skip judge+reflect: --no-judge
  💡 Review scores: btrans scores show work/broken-homes/
  💡 Revert a reflection: btrans prefer ch03_c02 translate
  💡 Assemble from first pass only: btrans assemble ... --from translate
```

#### 4.5 Флаги CLI

- `--no-judge` — пропустить judge + reflect целиком.
- `--no-reflect` — judge запускается (для статистики), но reflect нет.
- `--reflect-threshold N` — перекрыть `reflection.trigger_score`.
- `--reflect-all` — рефлексия для всех chunks независимо от score.

**Эстимейт.** +~$0.30–$1.00 на книгу (рефлексия 20–40% chunks).
Judge на Haiku 4.5 — ~$0.11 на книгу.

---

### Итерация 5 — Proofread / Style / Verify

**Статус: ✅ ВЫПОЛНЕНО (11 мая 2026)**

**Цель.** Три отдельных прохода post-translate для финального качества.
Все проходы хранятся в кэше как отдельные stage, участвуют в waterfall
сборки, могут быть откачены через `btrans prefer`.

- **Proofread** (`prompts/proofread.md`, Haiku 4.5): грамматика,
  пунктуация, опечатки. Stage: `proofread`.
- **Style** (`prompts/style.md`, Sonnet 4.6): кальки, канцелярит,
  авторский голос. Stage: `style`.
- **Verify** (`prompts/verify.md`, Sonnet 4.6): сверка с оригиналом,
  пропуски, искажения. Stage: `verify`.

**Каждый проход:**
- Берёт на вход текст из предыдущего stage по waterfall (verify читает
  результат style, style читает результат proofread, и т.д.).
- Сохраняет результат в кэше как свой stage.
- Не трогает предыдущие записи.

**Флаги CLI:** `--proofread/--no-proofread`, `--style/--no-style`,
`--verify/--no-verify`. В конфиге `stages.{proofread,style,verify}`.

**Сборка:** `btrans assemble book.epub --from proofread` даёт EPUB
после корректуры но до стилистики. Полезно для A/B сравнения этапов.

**Реализация (11 мая 2026):**

- Новый модуль: `src/booktranslator/postprocess.py` — класс
  `PostProcessor` с поддержкой кэширования, параллелизма, и
  хранения результатов как отдельных stage в waterfall.
- Три промпта: `prompts/proofread.md`, `prompts/style.md`,
  `prompts/verify.md`.
- Три standalone CLI-команды: `btrans proofread`, `btrans style`,
  `btrans verify`.
- Интеграция в `btrans translate`: автоматически запускаются после
  reflect (если не отключены через `--no-proofread` и т.д.).
- Хелперы в `pipeline_helpers.py`: `collect_waterfall_translations()`
  и `collect_waterfall_paragraphs()` для получения текста из
  предыдущего stage по waterfall.
- Тесты: `tests/test_postprocess.py` (30 тестов).

### Итерация 4.5 — Maintenance CLI (`btrans status` / `prefer` / `assemble`)

**Статус: ✅ ВЫПОЛНЕНО (11 мая 2026)**

**Цель.** Дать оператору полный контроль и обзор состояния перевода
пока процесс не устаканится. Read-only инспекция + точечное управление
сборкой.

**Реализуется ДО judge/reflect** — нужна рабочая инфраструктура для
инспекции кэша, которую judge/reflect будут использовать.

#### Команды

**`btrans status WORKDIR [--scores] [--diff CHUNK] [--assembly-map] [--below N]`**

Read-only инспекция:
- Без флагов: обзор (chunks, какие stages пройдены, стоимость, scores
  distribution если есть judge).
- `--scores`: таблица chunk_id | score | issues.
- `--scores --below N`: только chunks с оценкой < N.
- `--diff CHUNK [--stages S1,S2]`: показать текст chunk-а из разных
  stages для сравнения.
- `--assembly-map`: для каждого chunk показать, какой stage будет
  использован при сборке (waterfall + preferences).

**`btrans prefer CHUNK STAGE [--reason TEXT] [--reset] [--work WORKDIR]`**

Пометить предпочтительный stage для chunk-а. Не удаляет данные из кэша,
только пишет в `chunk_preferences`. `--reset` убирает override.

**`btrans assemble WORKDIR [--from STAGE] [--out PATH]`**

Собрать EPUB из кэша. По умолчанию waterfall, `--from STAGE` берёт
конкретный stage для всех chunks.

#### Пример вывода `btrans status`

```
Broken Homes — Ben Aaronovitch
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Chunks: 60 total, 2187 words avg

Passes in cache:
  translate   60/60  ✓  ($4.31, 12m 34s)
  judge       60/60  ✓  ($0.11, 1m 02s)
  reflect      8/60     ($0.89, 3m 15s)
  proofread    0/60     —
  style        0/60     —
  verify       0/60     —

Judge scores:
  ★5: 34  ★4: 18  ★3: 6  ★2: 2  ★1: 0
  Reflected: 8 chunks (score ≤ 3)

Preferences: 1 override (ch12_c01 → translate)

Assembly source (current waterfall):
  reflect: 7 chunks  |  translate: 53 chunks

Total cost: $5.31
```

#### Реализация

- Новый модуль: `src/booktranslator/status.py` — логика запросов к кэшу.
- Расширение `cache.py`: добавить таблицу `chunk_preferences`, методы
  `list_stages()`, `get_all_for_chunk()`, `get_preference()`,
  `set_preference()`.
- CLI: три новые команды в `cli.py` (или отдельный `status_app` typer group).

---

### Итерация 6 — Оптимизация: relevancy-фильтр глоссария

**Цель.** Уменьшить input tokens в 5-10× за счёт подачи в каждый chunk
только релевантных терминов.

- Для каждого chunk token-level match imён из series glossary в тексте.
- В промпт подаём только найденные + небольшой общий контекст.
- Ожидается: латентность ~15 сек на chunk даже при 300+ записях в серии.
- Риск: пропустить термин в тексте из-за морфологии (English этим
  страдает меньше). Мягкий fallback: при любом сомнении подаём полный
  глоссарий.

---

## 4bis. Отложено до появления внешнего триггера

### Pause-points + Telegram (ранее "Итерация 6")

**Идея.** Длинный прогон не должен требовать сидеть у экрана.
Pipeline делает паузу после glossary и после translate, шлёт
Telegram-уведомление, ждёт `--resume`.

**Почему отложено (4 мая 2026).**

На текущем profiling одного прогона:

- `btrans glossary extract …` — одна команда, ~2 минуты ожидания.
- Ручная вычитка `glossary.json` — 10–15 минут глазами
  (это не автоматизируется, pause-точка её не сокращает).
- `btrans glossary promote …` — одна команда.
- `btrans translate … -j 8` — одна команда, ~10–20 минут.

Итого 3 команды с коротким wall-clock между ними. Оператор и так
рядом с машиной, ручная вычитка глоссария никуда не делась.
Инженерная стоимость фичи — ~8–15 часов (state-machine в
`pipeline.py`, `state.json`, логика `--resume`, edge-кейсы,
Telegram-клиент, документация, тесты). ROI на текущем объёме
(5–6 книг в месяц) отрицательный.

**Когда разморозить.**

- Масштаб вырастет до 20+ книг в месяц.
- Появится удалённый/ночной запуск (CI, cron).
- Pipeline удлинится до нескольких часов wall-clock (например,
  если добавятся несколько проходов вычитки с Opus 4.7).
- Захочется передать инструмент другому человеку, которому не
  хочется учиться линейке команд.

**Лёгкая альтернатива на macOS** (бесплатно, не требует state-machine):
в zsh обернуть команду в `; osascript -e 'display notification …'`
или использовать `say "done"` — system-notification на завершение
без кода в проекте.

---

## 5. Roadmap v1.x и дальше

Полный список в `ARCHITECTURE.md` раздел 12. Выжимка:

### v1.1 — QoL

- `btrans estimate EPUB` — оценка стоимости до запуска.
- `btrans cache invalidate --chunk CH --stage STAGE`.
- Улучшенные отчёты: `cost-report.json` после DONE.

### v1.2 — HU → RU

- Language-aware промпты (в венгерском нет грамматического рода, его
  восстанавливает глоссарий).
- Тесты на HU-образце.
- Новый пресет `budget-hu.yaml` / `premium-hu.yaml`.

### v1.3 — Reader notes from glossary

- Поле `reader_note` в `SeriesGlossaryEntry`.
- По умолчанию комментируем только `type=concept` (заклинания Ааронович:
  vestigium, forma, sequestration...), scope `first-in-chapter`.
- Конфиг `reader_notes.types: [concept]` расширяется до `term` (для DCI,
  TSG) или `place` (для the Folly).
- Per-entry `comment_override: "always" | "never" | null`.
- Генерация: либо вручную, либо `prompts/notes_generate.md`.
- Реализация — вариант A: инжектим в промпт перевода и просим модель
  обернуть первое упоминание в EPUB footnote-ref.

### v2.0 — Any → Any

- Обобщение `source_lang`/`target_lang`.
- Набор language-specific промпт-модификаторов.
- Возможно, веб-интерфейс для правки глоссария.

### v2.x (идеи)

- Интеграция с Calibre-библиотекой для массового перевода серий.
- Авто-валидация через EPUBCheck.
- Экспорт bilingual EPUB (original + translation side-by-side).

---

## 6. Открытые вопросы и риски

### Качество

- **Параграф count mismatch.** Пока падает как failed chunk, без retry.
  План: при несовпадении делать один повтор с more explicit prompt
  (итерация 4).
- **Inline markup edge cases.** `<a href>` с нетекстовым содержимым,
  многоуровневая вложенность, `<br/>` в середине предложения. Пока
  работает, но на большом объёме могут вылезти сюрпризы.
- **Footnotes / sidenotes.** У Broken Homes их нет в spine, но у других
  книг серии (Foxglove Summer) возможны. План: в итерации 5 включить
  `<ul>`/`<ol>` как translatable, отдельно разобраться с элементами вне
  spine.

### Производительность и стоимость

- **Series glossary растёт** — на 9-й книге может быть 600–800 записей,
  ~30K токенов на каждый chunk. Решение в v1.7 (relevancy-фильтр).
- **Rate limits OpenRouter** на высоком параллелизме. Пока не наблюдал,
  но при `-j 16+` возможны 429. Есть tenacity retry, но стоит выставить
  разумный upper bound.
- **Hard limit по стоимости** (`cost.hard_limit_usd`) пока не проверяет
  накопленные расходы. Реализовать в итерации 4.

### Идемпотентность

- Повторный запуск `btrans translate` — всё из кэша, тот же EPUB.
  **Проверено.**
- Смена промпта (bump `version`) → инвалидация кэша. **Проверено.**
- Обновление series glossary → хэш меняется → кэш инвалидируется для
  всех последующих chunks. Возможно, захотим более гранулярно (только
  те chunks, где поменялись используемые термины), но пока ок.

### Git hygiene

- `work/` и `books/` в .gitignore — **работает**.
- `.env` с ключом в .gitignore — **работает**.
- Промпты и конфиги в git — **делаем**.

---

## 7. Полезные команды

### Разделить антологию

```bash
python tools/split_epub.py INPUT.epub --list
python tools/split_epub.py INPUT.epub --book 4
# → books/extracted/broken-homes-ben-aaronovitch.epub
```

### Глоссарий

```bash
# Первая книга серии
btrans glossary extract books/extracted/rivers-of-london-ben-aaronovitch.epub

# Ручной review work/rivers-of-london/glossary.json
# (выставить approved_by_human=true где всё ок)

# Инициализировать серию
btrans series init rivers-of-london \
  --title "Rivers of London" --author "Ben Aaronovitch"

# Промоутить approved записи
btrans glossary promote work/rivers-of-london --series rivers-of-london

# Посмотреть серию
btrans series show rivers-of-london

# Следующая книга с known terms
btrans glossary extract books/extracted/broken-homes-ben-aaronovitch.epub \
  --series rivers-of-london
```

### Перевод

```bash
# Полный перевод книги с параллелизмом 8
btrans translate books/extracted/broken-homes-ben-aaronovitch.epub \
  --series rivers-of-london -j 8

# Первые N chunks для превью
btrans translate ... --limit-chunks 7 --out preview.epub

# Свой out путь
btrans translate ... --out books/extracted/custom-ru.epub

# Переопределить модель
btrans translate ... --model anthropic/claude-haiku-4.5
```

### Полезное для отладки

```bash
# Принудительно перегнать глоссарий (игнор кэша)
btrans glossary extract ... --force

# Посмотреть таблицу серии
btrans series show rivers-of-london --limit 20

# Сколько в кэше
sqlite3 work/broken-homes/cache.sqlite \
  "SELECT stage, COUNT(*) FROM cache GROUP BY stage"

# Проверить собранный EPUB валидным парсером
python -c "from ebooklib import epub; import warnings; \
  warnings.filterwarnings('ignore'); \
  b = epub.read_epub('OUT.epub'); \
  print('title:', b.get_metadata('DC','title'), \
        'spine:', len(b.spine))"
```

---

*Документ обновляется вручную по итогам каждой итерации.
Следующее обновление — после запуска полного перевода Broken Homes
с параллелизмом и проверки качества.*
