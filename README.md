# book-translator

> CLI-инструмент для перевода EPUB-книг с английского на русский через
> OpenRouter. Сохраняет вёрстку, ведёт глоссарий уровня книги и серии
> для консистентности имён, кэширует переводы на уровне chunk-а.

Статус: v0.1. Работает end-to-end по упрощённой схеме
(`glossary → promote → translate`). Без автоматической вычитки, без
pause-точек — оператор ведёт процесс руками. См. [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md).

---

## Содержание

1. [Требования](#требования)
2. [Установка](#установка)
3. [Перевод книги: пошагово](#перевод-книги-пошагово)
4. [Полезные проверки по ходу](#полезные-проверки-по-ходу)
5. [Troubleshooting](#troubleshooting)
6. [Что сейчас не умеем](#что-сейчас-не-умеем)

---

## Требования

- Python 3.12+
- `uv` или `pip`
- Ключ OpenRouter — получить на <https://openrouter.ai/>.
- EPUB-файл (расшифрованный, без DRM).

Опционально: Calibre/Apple Books для просмотра готового EPUB.

---

## Установка

```bash
git clone <repo> ebook-translator
cd ebook-translator

# Python venv (один раз)
python3.12 -m venv .venv
source .venv/bin/activate

# Пакет + зависимости
pip install -e .

# API-ключ
cp .env.example .env
# Положить в .env: OPENROUTER_API_KEY=sk-or-...
```

После установки в venv появится команда `btrans`. Дальше во всех
командах предполагаем, что venv активирован
(или вызов через полный путь `.venv/bin/btrans`).

---

## Перевод книги: пошагово

Рабочий пример — книга 4 из "Rivers of London" Бена Ааронович.
Последовательность одинаковая для любой книги; отличаются только
пути и имена серии.

### Шаг 0 (опционально): извлечь книгу из антологии

Если исходный EPUB — сборник нескольких книг в одном файле
(`omnibus`, `collection`, «1-9 Collection»), достаём нужную книгу
отдельным файлом:

```bash
# Посмотреть, что внутри
python tools/split_epub.py path/to/omnibus.epub --list

# Извлечь книгу N (например, 4)
python tools/split_epub.py path/to/omnibus.epub --book 4
# → books/extracted/<book-slug>-<author>.epub
```

Если книга уже в отдельном EPUB — пропустить.

### Шаг 1: извлечь глоссарий книги

Глоссарий — это список имён, мест, заклинаний и терминов с
предложенными переводами. Нужен для консистентности на протяжении
книги и серии.

Для первой книги серии — без `--series`:

```bash
btrans glossary extract books/extracted/rivers-of-london-ben-aaronovitch.epub
```

Результат: `work/<book-slug>/glossary.json`.
Время: 1–3 минуты. Стоимость: ~$0.5–0.8 на книгу ~100K слов
(Claude Sonnet 4.6).

**Для второй и последующих книг серии** — с `--series`, чтобы модель
не придумывала новые переводы для уже известных терминов:

```bash
btrans glossary extract books/extracted/broken-homes-ben-aaronovitch.epub \
  --series rivers-of-london
```

### Шаг 2: вручную проверить глоссарий

Открыть `work/<book-slug>/glossary.json` в редакторе. Смотреть и
править в первую очередь записи с `"type": "person"` — ошибки в
именах персонажей самые заметные в тексте. Места и заклинания
обычно модель переводит разумно и их можно просто пробежать глазами.

Для записей, которые ты подтвердил (или поправил и согласен) —
выставить `"approved_by_human": true`. Только такие записи
пойдут в серию на следующем шаге (`promote` по умолчанию берёт
только approved).

Время: 10–15 минут для ~100–150 записей.

Если глоссарий тотально не устраивает — можно перегнать с более
строгим промптом: поправить `prompts/glossary_extract.md`, поднять
`version:` в frontmatter, запустить `btrans glossary extract …`
ещё раз (кэш инвалидируется по версии промпта, будет реальный
вызов LLM ~$0.60).

### Шаг 3 (один раз на серию): инициализировать серию

Это контейнер для канонических терминов, общих на все книги серии.
Делается один раз, потом переиспользуется.

```bash
btrans series init rivers-of-london \
  --title "Rivers of London" \
  --author "Ben Aaronovitch"
```

Создаёт `work/<slug>-series/series.glossary.json` (пустой).

### Шаг 4: перенести approved термины в серию (`promote`)

```bash
btrans glossary promote work/rivers-of-london \
  --series rivers-of-london
```

Что происходит:
- Берутся все записи с `approved_by_human: true` из
  `work/rivers-of-london/glossary.json`.
- Добавляются в series glossary, дедуплицируются по `original`.
- Если запись уже в серии с **другим** переводом — попадает в
  `conflicts`, series-вариант сохраняется, book-вариант показан
  для ревью.
- `--all` обходит фильтр по approved (осторожно).

После этого все последующие книги серии будут переводиться с
учётом этих канонических терминов.

### Шаг 5: перевод EPUB

```bash
btrans translate books/extracted/broken-homes-ben-aaronovitch.epub \
  --series rivers-of-london \
  -j 8
```

Что делает:
- Читает EPUB с сохранением lxml-деревьев всех XHTML-документов
  (не плоский текст — чтобы вернуть теги обратно).
- Разбивает главы на chunks по ~2000 слов с overlap в 1 параграф.
- Для каждого chunk — обращается к Sonnet 4.6 через OpenRouter
  с промптом, glossary серии, overlap-контекстом.
- Параллелит по `-j` потоков.
- Переведённые XHTML-фрагменты вставляются обратно в дерево главы,
  namespace сохраняется.
- Собирается новый EPUB: `books/extracted/<slug>-ru.epub` рядом с
  оригиналом (путь можно переопределить через `--out`).

Время на ~100K слов / 60 chunks: 10–20 минут при `-j 8`.
Стоимость: ~$4–6.

Повторный запуск той же команды — мгновенно из SQLite-кэша
(`work/<book-slug>/cache.sqlite`), LLM не зовётся.

Частые флаги:

```bash
# Превью первых N chunks (для быстрой проверки)
btrans translate ... --limit-chunks 7 --out preview.epub

# Свой путь для финального EPUB
btrans translate ... --out books/extracted/custom-ru.epub

# Переопределить модель (например, на Haiku для экономии)
btrans translate ... --model anthropic/claude-haiku-4.5
```

### Шаг 6: открыть готовый EPUB

Итог: `books/extracted/<slug>-ru.epub`. Открываем в Calibre,
Apple Books, Kindle (конвертация через Calibre) — любой ридер,
понимающий EPUB.

---

## Полезные проверки по ходу

### Посмотреть series glossary

```bash
btrans series show rivers-of-london
btrans series show rivers-of-london --limit 40
```

Печатает таблицу по типам (person / place / concept / term / other).

### Сколько переведено в кэше

```bash
sqlite3 work/broken-homes/cache.sqlite \
  "SELECT stage, COUNT(*) FROM cache GROUP BY stage"
```

### Сколько реально кириллицы в собранном EPUB

Грубая проверка, что все главы ушли на русский:

```bash
.venv/bin/python - << 'EOF'
import zipfile, re
from pathlib import Path
p = Path('books/extracted/broken-homes-ben-aaronovitch-ru.epub')
with zipfile.ZipFile(p) as z:
    for name in sorted(z.namelist()):
        if not (name.endswith('.xhtml') or name.endswith('.html')):
            continue
        data = z.read(name).decode('utf-8', errors='replace')
        cyr = len(re.findall(r'[а-яА-ЯёЁ]', data))
        lat = len(re.findall(r'[a-zA-Z]', data))
        if cyr + lat > 100:
            pct = 100 * cyr / (cyr + lat)
            print(f'{name:30s} cyr={cyr:>6} lat={lat:>6} ru={pct:5.1f}%')
EOF
```

Ожидаем 80–90% кириллицы в главах, 0% в `titlepage.xhtml`
(имя автора латиницей).

---

## Troubleshooting

### Chunk failed: "Translated fragment is not well-formed XML"

Чаще всего — голый `&` в выводе модели (бренды типа "M&S", "AT&T").
Фикс уже в `translator.py`: экранируем автоматически. Если снова
увидишь — проверь, что работаешь на последней версии (`git log`
должен показывать коммит `fix: escape bare ampersands …`).

### Chunk failed: "Expected N paragraphs, got M"

Модель склеила или разбила параграфы. Retry не встроен (план
итерации 4). Сейчас — пересобрать EPUB после другого прогона
(`--force` на глоссарий или ручная правка `prompts/translate.md` +
bump `version:`, чтобы пробить кэш только по translate-stage).

### Повторный запуск долгий, а должен быть из кэша

- Проверить `work/<book-slug>/cache.sqlite` на месте.
- Проверить, не менялись ли промпты (версия в frontmatter → ключ
  кэша).
- Проверить, не менялся ли series glossary (его хэш идёт в ключ).

### OpenRouter rate limit

`-j 16` и выше — риск 429. Снизить до `-j 8` или `-j 4`. В
провайдере (`provider.py`) стоит tenacity retry с exp backoff,
но он не спасёт от жёсткого лимита.

### Хочу посмотреть, что в кэше, без пересборки EPUB

```bash
sqlite3 work/broken-homes/cache.sqlite \
  "SELECT LENGTH(content), created_at FROM cache \
   WHERE stage='translate' ORDER BY created_at DESC LIMIT 10"
```

---

## Что сейчас не умеем

- **Автоматическая вычитка** (proofread / style / verify) — итерация
  5 по плану.
- **Judge + Reflect** — автоматический поиск слабых chunks и
  переделка — итерация 4.
- **Pause-точки + Telegram** — отложено
  ([см. `IMPLEMENTATION_PLAN.md`, §4bis](IMPLEMENTATION_PLAN.md)).
  Pipeline из 3 команд оператор ведёт руками, ~1 час wall-clock.
- **Оценка стоимости до запуска** (`btrans estimate`) — v1.1.
- **HU → RU** — v1.2.
- **Сноски для читателя из глоссария** (`reader_note`) — v1.3.
- **Венгерский, любые другие языковые пары** — v2.0.
- **Снятие DRM** — и не будем. Используй Calibre + DeDRM отдельно
  на легально приобретённых книгах.
- **Вёрстка PDF для типографии** — вне scope. На выходе только EPUB.

---

## Структура репозитория

```
ebook-translator/
├── README.md                # этот файл
├── ARCHITECTURE.md          # подробный дизайн pipeline
├── IMPLEMENTATION_PLAN.md   # статус, итерации, roadmap
├── AI-книги-от-А-до-Я.md    # первичный discovery-документ
├── pyproject.toml
├── .env.example
├── configs/default.yaml
├── prompts/
│   ├── glossary_extract.md
│   └── translate.md
├── src/booktranslator/      # код пакета
├── tools/split_epub.py      # standalone-скрипт для антологий
├── books/                   # gitignored: EPUB-файлы
└── work/                    # gitignored: glossary.json, cache.sqlite, state.json
```

---

## Лицензия и правовое

Инструмент предназначен для личного перевода легально приобретённых
книг. Снятие DRM — на твой страх и риск, делать отдельно через
Calibre + DeDRM.
