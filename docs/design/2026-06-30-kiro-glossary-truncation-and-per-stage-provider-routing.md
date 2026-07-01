# Design: маршрутизация провайдеров по стадиям, отключение thinking и guard от обрезки книги

> Status: **proposed** (план, не реализовано)
> Date: 2026-06-30
> Scope: `provider.py`, `prompts.py`, `models.py`, `pipeline_helpers.py`, `glossary.py`,
> `cli.py`, `configs/default.yaml`, все `prompts/*.md` текстовых стадий, тесты.
> Related rules: [AI_PROVIDER.md](../../AI_PROVIDER.md), [AI_PIPELINE.md](../../AI_PIPELINE.md),
> [ARCHITECTURE.md](../../ARCHITECTURE.md).

---

## 1. Задача

При запуске `btrans glossary extract <book>` через **kiro-gateway** падает извлечение глоссария:

```
Extraction failed: Model response is not valid JSON: Expecting value:
line 1 column 1 (char 0). First 200 chars: ''
```

Нужно разобраться в причине и сделать так, чтобы:
1. глоссарий извлекался корректно;
2. программа **громко падала** при обрезке книги, а не отдавала молча неполный глоссарий
   (CLAUDE.md: «никогда не порти книгу ради зелёной галочки»);
3. остальные стадии (перевод/чанки) продолжали работать через Kiro.

---

## 2. Диагноз (воспроизведено)

Цепочка проверок против `http://localhost:9000/v1` (kiro-gateway):

| Проверка | Результат |
| --- | --- |
| `GET /v1/models` | OK, `claude-sonnet-4.6` присутствует |
| Маленький запрос («ответь OK») | OK, `content="OK"` |
| Реальный вызов глоссария (вся книга) | **`content` пустой**, `finish_reason: stop` |

Детали провального вызова (реальный промпт + книга Bear Head):

```
system chars: 4125   user chars: 601525   approx input tokens: ~151412
finish_reason: stop
content len: 0                 ← пустая строка '' уходит в json.loads() → краш
reasoning len: 26327           ← 8254 completion-токенов ушли в "размышления"
usage: prompt_tokens=24865, completion_tokens=8254
```

### 2.1 Корневая причина №1 — thinking съедает бюджет вывода

Kiro-gateway гоняет Claude с **включённым extended thinking** и складывает рассуждения в
отдельное поле `reasoning_content`, а стандартное OpenAI-поле `content` остаётся пустым.
Провайдер читает только `content` ([provider.py:113](../../src/booktranslator/provider.py)):

```python
text = choice.message.content or ""   # -> '' когда всё ушло в reasoning_content
```

Хвост `reasoning_content` обрывается на полуслове — модель **израсходовала бюджет вывода
на размышления и не успела начать писать сам JSON**:

> «Let me now organize and write the final JSON... 1. Albedo... 2. Aslan, Keram John →»

Пустой `''` уходит в [glossary.py:43](../../src/booktranslator/glossary.py) → `json.loads('')`
→ «Expecting value: line 1 column 1».

**Рычаг найден.** Шлюз честно слушается OpenAI-параметра `reasoning_effort`:

```
reasoning_effort: "none"   → content корректный, reasoning_len: 0, latency 6.5s (вместо 152s)
```

### 2.2 Корневая причина №2 — Kiro обрезает большой ввод (~в 5 раз)

Тест с маркерами начала/конца книги: модель вернула **выдуманный** «конец текста»
(`...by default.\n</ip_reminder>` — это служебная вставка Kiro, не книга; реальная книга
кончается на «Head of Zeus Books … OceanofPDF.com»). `usage.prompt_tokens` на ~150k-токенном
вводе показал лишь **~31k** — то есть Kiro видит примерно **1/5** книги.

**Вывод:** извлечение глоссария по всей книге через Kiro принципиально не работает — глоссарий
покрывал бы только начало книги, тихо и незаметно.

### 2.3 Мёртвая конфигурация

`reflection.extended_thinking` объявлена в
[models.py:54](../../src/booktranslator/models.py), но **нигде не читается** в коде вызова.
Текущим thinking ничто в коде не управляет — Kiro включает его по умолчанию.

---

## 3. Решения, принятые до начала (заказчик)

1. **Глоссарий → OpenRouter.** Стадия glossary роутится на OpenRouter (контекст 200k, без
   обрезки); перевод и чанк-стадии остаются на Kiro.
2. **Маршрутизация провайдеров — универсальная, но как НЕОБЯЗАТЕЛЬНЫЕ оверрайды поверх
   дефолта `text`.** Любая стадия может переопределить провайдера через `providers.<stage>`;
   не задано — берётся `providers.text`. Никаких обязательных 7 блоков.
3. **Thinking выключаем на всех текстовых стадиях** через `reasoning_effort: none`. Качество
   даёт явный waterfall, а не скрытый per-call thinking; скрытый thinking непредсказуемо
   съедает output-бюджет и рискует нарушить контракт «N абзацев → N абзацев».
4. **Версии промптов: тронул промпт → забампил.** У заказчика сделаны только 2 книги, они не
   переделываются, все будущие прогоны — на новых книгах (новый текст = новый ключ кэша),
   поэтому цена инвалидации = 0. Правило простое и без ловушек.
5. **Чанкование глоссария — НЕ делаем сейчас.** OpenRouter решает проблему: Bear Head
   (~109k слов ≈ ~151k токенов) + ответ 16k влезают в 200k одним вызовом. Чанкование
   остаётся опциональным заделом для книг-гигантов (см. §9).

---

## 4. Изменения по файлам

### 4.1 `models.py` — провайдеры по стадиям (оверрайды)

[models.py:23](../../src/booktranslator/models.py), `ProvidersConfig`. Оставляем `text`
(дефолт всех текстовых стадий) и `image`. Добавляем **опциональные** оверрайды, зеркалящие
`ModelsConfig`:

```python
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
```

Обратная совместимость: старые конфиги с одним `text` продолжают работать (все оверрайды
`None` → `text`).

### 4.2 `pipeline_helpers.py` — фабрика по стадии

[pipeline_helpers.py:43](../../src/booktranslator/pipeline_helpers.py). Добавить общий
`create_stage_provider`; `create_provider` оставить тонкой обёрткой над `text`.

```python
_TEXT_STAGES = (
    "glossary", "translate", "judge", "reflect", "proofread", "style", "verify",
)

def create_stage_provider(cfg: Config | None, stage: str) -> OpenRouterProvider:
    """Provider for a named text stage: providers.<stage> if set, else providers.text."""
    if cfg is None:
        return OpenRouterProvider()
    override = getattr(cfg.providers, stage, None)
    return _create_provider_from_config(override or cfg.providers.text)
```

`create_provider(cfg)` оставить как есть (== `text`) ради остального кода и обратной
совместимости.

### 4.3 `prompts.py` — поле `reasoning_effort`

[prompts.py:35](../../src/booktranslator/prompts.py), dataclass `Prompt`:

```python
@dataclass
class Prompt:
    name: str
    version: str
    model: str | None
    temperature: float
    max_tokens: int | None
    reasoning_effort: str | None   # NEW: none|low|medium|high; None => не слать параметр
    system_tmpl: str
    user_tmpl: str
```

[prompts.py:62](../../src/booktranslator/prompts.py) `load_prompt`:

```python
    reasoning_effort=fm.get("reasoning_effort"),
```

`None` (поле не указано) → провайдер не отправляет параметр (поведение по умолчанию).
`reasoning_effort` **не входит** в ключ кэша (ключ — `text+model+prompt_version+glossary+stage`).

### 4.4 `provider.py` — проброс `reasoning_effort` + `finish_reason`

[provider.py:21](../../src/booktranslator/provider.py) `CompletionResult` — добавить
`finish_reason` для диагностики:

```python
@dataclass
class CompletionResult:
    text: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    model: str
    finish_reason: str | None   # NEW
    raw: dict
```

[provider.py:85](../../src/booktranslator/provider.py) `complete()` — новый аргумент и проброс
через `extra_body` (нестандартное для OpenAI SDK поле, поэтому именно `extra_body`, оно
сливается в тело запроса; проверено curl-ом, что шлюз его понимает):

```python
    def complete(
        self,
        model: str,
        system: str,
        user: str,
        *,
        temperature: float = 0.3,
        max_tokens: int | None = None,
        response_format: dict | None = None,
        reasoning_effort: str | None = None,   # NEW
    ) -> CompletionResult:
        ...
        if reasoning_effort is not None:
            kwargs["extra_body"] = {"reasoning_effort": reasoning_effort}
        ...
        choice = response.choices[0]
        text = choice.message.content or ""
        ...
        return CompletionResult(
            ...,
            finish_reason=getattr(choice, "finish_reason", None),
            raw=...,
        )
```

> **Важно — не кидать исключение на пустой `content` внутри `complete()`.** Retry-декоратор
> ([provider.py:79](../../src/booktranslator/provider.py)) повторяет на `Exception` до 5 раз →
> 5× оплата бесполезных вызовов. Понятную ошибку про пустой ответ отдаём в вызывающем коде
> (см. §4.5). Корневой фикс (`reasoning_effort: none`) и так не даёт `content` опустеть.

### 4.5 `glossary.py` — guard от обрезки + понятная ошибка

[glossary.py:85](../../src/booktranslator/glossary.py) `extract_glossary`.

**(а) bookend-nonce.** Перед рендером дописываем в конец `book_text` случайный маркер и
требуем вернуть его. Значение nonce в инструкции НЕ раскрываем — прочитать его можно только
увидев хвост текста (в ходе диагностики модель «угадывала» маркер, просто повторяя его имя из
инструкции; поэтому проверяем именно неизвестное значение):

```python
import secrets

bookend = secrets.token_hex(4)               # напр. "a9f3c1d7"
context["book_text"] = book.full_text() + f"\n\n[[BOOKEND::{bookend}]]\n"
```

**(б) Парсинг + проверки.** После получения `raw_text`:

```python
if not raw_text.strip():
    raise ValueError(
        "Model returned empty content"
        + (f" (finish_reason={result.finish_reason})" if result else "")
        + ". На gateway с extended thinking ответ мог уйти в reasoning_content — "
        "убедитесь, что в промпте задан reasoning_effort: none."
    )
```

Поле `bookend` достаём из распарсенного JSON (верхнего уровня) ДО валидации записей. Если
модель — это OpenRouter, обрезки не будет; guard — дешёвая страховка и защита, если стадию
когда-нибудь зароутят иначе:

```python
returned = (data.get("bookend") if isinstance(data, dict) else None)
if returned != bookend:
    raise TruncationError(
        f"Книга, похоже, обрезана провайдером: ожидался bookend={bookend!r}, "
        f"получено {returned!r}. Глоссарий по неполному тексту не сохраняем. "
        "Проверьте окно контекста провайдера стадии glossary."
    )
```

`TruncationError` — новый класс в `glossary.py` (подкласс `ValueError`, чтобы CLI-обработчик
в [cli.py:175](../../src/booktranslator/cli.py) его поймал и записал в `raw_glossary_path`,
как сейчас).

> Реализационная заметка: `_parse_llm_response` сейчас принимает либо top-level список, либо
> `{"entries": [...]}`. Для чтения `bookend` глоссарий должен возвращать **объект**
> `{"bookend": "...", "entries": [...]}`. Это закрепляем в промпте (§4.7) и в парсере:
> `bookend` извлекаем из объекта, `entries` — как раньше. Top-level-список остаётся
> поддержанным для обратной совместимости, но тогда `bookend` отсутствует → для glossary-стадии
> это ошибка обрезки. (Для OpenRouter модель легко вернёт объект.)

**(в) Проброс `reasoning_effort`** в вызов [glossary.py:133](../../src/booktranslator/glossary.py):

```python
result = provider.complete(
    model=chosen_model,
    system=system,
    user=user,
    temperature=prompt.temperature,
    max_tokens=prompt.max_tokens,
    reasoning_effort=prompt.reasoning_effort,   # NEW
)
```

### 4.6 `cli.py` — стадии берут провайдера по имени

Заменить `create_provider(cfg)` на `create_stage_provider(cfg, "<stage>")` в каждой
стадийной команде. **Проверено по коду** (semantic-index + чтение):

| Команда | Строка `create_provider` | Стадия |
| --- | --- | --- |
| `glossary_extract` | [cli.py:159](../../src/booktranslator/cli.py) | `"glossary"` |
| `translate` | [cli.py:643](../../src/booktranslator/cli.py) | `"translate"` |
| `judge_cmd` | [cli.py:1208](../../src/booktranslator/cli.py) | `"judge"` |
| `reflect_cmd` | [cli.py:1414](../../src/booktranslator/cli.py) | `"reflect"` |
| `_run_postprocess_cmd` (proofread/style/verify) | [cli.py:1770](../../src/booktranslator/cli.py) | `stage` (уже параметр!) |

`proofread`/`style`/`verify` обслуживает **одна** общая функция
`_run_postprocess_cmd(stage, ...)` ([cli.py:1669](../../src/booktranslator/cli.py)); `stage`
уже передаётся в неё, поэтому правка одна:
`provider=create_stage_provider(cfg, stage)` на [cli.py:1770](../../src/booktranslator/cli.py).
Reflect внутри себя зовёт и translate-промпт, и reflect-промпт — провайдер один (стадия
`reflect`), это нормально.

### 4.7 Промпты — `reasoning_effort: none` + bump версий

| Файл | Изменение | version |
| --- | --- | --- |
| [prompts/glossary_extract.md](../../prompts/glossary_extract.md) | `reasoning_effort: none`; формат ответа — **объект** `{"bookend": "<значение маркера [[BOOKEND::...]] из конца текста>", "entries": [...]}`; инструкция вернуть `bookend` дословно | **3 → 4** |
| [prompts/translate.md](../../prompts/translate.md) | `reasoning_effort: none` | 1 → 2 |
| [prompts/judge.md](../../prompts/judge.md) | `reasoning_effort: none` | 1 → 2 |
| [prompts/reflect.md](../../prompts/reflect.md) | `reasoning_effort: none` | 1 → 2 |
| [prompts/proofread.md](../../prompts/proofread.md) | `reasoning_effort: none` | 2 → 3 |
| [prompts/style.md](../../prompts/style.md) | `reasoning_effort: none` | 2 → 3 |
| [prompts/verify.md](../../prompts/verify.md) | `reasoning_effort: none` | 2 → 3 |

Текст инструкции про `bookend` в glossary-промпте (значение НЕ раскрываем):

> В самом конце предоставленного текста стоит строка-маркер вида `[[BOOKEND::xxxx]]`. Верни
> её значение `xxxx` в поле `bookend` верхнего уровня JSON. Если такого маркера в конце текста
> нет — значит текст пришёл не полностью; всё равно верни поле `bookend` с тем, что видишь в
> самом конце.

### 4.8 `configs/default.yaml` — оверрайд glossary на OpenRouter

```yaml
providers:
  text:                       # дефолт для всех текстовых стадий (перевод, judge, ...)
    base_url: "http://localhost:9000/v1"
    api_key_env: "KIRO_GATEWAY_API_KEY"
  glossary:                   # цельная книга -> нужен большой контекст без обрезки
    base_url: "https://openrouter.ai/api/v1"
    api_key_env: "OPENROUTER_API_KEY"
  image:
    base_url: "https://openrouter.ai/api/v1"
    api_key_env: "OPENROUTER_API_KEY"
```

Поправить комментарии в шапке файла: `text` = Kiro (перевод/чанки), `glossary` = OpenRouter
(вся книга одним вызовом).

**Имя модели glossary — обязательно с префиксом `anthropic/`.** Проверено по коду: нормализации
имени модели нигде нет (grep по `provider.py`/`translator.py`/`pipeline_helpers.py` пуст) —
голая строка из конфига уходит провайдеру дословно. Команда glossary берёт
`chosen_model = model or cfg.models.glossary` ([cli.py:160](../../src/booktranslator/cli.py)) и
передаёт его в `extract_glossary`, где значение из конфига **перекрывает** `model:` из
frontmatter промпта. Сейчас в default.yaml стоит `models.glossary: claude-sonnet-4.6` (формат
Kiro) — для OpenRouter это невалидно. **Поменять на `anthropic/claude-sonnet-4.6`.** Модели
Kiro-стадий (`translate`, `judge`, …) остаются в формате Kiro (`claude-sonnet-4.6`).

---

## 5. Контракты, которые НЕЛЬЗЯ нарушить

- **Paragraph-count** (translate/reflect/proofread/style/verify): `reasoning_effort: none`
  только убирает скрытый thinking; число абзацев на выходе не меняется. Существующее поведение
  waterfall/фолбэка трогать нельзя ([AI_PIPELINE.md](../../AI_PIPELINE.md)).
- **Не писать в кэш до валидации.** Проверено по коду: glossary кэширует `raw_text` на
  [glossary.py:141](../../src/booktranslator/glossary.py) — это **до** `_parse_llm_response`
  на [glossary.py:151](../../src/booktranslator/glossary.py). При обрезке это закэширует мусор.
  → **Переставить** `cache.put(...)` ПОСЛЕ успешной проверки `bookend` и парсинга, либо не
  кэшировать при `TruncationError`. (Иначе `--force` придётся гонять каждый раз.)
- **Round-trip EPUB** — не затрагивается (правок в epub_io нет).

---

## 6. Тесты

Новые/обновлённые (mock-провайдер, без реальных сетевых вызовов):

1. `test_glossary.py::test_truncation_detected` — провайдер возвращает JSON с **неверным**
   `bookend` → `TruncationError`, файл глоссария не пишется, кэш не заполнен.
2. `test_glossary.py::test_missing_bookend` — ответ-список без `bookend` → `TruncationError`.
3. `test_glossary.py::test_empty_content_message` — `text=""` → понятная ошибка с упоминанием
   `finish_reason`/reasoning, НЕ «invalid JSON».
4. `test_glossary.py::test_bookend_ok` — корректный `bookend` → глоссарий парсится, nonce из
   ответа удаляется из записей (если вдруг просочился).
5. `test_provider.py::test_reasoning_effort_passthrough` — при заданном `reasoning_effort`
   в `extra_body` улетает `{"reasoning_effort": "none"}`; при `None` — ключа нет.
6. `test_pipeline_helpers.py::test_create_stage_provider` — оверрайд берётся при наличии,
   иначе `text`; неизвестная стадия → `text`.
7. Прогнать существующие тесты стадий — убедиться, что новый аргумент `complete()` не ломает
   их моки (аргумент keyword-only с дефолтом `None`).

Гейты: `pytest`, `ruff check .`, `ruff format --check .`.

---

## 7. Проверено по коду / остаточные заметки

**Проверено semantic-index + чтением (уже учтено в §4):**

- ✅ **Нормализации имени модели нет** — голая строка из конфига уходит провайдеру дословно;
  для OpenRouter-glossary конфиг обязан содержать `anthropic/`-префикс (см. §4.8).
- ✅ **proofread/style/verify — одна функция** `_run_postprocess_cmd(stage, ...)`; `stage` уже
  параметр, правка одна (см. §4.6).
- ✅ **glossary кэширует до парсинга** ([glossary.py:141](../../src/booktranslator/glossary.py)
  vs `:151`) — переставить `cache.put` после валидации (см. §5).

**Требует внимания при кодировании:**

1. **`_parse_llm_response` и форма ответа.** Сейчас функция
   ([glossary.py:33](../../src/booktranslator/glossary.py)) принимает либо top-level список,
   либо `{"entries": [...]}`. Для `bookend` нужен объект `{"bookend": ..., "entries": [...]}`.
   Аккуратно извлечь `bookend` из объекта, не сломав поддержку series-overrides (`override`-поле
   записей внутри `entries`).
2. **Мёртвый `reflection.extended_thinking`** ([models.py:54](../../src/booktranslator/models.py)):
   нигде не читается (grep подтвердил). Либо удалить, либо зафиксировать, что управление thinking
   теперь только через `reasoning_effort` в промптах. Рекомендация — пометить deprecated/удалить
   в этом же PR.

---

## 8. План внедрения (порядок коммитов)

1. `provider.py` + `prompts.py` (+ тест passthrough) — фундамент, ничего не ломает.
2. `models.py` + `pipeline_helpers.py` (+ тест фабрики) — роутинг, обратно совместимо.
3. `glossary.py` + `prompts/glossary_extract.md` (+ тесты guard) — основной фикс.
4. `cli.py` — переключить стадии на `create_stage_provider`.
5. Остальные `prompts/*.md` — `reasoning_effort: none` + bump версий.
6. `configs/default.yaml` + комментарии.
7. Прогнать полный `pytest` + ruff; ручной прогон `btrans glossary extract` на Bear Head через
   OpenRouter — убедиться, что `bookend` совпадает и глоссарий покрывает всю книгу.

---

## 9. Опционально: чанкование глоссария (НЕ в этом PR)

**Когда понадобится.** Только для книг-гигантов, где даже окно OpenRouter/Claude (200k токенов)
мало: ориентировочно от **~140–150k слов** (≈ 190k+ токенов) вход + ответ перестают влезать.
Для обычных романов (Bear Head ~109k слов) — неактуально.

**Почему пока не нужно.** OpenRouter ведёт к настоящему Claude с контекстом 200k; вся книга +
ответ влезают одним вызовом. При превышении окна OpenRouter/Claude вернёт **честную ошибку**
(не молчаливую обрезку, как Kiro), а `bookend`-guard поймает любую обрезку — тихой порчи не
будет в любом случае.

**Эскиз, если когда-нибудь потребуется:**

```
chunks = split_book(book, target_tokens≈120k, overlap_paragraphs=N)
partial = [extract_glossary(chunk, ...) for chunk in chunks]   # каждый со своим bookend
glossary = merge_and_dedupe(partial)   # по (original, type); конфликт перевода -> flag/override
```

Ключевые сложности слияния: дедуп по `(original, type)`; разрешение конфликтов канонического
перевода между чанками (логика, близкая к series-override в
[glossary.py](../../src/booktranslator/glossary.py)); сохранение «один канонический перевод на
термин». Это отдельная фича с собственным дизайном и тестами — заводить отдельным документом,
когда реально появится книга, не влезающая в 200k.
