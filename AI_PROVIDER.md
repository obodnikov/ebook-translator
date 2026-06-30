# AI rules — Providers, prompts, config (Python)

Scope: `src/booktranslator/provider.py`, `prompts.py`, `config.py`, `models.py`. This layer is the
boundary to OpenRouter (and any OpenAI-compatible endpoint), the prompt loader, and the typed
config. See [ARCHITECTURE.md §5, §7, §8, §11](ARCHITECTURE.md); this file is the coding contract.

## Language & build

- Python 3.12+, full type hints, `ruff` clean (line length 100). No mypy gate.
- Config and data shapes are **pydantic v2** models in `models.py`; the single typed `Config`
  tree is the contract every other layer reads. Add fields there with defaults, don't pass loose
  dicts around.

## Text and image providers are SEPARATE and independently configurable

This is a deliberate split (`ProvidersConfig.text` vs `ProvidersConfig.image`, see
`docs/chats/configuring-multiple-openai-compatible-providers-*`). Honour it:

- Each provider has its own `base_url`, `api_key_env`, and `extra_headers` (`ProviderConfig`).
  Both default to OpenRouter but can point at different gateways. Never assume text and image
  share a client, key, or base URL.
- Text completions go through the **text** provider; cover/image generation goes through the
  **image** provider. Routing a call to the wrong provider is a bug even if both happen to point
  at OpenRouter today.
- The model is selected per stage via `ModelsConfig` (`glossary`, `translate`, `judge`, `reflect`,
  `proofread`, `style`, `verify`, `cover`). Switching models is a config/CLI override, not a code
  change.

## Provider client (`provider.py`)

- Text uses the `openai` SDK pointed at the configured `base_url`. Image generation uses `httpx`
  directly because the OpenAI SDK doesn't support the `modalities` + `image_config` fields — keep
  these two paths distinct; don't try to force image gen through the chat SDK.
- Retry transient failures (429, 500, 502, 503) with exponential backoff (`RetryConfig`,
  `tenacity`); do not retry permanent 4xx. Surface the final error rather than returning empty
  output that a later stage would treat as a valid (empty) translation.
- Log cost/tokens to `work/<book>/log.jsonl`, but **never log raw API keys or full book text**.
  `cost-report.json` carries aggregate cost only.

## Prompts (`prompts.py`) — frontmatter + Jinja2

- A prompt file is markdown with YAML frontmatter (`version`, `model`, `temperature`,
  `max_tokens`) plus `# System` / `# User` sections, loaded by `load_prompt` and rendered by
  `render_prompt`. Keep prompt *content* in `prompts/*.md`, not inlined in Python.
- `version` is part of the cache key (see [AI_PIPELINE.md](AI_PIPELINE.md)) — **bump it whenever
  you change a prompt's behaviour**. A prompt edit without a version bump silently serves stale
  cached output.
- All variable data reaches the prompt through documented Jinja2 placeholders
  (`{{ chunk_text }}`, `{{ glossary_json }}`, `{{ prev_overlap_text }}`, …). Don't introduce a new
  placeholder without wiring it in the render context; don't string-concatenate prompt text.
- Prompts instruct the model to return strict, parseable output (JSON for glossary/judge,
  paragraph-per-paragraph text for translate). Validate the response (pydantic / count check)
  before trusting it — the model returning prose preamble is an expected failure to guard against.

## Secrets & config (`config.py`, `models.py`)

- Keys come only from the environment / `.env` (`OPENROUTER_API_KEY`, `TELEGRAM_BOT_TOKEN`,
  `TELEGRAM_CHAT_ID`); `config.py` resolves `api_key_env` to the actual key at call time. Never
  hard-code, commit, echo, or log a key.
- Config is YAML merged with CLI overrides into the pydantic `Config`. Presets (`premium`,
  `budget`) are just different YAML files; keep preset logic data-driven, not branched in code.

## What stays out of this layer

- No EPUB tree work (that's [AI_EPUB.md](AI_EPUB.md)).
- No cache, stage, or pause/resume logic (that's [AI_PIPELINE.md](AI_PIPELINE.md)).
