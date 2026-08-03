"""Guards for the shipped config files under `configs/`.

These files are what users copy, and nothing else reads them, so drift in them
is invisible until a paid run fails. Commit 7a71698 is the worked example: the
`repair` stage had no `providers.repair` entry, so it fell through to
`providers.text` — the gateway — while `models.repair` carried an
OpenRouter-format name (`anthropic/claude-sonnet-4.6`, where the gateway wants
a bare `claude-sonnet-4.6`). Nothing failed until someone read the file.

Verifies:
- every shipped config parses into `Config`
- `providers` / `models` keys are ones the schema actually reads (pydantic drops
  unknown keys silently, so a typo would route a stage to the wrong endpoint)
- `ModelsConfig` covers every stage in `_TEXT_STAGES`, plus `cover`
- each stage's model name is written in its endpoint's format: bare name for the
  gateway, `vendor/model` for OpenRouter
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from booktranslator.config import load_config
from booktranslator.models import Config, ModelsConfig, ProvidersConfig
from booktranslator.pipeline_helpers import _TEXT_STAGES, create_stage_provider

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"

# Every file users are meant to copy or run with.
SHIPPED_CONFIGS = sorted(
    p for p in CONFIGS_DIR.iterdir() if p.suffix in {".yaml", ".example"} and p.is_file()
)

# How each known endpoint spells model names. OpenRouter namespaces them by
# vendor; kiro-gateway takes the bare name and rejects a prefixed one.
_PREFIXED = "prefixed"
_BARE = "bare"
_ENDPOINT_NAME_STYLE = {
    "openrouter.ai": _PREFIXED,
    "localhost:9000": _BARE,
    "127.0.0.1:9000": _BARE,
}


def _name_style(base_url: str) -> str | None:
    """Naming convention for `base_url`, or None if the endpoint is unknown."""
    for host, style in _ENDPOINT_NAME_STYLE.items():
        if host in base_url:
            return style
    return None


def _assert_name_matches_endpoint(config_name: str, stage: str, model: str, base_url: str) -> None:
    style = _name_style(base_url)
    if style is None:  # Unknown endpoint — we have no convention to check against.
        return
    if style is _PREFIXED and "/" not in model:
        pytest.fail(
            f"{config_name}: stage {stage!r} routes to {base_url} (OpenRouter), which "
            f"namespaces models by vendor, but models.{stage} is the bare name {model!r}. "
            f"Expected something like 'anthropic/{model}'."
        )
    if style is _BARE and "/" in model:
        pytest.fail(
            f"{config_name}: stage {stage!r} routes to {base_url} (kiro-gateway), which "
            f"takes bare model names, but models.{stage} is {model!r}. "
            f"Expected {model.split('/', 1)[1]!r}."
        )


def _raw(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


@contextmanager
def _dummy_api_keys(cfg: Config):
    """Satisfy the provider's key check so routing can be exercised offline.

    `create_stage_provider` builds a real provider, which refuses to start
    without its key. Every `api_key_env` the config names gets a placeholder.
    """
    sections = [cfg.providers.text, cfg.providers.image] + [
        section
        for stage in _TEXT_STAGES
        if (section := getattr(cfg.providers, stage, None)) is not None
    ]
    with patch.dict("os.environ", {s.api_key_env: "test-key" for s in sections}):
        yield


class TestSchemaCoversEveryStage:
    """Structural guards that hold regardless of any particular config file."""

    def test_at_least_one_config_is_shipped(self):
        # Guards the discovery glob itself: an empty list would make every
        # parametrized test below vacuously pass.
        assert SHIPPED_CONFIGS, f"no config files found under {CONFIGS_DIR}"

    def test_models_config_has_an_entry_per_text_stage(self):
        missing = set(_TEXT_STAGES) - set(ModelsConfig.model_fields)
        assert not missing, (
            f"stages {sorted(missing)} can be routed but have no models.<stage> entry — "
            f"add them to ModelsConfig"
        )

    def test_models_config_has_no_entries_beyond_the_stages_and_cover(self):
        extra = set(ModelsConfig.model_fields) - set(_TEXT_STAGES) - {"cover"}
        assert not extra, (
            f"models.{sorted(extra)} name no routable stage — either add them to "
            f"_TEXT_STAGES or drop them"
        )

    def test_providers_config_has_an_override_per_text_stage(self):
        missing = set(_TEXT_STAGES) - set(ProvidersConfig.model_fields)
        assert not missing, (
            f"stages {sorted(missing)} cannot be routed independently — add a "
            f"providers.<stage> field for them"
        )


@pytest.mark.parametrize("path", SHIPPED_CONFIGS, ids=lambda p: p.name)
class TestShippedConfig:
    def test_parses(self, path: Path):
        assert isinstance(load_config(path), Config)

    def test_provider_keys_are_read_by_the_schema(self, path: Path):
        # pydantic drops unknown keys without complaint, so a misspelled stage
        # name would leave that stage silently on providers.text.
        known = set(ProvidersConfig.model_fields)
        unknown = set(_raw(path).get("providers", {})) - known
        assert not unknown, (
            f"{path.name}: providers.{sorted(unknown)} are ignored by the loader — "
            f"those stages stay on providers.text. Known keys: {sorted(known)}"
        )

    def test_model_keys_are_read_by_the_schema(self, path: Path):
        known = set(ModelsConfig.model_fields)
        unknown = set(_raw(path).get("models", {})) - known
        assert not unknown, (
            f"{path.name}: models.{sorted(unknown)} are ignored by the loader. "
            f"Known keys: {sorted(known)}"
        )

    def test_model_names_match_their_endpoint(self, path: Path):
        cfg = load_config(path)
        with _dummy_api_keys(cfg):
            for stage in _TEXT_STAGES:
                _assert_name_matches_endpoint(
                    path.name,
                    stage,
                    getattr(cfg.models, stage),
                    create_stage_provider(cfg, stage)._base_url,
                )

    def test_cover_model_name_matches_the_image_endpoint(self, path: Path):
        # Cover translation goes through providers.image, not providers.text.
        cfg = load_config(path)
        _assert_name_matches_endpoint(
            path.name, "cover", cfg.models.cover, cfg.providers.image.base_url
        )
