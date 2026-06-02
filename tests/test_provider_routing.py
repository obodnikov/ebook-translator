"""Tests for provider routing: create_provider and create_image_provider.

Verifies:
- Default fallback (no config) uses OPENROUTER_API_KEY and OpenRouter URL
- Config-driven text provider uses providers.text settings
- Config-driven image provider uses providers.image settings
- Different env vars for text vs image providers
- Missing API key raises RuntimeError with correct env var name
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from booktranslator.models import Config, ProviderConfig, ProvidersConfig
from booktranslator.pipeline_helpers import create_image_provider, create_provider
from booktranslator.provider import DEFAULT_BASE_URL


class TestCreateProviderDefaults:
    """Tests for create_provider without config (backward compatibility)."""

    @patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key-default"})
    def test_default_uses_openrouter(self):
        provider = create_provider()
        assert provider._base_url == DEFAULT_BASE_URL
        assert provider._api_key == "test-key-default"

    @patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key-default"})
    def test_default_no_config(self):
        provider = create_provider(None)
        assert provider._base_url == DEFAULT_BASE_URL

    def test_default_missing_key_raises(self):
        with patch.dict("os.environ", {}, clear=True):
            import os

            backup = os.environ.get("OPENROUTER_API_KEY")
            if "OPENROUTER_API_KEY" in os.environ:
                del os.environ["OPENROUTER_API_KEY"]
            try:
                with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
                    create_provider()
            finally:
                if backup:
                    os.environ["OPENROUTER_API_KEY"] = backup


class TestCreateProviderWithConfig:
    """Tests for create_provider with explicit config."""

    @patch.dict("os.environ", {"MY_TEXT_KEY": "text-api-key-123"})
    def test_text_provider_uses_config(self):
        cfg = Config(
            providers=ProvidersConfig(
                text=ProviderConfig(
                    base_url="http://localhost:9000/v1",
                    api_key_env="MY_TEXT_KEY",
                ),
            )
        )
        provider = create_provider(cfg)
        assert provider._base_url == "http://localhost:9000/v1"
        assert provider._api_key == "text-api-key-123"

    @patch.dict("os.environ", {"MY_TEXT_KEY": "text-key"})
    def test_text_provider_extra_headers(self):
        cfg = Config(
            providers=ProvidersConfig(
                text=ProviderConfig(
                    base_url="http://localhost:9000/v1",
                    api_key_env="MY_TEXT_KEY",
                    extra_headers={"X-Custom": "hello"},
                ),
            )
        )
        provider = create_provider(cfg)
        assert "X-Custom" in provider._extra_headers
        assert provider._extra_headers["X-Custom"] == "hello"

    def test_text_provider_wrong_env_raises(self):
        """Missing env var referenced in config raises with correct name."""
        with patch.dict("os.environ", {}, clear=True):
            import os

            # Ensure the var is definitely not set
            os.environ.pop("NONEXISTENT_KEY", None)
            cfg = Config(
                providers=ProvidersConfig(
                    text=ProviderConfig(
                        base_url="http://example.com/v1",
                        api_key_env="NONEXISTENT_KEY",
                    ),
                )
            )
            with pytest.raises(RuntimeError, match="NONEXISTENT_KEY"):
                create_provider(cfg)


class TestCreateImageProviderWithConfig:
    """Tests for create_image_provider with explicit config."""

    @patch.dict("os.environ", {"MY_IMAGE_KEY": "image-api-key-456"})
    def test_image_provider_uses_config(self):
        cfg = Config(
            providers=ProvidersConfig(
                image=ProviderConfig(
                    base_url="https://openrouter.ai/api/v1",
                    api_key_env="MY_IMAGE_KEY",
                ),
            )
        )
        provider = create_image_provider(cfg)
        assert provider._base_url == "https://openrouter.ai/api/v1"
        assert provider._api_key == "image-api-key-456"

    @patch.dict("os.environ", {"OPENROUTER_API_KEY": "or-key"})
    def test_image_provider_default_fallback(self):
        provider = create_image_provider()
        assert provider._base_url == DEFAULT_BASE_URL
        assert provider._api_key == "or-key"


class TestTextAndImageProvidersSeparation:
    """Tests that text and image providers are truly independent."""

    @patch.dict(
        "os.environ",
        {"TEXT_KEY": "text-secret", "IMAGE_KEY": "image-secret"},
    )
    def test_different_providers(self):
        cfg = Config(
            providers=ProvidersConfig(
                text=ProviderConfig(
                    base_url="http://localhost:8000/v1",
                    api_key_env="TEXT_KEY",
                ),
                image=ProviderConfig(
                    base_url="https://openrouter.ai/api/v1",
                    api_key_env="IMAGE_KEY",
                ),
            )
        )
        text_prov = create_provider(cfg)
        image_prov = create_image_provider(cfg)

        assert text_prov._base_url == "http://localhost:8000/v1"
        assert text_prov._api_key == "text-secret"

        assert image_prov._base_url == "https://openrouter.ai/api/v1"
        assert image_prov._api_key == "image-secret"

    @patch.dict("os.environ", {"SHARED_KEY": "shared-secret"})
    def test_same_provider_both_slots(self):
        """Both slots can point to the same endpoint."""
        cfg = Config(
            providers=ProvidersConfig(
                text=ProviderConfig(
                    base_url="https://openrouter.ai/api/v1",
                    api_key_env="SHARED_KEY",
                ),
                image=ProviderConfig(
                    base_url="https://openrouter.ai/api/v1",
                    api_key_env="SHARED_KEY",
                ),
            )
        )
        text_prov = create_provider(cfg)
        image_prov = create_image_provider(cfg)

        assert text_prov._base_url == image_prov._base_url
        assert text_prov._api_key == image_prov._api_key
