"""Tests for pipeline_helpers.py: shared logic for judge/reflect orchestration.

Covers:
- collect_chunk_originals: correct filtering by chunk IDs
- collect_stage_translations: latest revision selection, stage filtering
- build_reflect_input: O(1) lookup, correct data assembly
- create_provider: factory works
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from booktranslator.cache import Cache
from booktranslator.pipeline_helpers import (
    ChunkerConfigMismatchError,
    build_reflect_input,
    collect_chunk_originals,
    collect_chunk_originals_with_paragraphs,
    collect_preferred_translations,
    collect_stage_translations,
    create_provider,
    create_stage_provider,
    save_chunker_params,
    verify_chunker_params,
)
from tests.epub_fixtures import write_minimal_epub


@pytest.fixture
def cache(tmp_path: Path) -> Cache:
    return Cache(tmp_path / "test.sqlite")


# ---------------------------------------------------------------------------
# Mock ChunkSet for testing
# ---------------------------------------------------------------------------


class MockChunk:
    def __init__(self, chunk_id: str):
        self.id = chunk_id


class MockChunkSet:
    def __init__(self, chunks_data: dict[str, list[str]]):
        """chunks_data: {chunk_id: [paragraph_fragments]}"""
        self._data = chunks_data
        self.chunks = [MockChunk(cid) for cid in chunks_data]

    def render_main(self, chunk) -> list[str]:
        return self._data.get(chunk.id, [])


# ---------------------------------------------------------------------------
# collect_chunk_originals
# ---------------------------------------------------------------------------


class TestCollectChunkOriginals:
    def test_filters_by_ids(self):
        cs = MockChunkSet(
            {
                "c1": ["<p>Hello</p>", "<p>World</p>"],
                "c2": ["<p>Foo</p>"],
                "c3": ["<p>Bar</p>"],
            }
        )
        result = collect_chunk_originals(cs, ["c1", "c3"])
        assert "c1" in result
        assert "c3" in result
        assert "c2" not in result

    def test_joins_paragraphs(self):
        cs = MockChunkSet(
            {
                "c1": ["<p>Line 1</p>", "<p>Line 2</p>"],
            }
        )
        result = collect_chunk_originals(cs, ["c1"])
        assert result["c1"] == "<p>Line 1</p>\n<p>Line 2</p>"

    def test_empty_ids(self):
        cs = MockChunkSet({"c1": ["<p>A</p>"]})
        result = collect_chunk_originals(cs, [])
        assert result == {}

    def test_missing_ids_ignored(self):
        cs = MockChunkSet({"c1": ["<p>A</p>"]})
        result = collect_chunk_originals(cs, ["c1", "c99"])
        assert "c1" in result
        assert "c99" not in result


# ---------------------------------------------------------------------------
# collect_chunk_originals_with_paragraphs
# ---------------------------------------------------------------------------


class TestCollectChunkOriginalsWithParagraphs:
    def test_returns_tuple(self):
        cs = MockChunkSet(
            {
                "c1": ["<p>A</p>", "<p>B</p>"],
            }
        )
        result = collect_chunk_originals_with_paragraphs(cs, ["c1"])
        text, paras = result["c1"]
        assert text == "<p>A</p>\n<p>B</p>"
        assert paras == ["<p>A</p>", "<p>B</p>"]


# ---------------------------------------------------------------------------
# collect_stage_translations
# ---------------------------------------------------------------------------


class TestCollectStageTranslations:
    def test_gets_translate_stage(self, cache: Cache):
        cache.put("k1", "translate", "m", "v1", "Translation 1", meta={"chunk_id": "c1"})
        cache.put("k2", "judge", "m", "v1", '{"score": 5}', meta={"chunk_id": "c1"})

        result = collect_stage_translations(cache, ["c1"], "translate")
        assert result == {"c1": "Translation 1"}

    def test_gets_latest_revision(self, cache: Cache):
        """When multiple translate entries exist, returns the latest."""
        cache.put("k1", "translate", "m", "v1", "Old translation", meta={"chunk_id": "c1"})
        cache.put("k2", "translate", "m", "v2", "New translation", meta={"chunk_id": "c1"})

        result = collect_stage_translations(cache, ["c1"], "translate")
        # Should get the latest (v2)
        assert result["c1"] == "New translation"

    def test_missing_chunk_not_in_result(self, cache: Cache):
        cache.put("k1", "translate", "m", "v1", "text", meta={"chunk_id": "c1"})

        result = collect_stage_translations(cache, ["c1", "c99"], "translate")
        assert "c1" in result
        assert "c99" not in result

    def test_wrong_stage_not_returned(self, cache: Cache):
        cache.put("k1", "reflect", "m", "v1", "reflected", meta={"chunk_id": "c1"})

        result = collect_stage_translations(cache, ["c1"], "translate")
        assert result == {}

    def test_gets_reflect_stage(self, cache: Cache):
        cache.put("k1", "translate", "m", "v1", "original", meta={"chunk_id": "c1"})
        cache.put("k2", "reflect", "m", "v1", "reflected", meta={"chunk_id": "c1"})

        result = collect_stage_translations(cache, ["c1"], "reflect")
        assert result == {"c1": "reflected"}


# ---------------------------------------------------------------------------
# collect_preferred_translations
# ---------------------------------------------------------------------------


class TestCollectPreferredTranslations:
    def test_waterfall_picks_latest_stage(self, cache: Cache):
        cache.put("k1", "translate", "m", "v1", "original", meta={"chunk_id": "c1"})
        cache.put("k2", "reflect", "m", "v1", "reflected", meta={"chunk_id": "c1"})

        result = collect_preferred_translations(cache, ["c1"])
        assert result["c1"] == "reflected"

    def test_preference_overrides_waterfall(self, cache: Cache):
        cache.put("k1", "translate", "m", "v1", "original", meta={"chunk_id": "c1"})
        cache.put("k2", "reflect", "m", "v1", "reflected", meta={"chunk_id": "c1"})
        cache.set_preference("c1", "translate")

        result = collect_preferred_translations(cache, ["c1"])
        assert result["c1"] == "original"

    def test_multiple_chunks(self, cache: Cache):
        cache.put("k1", "translate", "m", "v1", "t1", meta={"chunk_id": "c1"})
        cache.put("k2", "translate", "m", "v1", "t2", meta={"chunk_id": "c2"})
        cache.put("k3", "reflect", "m", "v1", "r1", meta={"chunk_id": "c1"})

        result = collect_preferred_translations(cache, ["c1", "c2"])
        assert result["c1"] == "r1"  # reflect wins
        assert result["c2"] == "t2"  # only translate available


# ---------------------------------------------------------------------------
# build_reflect_input
# ---------------------------------------------------------------------------


class TestBuildReflectInput:
    def test_builds_correct_structure(self, cache: Cache):
        cs = MockChunkSet(
            {
                "c1": ["<p>Hello</p>", "<p>World</p>"],
                "c2": ["<p>Foo</p>"],
            }
        )
        cache.put("k1", "translate", "m", "v1", "Translated c1", meta={"chunk_id": "c1"})
        cache.put("k2", "translate", "m", "v1", "Translated c2", meta={"chunk_id": "c2"})

        judge_by_id = {
            "c1": {"score": 2, "issues": ["accuracy: omission"]},
            "c2": {"score": 1, "issues": ["glossary: wrong"]},
        }

        result = build_reflect_input(cs, cache, {"c1", "c2"}, judge_by_id)

        assert len(result) == 2
        c1_data = next(r for r in result if r["chunk_id"] == "c1")
        assert c1_data["original_text"] == "<p>Hello</p>\n<p>World</p>"
        assert c1_data["original_paragraphs"] == ["<p>Hello</p>", "<p>World</p>"]
        assert c1_data["translated_text"] == "Translated c1"
        assert c1_data["judge_score"] == 2
        assert c1_data["judge_issues"] == ["accuracy: omission"]

    def test_skips_chunks_without_translation(self, cache: Cache):
        cs = MockChunkSet(
            {
                "c1": ["<p>Hello</p>"],
                "c2": ["<p>World</p>"],
            }
        )
        # Only c1 has a translation
        cache.put("k1", "translate", "m", "v1", "Translated", meta={"chunk_id": "c1"})

        judge_by_id = {
            "c1": {"score": 2, "issues": []},
            "c2": {"score": 1, "issues": []},
        }

        result = build_reflect_input(cs, cache, {"c1", "c2"}, judge_by_id)
        assert len(result) == 1
        assert result[0]["chunk_id"] == "c1"

    def test_uses_latest_translate_revision(self, cache: Cache):
        """build_reflect_input should use the latest translate revision."""
        cs = MockChunkSet({"c1": ["<p>Text</p>"]})
        cache.put("k1", "translate", "m", "v1", "Old translation", meta={"chunk_id": "c1"})
        cache.put("k2", "translate", "m", "v2", "New translation", meta={"chunk_id": "c1"})

        judge_by_id = {"c1": {"score": 2, "issues": []}}
        result = build_reflect_input(cs, cache, {"c1"}, judge_by_id)

        assert result[0]["translated_text"] == "New translation"

    def test_handles_missing_judge_data(self, cache: Cache):
        cs = MockChunkSet({"c1": ["<p>Text</p>"]})
        cache.put("k1", "translate", "m", "v1", "Translated", meta={"chunk_id": "c1"})

        # No judge data for c1
        result = build_reflect_input(cs, cache, {"c1"}, {})
        assert len(result) == 1
        assert result[0]["judge_score"] == 0
        assert result[0]["judge_issues"] == []


# ---------------------------------------------------------------------------
# create_provider
# ---------------------------------------------------------------------------


class TestCreateProvider:
    @patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key-123"})
    def test_creates_provider(self):
        provider = create_provider()
        assert provider is not None
        assert provider.client is not None

    def test_raises_without_key(self):
        """Without API key, provider creation should fail."""
        with patch.dict("os.environ", {}, clear=True):
            # Remove the key if it exists
            import os

            env_backup = os.environ.get("OPENROUTER_API_KEY")
            if "OPENROUTER_API_KEY" in os.environ:
                del os.environ["OPENROUTER_API_KEY"]
            try:
                with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
                    create_provider()
            finally:
                if env_backup:
                    os.environ["OPENROUTER_API_KEY"] = env_backup


# ---------------------------------------------------------------------------
# Chunker config persistence and mismatch detection
# ---------------------------------------------------------------------------


class TestChunkerConfigPersistence:
    def test_save_and_verify_matching(self, cache: Cache):
        """Matching params pass verification silently."""
        save_chunker_params(cache, target_words=2000, overlap_paragraphs=1)
        # Should not raise
        verify_chunker_params(cache, target_words=2000, overlap_paragraphs=1)

    def test_verify_mismatch_target_words(self, cache: Cache):
        """Different target_words raises ChunkerConfigMismatchError."""
        save_chunker_params(cache, target_words=2000, overlap_paragraphs=1)
        with pytest.raises(ChunkerConfigMismatchError, match="target_words=2000"):
            verify_chunker_params(cache, target_words=3000, overlap_paragraphs=1)

    def test_verify_mismatch_overlap(self, cache: Cache):
        """Different overlap_paragraphs raises ChunkerConfigMismatchError."""
        save_chunker_params(cache, target_words=2000, overlap_paragraphs=1)
        with pytest.raises(ChunkerConfigMismatchError, match="overlap=1"):
            verify_chunker_params(cache, target_words=2000, overlap_paragraphs=2)

    def test_verify_both_mismatch(self, cache: Cache):
        """Both params different raises error."""
        save_chunker_params(cache, target_words=2000, overlap_paragraphs=1)
        with pytest.raises(ChunkerConfigMismatchError):
            verify_chunker_params(cache, target_words=1500, overlap_paragraphs=3)

    def test_verify_no_saved_params_passes(self, cache: Cache):
        """Legacy cache without saved params passes silently."""
        # Don't save anything — simulates old cache
        # Should not raise
        verify_chunker_params(cache, target_words=9999, overlap_paragraphs=99)

    def test_save_overwrites_previous(self, cache: Cache):
        """Saving with different params raises error (no silent overwrite)."""
        save_chunker_params(cache, target_words=2000, overlap_paragraphs=1)
        with pytest.raises(ChunkerConfigMismatchError):
            save_chunker_params(cache, target_words=3000, overlap_paragraphs=2)

    def test_save_same_params_is_noop(self, cache: Cache):
        """Saving with same params is a no-op (idempotent)."""
        save_chunker_params(cache, target_words=2000, overlap_paragraphs=1)
        # Should not raise
        save_chunker_params(cache, target_words=2000, overlap_paragraphs=1)

    def test_save_first_time_stores(self, cache: Cache):
        """First save stores the params successfully."""
        save_chunker_params(cache, target_words=1500, overlap_paragraphs=2)
        # Verify they're stored
        verify_chunker_params(cache, target_words=1500, overlap_paragraphs=2)


# ---------------------------------------------------------------------------
# create_stage_provider
# ---------------------------------------------------------------------------


from booktranslator.models import Config, ProviderConfig, ProvidersConfig  # noqa: E402


class TestCreateStageProvider:
    @patch.dict("os.environ", {"OPENROUTER_API_KEY": "or-key"})
    def test_no_config_falls_back_to_defaults(self):
        """None config returns a default OpenRouterProvider."""
        provider = create_stage_provider(None, "glossary")
        assert provider is not None

    @patch.dict("os.environ", {"TEXT_KEY": "text-secret"})
    def test_no_override_uses_text(self):
        """Stage with no override falls back to providers.text."""
        cfg = Config(
            providers=ProvidersConfig(
                text=ProviderConfig(base_url="http://text/v1", api_key_env="TEXT_KEY"),
            )
        )
        provider = create_stage_provider(cfg, "translate")
        assert provider._base_url == "http://text/v1"
        assert provider._api_key == "text-secret"

    @patch.dict("os.environ", {"TEXT_KEY": "text-secret", "GLOSSARY_KEY": "glossary-secret"})
    def test_override_used_when_set(self):
        """Stage with explicit override uses that provider."""
        cfg = Config(
            providers=ProvidersConfig(
                text=ProviderConfig(base_url="http://text/v1", api_key_env="TEXT_KEY"),
                glossary=ProviderConfig(
                    base_url="https://openrouter.ai/api/v1", api_key_env="GLOSSARY_KEY"
                ),
            )
        )
        provider = create_stage_provider(cfg, "glossary")
        assert provider._base_url == "https://openrouter.ai/api/v1"
        assert provider._api_key == "glossary-secret"

    @patch.dict("os.environ", {"TEXT_KEY": "text-secret"})
    def test_unknown_stage_falls_back_to_text(self):
        """Unknown stage name falls back to providers.text silently."""
        cfg = Config(
            providers=ProvidersConfig(
                text=ProviderConfig(base_url="http://text/v1", api_key_env="TEXT_KEY"),
            )
        )
        provider = create_stage_provider(cfg, "nonexistent_stage")
        assert provider._base_url == "http://text/v1"

    @patch.dict("os.environ", {"TEXT_KEY": "text-secret", "JUDGE_KEY": "judge-secret"})
    def test_judge_override_independent_of_translate(self):
        """judge override does not affect translate (which uses text)."""
        cfg = Config(
            providers=ProvidersConfig(
                text=ProviderConfig(base_url="http://text/v1", api_key_env="TEXT_KEY"),
                judge=ProviderConfig(base_url="http://judge/v1", api_key_env="JUDGE_KEY"),
            )
        )
        judge_prov = create_stage_provider(cfg, "judge")
        translate_prov = create_stage_provider(cfg, "translate")
        assert judge_prov._base_url == "http://judge/v1"
        assert translate_prov._base_url == "http://text/v1"


# ---------------------------------------------------------------------------
# Originals must not be read from a tree that translation has written into
# ---------------------------------------------------------------------------


@pytest.fixture
def tiny_epub(tmp_path: Path) -> Path:
    return write_minimal_epub(tmp_path / "test-book.epub")


class TestOriginalsSurviveTranslation:
    """The translator splices each translated paragraph into the live tree
    (translator.py), so a ChunkSet that has been translated no longer holds the
    source text — asking it for originals hands back the translation.

    On the Bear Head run this fed the judge the Russian text in both slots. It
    compared the translation with itself, scored 102 of 110 chunks a perfect 5,
    and the four chunks that answered "the Original section is in Russian" were
    reporting the truth. Verify, whose whole job is comparing against the
    source, was hit the same way.

    The fix is that cli.translate builds `source_chunk_set` from a second read
    of the EPUB and hands that to judge, reflect and verify. These tests pin
    the property that fix relies on.
    """

    @staticmethod
    def _build(epub: Path):
        from booktranslator.chunker import chunk_book
        from booktranslator.epub_io import read_book_structured

        return chunk_book(read_book_structured(epub), target_words=2000, overlap_paragraphs=1)

    @staticmethod
    def _translate_in_place(chunk_set) -> None:
        """Do to the tree exactly what Translator does after a chunk returns."""
        from lxml import etree

        for chapter in chunk_set.book.chapters:
            for i, para in enumerate(chapter.paragraphs):
                new_el = etree.fromstring(
                    '<p xmlns="http://www.w3.org/1999/xhtml">Русский перевод абзаца.</p>'
                )
                para.getparent().replace(para, new_el)
                chapter.paragraphs[i] = new_el

    def test_translated_tree_no_longer_yields_originals(self, tiny_epub: Path):
        chunk_set = self._build(tiny_epub)
        ids = [c.id for c in chunk_set.chunks]
        assert "vestigium" in collect_chunk_originals(chunk_set, ids)[ids[0]]

        self._translate_in_place(chunk_set)

        after = collect_chunk_originals(chunk_set, ids)[ids[0]]
        assert "vestigium" not in after, (
            "translation has overwritten the source in this ChunkSet — this is the "
            "hazard cli.translate works around with source_chunk_set"
        )
        assert "Русский перевод" in after

    def test_a_second_chunk_set_still_holds_the_source(self, tiny_epub: Path):
        translated = self._build(tiny_epub)
        source = self._build(tiny_epub)  # what cli.translate calls source_chunk_set

        self._translate_in_place(translated)

        ids = [c.id for c in source.chunks]
        originals = collect_chunk_originals(source, ids)[ids[0]]
        assert "vestigium" in originals
        assert "Русский перевод" not in originals

    def test_originals_with_paragraphs_are_protected_too(self, tiny_epub: Path):
        """reflect reads through collect_chunk_originals_with_paragraphs."""
        translated = self._build(tiny_epub)
        source = self._build(tiny_epub)

        self._translate_in_place(translated)

        ids = {c.id for c in source.chunks}
        text, paragraphs = collect_chunk_originals_with_paragraphs(source, ids)[next(iter(ids))]
        assert "vestigium" in text
        assert any("vestigium" in p for p in paragraphs)
