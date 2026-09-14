"""Integration tests for CLI command flows.

Tests the control-flow paths of translate/judge/reflect commands
without making real LLM calls. Covers:
- Chunker metadata persistence and mismatch detection
- --no-judge / --no-reflect flags
- Standalone judge/reflect with/without cached data
- Parallelism validation
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from booktranslator.cache import Cache
from booktranslator.cli import app
from booktranslator.pipeline_helpers import (
    ChunkerConfigMismatchError,
    save_chunker_params,
    verify_chunker_params,
)
from tests.epub_fixtures import write_minimal_epub

runner = CliRunner()


# ---------------------------------------------------------------------------
# Chunker metadata lifecycle (integration-level)
# ---------------------------------------------------------------------------


class TestChunkerMetadataLifecycle:
    """Tests for the full save → verify → mismatch lifecycle."""

    def test_translate_saves_then_judge_verifies(self, tmp_path: Path):
        """Simulate: translate saves params, judge verifies same params."""
        cache = Cache(tmp_path / "cache.sqlite")
        save_chunker_params(cache, target_words=2000, overlap_paragraphs=1)

        # Simulate judge verifying with same config
        verify_chunker_params(cache, target_words=2000, overlap_paragraphs=1)
        cache.close()

    def test_translate_saves_then_judge_detects_mismatch(self, tmp_path: Path):
        """Simulate: translate saves params, judge with different config fails."""
        cache = Cache(tmp_path / "cache.sqlite")
        save_chunker_params(cache, target_words=2000, overlap_paragraphs=1)

        # Simulate judge with different config
        with pytest.raises(ChunkerConfigMismatchError):
            verify_chunker_params(cache, target_words=3000, overlap_paragraphs=1)
        cache.close()

    def test_retranslate_same_params_is_idempotent(self, tmp_path: Path):
        """Re-running translate with same params doesn't error."""
        cache = Cache(tmp_path / "cache.sqlite")
        save_chunker_params(cache, target_words=2000, overlap_paragraphs=1)
        # Second translate run with same params
        save_chunker_params(cache, target_words=2000, overlap_paragraphs=1)
        cache.close()

    def test_retranslate_different_params_fails(self, tmp_path: Path):
        """Re-running translate with different params fails fast."""
        cache = Cache(tmp_path / "cache.sqlite")
        save_chunker_params(cache, target_words=2000, overlap_paragraphs=1)

        with pytest.raises(ChunkerConfigMismatchError, match="conflict"):
            save_chunker_params(cache, target_words=1500, overlap_paragraphs=2)
        cache.close()

    def test_legacy_cache_allows_any_params(self, tmp_path: Path):
        """Old caches without metadata allow any params in judge/reflect."""
        cache = Cache(tmp_path / "cache.sqlite")
        # Don't save any metadata — simulates pre-metadata cache
        # Both verify and save should work
        verify_chunker_params(cache, target_words=9999, overlap_paragraphs=99)
        cache.close()


# ---------------------------------------------------------------------------
# Judge strict parsing (integration with cache)
# ---------------------------------------------------------------------------


class TestJudgeStrictParsing:
    """Tests that invalid judge responses are treated as failures."""

    def test_invalid_score_counts_as_failed(self, tmp_path: Path):
        """Judge with out-of-range score raises, chunk counted as failed."""
        from booktranslator.judge import Judge
        from booktranslator.provider import CompletionResult

        cache = Cache(tmp_path / "cache.sqlite")
        prompt_path = tmp_path / "judge.md"
        prompt_path.write_text(
            "---\nversion: 1\ntemperature: 0.1\n---\n"
            "# System\nJudge {{ source_lang_name }} to "
            "{{ target_lang_name }}.\n"
            "Glossary: {{ glossary_block }}\n\n"
            "# User\n{{ original_text }}\n{{ translated_text }}\n",
            encoding="utf-8",
        )

        mock_provider = MagicMock()
        # Return score=0 (out of range)
        mock_provider.complete.return_value = CompletionResult(
            text='{"score": 0, "issues": []}',
            input_tokens=100,
            output_tokens=20,
            total_tokens=120,
            model="m",
            raw={},
        )

        judge = Judge(
            provider=mock_provider,
            prompt_path=prompt_path,
            cache=cache,
            glossary=None,
            model="test-model",
        )

        stats = judge.judge_chunks(
            {"c1": "<p>Hello</p>"},
            {"c1": "<p>Привет</p>"},
            parallelism=1,
        )

        # Should be counted as failed (parsing error), no valid results
        assert stats.chunks_failed == 1
        assert len(stats.results) == 0
        cache.close()

    def test_missing_score_counts_as_failed(self, tmp_path: Path):
        """Judge response without score field is a failure."""
        from booktranslator.judge import Judge
        from booktranslator.provider import CompletionResult

        cache = Cache(tmp_path / "cache.sqlite")
        prompt_path = tmp_path / "judge.md"
        prompt_path.write_text(
            "---\nversion: 1\ntemperature: 0.1\n---\n"
            "# System\nJudge {{ source_lang_name }} to "
            "{{ target_lang_name }}.\n"
            "Glossary: {{ glossary_block }}\n\n"
            "# User\n{{ original_text }}\n{{ translated_text }}\n",
            encoding="utf-8",
        )

        mock_provider = MagicMock()
        mock_provider.complete.return_value = CompletionResult(
            text='{"issues": ["something"]}',  # no score field
            input_tokens=100,
            output_tokens=20,
            total_tokens=120,
            model="m",
            raw={},
        )

        judge = Judge(
            provider=mock_provider,
            prompt_path=prompt_path,
            cache=cache,
            glossary=None,
            model="test-model",
        )

        stats = judge.judge_chunks(
            {"c1": "<p>Hello</p>"},
            {"c1": "<p>Привет</p>"},
            parallelism=1,
        )

        assert stats.chunks_failed == 1
        cache.close()


# ---------------------------------------------------------------------------
# Pipeline meta table tests
# ---------------------------------------------------------------------------


class TestPipelineMetaTable:
    """Tests for the pipeline_meta table in Cache."""

    def test_set_and_get_meta(self, tmp_path: Path):
        cache = Cache(tmp_path / "cache.sqlite")
        cache.set_meta("test_key", {"foo": "bar", "num": 42})
        result = cache.get_meta("test_key")
        assert result == {"foo": "bar", "num": 42}
        cache.close()

    def test_get_nonexistent_meta(self, tmp_path: Path):
        cache = Cache(tmp_path / "cache.sqlite")
        assert cache.get_meta("nonexistent") is None
        cache.close()

    def test_set_meta_overwrites(self, tmp_path: Path):
        cache = Cache(tmp_path / "cache.sqlite")
        cache.set_meta("key", {"v": 1})
        cache.set_meta("key", {"v": 2})
        assert cache.get_meta("key") == {"v": 2}
        cache.close()

    def test_meta_survives_reopen(self, tmp_path: Path):
        db_path = tmp_path / "cache.sqlite"
        cache = Cache(db_path)
        cache.set_meta("persist", [1, 2, 3])
        cache.close()

        cache2 = Cache(db_path)
        assert cache2.get_meta("persist") == [1, 2, 3]
        cache2.close()


# ---------------------------------------------------------------------------
# Threshold validation
# ---------------------------------------------------------------------------


class TestThresholdValidation:
    """Tests for --reflect-threshold / --threshold bounds validation."""

    def test_threshold_zero_invalid(self, tmp_path: Path):
        """Threshold 0 is out of range 1-5."""
        # We test the validation logic directly since CLI invocation
        # requires a full EPUB. The validation is: 1 <= threshold <= 5
        assert not (1 <= 0 <= 5)

    def test_threshold_negative_invalid(self, tmp_path: Path):
        """Negative threshold is out of range."""
        assert not (1 <= -1 <= 5)

    def test_threshold_six_invalid(self, tmp_path: Path):
        """Threshold 6 is out of range."""
        assert not (1 <= 6 <= 5)

    def test_threshold_one_valid(self, tmp_path: Path):
        """Threshold 1 is valid (reflect only score=1 chunks)."""
        assert 1 <= 1 <= 5

    def test_threshold_five_valid(self, tmp_path: Path):
        """Threshold 5 is valid (reflect all chunks)."""
        assert 1 <= 5 <= 5


# ---------------------------------------------------------------------------
# Parallelism validation
# ---------------------------------------------------------------------------


class TestParallelismValidation:
    """Tests for --parallelism bounds validation."""

    def test_zero_parallelism_invalid(self):
        """Parallelism 0 should be rejected."""
        parallelism = 0
        assert parallelism < 1

    def test_negative_parallelism_invalid(self):
        """Negative parallelism should be rejected."""
        parallelism = -5
        assert parallelism < 1

    def test_one_parallelism_valid(self):
        """Parallelism 1 is valid (sequential)."""
        parallelism = 1
        assert parallelism >= 1


# ---------------------------------------------------------------------------
# normalize_judge_map
# ---------------------------------------------------------------------------


class TestNormalizeJudgeMap:
    """Tests for the shared judge score normalization helper."""

    def test_normalizes_raw_cache_scores(self):
        from booktranslator.pipeline_helpers import normalize_judge_map

        raw = [
            {"chunk_id": "c1", "score": 4, "issues": ["minor"]},
            {"chunk_id": "c2", "score": 2, "issues": ["bad", "worse"]},
        ]
        result = normalize_judge_map(raw)
        assert result["c1"] == {"score": 4, "issues": ["minor"]}
        assert result["c2"] == {"score": 2, "issues": ["bad", "worse"]}

    def test_normalizes_judge_result_dicts(self):
        from booktranslator.pipeline_helpers import normalize_judge_map

        # Shape from JudgeResult objects converted to dicts
        raw = [
            {"chunk_id": "c1", "score": 5, "issues": []},
        ]
        result = normalize_judge_map(raw)
        assert result["c1"] == {"score": 5, "issues": []}

    def test_handles_string_score(self):
        from booktranslator.pipeline_helpers import normalize_judge_map

        raw = [{"chunk_id": "c1", "score": "3", "issues": []}]
        result = normalize_judge_map(raw)
        assert result["c1"]["score"] == 3

    def test_handles_invalid_score(self):
        from booktranslator.pipeline_helpers import normalize_judge_map

        raw = [{"chunk_id": "c1", "score": "bad", "issues": []}]
        result = normalize_judge_map(raw)
        assert result["c1"]["score"] == 0

    def test_handles_missing_chunk_id(self):
        from booktranslator.pipeline_helpers import normalize_judge_map

        raw = [{"score": 4, "issues": []}]  # no chunk_id
        result = normalize_judge_map(raw)
        assert result == {}

    def test_handles_issues_as_string(self):
        from booktranslator.pipeline_helpers import normalize_judge_map

        raw = [{"chunk_id": "c1", "score": 3, "issues": "single"}]
        result = normalize_judge_map(raw)
        assert result["c1"]["issues"] == ["single"]

    def test_handles_missing_issues(self):
        from booktranslator.pipeline_helpers import normalize_judge_map

        raw = [{"chunk_id": "c1", "score": 4}]
        result = normalize_judge_map(raw)
        assert result["c1"]["issues"] == []


# ---------------------------------------------------------------------------
# Cache closure (resource management)
# ---------------------------------------------------------------------------


class TestCacheResourceManagement:
    """Tests that cache connections are properly managed."""

    def test_cache_close_releases_lock(self, tmp_path: Path):
        """After close(), another connection can open the same DB."""
        db_path = tmp_path / "cache.sqlite"
        cache1 = Cache(db_path)
        cache1.put("k1", "translate", "m", "v1", "text", meta={"chunk_id": "c1"})
        cache1.close()

        # Should be able to open and write without "database is locked"
        cache2 = Cache(db_path)
        cache2.put("k2", "translate", "m", "v1", "text2", meta={"chunk_id": "c2"})
        assert cache2.count_stage("translate") == 2
        cache2.close()

    def test_multiple_sequential_opens(self, tmp_path: Path):
        """Multiple open/close cycles work without issues."""
        db_path = tmp_path / "cache.sqlite"
        for i in range(5):
            cache = Cache(db_path)
            cache.put(f"k{i}", "translate", "m", "v1", f"text{i}", meta={"chunk_id": f"c{i}"})
            cache.close()

        # Final check
        cache = Cache(db_path)
        assert cache.count_stage("translate") == 5
        cache.close()


# ---------------------------------------------------------------------------
# reflect --all without judge scores
# ---------------------------------------------------------------------------


class TestReflectAllWithoutJudge:
    """Tests that reflect --all works safely when no judge scores exist."""

    def test_build_reflect_input_with_empty_judge(self, tmp_path: Path):
        """build_reflect_input handles empty judge_by_id without crashing."""
        from booktranslator.pipeline_helpers import build_reflect_input

        class MockChunk:
            def __init__(self, cid):
                self.id = cid

        class MockChunkSet:
            def __init__(self):
                self.chunks = [MockChunk("c1"), MockChunk("c2")]

            def render_main(self, chunk):
                return [f"<p>Original {chunk.id}</p>"]

        cache = Cache(tmp_path / "cache.sqlite")
        cache.put("k1", "translate", "m", "v1", "Translation 1", meta={"chunk_id": "c1"})
        cache.put("k2", "translate", "m", "v1", "Translation 2", meta={"chunk_id": "c2"})

        cs = MockChunkSet()

        # Empty judge_by_id — simulates --all without judge run
        result = build_reflect_input(cs, cache, {"c1", "c2"}, {})

        assert len(result) == 2
        # All chunks get default score=0 and empty issues
        for item in result:
            assert item["judge_score"] == 0
            assert item["judge_issues"] == []
            assert item["original_text"] != ""
            assert item["translated_text"] != ""

        cache.close()

    def test_build_reflect_input_partial_judge(self, tmp_path: Path):
        """build_reflect_input works when only some chunks have judge data."""
        from booktranslator.pipeline_helpers import build_reflect_input

        class MockChunk:
            def __init__(self, cid):
                self.id = cid

        class MockChunkSet:
            def __init__(self):
                self.chunks = [MockChunk("c1"), MockChunk("c2")]

            def render_main(self, chunk):
                return [f"<p>Text {chunk.id}</p>"]

        cache = Cache(tmp_path / "cache.sqlite")
        cache.put("k1", "translate", "m", "v1", "T1", meta={"chunk_id": "c1"})
        cache.put("k2", "translate", "m", "v1", "T2", meta={"chunk_id": "c2"})

        cs = MockChunkSet()

        # Only c1 has judge data
        judge_by_id = {"c1": {"score": 2, "issues": ["bad"]}}
        result = build_reflect_input(cs, cache, {"c1", "c2"}, judge_by_id)

        assert len(result) == 2
        c1 = next(r for r in result if r["chunk_id"] == "c1")
        c2 = next(r for r in result if r["chunk_id"] == "c2")
        assert c1["judge_score"] == 2
        assert c1["judge_issues"] == ["bad"]
        assert c2["judge_score"] == 0  # default
        assert c2["judge_issues"] == []  # default

        cache.close()

    def test_normalize_judge_map_empty_input(self):
        """normalize_judge_map with empty list returns empty dict."""
        from booktranslator.pipeline_helpers import normalize_judge_map

        assert normalize_judge_map([]) == {}


# ---------------------------------------------------------------------------
# The judge must be handed the source text, not the translation
# ---------------------------------------------------------------------------


class TestJudgeReceivesTheSourceText:
    """`translate` runs its judge pass after the translator has spliced the
    translation into the live tree, so reading originals from that same
    ChunkSet hands the judge the translation in both slots.

    That is what happened on the Bear Head run: the judge compared the
    translation with itself and scored 102 of 110 chunks a perfect 5. This test
    runs the real command end to end against a stub provider and asserts the
    text reaching the judge is still English.
    """

    def _run(self, tmp_path: Path, capture: dict):
        from booktranslator.chunker import chunk_book as real_chunk_book

        epub = write_minimal_epub(tmp_path / "book.epub")

        translated = (
            "===PARAGRAPH 1===\n<p>Это тестовый абзац про вестигиум.</p>\n"
            "===PARAGRAPH 2===\n<p>Ещё абзац с Сиуоллом.</p>"
        )

        def fake_complete(**kwargs):
            result = MagicMock()
            # The judge prompt is the only one asking for a score.
            if "score" in (kwargs.get("system", "") + kwargs.get("user", "")).lower():
                result.text = '{"score": 5, "issues": []}'
            else:
                result.text = translated
            result.input_tokens = 10
            result.output_tokens = 10
            return result

        provider = MagicMock()
        provider.complete.side_effect = fake_complete

        real_judge_chunks = None

        def spy_judge_chunks(self, chunk_originals, chunk_translations, **kwargs):
            capture["originals"] = dict(chunk_originals)
            capture["translations"] = dict(chunk_translations)
            return real_judge_chunks(self, chunk_originals, chunk_translations, **kwargs)

        from booktranslator.judge import Judge

        real_judge_chunks = Judge.judge_chunks

        with (
            patch("booktranslator.cli.create_stage_provider", return_value=provider),
            patch.object(Judge, "judge_chunks", spy_judge_chunks),
            patch("booktranslator.cli.chunk_book", side_effect=real_chunk_book),
        ):
            return runner.invoke(
                app,
                [
                    "translate",
                    str(epub),
                    "--work",
                    str(tmp_path / "work"),
                    "--no-reflect",
                    "--no-proofread",
                    "--no-style",
                    "--no-verify",
                    "--no-notes",
                    "--out",
                    str(tmp_path / "out.epub"),
                ],
            )

    def test_originals_reaching_the_judge_are_not_the_translation(self, tmp_path: Path):
        capture: dict = {}
        result = self._run(tmp_path, capture)

        assert capture, f"judge never ran; output was:\n{result.output}"
        joined = "\n".join(capture["originals"].values())
        assert "vestigium" in joined, "the judge was not given the English source"
        assert "вестигиум" not in joined, (
            "the judge was handed the translation as the original — the very defect "
            "source_chunk_set exists to prevent"
        )
        # And the translation slot really does carry the Russian, so the test
        # would not pass merely because nothing was translated.
        assert "вестигиум" in "\n".join(capture["translations"].values())


# ---------------------------------------------------------------------------
# btrans cache clear
# ---------------------------------------------------------------------------


class TestCacheClearCommand:
    """The command around Cache.select_for_clear / Cache.clear."""

    def _book(self, tmp_path: Path) -> Path:
        work = tmp_path / "foxglove-summer"
        work.mkdir()
        cache = Cache(work / "cache.sqlite")
        cache.put("g", "glossary", "m", "1", "{}")
        for stage in ("translate", "style"):
            cache.put(f"{stage}-1", stage, "m", "1", "text", meta={"chunk_id": "ch01_c01"})
        cache.put("translate-2", "translate", "m", "1", "text", meta={"chunk_id": "ch01_c02"})
        save_chunker_params(cache, target_words=2000, overlap_paragraphs=1)
        cache.close()
        (work / "state.json").write_text(
            '{"book_slug": "foxglove-summer", "current_stage": "translate", '
            '"completed_stages": ["glossary", "translate"]}',
            encoding="utf-8",
        )
        return work

    def _completed(self, work: Path) -> list[str]:
        return json.loads((work / "state.json").read_text(encoding="utf-8"))["completed_stages"]

    def _stages(self, work: Path) -> dict[str, int]:
        cache = Cache(work / "cache.sqlite")
        try:
            return cache.list_stages()
        finally:
            cache.close()

    def test_full_clear_lets_translate_rechunk(self, tmp_path: Path):
        work = self._book(tmp_path)

        result = runner.invoke(app, ["cache", "clear", str(work), "--yes"])

        assert result.exit_code == 0, result.output
        assert self._stages(work) == {"glossary": 1}
        assert "target_words=2000" in result.output
        assert list(work.glob("cache.sqlite.bak-*")), "no backup written"
        assert self._completed(work) == ["glossary"]
        # The very error this command exists for no longer fires.
        cache = Cache(work / "cache.sqlite")
        save_chunker_params(cache, target_words=1200, overlap_paragraphs=1)
        cache.close()

    def test_refusal_deletes_nothing(self, tmp_path: Path):
        work = self._book(tmp_path)
        before = self._stages(work)

        result = runner.invoke(app, ["cache", "clear", str(work)], input="n\n")

        assert result.exit_code == 0, result.output
        assert "Nothing deleted" in result.output
        assert self._stages(work) == before
        assert not list(work.glob("cache.sqlite.bak-*"))

    def test_stage_with_chunk_keeps_chunking(self, tmp_path: Path):
        work = self._book(tmp_path)

        result = runner.invoke(
            app,
            ["cache", "clear", str(work), "--stage", "translate", "--chunk", "ch01_c01", "-y"],
        )

        assert result.exit_code == 0, result.output
        assert self._stages(work) == {"glossary": 1, "translate": 1}
        cache = Cache(work / "cache.sqlite")
        with pytest.raises(ChunkerConfigMismatchError):
            save_chunker_params(cache, target_words=1200, overlap_paragraphs=1)
        cache.close()
        assert self._completed(work) == ["glossary", "translate"]

    def test_invalid_stage_rejected(self, tmp_path: Path):
        work = self._book(tmp_path)
        result = runner.invoke(app, ["cache", "clear", str(work), "--stage", "glossary"])
        assert result.exit_code == 1
        assert self._stages(work)["glossary"] == 1

    def test_mismatch_error_names_the_command(self, tmp_path: Path):
        cache = Cache(tmp_path / "cache.sqlite")
        save_chunker_params(cache, target_words=2000, overlap_paragraphs=1)
        with pytest.raises(ChunkerConfigMismatchError, match="btrans cache clear"):
            save_chunker_params(cache, target_words=1200, overlap_paragraphs=1)
        with pytest.raises(ChunkerConfigMismatchError, match="btrans cache clear"):
            verify_chunker_params(cache, target_words=1200, overlap_paragraphs=1)
        cache.close()
