"""Tests for judge.py: scoring logic, response parsing, caching."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from booktranslator.cache import Cache
from booktranslator.judge import Judge, JudgeStats
from booktranslator.provider import CompletionResult


@pytest.fixture
def cache(tmp_path: Path) -> Cache:
    return Cache(tmp_path / "test.sqlite")


@pytest.fixture
def mock_provider():
    return MagicMock()


@pytest.fixture
def judge_prompt_path(tmp_path: Path) -> Path:
    prompt = tmp_path / "judge.md"
    prompt.write_text(
        "---\n"
        "version: 1\n"
        "model: anthropic/claude-haiku-4.5\n"
        "temperature: 0.1\n"
        "max_tokens: 2000\n"
        "---\n"
        "# System\n"
        "You are a judge. Score {{ source_lang_name }} to {{ target_lang_name }} translation.\n"
        "Glossary: {{ glossary_block }}\n\n"
        "# User\n"
        "Original: {{ original_text }}\n"
        "Translation: {{ translated_text }}\n",
        encoding="utf-8",
    )
    return prompt


@pytest.fixture
def judge(mock_provider, judge_prompt_path, cache) -> Judge:
    return Judge(
        provider=mock_provider,
        prompt_path=judge_prompt_path,
        cache=cache,
        glossary=None,
        model="anthropic/claude-haiku-4.5",
        source_lang="en",
        target_lang="ru",
    )


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


class TestParseJudgeResponse:
    def test_valid_json(self, judge: Judge):
        text = '{"score": 4, "issues": ["naturalness: slight calque"]}'
        score, issues = judge._parse_judge_response(text)
        assert score == 4
        assert issues == ["naturalness: slight calque"]

    def test_perfect_score(self, judge: Judge):
        text = '{"score": 5, "issues": []}'
        score, issues = judge._parse_judge_response(text)
        assert score == 5
        assert issues == []

    def test_with_code_fences(self, judge: Judge):
        text = '```json\n{"score": 3, "issues": ["accuracy: omission"]}\n```'
        score, issues = judge._parse_judge_response(text)
        assert score == 3
        assert issues == ["accuracy: omission"]

    def test_score_clamped_to_range(self, judge: Judge):
        text = '{"score": 7, "issues": []}'
        with pytest.raises(ValueError, match="out of range"):
            judge._parse_judge_response(text)

    def test_score_clamped_below(self, judge: Judge):
        text = '{"score": 0, "issues": []}'
        with pytest.raises(ValueError, match="out of range"):
            judge._parse_judge_response(text)

    def test_invalid_json_raises(self, judge: Judge):
        with pytest.raises(ValueError, match="not valid JSON"):
            judge._parse_judge_response("This is not JSON at all")

    def test_issues_as_string(self, judge: Judge):
        text = '{"score": 2, "issues": "single issue"}'
        score, issues = judge._parse_judge_response(text)
        assert score == 2
        assert issues == ["single issue"]

    def test_issues_missing(self, judge: Judge):
        text = '{"score": 4}'
        score, issues = judge._parse_judge_response(text)
        assert score == 4
        assert issues == []

    def test_whitespace_handling(self, judge: Judge):
        text = '  \n  {"score": 5, "issues": []}  \n  '
        score, issues = judge._parse_judge_response(text)
        assert score == 5
        assert issues == []

    def test_missing_score_raises(self, judge: Judge):
        text = '{"issues": ["something"]}'
        with pytest.raises(ValueError, match="missing required"):
            judge._parse_judge_response(text)

    def test_non_integer_score_raises(self, judge: Judge):
        text = '{"score": "high", "issues": []}'
        with pytest.raises(ValueError, match="not a valid integer"):
            judge._parse_judge_response(text)

    def test_null_score_raises(self, judge: Judge):
        text = '{"score": null, "issues": []}'
        with pytest.raises(ValueError, match="missing required"):
            judge._parse_judge_response(text)


# ---------------------------------------------------------------------------
# Judge chunk (with mocked provider)
# ---------------------------------------------------------------------------


class TestJudgeChunk:
    def test_calls_provider_and_caches(self, judge: Judge, mock_provider, cache: Cache):
        mock_provider.complete.return_value = CompletionResult(
            text='{"score": 4, "issues": ["style: minor"]}',
            input_tokens=500,
            output_tokens=50,
            total_tokens=550,
            model="haiku",
            raw={},
        )

        stats = JudgeStats()
        result = judge.judge_chunk(
            "ch01_c01",
            "<p>Hello world</p>",
            "<p>Привет мир</p>",
            stats,
        )

        assert result is not None
        assert result.chunk_id == "ch01_c01"
        assert result.score == 4
        assert result.issues == ["style: minor"]
        assert stats.chunks_judged == 1
        assert stats.input_tokens == 500
        assert stats.output_tokens == 50
        mock_provider.complete.assert_called_once()

    def test_uses_cache_on_second_call(self, judge: Judge, mock_provider, cache: Cache):
        mock_provider.complete.return_value = CompletionResult(
            text='{"score": 5, "issues": []}',
            input_tokens=500,
            output_tokens=50,
            total_tokens=550,
            model="haiku",
            raw={},
        )

        stats1 = JudgeStats()
        judge.judge_chunk("ch01_c01", "<p>Hello</p>", "<p>Привет</p>", stats1)
        assert stats1.chunks_judged == 1

        # Second call should use cache
        stats2 = JudgeStats()
        judge.judge_chunk("ch01_c01", "<p>Hello</p>", "<p>Привет</p>", stats2)
        assert stats2.chunks_cached == 1
        assert stats2.chunks_judged == 0
        # Provider called only once total
        assert mock_provider.complete.call_count == 1

    def test_stores_in_cache_with_correct_stage(self, judge: Judge, mock_provider, cache: Cache):
        mock_provider.complete.return_value = CompletionResult(
            text='{"score": 3, "issues": ["accuracy: omission"]}',
            input_tokens=400,
            output_tokens=40,
            total_tokens=440,
            model="haiku",
            raw={},
        )

        stats = JudgeStats()
        judge.judge_chunk("ch02_c01", "<p>Test</p>", "<p>Тест</p>", stats)

        # Verify it's in the cache with stage=judge
        stages = cache.get_chunk_stages("ch02_c01")
        judge_stages = [s for s in stages if s.stage == "judge"]
        assert len(judge_stages) == 1
        assert "score" in judge_stages[0].content


# ---------------------------------------------------------------------------
# Judge multiple chunks
# ---------------------------------------------------------------------------


class TestJudgeChunks:
    def test_judges_all_chunks(self, judge: Judge, mock_provider):
        mock_provider.complete.return_value = CompletionResult(
            text='{"score": 4, "issues": []}',
            input_tokens=500,
            output_tokens=50,
            total_tokens=550,
            model="haiku",
            raw={},
        )

        originals = {
            "ch01_c01": "<p>First</p>",
            "ch01_c02": "<p>Second</p>",
            "ch01_c03": "<p>Third</p>",
        }
        translations = {
            "ch01_c01": "<p>Первый</p>",
            "ch01_c02": "<p>Второй</p>",
            "ch01_c03": "<p>Третий</p>",
        }

        stats = judge.judge_chunks(originals, translations, parallelism=1)

        assert stats.chunks_total == 3
        assert stats.chunks_judged == 3
        assert len(stats.results) == 3
        assert all(r.score == 4 for r in stats.results)

    def test_handles_provider_failure(self, judge: Judge, mock_provider):
        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 2:
                raise RuntimeError("API error")
            return CompletionResult(
                text='{"score": 5, "issues": []}',
                input_tokens=500,
                output_tokens=50,
                total_tokens=550,
                model="haiku",
                raw={},
            )

        mock_provider.complete.side_effect = side_effect

        originals = {"c1": "<p>A</p>", "c2": "<p>B</p>", "c3": "<p>C</p>"}
        translations = {"c1": "<p>А</p>", "c2": "<p>Б</p>", "c3": "<p>В</p>"}

        stats = judge.judge_chunks(originals, translations, parallelism=1)

        assert stats.chunks_failed == 1
        assert stats.chunks_judged == 2

    def test_progress_callback(self, judge: Judge, mock_provider):
        mock_provider.complete.return_value = CompletionResult(
            text='{"score": 4, "issues": []}',
            input_tokens=100,
            output_tokens=20,
            total_tokens=120,
            model="haiku",
            raw={},
        )

        progress_calls = []

        def on_progress(done, total, cid, stats):
            progress_calls.append((done, total, cid))

        originals = {"c1": "<p>A</p>", "c2": "<p>B</p>"}
        translations = {"c1": "<p>А</p>", "c2": "<p>Б</p>"}

        judge.judge_chunks(
            originals, translations,
            on_progress=on_progress,
            parallelism=1,
        )

        assert len(progress_calls) == 2
        assert progress_calls[-1][0] == 2  # done count
        assert progress_calls[-1][1] == 2  # total

    def test_parallel_execution(self, judge: Judge, mock_provider):
        """Parallel judge produces same results as sequential."""
        mock_provider.complete.return_value = CompletionResult(
            text='{"score": 3, "issues": ["test"]}',
            input_tokens=100,
            output_tokens=20,
            total_tokens=120,
            model="haiku",
            raw={},
        )

        originals = {f"c{i}": f"<p>Text {i}</p>" for i in range(10)}
        translations = {f"c{i}": f"<p>Текст {i}</p>" for i in range(10)}

        stats = judge.judge_chunks(originals, translations, parallelism=4)

        assert stats.chunks_total == 10
        assert stats.chunks_judged == 10
        assert len(stats.results) == 10


# ---------------------------------------------------------------------------
# Integration with cache scoring
# ---------------------------------------------------------------------------


class TestJudgeCacheIntegration:
    def test_judge_results_visible_in_cache(self, judge: Judge, mock_provider, cache: Cache):
        """Judge results stored in cache are retrievable via get_judge_scores."""
        mock_provider.complete.return_value = CompletionResult(
            text='{"score": 2, "issues": ["accuracy: omission", "glossary: wrong term"]}',
            input_tokens=500,
            output_tokens=50,
            total_tokens=550,
            model="haiku",
            raw={},
        )

        stats = JudgeStats()
        judge.judge_chunk("ch01_c01", "<p>Hello</p>", "<p>Привет</p>", stats)

        scores = cache.get_judge_scores()
        assert len(scores) == 1
        assert scores[0]["chunk_id"] == "ch01_c01"
        assert scores[0]["score"] == 2
        assert len(scores[0]["issues"]) == 2

    def test_judge_does_not_affect_waterfall(self, judge: Judge, mock_provider, cache: Cache):
        """Judge stage is metadata — it should not appear in waterfall resolution."""
        # Put a translate entry
        cache.put("t1", "translate", "m", "v1", "translated text",
                  meta={"chunk_id": "ch01_c01"})

        # Judge it
        mock_provider.complete.return_value = CompletionResult(
            text='{"score": 4, "issues": []}',
            input_tokens=500,
            output_tokens=50,
            total_tokens=550,
            model="haiku",
            raw={},
        )
        stats = JudgeStats()
        judge.judge_chunk("ch01_c01", "<p>Hello</p>", "<p>Привет</p>", stats)

        # Waterfall should still resolve to translate (judge is not in waterfall)
        resolved = cache.resolve_stage_for_chunk("ch01_c01")
        assert resolved == "translate"
