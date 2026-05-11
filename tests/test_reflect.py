"""Tests for reflect.py: reflection logic, two-step process, caching."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from booktranslator.cache import Cache
from booktranslator.provider import CompletionResult
from booktranslator.reflect import Reflector, ReflectStats


@pytest.fixture
def cache(tmp_path: Path) -> Cache:
    return Cache(tmp_path / "test.sqlite")


@pytest.fixture
def mock_provider():
    return MagicMock()


@pytest.fixture
def reflect_prompt_path(tmp_path: Path) -> Path:
    prompt = tmp_path / "reflect.md"
    prompt.write_text(
        "---\n"
        "version: 1\n"
        "model: anthropic/claude-sonnet-4.6\n"
        "temperature: 0.4\n"
        "max_tokens: 4000\n"
        "---\n"
        "# System\n"
        "You are a critic. Analyze {{ source_lang_name }} to {{ target_lang_name }} translation.\n"
        "Glossary: {{ glossary_block }}\n\n"
        "# User\n"
        "Original: {{ original_text }}\n"
        "Translation: {{ translated_text }}\n"
        "Score: {{ judge_score }}\n"
        "Issues: {{ judge_issues }}\n",
        encoding="utf-8",
    )
    return prompt


@pytest.fixture
def translate_prompt_path(tmp_path: Path) -> Path:
    prompt = tmp_path / "translate.md"
    prompt.write_text(
        "---\n"
        "version: 1\n"
        "model: anthropic/claude-sonnet-4.6\n"
        "temperature: 0.3\n"
        "max_tokens: 12000\n"
        "---\n"
        "# System\n"
        "Translate {{ source_lang_name }} to {{ target_lang_name }}.\n"
        "Glossary: {{ glossary_block }}\n\n"
        "# User\n"
        "{% for fragment in main_fragments %}\n"
        "===PARAGRAPH {{ loop.index }}===\n"
        "{{ fragment }}\n"
        "{% endfor %}\n"
        "{{ prev_overlap }}\n"
        "{{ next_overlap }}\n"
        "{% if reflection_notes %}\n"
        "Reflection: {{ reflection_notes }}\n"
        "{% endif %}\n",
        encoding="utf-8",
    )
    return prompt


@pytest.fixture
def reflector(mock_provider, reflect_prompt_path, translate_prompt_path, cache) -> Reflector:
    return Reflector(
        provider=mock_provider,
        reflect_prompt_path=reflect_prompt_path,
        translate_prompt_path=translate_prompt_path,
        cache=cache,
        glossary=None,
        model="anthropic/claude-sonnet-4.6",
        source_lang="en",
        target_lang="ru",
    )


# ---------------------------------------------------------------------------
# Reflection notes (step 1)
# ---------------------------------------------------------------------------


class TestGetReflectionNotes:
    def test_calls_provider_for_critique(self, reflector: Reflector, mock_provider):
        critique_json = json.dumps({
            "notes": [
                {
                    "paragraph": 1,
                    "category": "naturalness",
                    "original_fragment": "He was not happy",
                    "current_translation": "Он был не счастлив",
                    "suggestion": "Он был несчастен",
                    "explanation": "More natural Russian phrasing",
                }
            ],
            "general_notes": "",
        })
        mock_provider.complete.return_value = CompletionResult(
            text=critique_json,
            input_tokens=1000,
            output_tokens=200,
            total_tokens=1200,
            model="sonnet",
            raw={},
        )

        stats = ReflectStats()
        notes = reflector._get_reflection_notes(
            "ch01_c01",
            "<p>He was not happy.</p>",
            "<p>Он был не счастлив.</p>",
            2,
            ["naturalness: calque"],
            stats,
        )

        assert "notes" in notes
        assert stats.input_tokens == 1000
        assert stats.output_tokens == 200
        mock_provider.complete.assert_called_once()

    def test_caches_reflection_notes(self, reflector: Reflector, mock_provider, cache: Cache):
        mock_provider.complete.return_value = CompletionResult(
            text='{"notes": [], "general_notes": ""}',
            input_tokens=500,
            output_tokens=100,
            total_tokens=600,
            model="sonnet",
            raw={},
        )

        stats = ReflectStats()
        reflector._get_reflection_notes(
            "ch01_c01", "original", "translated", 3, [], stats
        )

        # Second call should use cache
        stats2 = ReflectStats()
        result = reflector._get_reflection_notes(
            "ch01_c01", "original", "translated", 3, [], stats2
        )

        # Provider called only once
        assert mock_provider.complete.call_count == 1
        assert result == '{"notes": [], "general_notes": ""}'


# ---------------------------------------------------------------------------
# Re-translation (step 2)
# ---------------------------------------------------------------------------


class TestRetranslateWithNotes:
    def test_calls_provider_with_notes(self, reflector: Reflector, mock_provider):
        mock_provider.complete.return_value = CompletionResult(
            text="===PARAGRAPH 1===\n<p>Улучшенный перевод</p>",
            input_tokens=2000,
            output_tokens=500,
            total_tokens=2500,
            model="sonnet",
            raw={},
        )

        stats = ReflectStats()
        result = reflector._retranslate_with_notes(
            "ch01_c01",
            ["<p>Original paragraph</p>"],
            '{"notes": [{"suggestion": "improve"}]}',
            stats,
        )

        assert "Улучшенный перевод" in result
        assert stats.chunks_reflected == 1
        assert stats.input_tokens == 2000

    def test_caches_retranslation(self, reflector: Reflector, mock_provider, cache: Cache):
        mock_provider.complete.return_value = CompletionResult(
            text="===PARAGRAPH 1===\n<p>Перевод</p>",
            input_tokens=1500,
            output_tokens=300,
            total_tokens=1800,
            model="sonnet",
            raw={},
        )

        stats = ReflectStats()
        reflector._retranslate_with_notes(
            "ch01_c01", ["<p>Hello</p>"], "notes", stats
        )

        # Second call uses cache
        stats2 = ReflectStats()
        reflector._retranslate_with_notes(
            "ch01_c01", ["<p>Hello</p>"], "notes", stats2
        )

        assert mock_provider.complete.call_count == 1
        assert stats2.chunks_cached == 1

    def test_stores_as_reflect_stage(self, reflector: Reflector, mock_provider, cache: Cache):
        mock_provider.complete.return_value = CompletionResult(
            text="===PARAGRAPH 1===\n<p>Перевод</p>",
            input_tokens=1500,
            output_tokens=300,
            total_tokens=1800,
            model="sonnet",
            raw={},
        )

        stats = ReflectStats()
        reflector._retranslate_with_notes(
            "ch01_c01", ["<p>Hello</p>"], "notes", stats
        )

        stages = cache.get_chunk_stages("ch01_c01")
        reflect_stages = [s for s in stages if s.stage == "reflect"]
        assert len(reflect_stages) == 1


# ---------------------------------------------------------------------------
# Full reflect_chunk (both steps)
# ---------------------------------------------------------------------------


class TestReflectChunk:
    def test_two_step_process(self, reflector: Reflector, mock_provider):
        """reflect_chunk calls provider twice: once for critique, once for retranslation."""
        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                # First call: reflection notes
                return CompletionResult(
                    text='{"notes": [{"suggestion": "fix calque"}], "general_notes": ""}',
                    input_tokens=1000,
                    output_tokens=200,
                    total_tokens=1200,
                    model="sonnet",
                    raw={},
                )
            else:
                # Second call: retranslation
                return CompletionResult(
                    text="===PARAGRAPH 1===\n<p>Улучшенный текст</p>",
                    input_tokens=1500,
                    output_tokens=300,
                    total_tokens=1800,
                    model="sonnet",
                    raw={},
                )

        mock_provider.complete.side_effect = side_effect

        stats = ReflectStats()
        result = reflector.reflect_chunk(
            chunk_id="ch01_c01",
            original_text="<p>Original text</p>",
            original_paragraphs=["<p>Original text</p>"],
            translated_text="<p>Плохой перевод</p>",
            judge_score=2,
            judge_issues=["naturalness: calque"],
            stats=stats,
        )

        assert result.chunk_id == "ch01_c01"
        assert result.improved is True
        assert "fix calque" in result.reflection_notes
        assert "Улучшенный текст" in result.new_translation
        assert mock_provider.complete.call_count == 2

    def test_unchanged_when_same_output(self, reflector: Reflector, mock_provider):
        """If retranslation is identical to original, improved=False."""
        original_translation = "===PARAGRAPH 1===\n<p>Тот же текст</p>"

        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return CompletionResult(
                    text='{"notes": [], "general_notes": "Translation is fine."}',
                    input_tokens=1000,
                    output_tokens=100,
                    total_tokens=1100,
                    model="sonnet",
                    raw={},
                )
            else:
                # Returns same text as input
                return CompletionResult(
                    text=original_translation,
                    input_tokens=1500,
                    output_tokens=300,
                    total_tokens=1800,
                    model="sonnet",
                    raw={},
                )

        mock_provider.complete.side_effect = side_effect

        stats = ReflectStats()
        result = reflector.reflect_chunk(
            chunk_id="ch01_c01",
            original_text="<p>Text</p>",
            original_paragraphs=["<p>Text</p>"],
            translated_text=original_translation,
            judge_score=3,
            judge_issues=[],
            stats=stats,
        )

        assert result.improved is False
        assert stats.chunks_unchanged == 1


# ---------------------------------------------------------------------------
# Reflect multiple chunks
# ---------------------------------------------------------------------------


class TestReflectChunks:
    def test_reflects_all_chunks(self, reflector: Reflector, mock_provider):
        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] % 2 == 1:
                return CompletionResult(
                    text='{"notes": [{"suggestion": "fix"}], "general_notes": ""}',
                    input_tokens=1000,
                    output_tokens=200,
                    total_tokens=1200,
                    model="sonnet",
                    raw={},
                )
            else:
                return CompletionResult(
                    text="===PARAGRAPH 1===\n<p>Improved</p>",
                    input_tokens=1500,
                    output_tokens=300,
                    total_tokens=1800,
                    model="sonnet",
                    raw={},
                )

        mock_provider.complete.side_effect = side_effect

        chunks_data = [
            {
                "chunk_id": f"ch01_c{i:02d}",
                "original_text": f"<p>Original {i}</p>",
                "original_paragraphs": [f"<p>Original {i}</p>"],
                "translated_text": f"<p>Перевод {i}</p>",
                "judge_score": 2,
                "judge_issues": ["accuracy: omission"],
            }
            for i in range(1, 4)
        ]

        stats = reflector.reflect_chunks(chunks_data, parallelism=1)

        assert stats.chunks_total == 3
        assert stats.chunks_improved == 3
        assert stats.chunks_failed == 0

    def test_handles_failure_gracefully(self, reflector: Reflector, mock_provider):
        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] <= 2:
                # First chunk succeeds (2 calls)
                if call_count[0] == 1:
                    return CompletionResult(
                        text='{"notes": [], "general_notes": ""}',
                        input_tokens=500, output_tokens=50,
                        total_tokens=550, model="s", raw={},
                    )
                return CompletionResult(
                    text="===PARAGRAPH 1===\n<p>OK</p>",
                    input_tokens=500, output_tokens=50,
                    total_tokens=550, model="s", raw={},
                )
            # Second chunk fails on first call
            raise RuntimeError("API timeout")

        mock_provider.complete.side_effect = side_effect

        chunks_data = [
            {
                "chunk_id": "c1",
                "original_text": "<p>A</p>",
                "original_paragraphs": ["<p>A</p>"],
                "translated_text": "<p>А</p>",
                "judge_score": 2,
                "judge_issues": [],
            },
            {
                "chunk_id": "c2",
                "original_text": "<p>B</p>",
                "original_paragraphs": ["<p>B</p>"],
                "translated_text": "<p>Б</p>",
                "judge_score": 1,
                "judge_issues": [],
            },
        ]

        stats = reflector.reflect_chunks(chunks_data, parallelism=1)

        assert stats.chunks_failed == 1
        assert len(stats.results) == 1  # Only successful one

    def test_progress_callback(self, reflector: Reflector, mock_provider):
        mock_provider.complete.return_value = CompletionResult(
            text='{"notes": [], "general_notes": ""}',
            input_tokens=500,
            output_tokens=50,
            total_tokens=550,
            model="s",
            raw={},
        )

        # Override retranslate to also return something
        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] % 2 == 1:
                return CompletionResult(
                    text='{"notes": [], "general_notes": ""}',
                    input_tokens=500, output_tokens=50,
                    total_tokens=550, model="s", raw={},
                )
            return CompletionResult(
                text="===PARAGRAPH 1===\n<p>Done</p>",
                input_tokens=500, output_tokens=50,
                total_tokens=550, model="s", raw={},
            )

        mock_provider.complete.side_effect = side_effect

        progress_calls = []

        def on_progress(done, total, cid, stats):
            progress_calls.append((done, total, cid))

        chunks_data = [
            {
                "chunk_id": "c1",
                "original_text": "<p>A</p>",
                "original_paragraphs": ["<p>A</p>"],
                "translated_text": "<p>А</p>",
                "judge_score": 2,
                "judge_issues": [],
            },
        ]

        reflector.reflect_chunks(
            chunks_data, on_progress=on_progress, parallelism=1
        )

        assert len(progress_calls) == 1
        assert progress_calls[0] == (1, 1, "c1")


# ---------------------------------------------------------------------------
# Cache integration
# ---------------------------------------------------------------------------


class TestReflectCacheIntegration:
    def test_reflect_stage_in_waterfall(self, reflector: Reflector, mock_provider, cache: Cache):
        """Reflect results participate in waterfall resolution."""
        # Put a translate entry
        cache.put("t1", "translate", "m", "v1", "original translation",
                  meta={"chunk_id": "ch01_c01"})

        # Reflect it
        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return CompletionResult(
                    text='{"notes": [{"suggestion": "fix"}], "general_notes": ""}',
                    input_tokens=1000, output_tokens=200,
                    total_tokens=1200, model="s", raw={},
                )
            return CompletionResult(
                text="===PARAGRAPH 1===\n<p>Better</p>",
                input_tokens=1500, output_tokens=300,
                total_tokens=1800, model="s", raw={},
            )

        mock_provider.complete.side_effect = side_effect

        stats = ReflectStats()
        reflector.reflect_chunk(
            "ch01_c01", "<p>Orig</p>", ["<p>Orig</p>"],
            "original translation", 2, ["issue"], stats,
        )

        # Waterfall should now resolve to reflect
        resolved = cache.resolve_stage_for_chunk("ch01_c01")
        assert resolved == "reflect"

    def test_original_translate_preserved(self, reflector: Reflector, mock_provider, cache: Cache):
        """Reflect does not modify the original translate entry."""
        cache.put("t1", "translate", "m", "v1", "original translation",
                  meta={"chunk_id": "ch01_c01"})

        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return CompletionResult(
                    text='{"notes": [], "general_notes": ""}',
                    input_tokens=500, output_tokens=50,
                    total_tokens=550, model="s", raw={},
                )
            return CompletionResult(
                text="===PARAGRAPH 1===\n<p>New</p>",
                input_tokens=500, output_tokens=50,
                total_tokens=550, model="s", raw={},
            )

        mock_provider.complete.side_effect = side_effect

        stats = ReflectStats()
        reflector.reflect_chunk(
            "ch01_c01", "<p>Orig</p>", ["<p>Orig</p>"],
            "original translation", 3, [], stats,
        )

        # Original translate entry still there
        stages = cache.get_chunk_stages("ch01_c01")
        translate_entries = [s for s in stages if s.stage == "translate"]
        assert len(translate_entries) == 1
        assert translate_entries[0].content == "original translation"

    def test_preference_can_revert_to_translate(
        self, reflector: Reflector, mock_provider, cache: Cache
    ):
        """After reflect, user can prefer translate via preferences."""
        cache.put("t1", "translate", "m", "v1", "original",
                  meta={"chunk_id": "ch01_c01"})

        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return CompletionResult(
                    text='{"notes": [], "general_notes": ""}',
                    input_tokens=500, output_tokens=50,
                    total_tokens=550, model="s", raw={},
                )
            return CompletionResult(
                text="===PARAGRAPH 1===\n<p>Reflected</p>",
                input_tokens=500, output_tokens=50,
                total_tokens=550, model="s", raw={},
            )

        mock_provider.complete.side_effect = side_effect

        stats = ReflectStats()
        reflector.reflect_chunk(
            "ch01_c01", "<p>Orig</p>", ["<p>Orig</p>"],
            "original", 2, [], stats,
        )

        # Default: reflect wins
        assert cache.resolve_stage_for_chunk("ch01_c01") == "reflect"

        # Set preference back to translate
        cache.set_preference("ch01_c01", "translate")
        assert cache.resolve_stage_for_chunk("ch01_c01") == "translate"
