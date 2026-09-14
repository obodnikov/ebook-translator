"""Tests for postprocess.py: proofread, style, verify passes."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from booktranslator.cache import Cache
from booktranslator.postprocess import PostProcessor, PostprocessStats
from booktranslator.provider import CompletionResult


@pytest.fixture
def cache(tmp_path: Path) -> Cache:
    return Cache(tmp_path / "test.sqlite")


@pytest.fixture
def mock_provider():
    return MagicMock()


@pytest.fixture
def proofread_prompt_path(tmp_path: Path) -> Path:
    prompt = tmp_path / "proofread.md"
    prompt.write_text(
        "---\n"
        "version: 1\n"
        "model: anthropic/claude-haiku-4.5\n"
        "temperature: 0.2\n"
        "max_tokens: 12000\n"
        "---\n"
        "# System\n"
        "Fix grammar in {{ target_lang_name }} text.\n"
        "Glossary: {{ glossary_block }}\n\n"
        "# User\n"
        "{% for fragment in main_fragments %}\n"
        "===PARAGRAPH {{ loop.index }}===\n"
        "{{ fragment }}\n"
        "{% endfor %}\n",
        encoding="utf-8",
    )
    return prompt


@pytest.fixture
def style_prompt_path(tmp_path: Path) -> Path:
    prompt = tmp_path / "style.md"
    prompt.write_text(
        "---\n"
        "version: 1\n"
        "model: anthropic/claude-sonnet-4.6\n"
        "temperature: 0.4\n"
        "max_tokens: 12000\n"
        "---\n"
        "# System\n"
        "Improve style in {{ target_lang_name }} text.\n"
        "Glossary: {{ glossary_block }}\n\n"
        "# User\n"
        "{% for fragment in main_fragments %}\n"
        "===PARAGRAPH {{ loop.index }}===\n"
        "{{ fragment }}\n"
        "{% endfor %}\n",
        encoding="utf-8",
    )
    return prompt


@pytest.fixture
def verify_prompt_path(tmp_path: Path) -> Path:
    prompt = tmp_path / "verify.md"
    prompt.write_text(
        "---\n"
        "version: 1\n"
        "model: anthropic/claude-sonnet-4.6\n"
        "temperature: 0.2\n"
        "max_tokens: 12000\n"
        "---\n"
        "# System\n"
        "Verify {{ target_lang_name }} translation against {{ source_lang_name }} original.\n"
        "Glossary: {{ glossary_block }}\n\n"
        "# User\n"
        "Original: {{ original_text }}\n"
        "{% for fragment in main_fragments %}\n"
        "===PARAGRAPH {{ loop.index }}===\n"
        "{{ fragment }}\n"
        "{% endfor %}\n",
        encoding="utf-8",
    )
    return prompt


@pytest.fixture
def proofreader(mock_provider, proofread_prompt_path, cache) -> PostProcessor:
    return PostProcessor(
        provider=mock_provider,
        prompt_path=proofread_prompt_path,
        cache=cache,
        glossary=None,
        stage="proofread",
        model="anthropic/claude-haiku-4.5",
        source_lang="en",
        target_lang="ru",
    )


@pytest.fixture
def styler(mock_provider, style_prompt_path, cache) -> PostProcessor:
    return PostProcessor(
        provider=mock_provider,
        prompt_path=style_prompt_path,
        cache=cache,
        glossary=None,
        stage="style",
        model="anthropic/claude-sonnet-4.6",
        source_lang="en",
        target_lang="ru",
    )


@pytest.fixture
def verifier(mock_provider, verify_prompt_path, cache) -> PostProcessor:
    return PostProcessor(
        provider=mock_provider,
        prompt_path=verify_prompt_path,
        cache=cache,
        glossary=None,
        stage="verify",
        model="anthropic/claude-sonnet-4.6",
        source_lang="en",
        target_lang="ru",
    )


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


class TestParseResponse:
    def test_valid_paragraphs(self, proofreader: PostProcessor):
        text = "===PARAGRAPH 1===\n<p>Привет мир.</p>\n===PARAGRAPH 2===\n<p>Второй.</p>"
        fragments = proofreader._parse_response(text, 2)
        assert len(fragments) == 2
        assert fragments[0] == "<p>Привет мир.</p>"
        assert fragments[1] == "<p>Второй.</p>"

    def test_with_code_fences(self, proofreader: PostProcessor):
        text = "```\n===PARAGRAPH 1===\n<p>Текст.</p>\n```"
        fragments = proofreader._parse_response(text, 1)
        assert len(fragments) == 1
        assert fragments[0] == "<p>Текст.</p>"

    def test_count_mismatch_raises(self, proofreader: PostProcessor):
        text = "===PARAGRAPH 1===\n<p>Only one.</p>"
        with pytest.raises(ValueError, match="Expected 2 paragraphs, got 1"):
            proofreader._parse_response(text, 2)

    def test_no_markers_raises(self, proofreader: PostProcessor):
        text = "<p>No markers here.</p>"
        with pytest.raises(ValueError, match="No.*markers found"):
            proofreader._parse_response(text, 1)

    def test_whitespace_handling(self, proofreader: PostProcessor):
        text = "  \n===PARAGRAPH 1===\n  <p>Spaced.</p>  \n  "
        fragments = proofreader._parse_response(text, 1)
        assert fragments[0] == "<p>Spaced.</p>"


class TestNoChangesRecognition:
    """The delta prompts print NO_CHANGES inside a code fence and call it
    `NO_CHANGES` in the rules, so models copy those shapes back. Rejecting
    them threw away correct verdicts on the Bear Head run: style ch19_c05
    answered with commentary plus a fenced NO_CHANGES, verify ch25_c03 with
    a back-quoted one. Both were logged as parse failures.
    """

    @pytest.mark.parametrize(
        "text",
        [
            "NO_CHANGES",
            "no changes",
            "NOCHANGES",
            "```\nNO_CHANGES\n```",
            "```NO_CHANGES```",
            "`NO_CHANGES`",  # verify ch25_c03
            "  NO_CHANGES  \n",
            # style ch19_c05: a sentence of commentary, then the fenced verdict
            "Перечитал абзац. Текст живой, канцелярита нет.\n\n```\nNO_CHANGES\n```",
        ],
    )
    def test_accepted_forms_leave_paragraphs_untouched(self, proofreader: PostProcessor, text):
        paragraphs = ["<p>Первый.</p>", "<p>Второй.</p>"]
        out, changed = proofreader._parse_delta_response(text, paragraphs)
        assert out == paragraphs
        assert changed is False

    @pytest.mark.parametrize(
        "text",
        [
            # A response carrying patches is never read as "nothing to change",
            # even when it mentions NO_CHANGES in passing — edits must not be
            # silently dropped.
            "Сначала думал ответить NO_CHANGES, но нет.\n"
            '```json\n[{"p": 1, "text": "<p>Правка.</p>"}]\n```',
            '[{"p": 1, "text": "<p>Правка.</p>"}]',
        ],
    )
    def test_patches_win_over_a_mention_of_no_changes(self, proofreader: PostProcessor, text):
        paragraphs = ["<p>Первый.</p>", "<p>Второй.</p>"]
        out, changed = proofreader._parse_delta_response(text, paragraphs)
        assert changed is True
        assert out[0] == "<p>Правка.</p>"
        assert out[1] == "<p>Второй.</p>"

    def test_unparseable_response_still_raises(self, proofreader: PostProcessor):
        with pytest.raises(ValueError, match="not valid JSON or NO_CHANGES"):
            proofreader._parse_delta_response("Ответа нет вовсе.", ["<p>Первый.</p>"])


# ---------------------------------------------------------------------------
# Process single chunk — proofread
# ---------------------------------------------------------------------------


class TestProcessChunkProofread:
    def test_calls_provider_and_stores(
        self, proofreader: PostProcessor, mock_provider, cache: Cache
    ):
        mock_provider.complete.return_value = CompletionResult(
            text="===PARAGRAPH 1===\n<p>Исправленный текст.</p>",
            input_tokens=300,
            output_tokens=50,
            total_tokens=350,
            model="haiku",
            raw={},
        )

        stats = PostprocessStats(stage="proofread")
        result = proofreader.process_chunk(
            "ch01_c01",
            ["<p>Исправленый текст.</p>"],
            stats,
        )

        assert result is not None
        assert result.chunk_id == "ch01_c01"
        assert result.stage == "proofread"
        assert result.changed is True
        assert stats.chunks_processed == 1
        assert stats.chunks_changed == 1
        assert stats.input_tokens == 300
        mock_provider.complete.assert_called_once()

    def test_unchanged_when_same_output(self, proofreader: PostProcessor, mock_provider):
        # Return the same text as input
        mock_provider.complete.return_value = CompletionResult(
            text="===PARAGRAPH 1===\n<p>Уже правильно.</p>",
            input_tokens=200,
            output_tokens=30,
            total_tokens=230,
            model="haiku",
            raw={},
        )

        stats = PostprocessStats(stage="proofread")
        result = proofreader.process_chunk(
            "ch01_c01",
            ["<p>Уже правильно.</p>"],
            stats,
        )

        assert result is not None
        assert result.changed is False
        assert stats.chunks_unchanged == 1

    def test_caches_result(self, proofreader: PostProcessor, mock_provider, cache: Cache):
        mock_provider.complete.return_value = CompletionResult(
            text="===PARAGRAPH 1===\n<p>Результат.</p>",
            input_tokens=200,
            output_tokens=30,
            total_tokens=230,
            model="haiku",
            raw={},
        )

        stats1 = PostprocessStats(stage="proofread")
        proofreader.process_chunk("ch01_c01", ["<p>Вход.</p>"], stats1)
        assert stats1.chunks_processed == 1

        # Second call uses cache
        stats2 = PostprocessStats(stage="proofread")
        proofreader.process_chunk("ch01_c01", ["<p>Вход.</p>"], stats2)
        assert stats2.chunks_cached == 1
        assert stats2.chunks_processed == 0
        assert mock_provider.complete.call_count == 1

    def test_stores_as_correct_stage(self, proofreader: PostProcessor, mock_provider, cache: Cache):
        mock_provider.complete.return_value = CompletionResult(
            text="===PARAGRAPH 1===\n<p>Текст.</p>",
            input_tokens=200,
            output_tokens=30,
            total_tokens=230,
            model="haiku",
            raw={},
        )

        stats = PostprocessStats(stage="proofread")
        proofreader.process_chunk("ch01_c01", ["<p>Текст.</p>"], stats)

        stages = cache.get_chunk_stages("ch01_c01")
        proofread_stages = [s for s in stages if s.stage == "proofread"]
        assert len(proofread_stages) == 1

    def test_multiple_paragraphs(self, proofreader: PostProcessor, mock_provider):
        mock_provider.complete.return_value = CompletionResult(
            text=(
                "===PARAGRAPH 1===\n<p>Первый.</p>\n"
                "===PARAGRAPH 2===\n<p>Второй.</p>\n"
                "===PARAGRAPH 3===\n<p>Третий.</p>"
            ),
            input_tokens=500,
            output_tokens=100,
            total_tokens=600,
            model="haiku",
            raw={},
        )

        stats = PostprocessStats(stage="proofread")
        result = proofreader.process_chunk(
            "ch01_c01",
            ["<p>Первый.</p>", "<p>Второй.</p>", "<p>Третий.</p>"],
            stats,
        )

        assert result is not None
        assert result.changed is False  # Same content


# ---------------------------------------------------------------------------
# Process single chunk — style
# ---------------------------------------------------------------------------


class TestProcessChunkStyle:
    def test_style_pass_stores_correctly(self, styler: PostProcessor, mock_provider, cache: Cache):
        mock_provider.complete.return_value = CompletionResult(
            text="===PARAGRAPH 1===\n<p>Улучшенный стиль.</p>",
            input_tokens=400,
            output_tokens=60,
            total_tokens=460,
            model="sonnet",
            raw={},
        )

        stats = PostprocessStats(stage="style")
        result = styler.process_chunk(
            "ch02_c01",
            ["<p>Плохой стиль.</p>"],
            stats,
        )

        assert result is not None
        assert result.stage == "style"
        assert result.changed is True

        stages = cache.get_chunk_stages("ch02_c01")
        style_stages = [s for s in stages if s.stage == "style"]
        assert len(style_stages) == 1


# ---------------------------------------------------------------------------
# Process single chunk — verify
# ---------------------------------------------------------------------------


class TestProcessChunkVerify:
    def test_verify_uses_original_text(self, verifier: PostProcessor, mock_provider):
        mock_provider.complete.return_value = CompletionResult(
            text="===PARAGRAPH 1===\n<p>Исправленный перевод с восстановленным предложением.</p>",
            input_tokens=600,
            output_tokens=80,
            total_tokens=680,
            model="sonnet",
            raw={},
        )

        stats = PostprocessStats(stage="verify")
        result = verifier.process_chunk(
            "ch03_c01",
            ["<p>Неполный перевод.</p>"],
            stats,
            original_text="<p>Full original text with more content.</p>",
        )

        assert result is not None
        assert result.stage == "verify"
        assert result.changed is True

        # Verify that original_text was passed to the prompt
        call_args = mock_provider.complete.call_args
        user_msg = call_args.kwargs.get("user") or call_args[1].get("user", "")
        assert "Full original text" in user_msg


# ---------------------------------------------------------------------------
# Process multiple chunks
# ---------------------------------------------------------------------------


class TestProcessChunks:
    def test_processes_all_chunks(self, proofreader: PostProcessor, mock_provider):
        mock_provider.complete.return_value = CompletionResult(
            text="===PARAGRAPH 1===\n<p>Fixed.</p>",
            input_tokens=200,
            output_tokens=30,
            total_tokens=230,
            model="haiku",
            raw={},
        )

        chunks_data = [
            {"chunk_id": f"ch01_c{i:02d}", "translated_paragraphs": [f"<p>Text {i}.</p>"]}
            for i in range(1, 4)
        ]

        stats = proofreader.process_chunks(chunks_data, parallelism=1)

        assert stats.chunks_total == 3
        assert stats.chunks_processed == 3
        assert stats.chunks_failed == 0
        assert len(stats.results) == 3

    def test_handles_failure_gracefully(self, proofreader: PostProcessor, mock_provider):
        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 2:
                raise RuntimeError("API timeout")
            return CompletionResult(
                text="===PARAGRAPH 1===\n<p>OK.</p>",
                input_tokens=200,
                output_tokens=30,
                total_tokens=230,
                model="haiku",
                raw={},
            )

        mock_provider.complete.side_effect = side_effect

        chunks_data = [
            {"chunk_id": "c1", "translated_paragraphs": ["<p>A.</p>"]},
            {"chunk_id": "c2", "translated_paragraphs": ["<p>B.</p>"]},
            {"chunk_id": "c3", "translated_paragraphs": ["<p>C.</p>"]},
        ]

        stats = proofreader.process_chunks(chunks_data, parallelism=1)

        assert stats.chunks_failed == 1
        assert stats.chunks_processed == 2
        assert len(stats.results) == 2

    def test_progress_callback(self, proofreader: PostProcessor, mock_provider):
        mock_provider.complete.return_value = CompletionResult(
            text="===PARAGRAPH 1===\n<p>Done.</p>",
            input_tokens=100,
            output_tokens=20,
            total_tokens=120,
            model="haiku",
            raw={},
        )

        progress_calls = []

        def on_progress(done, total, cid, stats):
            progress_calls.append((done, total, cid))

        chunks_data = [
            {"chunk_id": "c1", "translated_paragraphs": ["<p>A.</p>"]},
            {"chunk_id": "c2", "translated_paragraphs": ["<p>B.</p>"]},
        ]

        proofreader.process_chunks(chunks_data, on_progress=on_progress, parallelism=1)

        assert len(progress_calls) == 2
        assert progress_calls[-1][0] == 2  # done
        assert progress_calls[-1][1] == 2  # total

    def test_parallel_execution(self, proofreader: PostProcessor, mock_provider):
        mock_provider.complete.return_value = CompletionResult(
            text="===PARAGRAPH 1===\n<p>Parallel.</p>",
            input_tokens=100,
            output_tokens=20,
            total_tokens=120,
            model="haiku",
            raw={},
        )

        chunks_data = [
            {"chunk_id": f"c{i}", "translated_paragraphs": [f"<p>Text {i}.</p>"]} for i in range(8)
        ]

        stats = proofreader.process_chunks(chunks_data, parallelism=4)

        assert stats.chunks_total == 8
        assert stats.chunks_processed == 8
        assert stats.chunks_failed == 0


# ---------------------------------------------------------------------------
# Waterfall integration
# ---------------------------------------------------------------------------


class TestWaterfallIntegration:
    def test_proofread_in_waterfall(self, proofreader: PostProcessor, mock_provider, cache: Cache):
        """Proofread results participate in waterfall resolution."""
        # Put a translate entry
        cache.put(
            "t1",
            "translate",
            "m",
            "v1",
            "===PARAGRAPH 1===\n<p>Original.</p>",
            meta={"chunk_id": "ch01_c01"},
        )

        # Run proofread
        mock_provider.complete.return_value = CompletionResult(
            text="===PARAGRAPH 1===\n<p>Proofread.</p>",
            input_tokens=200,
            output_tokens=30,
            total_tokens=230,
            model="haiku",
            raw={},
        )

        stats = PostprocessStats(stage="proofread")
        proofreader.process_chunk("ch01_c01", ["<p>Original.</p>"], stats)

        # Waterfall should now resolve to proofread
        resolved = cache.resolve_stage_for_chunk("ch01_c01")
        assert resolved == "proofread"

    def test_style_after_proofread_in_waterfall(
        self, styler: PostProcessor, mock_provider, cache: Cache
    ):
        """Style after proofread: waterfall resolves to style."""
        cache.put(
            "t1",
            "translate",
            "m",
            "v1",
            "===PARAGRAPH 1===\n<p>Translate.</p>",
            meta={"chunk_id": "ch01_c01"},
        )
        cache.put(
            "p1",
            "proofread",
            "m",
            "v1",
            "===PARAGRAPH 1===\n<p>Proofread.</p>",
            meta={"chunk_id": "ch01_c01"},
        )

        mock_provider.complete.return_value = CompletionResult(
            text="===PARAGRAPH 1===\n<p>Styled.</p>",
            input_tokens=200,
            output_tokens=30,
            total_tokens=230,
            model="sonnet",
            raw={},
        )

        stats = PostprocessStats(stage="style")
        styler.process_chunk("ch01_c01", ["<p>Proofread.</p>"], stats)

        resolved = cache.resolve_stage_for_chunk("ch01_c01")
        assert resolved == "style"

    def test_verify_is_highest_priority(self, verifier: PostProcessor, mock_provider, cache: Cache):
        """Verify is the highest stage in waterfall."""
        cache.put(
            "t1",
            "translate",
            "m",
            "v1",
            "===PARAGRAPH 1===\n<p>T.</p>",
            meta={"chunk_id": "ch01_c01"},
        )
        cache.put(
            "p1",
            "proofread",
            "m",
            "v1",
            "===PARAGRAPH 1===\n<p>P.</p>",
            meta={"chunk_id": "ch01_c01"},
        )
        cache.put(
            "s1", "style", "m", "v1", "===PARAGRAPH 1===\n<p>S.</p>", meta={"chunk_id": "ch01_c01"}
        )

        mock_provider.complete.return_value = CompletionResult(
            text="===PARAGRAPH 1===\n<p>Verified.</p>",
            input_tokens=300,
            output_tokens=40,
            total_tokens=340,
            model="sonnet",
            raw={},
        )

        stats = PostprocessStats(stage="verify")
        verifier.process_chunk(
            "ch01_c01",
            ["<p>S.</p>"],
            stats,
            original_text="<p>Original.</p>",
        )

        resolved = cache.resolve_stage_for_chunk("ch01_c01")
        assert resolved == "verify"

    def test_preference_overrides_postprocess(
        self, proofreader: PostProcessor, mock_provider, cache: Cache
    ):
        """User preference can revert to translate even after postprocess."""
        cache.put(
            "t1",
            "translate",
            "m",
            "v1",
            "===PARAGRAPH 1===\n<p>T.</p>",
            meta={"chunk_id": "ch01_c01"},
        )

        mock_provider.complete.return_value = CompletionResult(
            text="===PARAGRAPH 1===\n<p>Proofread.</p>",
            input_tokens=200,
            output_tokens=30,
            total_tokens=230,
            model="haiku",
            raw={},
        )

        stats = PostprocessStats(stage="proofread")
        proofreader.process_chunk("ch01_c01", ["<p>T.</p>"], stats)

        # Default: proofread wins
        assert cache.resolve_stage_for_chunk("ch01_c01") == "proofread"

        # Set preference back to translate
        cache.set_preference("ch01_c01", "translate")
        assert cache.resolve_stage_for_chunk("ch01_c01") == "translate"

    def test_original_stages_preserved(
        self, proofreader: PostProcessor, mock_provider, cache: Cache
    ):
        """Post-processing does not modify earlier stage entries."""
        cache.put(
            "t1",
            "translate",
            "m",
            "v1",
            "===PARAGRAPH 1===\n<p>Original translate.</p>",
            meta={"chunk_id": "ch01_c01"},
        )

        mock_provider.complete.return_value = CompletionResult(
            text="===PARAGRAPH 1===\n<p>Proofread version.</p>",
            input_tokens=200,
            output_tokens=30,
            total_tokens=230,
            model="haiku",
            raw={},
        )

        stats = PostprocessStats(stage="proofread")
        proofreader.process_chunk("ch01_c01", ["<p>Original translate.</p>"], stats)

        # Original translate entry still intact
        stages = cache.get_chunk_stages("ch01_c01")
        translate_entries = [s for s in stages if s.stage == "translate"]
        assert len(translate_entries) == 1
        assert "Original translate" in translate_entries[0].content


# ---------------------------------------------------------------------------
# Pipeline helpers: collect_waterfall_paragraphs
# ---------------------------------------------------------------------------


class TestCollectWaterfallParagraphs:
    def test_skips_a_stage_with_malformed_xhtml(self, cache: Cache):
        """A post-processing pass must not be paid to patch a broken paragraph."""
        from booktranslator.pipeline_helpers import collect_waterfall_paragraphs

        cache.put(
            "t1",
            "translate",
            "m",
            "v1",
            "===PARAGRAPH 1===\n<p>First.</p>\n===PARAGRAPH 2===\n<p>Second.</p>",
            meta={"chunk_id": "ch01_c01"},
        )
        cache.put(
            "r1",
            "reflect",
            "m",
            "v1",
            "===PARAGRAPH 1===\n<p>First.</p>\n<p>Extra.</p>\n===PARAGRAPH 2===\n<p>Second.</p>",
            meta={"chunk_id": "ch01_c01"},
        )

        result = collect_waterfall_paragraphs(cache, ["ch01_c01"], "proofread", {"ch01_c01": 2})

        assert result["ch01_c01"] == ["<p>First.</p>", "<p>Second.</p>"]

    def test_chunk_with_no_well_formed_stage_is_left_out(self, cache: Cache):
        from booktranslator.pipeline_helpers import collect_waterfall_paragraphs

        cache.put(
            "t1",
            "translate",
            "m",
            "v1",
            "===PARAGRAPH 1===\n<p>First.\n===PARAGRAPH 2===\n<p>Second.",
            meta={"chunk_id": "ch01_c01"},
        )

        result = collect_waterfall_paragraphs(cache, ["ch01_c01"], "proofread", {"ch01_c01": 2})

        assert result == {}

    def test_gets_translate_when_no_later_stages(self, cache: Cache):
        from booktranslator.pipeline_helpers import collect_waterfall_paragraphs

        cache.put(
            "t1",
            "translate",
            "m",
            "v1",
            "===PARAGRAPH 1===\n<p>First.</p>\n===PARAGRAPH 2===\n<p>Second.</p>",
            meta={"chunk_id": "ch01_c01"},
        )

        result = collect_waterfall_paragraphs(cache, ["ch01_c01"], "proofread", {"ch01_c01": 2})

        assert "ch01_c01" in result
        assert result["ch01_c01"] == ["<p>First.</p>", "<p>Second.</p>"]

    def test_gets_proofread_for_style_stage(self, cache: Cache):
        from booktranslator.pipeline_helpers import collect_waterfall_paragraphs

        cache.put(
            "t1",
            "translate",
            "m",
            "v1",
            "===PARAGRAPH 1===\n<p>Translate.</p>",
            meta={"chunk_id": "ch01_c01"},
        )
        cache.put(
            "p1",
            "proofread",
            "m",
            "v1",
            "===PARAGRAPH 1===\n<p>Proofread.</p>",
            meta={"chunk_id": "ch01_c01"},
        )

        result = collect_waterfall_paragraphs(cache, ["ch01_c01"], "style", {"ch01_c01": 1})

        assert result["ch01_c01"] == ["<p>Proofread.</p>"]

    def test_gets_style_for_verify_stage(self, cache: Cache):
        from booktranslator.pipeline_helpers import collect_waterfall_paragraphs

        cache.put(
            "t1",
            "translate",
            "m",
            "v1",
            "===PARAGRAPH 1===\n<p>T.</p>",
            meta={"chunk_id": "ch01_c01"},
        )
        cache.put(
            "p1",
            "proofread",
            "m",
            "v1",
            "===PARAGRAPH 1===\n<p>P.</p>",
            meta={"chunk_id": "ch01_c01"},
        )
        cache.put(
            "s1", "style", "m", "v1", "===PARAGRAPH 1===\n<p>S.</p>", meta={"chunk_id": "ch01_c01"}
        )

        result = collect_waterfall_paragraphs(cache, ["ch01_c01"], "verify", {"ch01_c01": 1})

        assert result["ch01_c01"] == ["<p>S.</p>"]

    def test_skips_chunks_with_wrong_paragraph_count(self, cache: Cache):
        from booktranslator.pipeline_helpers import collect_waterfall_paragraphs

        # Content has 1 paragraph but we expect 2
        cache.put(
            "t1",
            "translate",
            "m",
            "v1",
            "===PARAGRAPH 1===\n<p>Only one.</p>",
            meta={"chunk_id": "ch01_c01"},
        )

        result = collect_waterfall_paragraphs(cache, ["ch01_c01"], "proofread", {"ch01_c01": 2})

        assert "ch01_c01" not in result

    def test_handles_reflect_in_waterfall(self, cache: Cache):
        from booktranslator.pipeline_helpers import collect_waterfall_paragraphs

        cache.put(
            "t1",
            "translate",
            "m",
            "v1",
            "===PARAGRAPH 1===\n<p>Translate.</p>",
            meta={"chunk_id": "ch01_c01"},
        )
        cache.put(
            "r1",
            "reflect",
            "m",
            "v1",
            "===PARAGRAPH 1===\n<p>Reflected.</p>",
            meta={"chunk_id": "ch01_c01"},
        )

        # For proofread, reflect is the latest preceding stage
        result = collect_waterfall_paragraphs(cache, ["ch01_c01"], "proofread", {"ch01_c01": 1})

        assert result["ch01_c01"] == ["<p>Reflected.</p>"]

    def test_empty_chunk_ids(self, cache: Cache):
        from booktranslator.pipeline_helpers import collect_waterfall_paragraphs

        result = collect_waterfall_paragraphs(cache, [], "proofread", {})
        assert result == {}

    def test_falls_back_when_latest_stage_has_wrong_count(self, cache: Cache):
        """If latest stage has wrong paragraph count, falls back to earlier."""
        from booktranslator.pipeline_helpers import collect_waterfall_paragraphs

        # translate has correct count (2)
        cache.put(
            "t1",
            "translate",
            "m",
            "v1",
            "===PARAGRAPH 1===\n<p>T1.</p>\n===PARAGRAPH 2===\n<p>T2.</p>",
            meta={"chunk_id": "ch01_c01"},
        )
        # reflect has wrong count (1 instead of 2)
        cache.put(
            "r1",
            "reflect",
            "m",
            "v1",
            "===PARAGRAPH 1===\n<p>Only one.</p>",
            meta={"chunk_id": "ch01_c01"},
        )

        result = collect_waterfall_paragraphs(cache, ["ch01_c01"], "proofread", {"ch01_c01": 2})

        # Should fall back to translate
        assert "ch01_c01" in result
        assert result["ch01_c01"] == ["<p>T1.</p>", "<p>T2.</p>"]

    def test_falls_back_when_latest_stage_has_no_markers(self, cache: Cache):
        """If latest stage has no paragraph markers, falls back to earlier."""
        from booktranslator.pipeline_helpers import collect_waterfall_paragraphs

        # Expected 2 paragraphs — "no markers" content can't satisfy that
        cache.put(
            "t1",
            "translate",
            "m",
            "v1",
            "===PARAGRAPH 1===\n<p>T1.</p>\n===PARAGRAPH 2===\n<p>T2.</p>",
            meta={"chunk_id": "ch01_c01"},
        )
        # reflect has no markers (malformed)
        cache.put(
            "r1",
            "reflect",
            "m",
            "v1",
            "Just some text without markers",
            meta={"chunk_id": "ch01_c01"},
        )

        result = collect_waterfall_paragraphs(cache, ["ch01_c01"], "proofread", {"ch01_c01": 2})

        # Should fall back to translate
        assert "ch01_c01" in result
        assert result["ch01_c01"] == ["<p>T1.</p>", "<p>T2.</p>"]

    def test_falls_back_through_multiple_invalid_stages(self, cache: Cache):
        """Falls back through multiple invalid stages to find valid one."""
        from booktranslator.pipeline_helpers import collect_waterfall_paragraphs

        # translate is valid (2 paragraphs)
        cache.put(
            "t1",
            "translate",
            "m",
            "v1",
            "===PARAGRAPH 1===\n<p>T1 OK.</p>\n===PARAGRAPH 2===\n<p>T2 OK.</p>",
            meta={"chunk_id": "ch01_c01"},
        )
        # reflect is malformed (wrong count: 1 instead of 2)
        cache.put(
            "r1",
            "reflect",
            "m",
            "v1",
            "===PARAGRAPH 1===\n<p>Only one.</p>",
            meta={"chunk_id": "ch01_c01"},
        )
        # proofread is malformed (no markers)
        cache.put(
            "p1",
            "proofread",
            "m",
            "v1",
            "garbage content without markers",
            meta={"chunk_id": "ch01_c01"},
        )

        # For style, preceding stages are: translate, reflect, proofread
        # proofread (latest) has no markers, reflect has wrong count (1 vs 2)
        # should fall back to translate
        result = collect_waterfall_paragraphs(cache, ["ch01_c01"], "style", {"ch01_c01": 2})

        assert "ch01_c01" in result
        assert result["ch01_c01"] == ["<p>T1 OK.</p>", "<p>T2 OK.</p>"]


# ---------------------------------------------------------------------------
# Pipeline helpers: collect_waterfall_translations
# ---------------------------------------------------------------------------


class TestCollectWaterfallTranslations:
    def test_basic_waterfall(self, cache: Cache):
        from booktranslator.pipeline_helpers import collect_waterfall_translations

        cache.put("t1", "translate", "m", "v1", "translate content", meta={"chunk_id": "ch01_c01"})

        result = collect_waterfall_translations(cache, ["ch01_c01"], "proofread")
        assert result["ch01_c01"] == "translate content"

    def test_picks_latest_preceding(self, cache: Cache):
        from booktranslator.pipeline_helpers import collect_waterfall_translations

        cache.put("t1", "translate", "m", "v1", "translate", meta={"chunk_id": "ch01_c01"})
        cache.put("r1", "reflect", "m", "v1", "reflect", meta={"chunk_id": "ch01_c01"})
        cache.put("p1", "proofread", "m", "v1", "proofread", meta={"chunk_id": "ch01_c01"})

        # For style, proofread is the latest preceding stage
        result = collect_waterfall_translations(cache, ["ch01_c01"], "style")
        assert result["ch01_c01"] == "proofread"

    def test_missing_chunk_not_in_result(self, cache: Cache):
        from booktranslator.pipeline_helpers import collect_waterfall_translations

        result = collect_waterfall_translations(cache, ["nonexistent"], "proofread")
        assert "nonexistent" not in result


# ---------------------------------------------------------------------------
# Integration: rehydrate_book_from_waterfall
# ---------------------------------------------------------------------------


class TestRehydrateBookFromWaterfall:
    """Test that rehydration updates in-memory trees from cache."""

    def _make_book_and_chunks(self, tmp_path):
        """Create a minimal StructuredBook + ChunkSet for testing."""
        from dataclasses import dataclass, field

        from lxml import etree

        from booktranslator.chunker import Chunk, ChunkSet

        # Minimal ChapterDoc-like object
        @dataclass
        class FakeChapterDoc:
            spine_index: int = 0
            archive_name: str = "ch1.xhtml"
            original_bytes: bytes = b""
            tree: object = None
            paragraphs: list = field(default_factory=list)

        # Minimal StructuredBook-like object
        @dataclass
        class FakeBook:
            meta: object = None
            source_path: object = None
            chapters: list = field(default_factory=list)

        # Build a chapter with 2 paragraphs
        p1 = etree.fromstring('<p xmlns="http://www.w3.org/1999/xhtml">Original 1</p>')
        p2 = etree.fromstring('<p xmlns="http://www.w3.org/1999/xhtml">Original 2</p>')

        # Create a parent element to hold them (needed for replace)
        body = etree.SubElement(
            etree.Element("{http://www.w3.org/1999/xhtml}html"),
            "{http://www.w3.org/1999/xhtml}body",
        )
        body.append(p1)
        body.append(p2)

        chapter = FakeChapterDoc(paragraphs=[p1, p2])
        book = FakeBook(chapters=[chapter])

        chunks = [
            Chunk(
                id="ch01_c01",
                chapter_index=0,
                paragraph_indexes=[0, 1],
                word_count=10,
            )
        ]
        chunk_set = ChunkSet(book=book, chunks=chunks)
        return chunk_set

    def test_rehydrates_from_proofread_stage(self, cache: Cache):
        from booktranslator.pipeline_helpers import rehydrate_book_from_waterfall

        chunk_set = self._make_book_and_chunks(None)

        # Put translate and proofread in cache
        cache.put(
            "t1",
            "translate",
            "m",
            "v1",
            "===PARAGRAPH 1===\n<p>Translated 1</p>\n===PARAGRAPH 2===\n<p>Translated 2</p>",
            meta={"chunk_id": "ch01_c01"},
        )
        cache.put(
            "p1",
            "proofread",
            "m",
            "v1",
            "===PARAGRAPH 1===\n<p>Proofread 1</p>\n===PARAGRAPH 2===\n<p>Proofread 2</p>",
            meta={"chunk_id": "ch01_c01"},
        )

        rehydrated, _ = rehydrate_book_from_waterfall(cache, chunk_set)

        assert rehydrated == 1
        # Check that in-memory paragraphs were updated
        from lxml import etree

        chapter = chunk_set.book.chapters[0]
        text1 = etree.tostring(chapter.paragraphs[0], encoding="unicode")
        text2 = etree.tostring(chapter.paragraphs[1], encoding="unicode")
        assert "Proofread 1" in text1
        assert "Proofread 2" in text2

    def test_skips_translate_only_chunks(self, cache: Cache):
        from booktranslator.pipeline_helpers import rehydrate_book_from_waterfall

        chunk_set = self._make_book_and_chunks(None)

        # Only translate stage — no rehydration needed
        cache.put(
            "t1",
            "translate",
            "m",
            "v1",
            "===PARAGRAPH 1===\n<p>Translated 1</p>\n===PARAGRAPH 2===\n<p>Translated 2</p>",
            meta={"chunk_id": "ch01_c01"},
        )

        rehydrated, _ = rehydrate_book_from_waterfall(cache, chunk_set)
        assert rehydrated == 0

    def test_handles_verify_as_highest_stage(self, cache: Cache):
        from booktranslator.pipeline_helpers import rehydrate_book_from_waterfall

        chunk_set = self._make_book_and_chunks(None)

        cache.put(
            "t1",
            "translate",
            "m",
            "v1",
            "===PARAGRAPH 1===\n<p>T1</p>\n===PARAGRAPH 2===\n<p>T2</p>",
            meta={"chunk_id": "ch01_c01"},
        )
        cache.put(
            "p1",
            "proofread",
            "m",
            "v1",
            "===PARAGRAPH 1===\n<p>P1</p>\n===PARAGRAPH 2===\n<p>P2</p>",
            meta={"chunk_id": "ch01_c01"},
        )
        cache.put(
            "s1",
            "style",
            "m",
            "v1",
            "===PARAGRAPH 1===\n<p>S1</p>\n===PARAGRAPH 2===\n<p>S2</p>",
            meta={"chunk_id": "ch01_c01"},
        )
        cache.put(
            "v1",
            "verify",
            "m",
            "v1",
            "===PARAGRAPH 1===\n<p>Verified 1</p>\n===PARAGRAPH 2===\n<p>Verified 2</p>",
            meta={"chunk_id": "ch01_c01"},
        )

        rehydrated, _ = rehydrate_book_from_waterfall(cache, chunk_set)

        assert rehydrated == 1
        from lxml import etree

        chapter = chunk_set.book.chapters[0]
        text1 = etree.tostring(chapter.paragraphs[0], encoding="unicode")
        assert "Verified 1" in text1

    def test_handles_paragraph_count_mismatch(self, cache: Cache):
        from booktranslator.pipeline_helpers import rehydrate_book_from_waterfall

        chunk_set = self._make_book_and_chunks(None)

        cache.put(
            "t1",
            "translate",
            "m",
            "v1",
            "===PARAGRAPH 1===\n<p>T1</p>\n===PARAGRAPH 2===\n<p>T2</p>",
            meta={"chunk_id": "ch01_c01"},
        )
        # Proofread has wrong paragraph count (1 instead of 2)
        cache.put(
            "p1",
            "proofread",
            "m",
            "v1",
            "===PARAGRAPH 1===\n<p>Only one</p>",
            meta={"chunk_id": "ch01_c01"},
        )

        rehydrated, _ = rehydrate_book_from_waterfall(cache, chunk_set)
        # Should fall back to translate (which is stage='translate', skipped)
        # Since resolved stage is proofread but it's invalid, and reflect
        # doesn't exist, there's nothing valid to rehydrate from
        # (translate is excluded from rehydration since it's already in-memory)
        assert rehydrated == 0

    def test_rehydration_falls_back_to_earlier_valid_stage(self, cache: Cache):
        """If resolved stage is unparsable, rehydration falls back."""
        from booktranslator.pipeline_helpers import rehydrate_book_from_waterfall

        chunk_set = self._make_book_and_chunks(None)

        cache.put(
            "t1",
            "translate",
            "m",
            "v1",
            "===PARAGRAPH 1===\n<p>T1</p>\n===PARAGRAPH 2===\n<p>T2</p>",
            meta={"chunk_id": "ch01_c01"},
        )
        # reflect is valid
        cache.put(
            "r1",
            "reflect",
            "m",
            "v1",
            "===PARAGRAPH 1===\n<p>Reflect 1</p>\n===PARAGRAPH 2===\n<p>Reflect 2</p>",
            meta={"chunk_id": "ch01_c01"},
        )
        # proofread is malformed (wrong count) — this is the resolved stage
        cache.put(
            "p1",
            "proofread",
            "m",
            "v1",
            "===PARAGRAPH 1===\n<p>Only one paragraph</p>",
            meta={"chunk_id": "ch01_c01"},
        )

        rehydrated, _ = rehydrate_book_from_waterfall(cache, chunk_set)

        # Should fall back to reflect (valid, earlier stage)
        assert rehydrated == 1
        from lxml import etree

        chapter = chunk_set.book.chapters[0]
        text1 = etree.tostring(chapter.paragraphs[0], encoding="unicode")
        assert "Reflect 1" in text1

    def test_rehydration_falls_back_on_xml_error(self, cache: Cache):
        """If resolved stage has broken XML, falls back to earlier stage."""
        from booktranslator.pipeline_helpers import rehydrate_book_from_waterfall

        chunk_set = self._make_book_and_chunks(None)

        cache.put(
            "t1",
            "translate",
            "m",
            "v1",
            "===PARAGRAPH 1===\n<p>T1</p>\n===PARAGRAPH 2===\n<p>T2</p>",
            meta={"chunk_id": "ch01_c01"},
        )
        # reflect is valid
        cache.put(
            "r1",
            "reflect",
            "m",
            "v1",
            "===PARAGRAPH 1===\n<p>Good 1</p>\n===PARAGRAPH 2===\n<p>Good 2</p>",
            meta={"chunk_id": "ch01_c01"},
        )
        # proofread has broken XML in second fragment
        cache.put(
            "p1",
            "proofread",
            "m",
            "v1",
            "===PARAGRAPH 1===\n<p>OK</p>\n===PARAGRAPH 2===\n<p>Broken <unclosed",
            meta={"chunk_id": "ch01_c01"},
        )

        rehydrated, _ = rehydrate_book_from_waterfall(cache, chunk_set)

        # Should fall back to reflect
        assert rehydrated == 1
        from lxml import etree

        chapter = chunk_set.book.chapters[0]
        text1 = etree.tostring(chapter.paragraphs[0], encoding="unicode")
        assert "Good 1" in text1

    def test_fresh_tree_falls_back_to_translate(self, cache: Cache):
        """In assemble the tree holds the source, so translate is a real fallback."""
        from lxml import etree

        from booktranslator.pipeline_helpers import rehydrate_book_from_waterfall

        chunk_set = self._make_book_and_chunks(None)
        cache.put(
            "t1",
            "translate",
            "m",
            "v1",
            "===PARAGRAPH 1===\n<p>T1</p>\n===PARAGRAPH 2===\n<p>T2</p>",
            meta={"chunk_id": "ch01_c01"},
        )
        cache.put(
            "v1",
            "verify",
            "m",
            "v1",
            "===PARAGRAPH 1===\n<p>V1</p>\n===PARAGRAPH 2===\n<p>Broken <unclosed",
            meta={"chunk_id": "ch01_c01"},
        )

        rehydrated, failed = rehydrate_book_from_waterfall(cache, chunk_set, rehydrate_all=True)

        assert (rehydrated, failed) == (1, [])
        chapter = chunk_set.book.chapters[0]
        assert "T1" in etree.tostring(chapter.paragraphs[0], encoding="unicode")

    def test_atomic_no_partial_update_on_xml_error(self, cache: Cache):
        """If second fragment has invalid XML, neither fragment is applied."""
        from booktranslator.pipeline_helpers import rehydrate_book_from_waterfall

        chunk_set = self._make_book_and_chunks(None)

        cache.put(
            "t1",
            "translate",
            "m",
            "v1",
            "===PARAGRAPH 1===\n<p>T1</p>\n===PARAGRAPH 2===\n<p>T2</p>",
            meta={"chunk_id": "ch01_c01"},
        )
        # Proofread: first fragment valid, second is broken XML
        cache.put(
            "p1",
            "proofread",
            "m",
            "v1",
            "===PARAGRAPH 1===\n<p>Good</p>\n===PARAGRAPH 2===\n<p>Broken <unclosed",
            meta={"chunk_id": "ch01_c01"},
        )

        rehydrated, _ = rehydrate_book_from_waterfall(cache, chunk_set)

        # Should NOT rehydrate — atomic: all-or-nothing
        assert rehydrated == 0

        # Verify first paragraph was NOT changed (no partial mutation)
        from lxml import etree

        chapter = chunk_set.book.chapters[0]
        text1 = etree.tostring(chapter.paragraphs[0], encoding="unicode")
        # Should still contain original content, not "Good"
        assert "Original 1" in text1

    def test_stale_cache_ids_ignored(self, cache: Cache):
        """Chunks in cache but not in chunk_set are not rehydrated."""
        from booktranslator.pipeline_helpers import rehydrate_book_from_waterfall

        chunk_set = self._make_book_and_chunks(None)
        # chunk_set only has ch01_c01

        # Put a stale chunk in cache that doesn't exist in chunk_set
        cache.put(
            "t_stale",
            "translate",
            "m",
            "v1",
            "===PARAGRAPH 1===\n<p>Stale</p>",
            meta={"chunk_id": "ch99_c99"},
        )
        cache.put(
            "p_stale",
            "proofread",
            "m",
            "v1",
            "===PARAGRAPH 1===\n<p>Stale proofread</p>",
            meta={"chunk_id": "ch99_c99"},
        )

        # Also put valid data for the real chunk
        cache.put(
            "t1",
            "translate",
            "m",
            "v1",
            "===PARAGRAPH 1===\n<p>T1</p>\n===PARAGRAPH 2===\n<p>T2</p>",
            meta={"chunk_id": "ch01_c01"},
        )
        cache.put(
            "p1",
            "proofread",
            "m",
            "v1",
            "===PARAGRAPH 1===\n<p>P1</p>\n===PARAGRAPH 2===\n<p>P2</p>",
            meta={"chunk_id": "ch01_c01"},
        )

        rehydrated, _ = rehydrate_book_from_waterfall(cache, chunk_set)

        # Only ch01_c01 should be rehydrated, not ch99_c99
        assert rehydrated == 1

    def test_respects_preference_override(self, cache: Cache):
        from booktranslator.pipeline_helpers import rehydrate_book_from_waterfall

        chunk_set = self._make_book_and_chunks(None)

        cache.put(
            "t1",
            "translate",
            "m",
            "v1",
            "===PARAGRAPH 1===\n<p>Translate</p>\n===PARAGRAPH 2===\n<p>T2</p>",
            meta={"chunk_id": "ch01_c01"},
        )
        cache.put(
            "p1",
            "proofread",
            "m",
            "v1",
            "===PARAGRAPH 1===\n<p>Proofread</p>\n===PARAGRAPH 2===\n<p>P2</p>",
            meta={"chunk_id": "ch01_c01"},
        )

        # Set preference to translate — should NOT rehydrate
        cache.set_preference("ch01_c01", "translate")

        rehydrated, _ = rehydrate_book_from_waterfall(cache, chunk_set)
        assert rehydrated == 0


# ---------------------------------------------------------------------------
# Integration: batch cache method
# ---------------------------------------------------------------------------


class TestGetStagesForChunksBulk:
    def test_returns_all_stages_for_multiple_chunks(self, cache: Cache):
        cache.put("t1", "translate", "m", "v1", "content1", meta={"chunk_id": "ch01_c01"})
        cache.put("p1", "proofread", "m", "v1", "content2", meta={"chunk_id": "ch01_c01"})
        cache.put("t2", "translate", "m", "v1", "content3", meta={"chunk_id": "ch01_c02"})

        result = cache.get_stages_for_chunks_bulk(["ch01_c01", "ch01_c02"])

        assert "ch01_c01" in result
        assert "ch01_c02" in result
        assert len(result["ch01_c01"]) == 2
        assert len(result["ch01_c02"]) == 1
        stages = {s.stage for s in result["ch01_c01"]}
        assert stages == {"translate", "proofread"}

    def test_empty_input(self, cache: Cache):
        result = cache.get_stages_for_chunks_bulk([])
        assert result == {}

    def test_nonexistent_chunks(self, cache: Cache):
        result = cache.get_stages_for_chunks_bulk(["nonexistent"])
        assert "nonexistent" not in result

    def test_large_batch(self, cache: Cache):
        """Test batching works for > 500 chunks."""
        for i in range(600):
            cache.put(
                f"key_{i}",
                "translate",
                "m",
                "v1",
                f"content_{i}",
                meta={"chunk_id": f"ch{i:04d}"},
            )

        chunk_ids = [f"ch{i:04d}" for i in range(600)]
        result = cache.get_stages_for_chunks_bulk(chunk_ids)

        assert len(result) == 600


# ---------------------------------------------------------------------------
# Integration: cache lifecycle (try/finally)
# ---------------------------------------------------------------------------


class TestCacheLifecycle:
    def test_cache_closes_on_exception(self, tmp_path: Path):
        """Verify cache is properly closed even when exceptions occur."""
        cache = Cache(tmp_path / "lifecycle.sqlite")
        cache.put("k1", "translate", "m", "v1", "test", meta={"chunk_id": "c1"})

        # Simulate what _run_postprocess_cmd does with try/finally
        try:
            raise RuntimeError("simulated failure")
        except RuntimeError:
            pass
        finally:
            cache.close()

        # Verify we can reopen the cache (no lock held)
        cache2 = Cache(tmp_path / "lifecycle.sqlite")
        entry = cache2.get("k1")
        assert entry is not None
        assert entry.content == "test"
        cache2.close()


class TestDialogueDashIsProtected:
    """Russian dialogue opens a paragraph with an em dash; guillemets are for
    speech quoted inline. The proofread prompt asked for « » outright, models
    obliged, and 8 paragraphs of Bear Head lost their dashes. The prompt says
    the opposite now (v5); this refuses the change if a model does it anyway.

    One patch is dropped, not the whole chunk: rejecting everything would throw
    away the pass's real corrections over a punctuation slip.
    """

    @pytest.mark.parametrize(
        "before,after",
        [
            # the three shapes seen in the book
            (
                '<p class="indent">— Он своё получит.</p>',
                '<p class="indent">« Он своё получит.»</p>',
            ),
            ("<p>— Выйти из машины!</p>", "<p>« Выйти из машины! »</p>"),
            (
                "<p>— Доктор — он своё получит, — пробормотал Бойо.</p>",
                "<p>« Доктор — он своё получит, — пробормотал Бойо.</p>",
            ),
        ],
    )
    def test_patch_is_dropped_and_paragraph_kept(self, proofreader: PostProcessor, before, after):
        paragraphs = [before, "<p>Второй абзац.</p>"]
        patches = json.dumps([{"p": 1, "text": after}])
        out, changed = proofreader._parse_delta_response(patches, paragraphs)
        assert out[0] == before
        assert changed is False

    def test_other_patches_in_the_same_chunk_still_apply(self, proofreader: PostProcessor):
        paragraphs = ["<p>— Он своё получит.</p>", "<p>Опечатка тут.</p>"]
        patches = json.dumps(
            [
                {"p": 1, "text": "<p>« Он своё получит. »</p>"},
                {"p": 2, "text": "<p>Опечатки тут нет.</p>"},
            ]
        )
        out, changed = proofreader._parse_delta_response(patches, paragraphs)
        assert out[0] == paragraphs[0], "the damaging patch was refused"
        assert out[1] == "<p>Опечатки тут нет.</p>", "the good patch went through"
        assert changed is True

    @pytest.mark.parametrize(
        "before,after",
        [
            # edits that must not be mistaken for the defect
            ("<p>— Ну, это совсем не так.</p>", "<p>— Ну, это совсем не так!</p>"),
            ("<p>Он сказал «да» и ушёл.</p>", "<p>Он сказал «нет» и ушёл.</p>"),
            (
                "<p>«Правда» — так называлась газета.</p>",
                "<p>«Известия» — так называлась газета.</p>",
            ),
            ("<p>Он сказал.</p>", "<p>— Он сказал.</p>"),
        ],
    )
    def test_legitimate_edits_are_untouched(self, proofreader: PostProcessor, before, after):
        out, changed = proofreader._parse_delta_response(
            json.dumps([{"p": 1, "text": after}]), [before]
        )
        assert out[0] == after
        assert changed is True


# ---------------------------------------------------------------------------
# Replies that ignore the delta format
# ---------------------------------------------------------------------------


def _reply(text: str, *, in_tokens: int = 300, out_tokens: int = 50, finish: str | None = "stop"):
    return CompletionResult(
        text=text,
        input_tokens=in_tokens,
        output_tokens=out_tokens,
        total_tokens=in_tokens + out_tokens,
        model="haiku",
        raw={},
        finish_reason=finish,
    )


PATCH = '[{"p": 1, "text": "<p>Исправлено.</p>"}]'


class TestPreambleBeforeThePatchArray:
    def test_commentary_then_array(self, proofreader: PostProcessor):
        text = "Looking through the text carefully:\n\n" + PATCH
        out, changed = proofreader._parse_delta_response(text, ["<p>Исходное.</p>"])
        assert out == ["<p>Исправлено.</p>"]
        assert changed is True

    def test_a_bracket_in_the_commentary_does_not_hide_the_array(self, proofreader: PostProcessor):
        """The first `[` is a false start — the parser must keep looking."""
        text = "Paragraph [14] contains a gender error, see [note]:\n\n" + PATCH
        out, changed = proofreader._parse_delta_response(text, ["<p>Исходное.</p>"])
        assert out == ["<p>Исправлено.</p>"]
        assert changed is True

    def test_commentary_without_an_array_still_fails(self, proofreader: PostProcessor):
        with pytest.raises(ValueError, match="not valid JSON or NO_CHANGES"):
            proofreader._parse_delta_response(
                "Looking at paragraph 18, the translation omits a long speech.",
                ["<p>Исходное.</p>"],
            )


class TestFormatRetry:
    def test_a_reply_of_pure_commentary_is_asked_again(
        self, proofreader: PostProcessor, mock_provider, cache: Cache
    ):
        mock_provider.complete.side_effect = [
            _reply("Looking through the text carefully, paragraph 14 needs work."),
            _reply(PATCH, in_tokens=310, out_tokens=20),
        ]

        stats = PostprocessStats(stage="proofread", chunks_total=10)
        result = proofreader.process_chunk("ch01_c01", ["<p>Исходное.</p>"], stats)

        assert result is not None
        assert "Исправлено" in result.content
        assert stats.format_retries == 1
        assert mock_provider.complete.call_count == 2
        # The second call is not free and must show up in the totals.
        assert stats.input_tokens == 610
        assert stats.output_tokens == 70

    def test_the_retry_restates_the_format(self, proofreader: PostProcessor, mock_provider):
        mock_provider.complete.side_effect = [_reply("Просто рассуждение."), _reply(PATCH)]

        proofreader.process_chunk(
            "ch01_c01", ["<p>Исходное.</p>"], PostprocessStats(stage="proofread", chunks_total=10)
        )

        first_user = mock_provider.complete.call_args_list[0].kwargs["user"]
        retry_user = mock_provider.complete.call_args_list[1].kwargs["user"]
        assert "previous reply was rejected" in retry_user
        assert retry_user.startswith(first_user), "the retry asks the same question"

    def test_a_second_bad_reply_fails_the_chunk(
        self, proofreader: PostProcessor, mock_provider, cache: Cache
    ):
        mock_provider.complete.side_effect = [
            _reply("Рассуждение раз."),
            _reply("Рассуждение два."),
        ]

        stats = PostprocessStats(stage="proofread", chunks_total=10)
        with pytest.raises(ValueError):
            proofreader.process_chunk("ch01_c01", ["<p>Исходное.</p>"], stats)

        assert mock_provider.complete.call_count == 2
        assert cache.get_chunk_stages("ch01_c01") == []

    def test_a_cached_reply_is_never_re_bought(
        self, proofreader: PostProcessor, mock_provider, cache: Cache
    ):
        """A cached row that stopped parsing is a failure, not a purchase."""
        mock_provider.complete.side_effect = [_reply("Рассуждение."), _reply(PATCH)]

        stats = PostprocessStats(stage="proofread", chunks_total=10)
        proofreader.process_chunk("ch01_c01", ["<p>Исходное.</p>"], stats)
        assert mock_provider.complete.call_count == 2

        stats2 = PostprocessStats(stage="proofread", chunks_total=10)
        proofreader.process_chunk("ch01_c01", ["<p>Исходное.</p>"], stats2)
        assert stats2.chunks_cached == 1
        assert mock_provider.complete.call_count == 2, "the cached run bought nothing"

    def test_the_retry_budget_is_bounded(self, proofreader: PostProcessor, mock_provider):
        """A model that has stopped following the format must not double the bill."""
        mock_provider.complete.return_value = _reply("Одно рассуждение за другим.")

        stats = PostprocessStats(stage="proofread", chunks_total=10)
        for i in range(8):
            with pytest.raises(ValueError):
                proofreader.process_chunk(f"ch01_c{i:02d}", ["<p>Исходное.</p>"], stats)

        assert stats.format_retries == 3, "budget for a 10-chunk run"
        assert mock_provider.complete.call_count == 8 + 3


class TestFailedReplyIsKept:
    def test_the_reply_lands_next_to_the_cache(
        self, proofreader: PostProcessor, mock_provider, cache: Cache, tmp_path: Path
    ):
        mock_provider.complete.return_value = _reply(
            "Looking at paragraph 18, the translation omits a long speech.",
            finish="length",
        )

        stats = PostprocessStats(stage="proofread", chunks_total=10)
        with pytest.raises(ValueError):
            proofreader.process_chunk("ch70_c01", ["<p>Исходное.</p>"], stats)

        dumped = tmp_path / "failed" / "proofread-ch70_c01.txt"
        assert dumped.exists()
        body = dumped.read_text(encoding="utf-8")
        assert "omits a long speech" in body, "the whole reply, not the first 200 chars"
        assert "finish_reason: 'length'" in body
        assert "--- reply after the format retry ---" in body
