"""Tests for status.py: report building, score normalization, diff."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from booktranslator.cache import Cache
from booktranslator.status import (
    build_status_report,
    get_assembly_map,
    get_chunk_diff,
    get_scores,
    get_stage_overviews,
    get_total_chunks,
)


@pytest.fixture
def cache_with_data(tmp_path: Path) -> Cache:
    """Cache with a realistic set of entries for testing."""
    cache = Cache(tmp_path / "test.sqlite")
    # 3 translated chunks
    for i in range(1, 4):
        cache.put(f"t{i}", "translate", "sonnet", "v1", f"Translation {i}",
                  input_tokens=1000, output_tokens=500, cost_usd=0.05,
                  meta={"chunk_id": f"ch01_c{i:02d}"})
    # Judge scores for all 3
    cache.put("j1", "judge", "haiku", "v1",
              json.dumps({"score": 5, "issues": []}),
              input_tokens=500, output_tokens=50, cost_usd=0.001,
              meta={"chunk_id": "ch01_c01"})
    cache.put("j2", "judge", "haiku", "v1",
              json.dumps({"score": 2, "issues": ["calque", "omission"]}),
              input_tokens=500, output_tokens=50, cost_usd=0.001,
              meta={"chunk_id": "ch01_c02"})
    cache.put("j3", "judge", "haiku", "v1",
              json.dumps({"score": 4, "issues": ["minor phrasing"]}),
              input_tokens=500, output_tokens=50, cost_usd=0.001,
              meta={"chunk_id": "ch01_c03"})
    # Reflect for chunk 2 (low score)
    cache.put("r2", "reflect", "sonnet", "v1", "Improved translation 2",
              input_tokens=2000, output_tokens=600, cost_usd=0.08,
              meta={"chunk_id": "ch01_c02"})
    return cache


# ---------------------------------------------------------------------------
# get_total_chunks
# ---------------------------------------------------------------------------


class TestGetTotalChunks:
    def test_counts_translate_entries(self, cache_with_data: Cache):
        assert get_total_chunks(cache_with_data) == 3

    def test_empty_cache(self, tmp_path: Path):
        cache = Cache(tmp_path / "empty.sqlite")
        assert get_total_chunks(cache) == 0
        cache.close()

    def test_legacy_data_fallback(self, tmp_path: Path):
        """When no chunk_ids exist, falls back to count_stage."""
        cache = Cache(tmp_path / "legacy.sqlite")
        # Put entries without meta (no chunk_id)
        cache.put("k1", "translate", "m", "v1", "text1")
        cache.put("k2", "translate", "m", "v1", "text2")
        assert get_total_chunks(cache) == 2
        cache.close()


# ---------------------------------------------------------------------------
# get_stage_overviews
# ---------------------------------------------------------------------------


class TestGetStageOverviews:
    def test_shows_all_stages(self, cache_with_data: Cache):
        overviews = get_stage_overviews(cache_with_data, 3)
        stage_names = [o.stage for o in overviews]
        assert stage_names == ["translate", "judge", "reflect", "proofread", "style", "verify"]

    def test_counts_are_correct(self, cache_with_data: Cache):
        overviews = get_stage_overviews(cache_with_data, 3)
        by_stage = {o.stage: o for o in overviews}
        assert by_stage["translate"].count == 3
        assert by_stage["judge"].count == 3
        assert by_stage["reflect"].count == 1
        assert by_stage["proofread"].count == 0

    def test_cost_aggregation(self, cache_with_data: Cache):
        overviews = get_stage_overviews(cache_with_data, 3)
        by_stage = {o.stage: o for o in overviews}
        assert by_stage["translate"].cost_usd == pytest.approx(0.15)  # 3 * 0.05
        assert by_stage["reflect"].cost_usd == pytest.approx(0.08)


# ---------------------------------------------------------------------------
# get_scores — normalization tests
# ---------------------------------------------------------------------------


class TestGetScores:
    def test_normal_scores(self, cache_with_data: Cache):
        scores = get_scores(cache_with_data)
        assert scores is not None
        assert len(scores) == 3

    def test_sorted_by_score_ascending(self, cache_with_data: Cache):
        scores = get_scores(cache_with_data)
        score_values = [s.score for s in scores]
        assert score_values == sorted(score_values)

    def test_reflected_flag(self, cache_with_data: Cache):
        scores = get_scores(cache_with_data)
        by_id = {s.chunk_id: s for s in scores}
        assert by_id["ch01_c02"].reflected is True
        assert by_id["ch01_c01"].reflected is False

    def test_no_judge_data_returns_none(self, tmp_path: Path):
        cache = Cache(tmp_path / "no_judge.sqlite")
        cache.put("t1", "translate", "m", "v1", "text", meta={"chunk_id": "c1"})
        assert get_scores(cache) is None
        cache.close()

    def test_score_as_string_coerced_to_int(self, tmp_path: Path):
        cache = Cache(tmp_path / "str_score.sqlite")
        cache.put("j1", "judge", "m", "v1",
                  json.dumps({"score": "3", "issues": []}),
                  meta={"chunk_id": "c1"})
        scores = get_scores(cache)
        assert scores[0].score == 3

    def test_score_as_none_becomes_zero(self, tmp_path: Path):
        cache = Cache(tmp_path / "none_score.sqlite")
        cache.put("j1", "judge", "m", "v1",
                  json.dumps({"score": None, "issues": []}),
                  meta={"chunk_id": "c1"})
        scores = get_scores(cache)
        assert scores[0].score == 0

    def test_score_as_float_coerced(self, tmp_path: Path):
        cache = Cache(tmp_path / "float_score.sqlite")
        cache.put("j1", "judge", "m", "v1",
                  json.dumps({"score": 3.7, "issues": []}),
                  meta={"chunk_id": "c1"})
        scores = get_scores(cache)
        assert scores[0].score == 3

    def test_issues_as_none_becomes_empty_list(self, tmp_path: Path):
        cache = Cache(tmp_path / "none_issues.sqlite")
        cache.put("j1", "judge", "m", "v1",
                  json.dumps({"score": 4, "issues": None}),
                  meta={"chunk_id": "c1"})
        scores = get_scores(cache)
        assert scores[0].issues == []

    def test_issues_as_string_becomes_list(self, tmp_path: Path):
        cache = Cache(tmp_path / "str_issues.sqlite")
        cache.put("j1", "judge", "m", "v1",
                  json.dumps({"score": 2, "issues": "single issue"}),
                  meta={"chunk_id": "c1"})
        scores = get_scores(cache)
        assert scores[0].issues == ["single issue"]

    def test_issues_as_object_becomes_stringified(self, tmp_path: Path):
        cache = Cache(tmp_path / "obj_issues.sqlite")
        cache.put("j1", "judge", "m", "v1",
                  json.dumps({"score": 1, "issues": {"type": "critical"}}),
                  meta={"chunk_id": "c1"})
        scores = get_scores(cache)
        assert len(scores[0].issues) == 1
        assert "critical" in scores[0].issues[0]

    def test_malformed_json_content(self, tmp_path: Path):
        cache = Cache(tmp_path / "bad_json.sqlite")
        cache.put("j1", "judge", "m", "v1", "not json at all",
                  meta={"chunk_id": "c1"})
        scores = get_scores(cache)
        assert scores[0].score == 0
        assert "parse error" in scores[0].issues

    def test_issues_list_with_non_string_items(self, tmp_path: Path):
        cache = Cache(tmp_path / "mixed_issues.sqlite")
        cache.put("j1", "judge", "m", "v1",
                  json.dumps({"score": 3, "issues": [42, None, "text"]}),
                  meta={"chunk_id": "c1"})
        scores = get_scores(cache)
        assert scores[0].issues == ["42", "None", "text"]


# ---------------------------------------------------------------------------
# get_assembly_map
# ---------------------------------------------------------------------------


class TestGetAssemblyMap:
    def test_basic_map(self, cache_with_data: Cache):
        amap = get_assembly_map(cache_with_data, 3)
        # ch01_c01: only translate -> translate
        # ch01_c02: translate + reflect -> reflect
        # ch01_c03: only translate -> translate
        assert amap == {"translate": 2, "reflect": 1}

    def test_preference_affects_map(self, cache_with_data: Cache):
        cache_with_data.set_preference("ch01_c02", "translate")
        amap = get_assembly_map(cache_with_data, 3)
        assert amap == {"translate": 3}

    def test_empty_cache(self, tmp_path: Path):
        cache = Cache(tmp_path / "empty.sqlite")
        assert get_assembly_map(cache, 0) == {}
        cache.close()


# ---------------------------------------------------------------------------
# get_chunk_diff
# ---------------------------------------------------------------------------


class TestGetChunkDiff:
    def test_returns_all_stages(self, cache_with_data: Cache):
        diff = get_chunk_diff(cache_with_data, "ch01_c02")
        stages = [d["stage"] for d in diff]
        assert "translate" in stages
        assert "reflect" in stages

    def test_filter_by_stages(self, cache_with_data: Cache):
        diff = get_chunk_diff(cache_with_data, "ch01_c02", ["translate"])
        assert len(diff) == 1
        assert diff[0]["stage"] == "translate"

    def test_nonexistent_chunk(self, cache_with_data: Cache):
        diff = get_chunk_diff(cache_with_data, "nonexistent")
        assert diff == []

    def test_content_is_included(self, cache_with_data: Cache):
        diff = get_chunk_diff(cache_with_data, "ch01_c02")
        by_stage = {d["stage"]: d for d in diff}
        assert by_stage["translate"]["content"] == "Translation 2"
        assert by_stage["reflect"]["content"] == "Improved translation 2"


# ---------------------------------------------------------------------------
# build_status_report
# ---------------------------------------------------------------------------


class TestBuildStatusReport:
    def test_report_structure(self, cache_with_data: Cache, tmp_path: Path):
        report = build_status_report(cache_with_data, tmp_path)
        assert report.total_chunks == 3
        assert len(report.stages) == 6
        assert report.scores is not None
        assert len(report.scores) == 3
        assert report.total_cost_usd > 0

    def test_report_assembly_map(self, cache_with_data: Cache, tmp_path: Path):
        report = build_status_report(cache_with_data, tmp_path)
        assert report.assembly_map == {"translate": 2, "reflect": 1}

    def test_report_preferences(self, cache_with_data: Cache, tmp_path: Path):
        cache_with_data.set_preference("ch01_c01", "translate", "test")
        report = build_status_report(cache_with_data, tmp_path)
        assert len(report.preferences) == 1
        assert report.preferences[0]["chunk_id"] == "ch01_c01"


# ---------------------------------------------------------------------------
# Mixed legacy/new data tests
# ---------------------------------------------------------------------------


class TestMixedData:
    """Tests for status reporting with mixed legacy and new cache entries."""

    def test_total_chunks_includes_legacy(self, tmp_path: Path):
        """get_total_chunks counts both legacy and new entries."""
        cache = Cache(tmp_path / "mixed.sqlite")
        cache.put("k1", "translate", "m", "v1", "text1", meta={"chunk_id": "c1"})
        cache.put("k2", "translate", "m", "v1", "text2")  # legacy, no chunk_id
        cache.put("k3", "translate", "m", "v1", "text3")  # legacy, no chunk_id
        assert get_total_chunks(cache) == 3
        cache.close()

    def test_assembly_map_includes_legacy_as_translate(self, tmp_path: Path):
        """Legacy entries without chunk_id count as 'translate' in assembly map."""
        cache = Cache(tmp_path / "mixed.sqlite")
        cache.put("k1", "translate", "m", "v1", "text1", meta={"chunk_id": "c1"})
        cache.put("k2", "reflect", "m", "v1", "text1b", meta={"chunk_id": "c1"})
        cache.put("k3", "translate", "m", "v1", "text2")  # legacy
        cache.put("k4", "translate", "m", "v1", "text3")  # legacy
        amap = get_assembly_map(cache, 3)
        # c1 -> reflect (waterfall), 2 legacy -> translate
        assert amap == {"reflect": 1, "translate": 2}
        cache.close()

    def test_stage_overview_uses_total_count(self, tmp_path: Path):
        """Stage overviews use count_stage which includes legacy."""
        cache = Cache(tmp_path / "mixed.sqlite")
        cache.put("k1", "translate", "m", "v1", "text1", meta={"chunk_id": "c1"})
        cache.put("k2", "translate", "m", "v1", "text2")  # legacy
        overviews = get_stage_overviews(cache, 2)
        translate_ov = next(o for o in overviews if o.stage == "translate")
        assert translate_ov.count == 2
        cache.close()

    def test_report_with_mixed_data(self, tmp_path: Path):
        """Full report handles mixed data correctly."""
        cache = Cache(tmp_path / "mixed.sqlite")
        cache.put("k1", "translate", "m", "v1", "text1",
                  cost_usd=0.05, meta={"chunk_id": "c1"})
        cache.put("k2", "translate", "m", "v1", "text2", cost_usd=0.05)  # legacy
        report = build_status_report(cache, tmp_path)
        assert report.total_chunks == 2
        assert report.assembly_map == {"translate": 2}
        cache.close()


# ---------------------------------------------------------------------------
# Score filtering edge cases
# ---------------------------------------------------------------------------


class TestScoreFilteringEdgeCases:
    def test_below_zero_filters_everything(self, tmp_path: Path):
        """--below 0 should filter all chunks (all scores >= 0)."""
        cache = Cache(tmp_path / "scores.sqlite")
        cache.put("j1", "judge", "m", "v1",
                  json.dumps({"score": 0, "issues": []}),
                  meta={"chunk_id": "c1"})
        scores = get_scores(cache)
        assert scores is not None
        # Filter with below=0: score < 0 -> nothing passes
        filtered = [s for s in scores if s.score < 0]
        assert filtered == []
        cache.close()

    def test_below_one_catches_zero_scores(self, tmp_path: Path):
        """--below 1 should catch chunks with score 0."""
        cache = Cache(tmp_path / "scores.sqlite")
        cache.put("j1", "judge", "m", "v1",
                  json.dumps({"score": 0, "issues": ["broken"]}),
                  meta={"chunk_id": "c1"})
        cache.put("j2", "judge", "m", "v1",
                  json.dumps({"score": 1, "issues": []}),
                  meta={"chunk_id": "c2"})
        scores = get_scores(cache)
        filtered = [s for s in scores if s.score < 1]
        assert len(filtered) == 1
        assert filtered[0].chunk_id == "c1"
        cache.close()


# ---------------------------------------------------------------------------
# Duplicate rows / rerun tests
# ---------------------------------------------------------------------------


class TestDuplicateRows:
    """Tests for status reporting with duplicate cache entries per chunk."""

    def test_total_chunks_with_reruns(self, tmp_path: Path):
        """Reruns (same chunk_id, different key) don't inflate total."""
        cache = Cache(tmp_path / "reruns.sqlite")
        cache.put("k1", "translate", "sonnet", "v1", "text1", meta={"chunk_id": "c1"})
        cache.put("k2", "translate", "sonnet", "v2", "text1b", meta={"chunk_id": "c1"})
        cache.put("k3", "translate", "sonnet", "v1", "text2", meta={"chunk_id": "c2"})
        assert get_total_chunks(cache) == 2
        cache.close()

    def test_stage_overview_with_reruns(self, tmp_path: Path):
        """Stage overview shows distinct chunk count, not row count."""
        cache = Cache(tmp_path / "reruns.sqlite")
        cache.put("k1", "translate", "m", "v1", "t1", meta={"chunk_id": "c1"})
        cache.put("k2", "translate", "m", "v2", "t1b", meta={"chunk_id": "c1"})
        cache.put("k3", "translate", "m", "v1", "t2", meta={"chunk_id": "c2"})
        overviews = get_stage_overviews(cache, 2)
        translate_ov = next(o for o in overviews if o.stage == "translate")
        assert translate_ov.count == 2  # distinct chunks, not 3 rows
        cache.close()

    def test_assembly_map_with_reruns(self, tmp_path: Path):
        """Assembly map counts distinct chunks, not rows."""
        cache = Cache(tmp_path / "reruns.sqlite")
        cache.put("k1", "translate", "m", "v1", "t1", meta={"chunk_id": "c1"})
        cache.put("k2", "translate", "m", "v2", "t1b", meta={"chunk_id": "c1"})
        cache.put("k3", "translate", "m", "v1", "t2", meta={"chunk_id": "c2"})
        amap = get_assembly_map(cache, 2)
        assert amap == {"translate": 2}  # 2 unique chunks
        cache.close()
