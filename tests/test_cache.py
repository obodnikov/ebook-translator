"""Tests for cache.py: schema migration, chunk_preferences, waterfall logic."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from booktranslator.cache import STAGE_WATERFALL, Cache


@pytest.fixture
def fresh_cache(tmp_path: Path) -> Cache:
    """A brand-new cache with the full schema."""
    return Cache(tmp_path / "cache.sqlite")


@pytest.fixture
def legacy_db(tmp_path: Path) -> Path:
    """Create a legacy SQLite DB without chunk_id column or preferences table."""
    db_path = tmp_path / "legacy.sqlite"
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE cache (
            key TEXT PRIMARY KEY,
            stage TEXT NOT NULL,
            model TEXT NOT NULL,
            prompt_version TEXT NOT NULL,
            input_tokens INTEGER,
            output_tokens INTEGER,
            cost_usd REAL,
            created_at TEXT DEFAULT (datetime('now')),
            content TEXT NOT NULL,
            meta_json TEXT
        )
    """)
    # Insert some legacy rows (no chunk_id column, no meta_json)
    conn.execute(
        "INSERT INTO cache (key, stage, model, prompt_version, content) VALUES (?, ?, ?, ?, ?)",
        ("abc123", "translate", "anthropic/claude-sonnet-4.6", "v1", "Привет мир"),
    )
    conn.execute(
        "INSERT INTO cache (key, stage, model, prompt_version, content) VALUES (?, ?, ?, ?, ?)",
        ("def456", "translate", "anthropic/claude-sonnet-4.6", "v1", "Второй чанк"),
    )
    # One with meta_json containing chunk_id (simulates partially migrated data)
    conn.execute(
        "INSERT INTO cache (key, stage, model, prompt_version, content, meta_json) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (
            "ghi789",
            "translate",
            "anthropic/claude-sonnet-4.6",
            "v1",
            "Третий",
            json.dumps({"chunk_id": "ch01_c03"}),
        ),
    )
    conn.commit()
    conn.close()
    return db_path


# ---------------------------------------------------------------------------
# Schema migration tests
# ---------------------------------------------------------------------------


class TestSchemaMigration:
    def test_legacy_db_gets_chunk_id_column(self, legacy_db: Path):
        """Opening a legacy DB adds the chunk_id column."""
        cache = Cache(legacy_db)
        cols = {row[1] for row in cache.conn.execute("PRAGMA table_info(cache)").fetchall()}
        assert "chunk_id" in cols
        cache.close()

    def test_legacy_db_creates_preferences_table(self, legacy_db: Path):
        """Opening a legacy DB creates chunk_preferences table."""
        cache = Cache(legacy_db)
        tables = {
            row[0]
            for row in cache.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "chunk_preferences" in tables
        cache.close()

    def test_legacy_db_backfills_chunk_id_from_meta(self, legacy_db: Path):
        """Entries with chunk_id in meta_json get backfilled."""
        cache = Cache(legacy_db)
        row = cache.conn.execute("SELECT chunk_id FROM cache WHERE key = 'ghi789'").fetchone()
        assert row[0] == "ch01_c03"
        cache.close()

    def test_legacy_db_null_chunk_id_for_entries_without_meta(self, legacy_db: Path):
        """Entries without meta_json keep chunk_id as NULL."""
        cache = Cache(legacy_db)
        row = cache.conn.execute("SELECT chunk_id FROM cache WHERE key = 'abc123'").fetchone()
        assert row[0] is None
        cache.close()

    def test_migration_is_idempotent(self, legacy_db: Path):
        """Opening the same DB twice doesn't crash or duplicate data."""
        cache1 = Cache(legacy_db)
        cache1.close()
        cache2 = Cache(legacy_db)
        count = cache2.conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
        assert count == 3
        cache2.close()

    def test_fresh_db_has_full_schema(self, fresh_cache: Cache):
        """A new DB has chunk_id column and preferences table from the start."""
        cols = {row[1] for row in fresh_cache.conn.execute("PRAGMA table_info(cache)").fetchall()}
        assert "chunk_id" in cols
        tables = {
            row[0]
            for row in fresh_cache.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "chunk_preferences" in tables
        fresh_cache.close()

    def test_index_exists_after_migration(self, legacy_db: Path):
        """The idx_cache_chunk_stage index is created."""
        cache = Cache(legacy_db)
        indexes = {row[1] for row in cache.conn.execute("PRAGMA index_list(cache)").fetchall()}
        assert "idx_cache_chunk_stage" in indexes
        cache.close()

    def test_malformed_meta_json_does_not_crash_migration(self, tmp_path: Path):
        """Entries with invalid JSON in meta_json don't break migration."""
        db_path = tmp_path / "bad_meta.sqlite"
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE cache (
                key TEXT PRIMARY KEY,
                stage TEXT NOT NULL,
                model TEXT NOT NULL,
                prompt_version TEXT NOT NULL,
                input_tokens INTEGER,
                output_tokens INTEGER,
                cost_usd REAL,
                created_at TEXT DEFAULT (datetime('now')),
                content TEXT NOT NULL,
                meta_json TEXT
            )
        """)
        conn.execute(
            "INSERT INTO cache (key, stage, model, prompt_version, content, meta_json) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("bad1", "translate", "m", "v1", "text", "not valid json {{{"),
        )
        conn.execute(
            "INSERT INTO cache (key, stage, model, prompt_version, content, meta_json) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("bad2", "translate", "m", "v1", "text", "null"),
        )
        conn.commit()
        conn.close()

        # Should not raise
        cache = Cache(db_path)
        assert cache.count_stage("translate") == 2
        cache.close()


# ---------------------------------------------------------------------------
# Chunk preferences tests
# ---------------------------------------------------------------------------


class TestChunkPreferences:
    def test_set_and_get_preference(self, fresh_cache: Cache):
        cache = fresh_cache
        cache.set_preference("ch01_c01", "translate", "reflect was worse")
        pref = cache.get_preference("ch01_c01")
        assert pref is not None
        assert pref.preferred_stage == "translate"
        assert pref.reason == "reflect was worse"

    def test_get_nonexistent_preference(self, fresh_cache: Cache):
        assert fresh_cache.get_preference("nonexistent") is None

    def test_reset_preference(self, fresh_cache: Cache):
        cache = fresh_cache
        cache.set_preference("ch01_c01", "translate", "test")
        cache.reset_preference("ch01_c01")
        assert cache.get_preference("ch01_c01") is None

    def test_list_preferences(self, fresh_cache: Cache):
        cache = fresh_cache
        cache.set_preference("ch02_c01", "style", None)
        cache.set_preference("ch01_c01", "translate", "reason")
        prefs = cache.list_preferences()
        assert len(prefs) == 2
        # Sorted by chunk_id
        assert prefs[0].chunk_id == "ch01_c01"
        assert prefs[1].chunk_id == "ch02_c01"

    def test_preference_overwrite(self, fresh_cache: Cache):
        cache = fresh_cache
        cache.set_preference("ch01_c01", "translate", "first")
        cache.set_preference("ch01_c01", "reflect", "changed mind")
        pref = cache.get_preference("ch01_c01")
        assert pref.preferred_stage == "reflect"
        assert pref.reason == "changed mind"


# ---------------------------------------------------------------------------
# Waterfall resolution tests
# ---------------------------------------------------------------------------


class TestWaterfallResolution:
    def test_single_stage_resolves_to_itself(self, fresh_cache: Cache):
        cache = fresh_cache
        cache.put("k1", "translate", "m", "v1", "text", meta={"chunk_id": "ch01_c01"})
        assert cache.resolve_stage_for_chunk("ch01_c01") == "translate"

    def test_later_stage_wins(self, fresh_cache: Cache):
        cache = fresh_cache
        cache.put("k1", "translate", "m", "v1", "v1", meta={"chunk_id": "ch01_c01"})
        cache.put("k2", "reflect", "m", "v1", "v2", meta={"chunk_id": "ch01_c01"})
        assert cache.resolve_stage_for_chunk("ch01_c01") == "reflect"

    def test_full_waterfall(self, fresh_cache: Cache):
        cache = fresh_cache
        for stage in STAGE_WATERFALL:
            cache.put(
                f"k_{stage}", stage, "m", "v1", f"text_{stage}", meta={"chunk_id": "ch01_c01"}
            )
        assert cache.resolve_stage_for_chunk("ch01_c01") == "verify"

    def test_preference_overrides_waterfall(self, fresh_cache: Cache):
        cache = fresh_cache
        cache.put("k1", "translate", "m", "v1", "v1", meta={"chunk_id": "ch01_c01"})
        cache.put("k2", "reflect", "m", "v1", "v2", meta={"chunk_id": "ch01_c01"})
        cache.put("k3", "style", "m", "v1", "v3", meta={"chunk_id": "ch01_c01"})
        # Without preference: style wins (latest in waterfall)
        assert cache.resolve_stage_for_chunk("ch01_c01") == "style"
        # With preference: translate wins
        cache.set_preference("ch01_c01", "translate")
        assert cache.resolve_stage_for_chunk("ch01_c01") == "translate"

    def test_preference_for_nonexistent_stage_falls_back(self, fresh_cache: Cache):
        """If preferred stage doesn't exist in cache, fall back to waterfall."""
        cache = fresh_cache
        cache.put("k1", "translate", "m", "v1", "text", meta={"chunk_id": "ch01_c01"})
        cache.set_preference("ch01_c01", "verify")  # verify not in cache
        # Should fall back to waterfall (translate is the only one)
        assert cache.resolve_stage_for_chunk("ch01_c01") == "translate"

    def test_nonexistent_chunk_returns_none(self, fresh_cache: Cache):
        assert fresh_cache.resolve_stage_for_chunk("nonexistent") is None


# ---------------------------------------------------------------------------
# Query method tests
# ---------------------------------------------------------------------------


class TestQueryMethods:
    def test_count_stage(self, fresh_cache: Cache):
        cache = fresh_cache
        cache.put("k1", "translate", "m", "v1", "a", meta={"chunk_id": "c1"})
        cache.put("k2", "translate", "m", "v1", "b", meta={"chunk_id": "c2"})
        cache.put("k3", "judge", "m", "v1", "{}", meta={"chunk_id": "c1"})
        assert cache.count_stage("translate") == 2
        assert cache.count_stage("judge") == 1
        assert cache.count_stage("reflect") == 0

    def test_get_all_chunk_ids_for_stage(self, fresh_cache: Cache):
        cache = fresh_cache
        cache.put("k1", "translate", "m", "v1", "a", meta={"chunk_id": "c1"})
        cache.put("k2", "translate", "m", "v1", "b", meta={"chunk_id": "c2"})
        cache.put("k3", "translate", "m", "v1", "c")  # no meta -> no chunk_id
        ids = set(cache.get_all_chunk_ids_for_stage("translate"))
        assert ids == {"c1", "c2"}

    def test_get_chunk_stages(self, fresh_cache: Cache):
        cache = fresh_cache
        cache.put("k1", "translate", "model-a", "v1", "text1", meta={"chunk_id": "c1"})
        cache.put("k2", "reflect", "model-b", "v1", "text2", meta={"chunk_id": "c1"})
        stages = cache.get_chunk_stages("c1")
        assert len(stages) == 2
        stage_names = {s.stage for s in stages}
        assert stage_names == {"translate", "reflect"}

    def test_get_judge_scores(self, fresh_cache: Cache):
        cache = fresh_cache
        cache.put(
            "j1",
            "judge",
            "m",
            "v1",
            json.dumps({"score": 4, "issues": ["minor"]}),
            meta={"chunk_id": "c1"},
        )
        cache.put(
            "j2",
            "judge",
            "m",
            "v1",
            json.dumps({"score": 2, "issues": ["bad", "worse"]}),
            meta={"chunk_id": "c2"},
        )
        scores = cache.get_judge_scores()
        assert len(scores) == 2
        by_id = {s["chunk_id"]: s for s in scores}
        assert by_id["c1"]["score"] == 4
        assert by_id["c2"]["issues"] == ["bad", "worse"]

    def test_list_stages(self, fresh_cache: Cache):
        cache = fresh_cache
        cache.put("k1", "translate", "m", "v1", "a", meta={"chunk_id": "c1"})
        cache.put("k2", "translate", "m", "v1", "b", meta={"chunk_id": "c2"})
        cache.put("k3", "judge", "m", "v1", "{}", meta={"chunk_id": "c1"})
        stages = cache.list_stages()
        assert stages == {"translate": 2, "judge": 1}


# ---------------------------------------------------------------------------
# Mixed legacy/new data tests
# ---------------------------------------------------------------------------


class TestMixedData:
    """Tests for caches with both legacy (NULL chunk_id) and new entries."""

    def test_count_stage_includes_both(self, fresh_cache: Cache):
        """count_stage counts all entries regardless of chunk_id."""
        cache = fresh_cache
        cache.put("k1", "translate", "m", "v1", "text1", meta={"chunk_id": "c1"})
        cache.put("k2", "translate", "m", "v1", "text2")  # no chunk_id
        cache.put("k3", "translate", "m", "v1", "text3")  # no chunk_id
        assert cache.count_stage("translate") == 3

    def test_get_all_chunk_ids_excludes_null(self, fresh_cache: Cache):
        """get_all_chunk_ids_for_stage only returns non-NULL chunk_ids."""
        cache = fresh_cache
        cache.put("k1", "translate", "m", "v1", "text1", meta={"chunk_id": "c1"})
        cache.put("k2", "translate", "m", "v1", "text2")  # no chunk_id
        ids = cache.get_all_chunk_ids_for_stage("translate")
        assert ids == ["c1"]

    def test_resolve_stage_only_works_for_tracked_chunks(self, fresh_cache: Cache):
        """resolve_stage_for_chunk returns None for chunks not in DB."""
        cache = fresh_cache
        cache.put("k1", "translate", "m", "v1", "text1")  # no chunk_id
        assert cache.resolve_stage_for_chunk("anything") is None


# ---------------------------------------------------------------------------
# Duplicate rows / rerun tests
# ---------------------------------------------------------------------------


class TestDuplicateRows:
    """Tests for caches with multiple entries per chunk (reruns)."""

    def test_count_distinct_chunks_deduplicates(self, fresh_cache: Cache):
        """Multiple translate rows for same chunk_id count as one."""
        cache = fresh_cache
        # Same chunk, different keys (different model/prompt versions)
        cache.put("k1", "translate", "sonnet", "v1", "text1", meta={"chunk_id": "c1"})
        cache.put("k2", "translate", "sonnet", "v2", "text1b", meta={"chunk_id": "c1"})
        cache.put("k3", "translate", "sonnet", "v1", "text2", meta={"chunk_id": "c2"})
        assert cache.count_distinct_chunks("translate") == 2

    def test_count_distinct_chunks_legacy_rows_not_counted(self, fresh_cache: Cache):
        """count_distinct_chunks only counts tracked entries (non-NULL chunk_id)."""
        cache = fresh_cache
        cache.put("k1", "translate", "m", "v1", "text1")  # legacy
        cache.put("k2", "translate", "m", "v1", "text2")  # legacy
        cache.put("k3", "translate", "m", "v1", "text3", meta={"chunk_id": "c1"})
        # Only 1 tracked chunk
        assert cache.count_distinct_chunks("translate") == 1
        # 2 legacy rows
        assert cache.count_legacy_rows("translate") == 2
        # Total (best-effort): 1 + 2 = 3
        assert cache.count_chunks_total("translate") == 3

    def test_count_stage_counts_all_rows(self, fresh_cache: Cache):
        """count_stage is raw row count (may overcount with reruns)."""
        cache = fresh_cache
        cache.put("k1", "translate", "sonnet", "v1", "text1", meta={"chunk_id": "c1"})
        cache.put("k2", "translate", "sonnet", "v2", "text1b", meta={"chunk_id": "c1"})
        assert cache.count_stage("translate") == 2  # raw rows
        assert cache.count_distinct_chunks("translate") == 1  # unique chunks

    def test_resolve_stages_bulk(self, fresh_cache: Cache):
        """Bulk resolution works correctly."""
        cache = fresh_cache
        cache.put("k1", "translate", "m", "v1", "t1", meta={"chunk_id": "c1"})
        cache.put("k2", "translate", "m", "v1", "t2", meta={"chunk_id": "c2"})
        cache.put("k3", "reflect", "m", "v1", "r1", meta={"chunk_id": "c1"})
        resolved = cache.resolve_stages_bulk(["c1", "c2"])
        assert resolved == {"c1": "reflect", "c2": "translate"}

    def test_resolve_stages_bulk_with_preferences(self, fresh_cache: Cache):
        """Bulk resolution respects preferences."""
        cache = fresh_cache
        cache.put("k1", "translate", "m", "v1", "t1", meta={"chunk_id": "c1"})
        cache.put("k2", "reflect", "m", "v1", "r1", meta={"chunk_id": "c1"})
        cache.set_preference("c1", "translate")
        resolved = cache.resolve_stages_bulk(["c1"])
        assert resolved == {"c1": "translate"}

    def test_resolve_stages_bulk_empty(self, fresh_cache: Cache):
        """Bulk resolution with empty list returns empty dict."""
        assert fresh_cache.resolve_stages_bulk([]) == {}


# ---------------------------------------------------------------------------
# Bulk resolution at scale (SQLite variable limit)
# ---------------------------------------------------------------------------


class TestBulkResolutionScale:
    """Tests for resolve_stages_bulk with large datasets."""

    def test_bulk_over_sqlite_limit(self, fresh_cache: Cache):
        """resolve_stages_bulk handles >999 chunk_ids via batching."""
        cache = fresh_cache
        n = 1200  # Above SQLite's 999 variable limit
        chunk_ids = [f"ch{i:04d}_c01" for i in range(n)]

        # Insert translate entries for all chunks
        for i, cid in enumerate(chunk_ids):
            cache.put(f"k{i}", "translate", "m", "v1", f"text{i}", meta={"chunk_id": cid})

        # Add reflect for first 100
        for i in range(100):
            cache.put(
                f"r{i}", "reflect", "m", "v1", f"reflected{i}", meta={"chunk_id": chunk_ids[i]}
            )

        resolved = cache.resolve_stages_bulk(chunk_ids)
        assert len(resolved) == n
        # First 100 should resolve to reflect
        for i in range(100):
            assert resolved[chunk_ids[i]] == "reflect"
        # Rest should resolve to translate
        for i in range(100, n):
            assert resolved[chunk_ids[i]] == "translate"

    def test_bulk_with_preferences_over_limit(self, fresh_cache: Cache):
        """Preferences work correctly with batched bulk resolution."""
        cache = fresh_cache
        n = 1100
        chunk_ids = [f"ch{i:04d}_c01" for i in range(n)]

        for i, cid in enumerate(chunk_ids):
            cache.put(f"k{i}", "translate", "m", "v1", f"text{i}", meta={"chunk_id": cid})
            cache.put(f"r{i}", "reflect", "m", "v1", f"ref{i}", meta={"chunk_id": cid})

        # Set preference for chunk 500 to use translate
        cache.set_preference(chunk_ids[500], "translate")

        resolved = cache.resolve_stages_bulk(chunk_ids)
        assert resolved[chunk_ids[500]] == "translate"  # preference
        assert resolved[chunk_ids[0]] == "reflect"  # waterfall
        assert resolved[chunk_ids[999]] == "reflect"  # waterfall
