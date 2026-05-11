"""Simple SQLite cache keyed by sha256(inputs).

Iteration 1 usage: cache the whole-book glossary extraction so re-runs
with the same book + model + prompt_version don't cost money.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS cache (
    key TEXT PRIMARY KEY,
    chunk_id TEXT,
    stage TEXT NOT NULL,
    model TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    input_tokens INTEGER,
    output_tokens INTEGER,
    cost_usd REAL,
    created_at TEXT DEFAULT (datetime('now')),
    content TEXT NOT NULL,
    meta_json TEXT
);
"""

SCHEMA_PREFERENCES = """
CREATE TABLE IF NOT EXISTS chunk_preferences (
    chunk_id TEXT PRIMARY KEY,
    preferred_stage TEXT NOT NULL,
    reason TEXT,
    updated_at TEXT DEFAULT (datetime('now'))
);
"""

SCHEMA_PIPELINE_META = """
CREATE TABLE IF NOT EXISTS pipeline_meta (
    key TEXT PRIMARY KEY,
    value_json TEXT NOT NULL,
    updated_at TEXT DEFAULT (datetime('now'))
);
"""

# Waterfall order: later stages take priority over earlier ones.
STAGE_WATERFALL = ["translate", "reflect", "proofread", "style", "verify"]


@dataclass
class CachedEntry:
    key: str
    content: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    meta: dict[str, Any]


@dataclass
class ChunkStageInfo:
    """Summary of what's in the cache for a given chunk_id and stage."""
    chunk_id: str
    stage: str
    model: str
    cost_usd: float
    input_tokens: int
    output_tokens: int
    created_at: str
    content: str


@dataclass
class Preference:
    chunk_id: str
    preferred_stage: str
    reason: str | None
    updated_at: str


class Cache:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False lets us share the connection across
        # worker threads; our Translator serialises cache put/get with
        # its own lock, so concurrent access is safe.
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        """Create tables and migrate schema if needed."""
        # Create tables if they don't exist
        self.conn.executescript(SCHEMA)
        self.conn.executescript(SCHEMA_PREFERENCES)
        self.conn.executescript(SCHEMA_PIPELINE_META)

        # Migration: add chunk_id column if missing (legacy DBs)
        cols = {
            row[1]
            for row in self.conn.execute("PRAGMA table_info(cache)").fetchall()
        }
        if "chunk_id" not in cols:
            self.conn.execute("ALTER TABLE cache ADD COLUMN chunk_id TEXT")
            # Try to backfill chunk_id from meta_json
            rows = self.conn.execute(
                "SELECT key, meta_json FROM cache WHERE meta_json IS NOT NULL"
            ).fetchall()
            for key, meta_json in rows:
                try:
                    meta = json.loads(meta_json)
                    if isinstance(meta, dict):
                        cid = meta.get("chunk_id")
                        if cid:
                            self.conn.execute(
                                "UPDATE cache SET chunk_id = ? WHERE key = ?",
                                (cid, key),
                            )
                except (json.JSONDecodeError, TypeError):
                    pass
            self.conn.commit()

        # Ensure index exists (after migration so column is guaranteed present)
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_cache_chunk_stage "
            "ON cache(chunk_id, stage)"
        )
        self.conn.commit()

    @staticmethod
    def make_key(*parts: str) -> str:
        h = hashlib.sha256()
        for p in parts:
            h.update(p.encode("utf-8"))
            h.update(b"\x1f")  # unit separator
        return h.hexdigest()

    def get(self, key: str) -> CachedEntry | None:
        row = self.conn.execute(
            "SELECT key, content, input_tokens, output_tokens, cost_usd, meta_json "
            "FROM cache WHERE key = ?",
            (key,),
        ).fetchone()
        if row is None:
            return None
        return CachedEntry(
            key=row[0],
            content=row[1],
            input_tokens=row[2] or 0,
            output_tokens=row[3] or 0,
            cost_usd=row[4] or 0.0,
            meta=json.loads(row[5]) if row[5] else {},
        )

    def put(
        self,
        key: str,
        stage: str,
        model: str,
        prompt_version: str,
        content: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost_usd: float = 0.0,
        meta: dict[str, Any] | None = None,
    ) -> None:
        chunk_id = meta.get("chunk_id") if meta else None
        self.conn.execute(
            "INSERT OR REPLACE INTO cache "
            "(key, chunk_id, stage, model, prompt_version, input_tokens, "
            " output_tokens, cost_usd, content, meta_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                key,
                chunk_id,
                stage,
                model,
                prompt_version,
                input_tokens,
                output_tokens,
                cost_usd,
                content,
                json.dumps(meta) if meta else None,
            ),
        )
        self.conn.commit()

    # ------------------------------------------------------------------
    # Multi-pass inspection methods
    # ------------------------------------------------------------------

    def list_stages(self) -> dict[str, int]:
        """Return {stage: count} for all entries in the cache."""
        rows = self.conn.execute(
            "SELECT stage, COUNT(*) FROM cache GROUP BY stage"
        ).fetchall()
        return {row[0]: row[1] for row in rows}

    def stage_stats(self, stage: str) -> dict[str, Any]:
        """Aggregate stats for a stage: count, total cost, total tokens."""
        row = self.conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(cost_usd), 0), "
            "COALESCE(SUM(input_tokens), 0), COALESCE(SUM(output_tokens), 0) "
            "FROM cache WHERE stage = ?",
            (stage,),
        ).fetchone()
        return {
            "count": row[0],
            "cost_usd": row[1],
            "input_tokens": row[2],
            "output_tokens": row[3],
        }

    def get_chunk_stages(self, chunk_id: str) -> list[ChunkStageInfo]:
        """Get all cached stages for a specific chunk."""
        rows = self.conn.execute(
            "SELECT stage, model, cost_usd, input_tokens, output_tokens, "
            "created_at, content "
            "FROM cache WHERE chunk_id = ? ORDER BY created_at",
            (chunk_id,),
        ).fetchall()
        results = []
        for row in rows:
            results.append(ChunkStageInfo(
                chunk_id=chunk_id,
                stage=row[0],
                model=row[1],
                cost_usd=row[2] or 0.0,
                input_tokens=row[3] or 0,
                output_tokens=row[4] or 0,
                created_at=row[5] or "",
                content=row[6],
            ))
        return results

    def get_stages_for_chunks_bulk(
        self, chunk_ids: list[str]
    ) -> dict[str, list[ChunkStageInfo]]:
        """Get all cached stages for multiple chunks in one query.

        Returns {chunk_id: [ChunkStageInfo, ...]} for each chunk that
        has at least one entry. More efficient than calling
        get_chunk_stages() per chunk.
        """
        if not chunk_ids:
            return {}

        batch_size = 500
        all_rows: list[tuple] = []

        for i in range(0, len(chunk_ids), batch_size):
            batch = chunk_ids[i:i + batch_size]
            placeholders = ",".join("?" * len(batch))
            rows = self.conn.execute(
                f"SELECT chunk_id, stage, model, cost_usd, input_tokens, "
                f"output_tokens, created_at, content "
                f"FROM cache WHERE chunk_id IN ({placeholders}) "
                f"AND chunk_id IS NOT NULL "
                f"ORDER BY chunk_id, created_at",
                batch,
            ).fetchall()
            all_rows.extend(rows)

        result: dict[str, list[ChunkStageInfo]] = {}
        for row in all_rows:
            cid = row[0]
            info = ChunkStageInfo(
                chunk_id=cid,
                stage=row[1],
                model=row[2],
                cost_usd=row[3] or 0.0,
                input_tokens=row[4] or 0,
                output_tokens=row[5] or 0,
                created_at=row[6] or "",
                content=row[7],
            )
            result.setdefault(cid, []).append(info)

        return result

    def get_all_chunk_ids_for_stage(self, stage: str) -> list[str]:
        """Get all chunk_ids that have entries for a given stage.

        Only returns chunks that have a non-NULL chunk_id. Legacy entries
        without chunk_id won't appear here but are still counted by
        count_stage().
        """
        rows = self.conn.execute(
            "SELECT DISTINCT chunk_id FROM cache "
            "WHERE stage = ? AND chunk_id IS NOT NULL",
            (stage,),
        ).fetchall()
        return [r[0] for r in rows]

    def count_stage(self, stage: str) -> int:
        """Count total rows for a stage. May overcount if reruns exist."""
        row = self.conn.execute(
            "SELECT COUNT(*) FROM cache WHERE stage = ?",
            (stage,),
        ).fetchone()
        return row[0] if row else 0

    def count_distinct_chunks(self, stage: str) -> int:
        """Count unique tracked chunks for a stage (chunk_id IS NOT NULL).

        Only counts entries with a known chunk_id. Legacy entries (NULL
        chunk_id) are excluded — use count_legacy_rows() to get those
        separately.
        """
        row = self.conn.execute(
            "SELECT COUNT(DISTINCT chunk_id) FROM cache "
            "WHERE stage = ? AND chunk_id IS NOT NULL",
            (stage,),
        ).fetchone()
        return row[0] or 0

    def count_legacy_rows(self, stage: str) -> int:
        """Count rows with NULL chunk_id for a stage (legacy/untracked)."""
        row = self.conn.execute(
            "SELECT COUNT(*) FROM cache "
            "WHERE stage = ? AND chunk_id IS NULL",
            (stage,),
        ).fetchone()
        return row[0] or 0

    def count_chunks_total(self, stage: str) -> int:
        """Best-effort total: distinct tracked + legacy rows.

        For fully tracked data this is exact. For mixed data it may
        overcount legacy rows (if reruns exist without chunk_id), but
        never undercounts.
        """
        return self.count_distinct_chunks(stage) + self.count_legacy_rows(stage)

    def get_judge_scores(self) -> list[dict[str, Any]]:
        """Get all judge results as parsed dicts with chunk_id, score, issues."""
        rows = self.conn.execute(
            "SELECT chunk_id, content FROM cache WHERE stage = 'judge'"
        ).fetchall()
        results = []
        decoder = json.JSONDecoder()
        for chunk_id, content in rows:
            # Judge content is JSON: {score: N, issues: [...]}
            # Model often wraps in code fences and/or appends explanation.
            parsed: dict[str, Any]
            text = (content or "").strip()

            # Strip code fences if present
            if text.startswith("```"):
                lines = text.splitlines()
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].startswith("```"):
                    lines = lines[:-1]
                text = "\n".join(lines).strip()

            try:
                parsed = json.loads(text)
            except (json.JSONDecodeError, TypeError):
                # Trailing text after valid JSON — use raw_decode
                try:
                    parsed, _ = decoder.raw_decode(text)
                except (json.JSONDecodeError, TypeError, ValueError):
                    parsed = {"score": 0, "issues": ["parse error"]}
            parsed["chunk_id"] = chunk_id or ""
            results.append(parsed)
        return results

    # ------------------------------------------------------------------
    # Chunk preferences
    # ------------------------------------------------------------------

    def get_preference(self, chunk_id: str) -> Preference | None:
        row = self.conn.execute(
            "SELECT chunk_id, preferred_stage, reason, updated_at "
            "FROM chunk_preferences WHERE chunk_id = ?",
            (chunk_id,),
        ).fetchone()
        if row is None:
            return None
        return Preference(
            chunk_id=row[0],
            preferred_stage=row[1],
            reason=row[2],
            updated_at=row[3] or "",
        )

    def set_preference(self, chunk_id: str, stage: str, reason: str | None = None) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO chunk_preferences "
            "(chunk_id, preferred_stage, reason, updated_at) "
            "VALUES (?, ?, ?, datetime('now'))",
            (chunk_id, stage, reason),
        )
        self.conn.commit()

    def reset_preference(self, chunk_id: str) -> None:
        self.conn.execute(
            "DELETE FROM chunk_preferences WHERE chunk_id = ?",
            (chunk_id,),
        )
        self.conn.commit()

    def list_preferences(self) -> list[Preference]:
        rows = self.conn.execute(
            "SELECT chunk_id, preferred_stage, reason, updated_at "
            "FROM chunk_preferences ORDER BY chunk_id"
        ).fetchall()
        return [
            Preference(chunk_id=r[0], preferred_stage=r[1], reason=r[2], updated_at=r[3] or "")
            for r in rows
        ]

    def resolve_stage_for_chunk(self, chunk_id: str) -> str | None:
        """Determine which stage to use for a chunk: preference > waterfall.

        Returns None if no cached content exists for this chunk.
        """
        # Single query to get all stages for this chunk
        stages = self.get_chunk_stages(chunk_id)
        if not stages:
            return None
        stage_names = {s.stage for s in stages}

        # Check preference first
        pref = self.get_preference(chunk_id)
        if pref and pref.preferred_stage in stage_names:
            return pref.preferred_stage

        # Waterfall: pick the latest available stage
        for stage in reversed(STAGE_WATERFALL):
            if stage in stage_names:
                return stage
        return None

    def resolve_stages_bulk(self, chunk_ids: list[str]) -> dict[str, str]:
        """Resolve the assembly stage for multiple chunks in bulk.

        Returns {chunk_id: stage} for each chunk that has cached content.
        More efficient than calling resolve_stage_for_chunk per chunk.
        Handles SQLite variable limit by batching.
        """
        if not chunk_ids:
            return {}

        # SQLite has a variable limit (commonly 999). Batch to stay safe.
        batch_size = 500
        all_rows: list[tuple[str, str]] = []
        all_pref_rows: list[tuple[str, str]] = []

        for i in range(0, len(chunk_ids), batch_size):
            batch = chunk_ids[i:i + batch_size]
            placeholders = ",".join("?" * len(batch))

            rows = self.conn.execute(
                f"SELECT chunk_id, stage FROM cache "
                f"WHERE chunk_id IN ({placeholders}) AND chunk_id IS NOT NULL",
                batch,
            ).fetchall()
            all_rows.extend(rows)

            pref_rows = self.conn.execute(
                f"SELECT chunk_id, preferred_stage FROM chunk_preferences "
                f"WHERE chunk_id IN ({placeholders})",
                batch,
            ).fetchall()
            all_pref_rows.extend(pref_rows)

        # Build {chunk_id: set(stages)}
        chunk_stages: dict[str, set[str]] = {}
        for cid, stage in all_rows:
            chunk_stages.setdefault(cid, set()).add(stage)

        # Build preferences map
        prefs = {r[0]: r[1] for r in all_pref_rows}

        # Resolve each chunk
        result: dict[str, str] = {}
        for cid in chunk_ids:
            stages = chunk_stages.get(cid)
            if not stages:
                continue
            # Preference takes priority if the stage exists
            pref_stage = prefs.get(cid)
            if pref_stage and pref_stage in stages:
                result[cid] = pref_stage
                continue
            # Waterfall
            for stage in reversed(STAGE_WATERFALL):
                if stage in stages:
                    result[cid] = stage
                    break

        return result

    # ------------------------------------------------------------------
    # Pipeline metadata (chunker params, etc.)
    # ------------------------------------------------------------------

    def set_meta(self, key: str, value: Any) -> None:
        """Store a pipeline metadata value (JSON-serializable)."""
        self.conn.execute(
            "INSERT OR REPLACE INTO pipeline_meta "
            "(key, value_json, updated_at) VALUES (?, ?, datetime('now'))",
            (key, json.dumps(value)),
        )
        self.conn.commit()

    def get_meta(self, key: str) -> Any | None:
        """Retrieve a pipeline metadata value. Returns None if not set."""
        row = self.conn.execute(
            "SELECT value_json FROM pipeline_meta WHERE key = ?",
            (key,),
        ).fetchone()
        if row is None:
            return None
        return json.loads(row[0])

    def close(self) -> None:
        self.conn.close()
