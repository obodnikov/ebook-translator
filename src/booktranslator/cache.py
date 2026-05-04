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


@dataclass
class CachedEntry:
    key: str
    content: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    meta: dict[str, Any]


class Cache:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False lets us share the connection across
        # worker threads; our Translator serialises cache put/get with
        # its own lock, so concurrent access is safe.
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.execute(SCHEMA)
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
        self.conn.execute(
            "INSERT OR REPLACE INTO cache "
            "(key, stage, model, prompt_version, input_tokens, output_tokens, "
            " cost_usd, content, meta_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                key,
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

    def close(self) -> None:
        self.conn.close()
