"""Status inspection logic for btrans status command.

Provides read-only queries over the cache to give the operator
full visibility into translation passes, scores, and assembly state.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .cache import STAGE_WATERFALL, Cache


@dataclass
class StageOverview:
    """Summary of one stage's presence in the cache."""
    stage: str
    count: int
    total_chunks: int
    cost_usd: float
    input_tokens: int
    output_tokens: int


@dataclass
class ScoreEntry:
    """One chunk's judge result."""
    chunk_id: str
    score: int
    issues: list[str]
    reflected: bool


@dataclass
class AssemblyChoice:
    """Which stage will be used for a chunk at assembly time."""
    chunk_id: str
    stage: str
    reason: str  # "waterfall", "preference"


@dataclass
class StatusReport:
    """Full status report for a book's work directory."""
    work_dir: Path
    total_chunks: int
    stages: list[StageOverview]
    scores: list[ScoreEntry] | None  # None if no judge data
    preferences: list[dict[str, str]]
    assembly_map: dict[str, int]  # stage -> count of chunks using it
    total_cost_usd: float


def get_total_chunks(cache: Cache) -> int:
    """Count unique translated chunks in the cache.

    Uses count_chunks_total: exact for tracked data, best-effort for mixed.
    """
    return cache.count_chunks_total("translate")


def get_stage_overviews(cache: Cache, total_chunks: int) -> list[StageOverview]:
    """Build overview for each stage in the waterfall + judge."""
    overviews = []

    # Always show all stages in order, plus judge
    for stage in ["translate", "judge", "reflect", "proofread", "style", "verify"]:
        count = cache.count_chunks_total(stage)
        if count > 0:
            stats = cache.stage_stats(stage)
            overviews.append(StageOverview(
                stage=stage,
                count=count,
                total_chunks=total_chunks,
                cost_usd=stats["cost_usd"],
                input_tokens=stats["input_tokens"],
                output_tokens=stats["output_tokens"],
            ))
        else:
            overviews.append(StageOverview(
                stage=stage,
                count=0,
                total_chunks=total_chunks,
                cost_usd=0.0,
                input_tokens=0,
                output_tokens=0,
            ))

    return overviews


def get_scores(cache: Cache) -> list[ScoreEntry] | None:
    """Get judge scores with reflection status. Returns None if no judge data."""
    judge_results = cache.get_judge_scores()
    if not judge_results:
        return None

    reflected_ids = set(cache.get_all_chunk_ids_for_stage("reflect"))

    entries = []
    for result in judge_results:
        chunk_id = result.get("chunk_id") or ""

        # Normalize score: coerce to int, fallback to 0
        raw_score = result.get("score", 0)
        try:
            score = int(raw_score)
        except (TypeError, ValueError):
            score = 0

        # Normalize issues: coerce to list[str]
        raw_issues = result.get("issues")
        if raw_issues is None:
            issues: list[str] = []
        elif isinstance(raw_issues, str):
            issues = [raw_issues]
        elif isinstance(raw_issues, list):
            issues = [str(i) for i in raw_issues]
        else:
            issues = [str(raw_issues)]

        entries.append(ScoreEntry(
            chunk_id=chunk_id,
            score=score,
            issues=issues,
            reflected=chunk_id in reflected_ids,
        ))

    # Sort by score ascending (worst first), then by chunk_id
    entries.sort(key=lambda e: (e.score, e.chunk_id))
    return entries


def get_assembly_map(cache: Cache, total_chunks: int) -> dict[str, int]:
    """For each chunk, determine which stage will be used, return counts.

    Tracked chunks (with chunk_id) get proper waterfall resolution.
    Legacy chunks (NULL chunk_id) are counted as 'translate' (approximate).
    """
    translate_ids = cache.get_all_chunk_ids_for_stage("translate")

    # Bulk resolve for tracked chunks
    resolved = cache.resolve_stages_bulk(translate_ids)
    stage_counts: dict[str, int] = {}
    for stage in resolved.values():
        stage_counts[stage] = stage_counts.get(stage, 0) + 1

    # Account for legacy rows without chunk_id
    legacy_count = cache.count_legacy_rows("translate")
    if legacy_count > 0:
        stage_counts["translate"] = stage_counts.get("translate", 0) + legacy_count

    return stage_counts


def build_status_report(cache: Cache, work_dir: Path) -> StatusReport:
    """Build a complete status report for a work directory."""
    total_chunks = get_total_chunks(cache)
    stages = get_stage_overviews(cache, total_chunks)
    scores = get_scores(cache)
    prefs = cache.list_preferences()
    assembly_map = get_assembly_map(cache, total_chunks)

    total_cost = sum(s.cost_usd for s in stages)

    return StatusReport(
        work_dir=work_dir,
        total_chunks=total_chunks,
        stages=stages,
        scores=scores,
        preferences=[
            {"chunk_id": p.chunk_id, "stage": p.preferred_stage, "reason": p.reason or ""}
            for p in prefs
        ],
        assembly_map=assembly_map,
        total_cost_usd=total_cost,
    )


def get_chunk_diff(cache: Cache, chunk_id: str, stages_filter: list[str] | None = None) -> list[dict[str, Any]]:
    """Get all stage contents for a chunk, optionally filtered to specific stages."""
    all_stages = cache.get_chunk_stages(chunk_id)

    if stages_filter:
        all_stages = [s for s in all_stages if s.stage in stages_filter]

    return [
        {
            "stage": s.stage,
            "model": s.model,
            "created_at": s.created_at,
            "content": s.content,
        }
        for s in all_stages
    ]
