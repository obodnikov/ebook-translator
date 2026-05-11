"""Shared helpers for judge/reflect pipeline orchestration.

Extracted to avoid duplication between `translate`, `judge`, and `reflect`
CLI commands. Addresses code review concerns about:
- Consistent provider initialization
- Correct preferred-stage selection (not stale translations)
- O(1) chunk lookup instead of O(n) linear scan
- Chunker config persistence and mismatch detection
"""

from __future__ import annotations

from typing import Any

from .cache import Cache
from .chunker import ChunkSet
from .provider import OpenRouterProvider

# Key used to store chunker parameters in pipeline_meta
CHUNKER_META_KEY = "chunker_params"


class ChunkerConfigMismatchError(Exception):
    """Raised when judge/reflect chunker config differs from translate time."""
    pass


def create_provider() -> OpenRouterProvider:
    """Create the LLM provider through a single factory point.

    All commands should use this instead of calling OpenRouterProvider()
    directly, so provider configuration changes propagate everywhere.
    """
    return OpenRouterProvider()


def save_chunker_params(
    cache: Cache, target_words: int, overlap_paragraphs: int
) -> None:
    """Persist chunker parameters used during translate.

    Called at translate time so that judge/reflect can verify they're
    using the same chunking boundaries.

    If metadata already exists with DIFFERENT values, raises
    ChunkerConfigMismatchError to prevent mixed-cache state. This
    protects against accidentally re-translating with different chunker
    settings into the same workdir.

    If metadata already exists with the SAME values, this is a no-op.
    If no metadata exists yet, writes the new values.
    """
    saved = cache.get_meta(CHUNKER_META_KEY)
    if saved is not None:
        saved_words = saved.get("target_words")
        saved_overlap = saved.get("overlap_paragraphs")
        if saved_words == target_words and saved_overlap == overlap_paragraphs:
            return  # Already stored with same values — no-op
        raise ChunkerConfigMismatchError(
            f"Chunker config conflict: this workdir was previously "
            f"translated with target_words={saved_words}, "
            f"overlap={saved_overlap}; current config has "
            f"target_words={target_words}, overlap={overlap_paragraphs}. "
            f"Use a different workdir or clear the cache to re-translate "
            f"with new settings."
        )
    cache.set_meta(CHUNKER_META_KEY, {
        "target_words": target_words,
        "overlap_paragraphs": overlap_paragraphs,
    })


def verify_chunker_params(
    cache: Cache, target_words: int, overlap_paragraphs: int
) -> None:
    """Verify current chunker params match those used during translate.

    Raises ChunkerConfigMismatchError if they differ. If no saved params
    exist (legacy cache from before this check), silently passes.
    """
    saved = cache.get_meta(CHUNKER_META_KEY)
    if saved is None:
        # Legacy cache — no metadata stored. Can't validate.
        return

    saved_words = saved.get("target_words")
    saved_overlap = saved.get("overlap_paragraphs")

    if saved_words != target_words or saved_overlap != overlap_paragraphs:
        raise ChunkerConfigMismatchError(
            f"Chunker config mismatch: translate used "
            f"target_words={saved_words}, overlap={saved_overlap}; "
            f"current config has target_words={target_words}, "
            f"overlap={overlap_paragraphs}. "
            f"Chunk IDs will not match cached translations. "
            f"Use the same chunker settings or re-translate."
        )


def collect_chunk_originals(
    chunk_set: ChunkSet,
    chunk_ids: set[str] | list[str],
) -> dict[str, str]:
    """Build {chunk_id: original XHTML text} for the given chunk IDs.

    Uses a set for O(1) membership checks.
    """
    target_ids = set(chunk_ids)
    originals: dict[str, str] = {}
    for chunk in chunk_set.chunks:
        if chunk.id in target_ids:
            frags = chunk_set.render_main(chunk)
            originals[chunk.id] = "\n".join(frags)
    return originals


def collect_chunk_originals_with_paragraphs(
    chunk_set: ChunkSet,
    chunk_ids: set[str] | list[str],
) -> dict[str, tuple[str, list[str]]]:
    """Build {chunk_id: (full_text, [paragraph_fragments])} for given IDs.

    Returns both the joined text and the individual paragraph list,
    needed by the reflect pass.
    """
    target_ids = set(chunk_ids)
    result: dict[str, tuple[str, list[str]]] = {}
    for chunk in chunk_set.chunks:
        if chunk.id in target_ids:
            frags = chunk_set.render_main(chunk)
            result[chunk.id] = ("\n".join(frags), frags)
    return result


def collect_preferred_translations(
    cache: Cache,
    chunk_ids: list[str],
) -> dict[str, str]:
    """Get the preferred/latest translation content for each chunk.

    Uses waterfall resolution: if a chunk has been reflected, returns
    the reflect content. If a preference is set, respects it. This
    ensures judge/reflect always operate on the currently-active
    translation, not a stale first-pass version.

    For judge specifically, we want to evaluate the 'translate' stage
    (the raw first translation), not the reflected version. Use
    collect_stage_translations() for that.
    """
    resolved_stages = cache.resolve_stages_bulk(chunk_ids)
    translations: dict[str, str] = {}

    for cid in chunk_ids:
        stage = resolved_stages.get(cid)
        if not stage:
            continue
        # Get the content for the resolved stage
        stages = cache.get_chunk_stages(cid)
        for s in stages:
            if s.stage == stage:
                translations[cid] = s.content
                break

    return translations


def collect_stage_translations(
    cache: Cache,
    chunk_ids: list[str],
    stage: str = "translate",
) -> dict[str, str]:
    """Get translation content from a specific stage for each chunk.

    When multiple entries exist for the same chunk+stage (reruns with
    different prompt versions), returns the most recent one (last in
    created_at order from get_chunk_stages).
    """
    translations: dict[str, str] = {}

    for cid in chunk_ids:
        stages = cache.get_chunk_stages(cid)
        # get_chunk_stages returns ordered by created_at; take the last
        # matching entry (most recent revision).
        for s in reversed(stages):
            if s.stage == stage:
                translations[cid] = s.content
                break

    return translations


def build_reflect_input(
    chunk_set: ChunkSet,
    cache: Cache,
    chunk_ids_to_reflect: set[str],
    judge_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build the input data list for Reflector.reflect_chunks().

    Uses O(1) lookups for both originals and translations.
    """
    # Collect originals with paragraphs (O(n) over chunk_set, once)
    originals_map = collect_chunk_originals_with_paragraphs(
        chunk_set, chunk_ids_to_reflect
    )

    # Collect translations — for reflect, we want the 'translate' stage
    # (the raw first translation that was judged), not a later stage.
    translations_map = collect_stage_translations(
        cache, list(chunk_ids_to_reflect), stage="translate"
    )

    chunks_data: list[dict[str, Any]] = []
    for cid in sorted(chunk_ids_to_reflect):
        orig_data = originals_map.get(cid)
        translated_text = translations_map.get(cid, "")

        if not orig_data or not translated_text:
            continue

        original_text, original_paragraphs = orig_data

        judge_data = judge_by_id.get(cid, {})
        judge_score = int(judge_data.get("score", 0))
        raw_issues = judge_data.get("issues", [])
        judge_issues = (
            [str(i) for i in raw_issues]
            if isinstance(raw_issues, list)
            else []
        )

        chunks_data.append({
            "chunk_id": cid,
            "original_text": original_text,
            "original_paragraphs": original_paragraphs,
            "translated_text": translated_text,
            "judge_score": judge_score,
            "judge_issues": judge_issues,
        })

    return chunks_data


def normalize_judge_map(
    raw_scores: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Normalize raw judge score dicts into a canonical shape.

    Input: list of dicts from cache.get_judge_scores() or JudgeResult objects.
    Output: {chunk_id: {"score": int, "issues": list[str]}}

    This ensures both `translate` (which has JudgeResult objects) and
    `reflect_cmd` (which has raw cache dicts) produce the same shape
    for build_reflect_input().
    """
    result: dict[str, dict[str, Any]] = {}
    for entry in raw_scores:
        cid = entry.get("chunk_id", "")
        if not cid:
            continue

        raw_score = entry.get("score", 0)
        try:
            score = int(raw_score)
        except (TypeError, ValueError):
            score = 0

        raw_issues = entry.get("issues", [])
        if isinstance(raw_issues, list):
            issues = [str(i) for i in raw_issues]
        elif isinstance(raw_issues, str):
            issues = [raw_issues]
        else:
            issues = []

        result[cid] = {"score": score, "issues": issues}
    return result
