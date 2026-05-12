"""Shared helpers for judge/reflect pipeline orchestration.

Extracted to avoid duplication between `translate`, `judge`, and `reflect`
CLI commands. Addresses code review concerns about:
- Consistent provider initialization
- Correct preferred-stage selection (not stale translations)
- O(1) chunk lookup instead of O(n) linear scan
- Chunker config persistence and mismatch detection
"""

from __future__ import annotations

import logging
import re
from typing import Any

from .cache import Cache
from .chunker import ChunkSet
from .provider import OpenRouterProvider

logger = logging.getLogger(__name__)

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


def collect_waterfall_translations(
    cache: Cache,
    chunk_ids: list[str],
    target_stage: str,
) -> dict[str, str]:
    """Get the best available translation for each chunk BEFORE target_stage.

    The waterfall order is: translate → reflect → proofread → style → verify.
    For a given target_stage, we pick the latest stage that precedes it.

    Example: if target_stage='style', we look for proofread first, then
    reflect, then translate — and return the first one found for each chunk.

    Returns {chunk_id: content} for chunks that have at least one
    preceding stage available.
    """
    from .cache import STAGE_WATERFALL

    if not chunk_ids:
        return {}

    # Determine which stages precede the target
    if target_stage not in STAGE_WATERFALL:
        # Fallback: just get the best available
        preceding_stages = STAGE_WATERFALL[:]
    else:
        target_idx = STAGE_WATERFALL.index(target_stage)
        preceding_stages = STAGE_WATERFALL[:target_idx]

    # Batch-fetch all stages for all chunks in one query
    all_chunk_stages = cache.get_stages_for_chunks_bulk(chunk_ids)

    # For each chunk, find the latest preceding stage that has content
    translations: dict[str, str] = {}

    for cid in chunk_ids:
        stages = all_chunk_stages.get(cid, [])
        stage_content: dict[str, str] = {}
        for s in stages:
            if s.stage in preceding_stages:
                stage_content[s.stage] = s.content

        # Pick the latest available preceding stage
        for stage in reversed(preceding_stages):
            if stage in stage_content:
                translations[cid] = stage_content[stage]
                break

    return translations


def collect_waterfall_paragraphs(
    cache: Cache,
    chunk_ids: list[str],
    target_stage: str,
    paragraph_counts: dict[str, int],
) -> dict[str, list[str]]:
    """Get waterfall translations parsed into paragraph lists.

    For each chunk, iterates preceding stages from newest to oldest and
    accepts the first stage whose content parses correctly and matches
    the expected paragraph count. Falls back to earlier stages if the
    latest one is malformed.

    Args:
        cache: Cache instance.
        chunk_ids: Chunks to collect.
        target_stage: The stage we're about to run (determines waterfall).
        paragraph_counts: {chunk_id: expected_paragraph_count} for validation.

    Returns {chunk_id: [paragraph_fragments]} for chunks with valid content.
    """
    import re

    from .cache import STAGE_WATERFALL

    _MARKER_RE = re.compile(r"^===PARAGRAPH\s+\d+===\s*$", re.MULTILINE)

    if not chunk_ids:
        return {}

    # Determine which stages precede the target
    if target_stage not in STAGE_WATERFALL:
        preceding_stages = STAGE_WATERFALL[:]
    else:
        target_idx = STAGE_WATERFALL.index(target_stage)
        preceding_stages = STAGE_WATERFALL[:target_idx]

    # Batch-fetch all stages for all chunks
    all_chunk_stages = cache.get_stages_for_chunks_bulk(chunk_ids)

    result: dict[str, list[str]] = {}
    for cid in chunk_ids:
        stages = all_chunk_stages.get(cid, [])
        # Build {stage_name: content} for preceding stages
        stage_content: dict[str, str] = {}
        for s in stages:
            if s.stage in preceding_stages:
                stage_content[s.stage] = s.content

        expected = paragraph_counts.get(cid)

        # Try stages from newest to oldest, accept first valid one
        for stage in reversed(preceding_stages):
            content = stage_content.get(stage)
            if content is None:
                continue

            fragments = _parse_waterfall_content(content, _MARKER_RE)
            if fragments is None:
                # No markers and not a single-paragraph case
                if expected == 1:
                    # Treat entire content as single paragraph
                    text = content.strip()
                    if text:
                        result[cid] = [text]
                        break
                continue

            if expected is not None and len(fragments) != expected:
                # Paragraph count mismatch — try earlier stage
                logger.debug(
                    "Waterfall %s/%s: %d paragraphs, expected %d; trying earlier stage",
                    cid, stage, len(fragments), expected,
                )
                continue

            result[cid] = fragments
            break

    return result


def _parse_waterfall_content(
    content: str, marker_re: re.Pattern[str]
) -> list[str] | None:
    """Parse ===PARAGRAPH N=== markers from cached content.

    Returns list of fragments, or None if no markers found.
    """
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines)

    matches = list(marker_re.finditer(text))
    if not matches:
        return None

    fragments: list[str] = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        fragment = text[start:end].strip()
        fragments.append(fragment)

    return fragments


def rehydrate_book_from_waterfall(
    cache: Cache,
    chunk_set: ChunkSet,
    forced_stage: str | None = None,
    rehydrate_all: bool = False,
) -> tuple[int, list[str]]:
    """Update in-memory book trees with the latest waterfall content.

    After post-processing passes write results to cache, the in-memory
    lxml trees still contain the translate-stage content. This function
    reads the final waterfall result for each chunk, parses the XHTML
    fragments, and splices them back into the chapter trees so that
    write_translated_epub() produces the correct output.

    Parameters
    ----------
    forced_stage : str | None
        If set, use ONLY this stage for all chunks (strict mode).
        Ignores waterfall resolution and preferences.
    rehydrate_all : bool
        If True, rehydrate ALL chunks including those resolved to
        'translate'. Use this when the book trees are freshly loaded
        (e.g. in assemble flow) and don't have any translations yet.

    Returns
    -------
    tuple[int, list[str]]
        (rehydrated_count, failed_chunk_ids). In non-strict mode,
        failed_chunk_ids is always empty. In strict mode (forced_stage),
        it contains IDs of chunks that could not be rehydrated.
    """
    import re

    from lxml import etree

    from .cache import STAGE_WATERFALL

    _MARKER_RE = re.compile(r"^===PARAGRAPH\s+\d+===\s*$", re.MULTILINE)

    # Matches a `&` that does NOT start a valid XML entity
    _BARE_AMP_RE = re.compile(
        r"&(?!(?:amp|lt|gt|quot|apos|#\d+|#x[0-9a-fA-F]+);)"
    )

    # Build chunk lookup: id -> Chunk
    chunk_by_id = {c.id: c for c in chunk_set.chunks}

    # Get all chunk IDs that have translations
    all_chunk_ids = list(chunk_by_id.keys())

    # Resolve the final stage for each chunk via waterfall (or forced)
    if forced_stage:
        # Strict mode: use only the forced stage for all chunks
        resolved_stages = {cid: forced_stage for cid in all_chunk_ids}
        # Rehydrate ALL chunks (even translate) since we're assembling fresh
        chunks_to_rehydrate = list(all_chunk_ids)
    else:
        resolved_stages = cache.resolve_stages_bulk(all_chunk_ids)
        if rehydrate_all:
            # Assembly flow: book loaded fresh, ALL chunks need rehydration
            chunks_to_rehydrate = list(all_chunk_ids)
        else:
            # Translate flow: in-memory trees already have translate content,
            # only rehydrate chunks with later stages.
            chunks_to_rehydrate = [
                cid for cid, stage in resolved_stages.items()
                if stage != "translate"
            ]

    if not chunks_to_rehydrate:
        return 0, []

    # Batch-fetch stages for chunks that need rehydration
    all_stages_map = cache.get_stages_for_chunks_bulk(chunks_to_rehydrate)

    rehydrated = 0
    _rehydrated_set: set[str] = set()
    for cid in chunks_to_rehydrate:
        chunk = chunk_by_id.get(cid)
        if not chunk:
            continue

        resolved_stage = resolved_stages.get(cid)
        if not resolved_stage:
            continue

        # Try the resolved stage first, then fall back to earlier stages
        stages = all_stages_map.get(cid, [])
        stage_content: dict[str, str] = {}
        for s in stages:
            if s.stage in STAGE_WATERFALL:
                stage_content[s.stage] = s.content

        # Build ordered list of stages to try
        if forced_stage:
            # Strict mode: only try the forced stage, no fallback
            stages_to_try: list[str] = [forced_stage]
        else:
            # Waterfall: resolved first, then earlier stages
            stages_to_try: list[str] = [resolved_stage]
            resolved_idx = (
                STAGE_WATERFALL.index(resolved_stage)
                if resolved_stage in STAGE_WATERFALL
                else len(STAGE_WATERFALL)
            )
            for stage in reversed(STAGE_WATERFALL[:resolved_idx]):
                if stage != "translate" and stage in stage_content:
                    stages_to_try.append(stage)

        expected = len(chunk.paragraph_indexes)
        chapter = chunk_set.book.chapters[chunk.chapter_index]
        applied = False

        for try_stage in stages_to_try:
            content = stage_content.get(try_stage)
            if content is None:
                continue

            # Parse paragraph markers
            text = content.strip()
            if text.startswith("```"):
                lines = text.splitlines()
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].startswith("```"):
                    lines = lines[:-1]
                text = "\n".join(lines)

            matches = list(_MARKER_RE.finditer(text))
            if not matches:
                continue

            candidate: list[str] = []
            for i, m in enumerate(matches):
                start = m.end()
                end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
                candidate.append(text[start:end].strip())

            if len(candidate) != expected:
                continue

            # Validate ALL fragments parse as XML before mutating
            parsed_elements: list[tuple[int, etree._Element]] = []
            all_ok = True

            for para_idx, frag_str in zip(
                chunk.paragraph_indexes, candidate, strict=True
            ):
                frag_str = _BARE_AMP_RE.sub("&amp;", frag_str)
                try:
                    new_el = etree.fromstring(frag_str)
                except etree.XMLSyntaxError:
                    all_ok = False
                    break

                old_el = chapter.paragraphs[para_idx]
                original_tag = old_el.tag
                if isinstance(original_tag, str) and "}" in original_tag:
                    ns_uri = original_tag.split("}", 1)[0].lstrip("{")
                    for sub in new_el.iter():
                        if isinstance(sub.tag, str) and "}" not in sub.tag:
                            sub.tag = f"{{{ns_uri}}}{sub.tag}"

                parsed_elements.append((para_idx, new_el))

            if not all_ok or len(parsed_elements) != len(candidate):
                # This stage failed XML validation — try earlier stage
                continue

            # All fragments validated — perform tree mutations
            for para_idx, new_el in parsed_elements:
                old_el = chapter.paragraphs[para_idx]
                new_el.tail = old_el.tail
                parent = old_el.getparent()
                if parent is not None:
                    parent.replace(old_el, new_el)
                    chapter.paragraphs[para_idx] = new_el

            applied = True
            break

        if applied:
            rehydrated += 1
            _rehydrated_set.add(cid)

    if forced_stage or rehydrate_all:
        # Report chunks that failed to rehydrate
        failed_ids = [
            cid for cid in chunks_to_rehydrate
            if cid not in _rehydrated_set
        ]
        return rehydrated, failed_ids

    return rehydrated, []


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
