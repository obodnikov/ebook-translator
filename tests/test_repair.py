"""Tests for the deterministic half of the repair stage.

The judge quotes the fragment it objects to and gives a correction. Where that
is an unambiguous single substitution, no model is involved — so the guarantees
here are absolute rather than statistical: the paragraph count never changes,
nothing outside the quoted fragment is touched, and an issue that does not fit
exactly is refused rather than guessed at.

Background: the judge invents a quote in roughly 6% of issues, so refusing is a
correctness requirement, not caution.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from booktranslator.cache import Cache
from booktranslator.prompts import render_prompt
from booktranslator.provider import CompletionResult
from booktranslator.repair import (
    MECHANICAL_CATEGORIES,
    Repairer,
    RepairStats,
    Skip,
    apply_candidates,
    parse_issue,
    propose,
)


def repaired(paragraphs, issues, categories=MECHANICAL_CATEGORIES):
    """Propose and apply everything proposed — the old all-in-one behaviour.

    Only for tests of the mechanical layer. In the pipeline a model vets each
    candidate in between, which is the whole point of the split.
    """
    p = propose(paragraphs, issues, categories)
    return p, apply_candidates(paragraphs, p.candidates)


# Real issues from the Bear Head judge run.
GENDER = "grammar: p.1 «Рой мёртв» → «Рой мертва»"
NUMERAL = "grammar: p.2 «было тридцать одного года» → «было тридцать один год»"
GLOSSARY = "glossary: p.1 «толпой Плохих» → «Медведи Беды»"
REGISTER = "register: p.1 «куда больше» — оригинал грубее"


class TestParsing:
    def test_splits_a_full_issue(self):
        issue = parse_issue(GENDER)
        assert issue.category == "grammar"
        assert issue.paragraph == 1
        assert issue.old == "Рой мёртв"
        assert issue.new == "Рой мертва"

    def test_accepts_ascii_arrow(self):
        """The judge is asked for → but writes -> often enough to matter."""
        issue = parse_issue("grammar: p.3 «был» -> «была»")
        assert issue.old == "был"
        assert issue.new == "была"

    def test_issue_without_a_pair(self):
        issue = parse_issue(REGISTER)
        assert issue.category == "register"
        assert issue.old is None

    def test_subcategory_form(self):
        """Planned accuracy subcategories must not break category detection."""
        assert parse_issue("accuracy/omission: p.4 «а» → «б»").category == "accuracy"


class TestSubstitution:
    def test_applies_a_single_match(self):
        paras = ["<p>Не то. Рой мёртв. Или Рой занята.</p>", "<p>Второй абзац.</p>"]
        out, out_paras = repaired(paras, [GENDER])
        assert out_paras[0] == "<p>Не то. Рой мертва. Или Рой занята.</p>"
        assert len(out.candidates) == 1
        assert bool(out.candidates)

    def test_leaves_other_paragraphs_byte_identical(self):
        paras = ["<p>Рой мёртв.</p>", "<p>Нетронутый абзац.</p>"]
        out, out_paras = repaired(paras, [GENDER])
        assert out_paras[1] == paras[1]

    def test_paragraph_count_never_changes(self):
        paras = ["<p>Рой мёртв.</p>", "<p>Два.</p>", "<p>Три.</p>"]
        out, out_paras = repaired(paras, [GENDER, NUMERAL, GLOSSARY])
        assert len(out_paras) == len(paras)

    def test_input_list_is_not_mutated(self):
        paras = ["<p>Рой мёртв.</p>"]
        repaired(paras, [GENDER])
        assert paras == ["<p>Рой мёртв.</p>"]

    def test_several_issues_in_one_chunk(self):
        paras = ["<p>Рой мёртв.</p>", "<p>Ей было тридцать одного года.</p>"]
        out, out_paras = repaired(paras, [GENDER, NUMERAL])
        assert out_paras[0] == "<p>Рой мертва.</p>"
        assert out_paras[1] == "<p>Ей было тридцать один год.</p>"
        assert len(out.candidates) == 2

    def test_markup_inside_the_quote_survives(self):
        paras = ['<p>Он сказал: <span class="italic">Рой мёртв</span>.</p>']
        out, out_paras = repaired(paras, [GENDER])
        assert out_paras[0] == '<p>Он сказал: <span class="italic">Рой мертва</span>.</p>'


class TestRefusal:
    def test_refuses_a_quote_that_is_not_there(self):
        """The judge invents quotes; an invented one must change nothing."""
        paras = ["<p>Здесь такого текста нет.</p>"]
        out, out_paras = repaired(paras, [GENDER])
        assert out_paras == paras
        assert out.skipped[0][1] is Skip.QUOTE_ABSENT
        assert not out.candidates

    def test_refuses_an_ambiguous_quote(self):
        """Two occurrences: guessing could corrupt the wrong sentence."""
        paras = ["<p>Рой мёртв. Все повторяли: Рой мёртв.</p>"]
        out, out_paras = repaired(paras, [GENDER])
        assert out_paras == paras
        assert out.skipped[0][1] is Skip.QUOTE_AMBIGUOUS

    def test_refuses_an_out_of_range_paragraph(self):
        out, out_paras = repaired(["<p>Один.</p>"], ["grammar: p.9 «а» → «б»"])
        assert out.skipped[0][1] is Skip.BAD_PARAGRAPH

    def test_refuses_an_issue_with_no_paragraph_number(self):
        out, out_paras = repaired(["<p>Рой мёртв.</p>"], ["grammar: «Рой мёртв» → «Рой мертва»"])
        assert out.skipped[0][1] is Skip.NO_PARAGRAPH

    def test_refuses_a_prose_issue(self):
        out, out_paras = repaired(["<p>куда больше</p>"], [REGISTER])
        assert out.skipped[0][1] is Skip.OUT_OF_SCOPE

    def test_refuses_a_category_outside_the_list(self):
        """register and naturalness are deliberately out of the first pass."""
        out, out_paras = repaired(["<p>текст</p>"], ["naturalness: p.1 «а» → «б»"])
        assert out.skipped[0][1] is Skip.OUT_OF_SCOPE

    def test_refuses_a_no_op(self):
        out, out_paras = repaired(["<p>Рой мёртв.</p>"], ["grammar: p.1 «Рой» → «Рой»"])
        assert out.skipped[0][1] is Skip.NO_OP
        assert not out.candidates

    def test_one_bad_issue_does_not_block_a_good_one(self):
        paras = ["<p>Рой мёртв.</p>", "<p>Ей было тридцать одного года.</p>"]
        out, out_paras = repaired(paras, ["grammar: p.1 «нет такого» → «х»", NUMERAL])
        assert out_paras[1] == "<p>Ей было тридцать один год.</p>"
        assert len(out.candidates) == 1
        assert len(out.skipped) == 1


class TestDeferral:
    def test_ambiguous_quote_goes_to_the_model(self):
        paras = ["<p>Рой мёртв. Рой мёртв.</p>"]
        out, out_paras = repaired(paras, [GENDER])
        assert [i.raw for i in out.deferred] == [GENDER]

    def test_prose_issue_in_scope_goes_to_the_model(self):
        """In-scope but not a substitution — exactly what the model pass is for."""
        paras = ["<p>текст</p>"]
        out, out_paras = repaired(paras, ["grammar: p.1 согласование сбито во всём абзаце"])
        assert len(out.deferred) == 1

    def test_invented_quote_is_not_deferred(self):
        """A larger model cannot make an invented quote real."""
        out, out_paras = repaired(["<p>Здесь ничего нет.</p>"], [GENDER])
        assert out.deferred == []

    def test_out_of_scope_category_is_not_deferred(self):
        out, out_paras = repaired(["<p>текст</p>"], [REGISTER])
        assert out.deferred == []


class TestScope:
    def test_default_categories_are_the_mechanical_three(self):
        # Set by measurement, not taste: on Bear Head the gatekeeper rejected
        # 3 of 11 grammar candidates, 7 of 20 accuracy, and 4 of 5 glossary.
        # Accuracy behaves like grammar; glossary's notes are judgement calls.
        assert MECHANICAL_CATEGORIES == ("grammar", "markup", "accuracy")

    def test_glossary_is_not_repaired_by_default(self):
        paras = ["<p>Мартен Джеймс Каспиан</p>"]
        out, out_paras = repaired(
            paras, ["glossary: p.1 «Мартен Джеймс Каспиан» → «Джеймс Каспиан Мартен»"]
        )
        assert out_paras == paras
        assert not out.candidates

    def test_scope_can_be_widened_by_the_caller(self):
        paras = ["<p>куда больше</p>"]
        out, out_paras = repaired(
            paras,
            ["register: p.1 «куда больше» → «чёртова уйма»"],
            categories=("register",),
        )
        assert out_paras[0] == "<p>чёртова уйма</p>"

    def test_empty_issue_list_is_a_no_op(self):
        paras = ["<p>Один.</p>", "<p>Два.</p>"]
        out, out_paras = repaired(paras, [])
        assert out_paras == paras
        assert not out.candidates


# ---------------------------------------------------------------------------
# The stage class: verdict parsing and how it decides what to apply
# ---------------------------------------------------------------------------


class TestVerdictParsing:
    def test_parses_a_plain_array(self):
        out = Repairer.parse_verdicts(
            '[{"n": 1, "verdict": "accept", "why": "род"},'
            ' {"n": 2, "verdict": "reject", "why": "смысл"}]'
        )
        assert out == {1: (True, "род"), 2: (False, "смысл")}

    def test_strips_code_fences(self):
        out = Repairer.parse_verdicts('```json\n[{"n": 1, "verdict": "accept"}]\n```')
        assert out[1][0] is True

    def test_ignores_surrounding_commentary(self):
        out = Repairer.parse_verdicts('Вот разбор:\n[{"n": 1, "verdict": "reject"}]\nГотово.')
        assert out[1][0] is False

    def test_unparseable_response_raises(self):
        """A verdict list we cannot read must never mean "accept everything"."""
        with pytest.raises(ValueError):
            Repairer.parse_verdicts("Не могу выполнить эту задачу.")

    def test_malformed_json_raises(self):
        with pytest.raises(ValueError):
            Repairer.parse_verdicts('[{"n": 1, "verdict": ]')

    def test_entry_without_a_number_is_dropped(self):
        out = Repairer.parse_verdicts('[{"verdict": "accept"}, {"n": 2, "verdict": "accept"}]')
        assert out == {2: (True, "")}

    def test_unknown_verdict_word_is_not_acceptance(self):
        out = Repairer.parse_verdicts('[{"n": 1, "verdict": "maybe"}]')
        assert out[1][0] is False


class TestApplyingVerdicts:
    """The decision rule, exercised without a provider."""

    @staticmethod
    def _decide(candidates, verdicts):
        accepted = [c for i, c in enumerate(candidates, 1) if verdicts.get(i, (False, ""))[0]]
        return apply_candidates(
            ["<p>Рой мёртв.</p>", "<p>Ей было тридцать одного года.</p>"], accepted
        )

    def test_only_accepted_candidates_are_applied(self):
        paras = ["<p>Рой мёртв.</p>", "<p>Ей было тридцать одного года.</p>"]
        cands = propose(paras, [GENDER, NUMERAL]).candidates
        out = self._decide(cands, {1: (True, ""), 2: (False, "")})
        assert out[0] == "<p>Рой мертва.</p>"
        assert out[1] == paras[1]

    def test_a_missing_verdict_means_reject(self):
        """Silence is not consent: an unanswered candidate is left alone."""
        paras = ["<p>Рой мёртв.</p>", "<p>Ей было тридцать одного года.</p>"]
        cands = propose(paras, [GENDER, NUMERAL]).candidates
        out = self._decide(cands, {1: (True, "")})
        assert out[1] == paras[1]

    def test_rejecting_everything_leaves_the_chunk_untouched(self):
        paras = ["<p>Рой мёртв.</p>", "<p>Ей было тридцать одного года.</p>"]
        cands = propose(paras, [GENDER, NUMERAL]).candidates
        assert self._decide(cands, {}) == paras


class TestLoneVerdictObject:
    """With one proposal, a model answered `{…}` instead of `[{…}]` on Foxglove
    Summer ch12_c02 and the chunk failed. The verdict is the same either way."""

    def test_a_single_object_is_one_verdict(self):
        out = Repairer.parse_verdicts('{"n": 1, "verdict": "reject", "why": "форма верна"}')
        assert out == {1: (False, "форма верна")}

    def test_a_single_object_in_fences(self):
        out = Repairer.parse_verdicts('```json\n{"n": 1, "verdict": "accept"}\n```')
        assert out == {1: (True, "")}

    def test_an_object_that_is_not_a_verdict_still_raises(self):
        with pytest.raises(ValueError, match="not a JSON array"):
            Repairer.parse_verdicts('{"result": "ok"}')


# ---------------------------------------------------------------------------
# The model pass: caching, re-runs, parallel chunks
# ---------------------------------------------------------------------------

REPAIR_PROMPT = Path(__file__).resolve().parents[1] / "prompts" / "repair.md"
PARAS = ["<p>Рой мёртв.</p>", "<p>Ей было тридцать одного года.</p>"]


def _verdicts(*accepts: bool) -> CompletionResult:
    body = ",".join(
        f'{{"n": {i}, "verdict": "{"accept" if ok else "reject"}", "why": "-"}}'
        for i, ok in enumerate(accepts, 1)
    )
    return CompletionResult(
        text=f"[{body}]", input_tokens=100, output_tokens=10, total_tokens=110, model="m", raw={}
    )


@pytest.fixture
def cache(tmp_path: Path) -> Cache:
    return Cache(tmp_path / "cache.sqlite")


def _repairer(provider, cache: Cache) -> Repairer:
    return Repairer(provider, REPAIR_PROMPT, cache, None, model="m")


class TestRepairReruns:
    def test_rerun_after_an_accepted_fix_is_free_and_same(self, cache):
        provider = MagicMock()
        provider.complete.return_value = _verdicts(True, False)
        first = _repairer(provider, cache).repair_chunk(
            "c1", PARAS, [GENDER, NUMERAL], "Bees is dead.", RepairStats()
        )

        stats = RepairStats()
        second = _repairer(provider, cache).repair_chunk(
            "c1", PARAS, [GENDER, NUMERAL], "Bees is dead.", stats
        )

        assert provider.complete.call_count == 1
        assert second.paragraphs == first.paragraphs == ["<p>Рой мертва.</p>", PARAS[1]]
        assert (stats.chunks_cached, stats.accepted, stats.rejected) == (1, 1, 1)

    def test_rejecting_everything_is_remembered_but_is_not_a_stage(self, cache):
        provider = MagicMock()
        provider.complete.return_value = _verdicts(False, False)
        for _ in range(2):
            _repairer(provider, cache).repair_chunk(
                "c1", PARAS, [GENDER, NUMERAL], "Bees is dead.", RepairStats()
            )

        assert provider.complete.call_count == 1, "the refusal was paid for twice"
        # No `repair` entry: an unchanged copy would outrank a later verify.
        assert cache.list_stages() == {"repair_verdicts": 1}

    def test_a_row_from_before_verdicts_were_kept_is_used_as_it_stands(self, cache):
        provider = MagicMock()
        repairer = _repairer(provider, cache)
        # Reproduce the old layout: the repaired text under the call's own key.
        proposal = propose(PARAS, [GENDER])
        context = {
            "source_lang": "en",
            "source_lang_name": "English",
            "target_lang": "ru",
            "target_lang_name": "Russian",
            "glossary_block": "(no glossary provided)",
            "original_text": "Bees is dead.",
            "candidates": [
                {
                    "paragraph": c.paragraph,
                    "note": c.issue.raw,
                    "paragraph_text": PARAS[c.paragraph - 1],
                    "old": c.old,
                    "new": c.new,
                }
                for c in proposal.candidates
            ],
        }
        system, user = render_prompt(repairer.prompt, context)
        key = Cache.make_key("repair", "m", repairer.prompt.version, system, user)
        old_text = "===PARAGRAPH 1===\n<p>Рой мертва.</p>\n===PARAGRAPH 2===\n" + PARAS[1]
        cache.put(key, "repair", "m", "1", old_text, meta={"chunk_id": "c1", "accepted": 1})

        stats = RepairStats()
        result = repairer.repair_chunk("c1", PARAS, [GENDER], "Bees is dead.", stats)

        provider.complete.assert_not_called()
        assert result.paragraphs == ["<p>Рой мертва.</p>", PARAS[1]]
        assert (stats.chunks_cached, stats.accepted, stats.chunks_repaired) == (1, 1, 1)

    def test_unhandled_issues_carry_their_reason(self, cache):
        stats = RepairStats()
        _repairer(MagicMock(), cache).repair_chunk(
            "c1", ["<p>Рой мёртв. Рой мёртв.</p>"], [GENDER], "", stats
        )
        assert stats.unhandled == [("c1", GENDER, Skip.QUOTE_AMBIGUOUS.value)]


class TestParallelRepair:
    def test_all_chunks_done_and_a_failure_does_not_stop_the_rest(self, cache):
        provider = MagicMock()

        def complete(**kwargs):
            if "Chunk three" in kwargs["user"]:
                raise RuntimeError("boom")
            return _verdicts(True)

        provider.complete.side_effect = complete
        jobs = [
            (f"c{i}", [f"<p>Рой мёртв. {name}</p>"], [GENDER], f"Chunk {name}")
            for i, name in enumerate(["one", "two", "three", "four"], 1)
        ]
        stats = RepairStats(chunks_total=len(jobs))
        seen: list[tuple[str, bool]] = []

        _repairer(provider, cache).repair_chunks(
            jobs,
            stats,
            parallelism=3,
            on_done=lambda done, cid, err: seen.append((cid, err is None)),
        )

        assert sorted(seen) == [("c1", True), ("c2", True), ("c3", False), ("c4", True)]
        assert (stats.chunks_failed, stats.chunks_repaired, stats.accepted) == (1, 3, 3)
