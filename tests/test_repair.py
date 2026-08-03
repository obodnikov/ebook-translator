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

from booktranslator.repair import (
    MECHANICAL_CATEGORIES,
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
        assert MECHANICAL_CATEGORIES == ("grammar", "glossary", "markup")

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
