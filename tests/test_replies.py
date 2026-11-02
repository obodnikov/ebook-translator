"""Tests for replies.py: what makes a reply unusable, and the one thing repaired in place.

The shapes are taken from the first run of The Tainted Cup on Sonnet 5.5, with the
text rewritten. That book mixes narration and speech in one paragraph, and the model
splits them. When it notices the count is off, it keeps the count right by dropping
text, which leaves N well-formed elements and a hole in the book.
"""

from __future__ import annotations

import pytest

from booktranslator.replies import (
    RETRY_NOTE,
    check_alignment,
    join_split_paragraphs,
    parse_fragment,
    parse_paragraphs,
    rebuild_over_split_reply,
    split_markers,
)

LONG_NARRATION = (
    "I walked slowly through the house and found her kneeling beside the machine, "
    "which rattled and clanged as if it had lost its mind."
)
RU_LONG_NARRATION = (
    "Я медленно прошёл по дому и нашёл её на коленях перед машиной, которая гремела "
    "и лязгала так, будто сошла с ума."
)
SHORT_NARRATION = "He nodded and looked out of the window."
RU_SHORT_NARRATION = "Он кивнул и посмотрел в окно."
RU_OTHER_NARRATION = (
    "Над городом висел тяжёлый туман, и фонари вдоль канала едва освещали мокрую мостовую "
    "под ногами редких прохожих."
)
RU_THIRD_NARRATION = (
    "Экономка молча провела нас через кухню, где на длинном столе остывали хлеб и "
    "несколько мисок с густым супом."
)


class TestCheckAlignment:
    def test_a_faithful_translation_passes(self):
        check_alignment(
            [
                f"<p>{LONG_NARRATION}</p>",
                "<p>“Who?”</p>",
                "<p>She shrugged. “Nobody you would know, I am sure of that much at least.”</p>",
                f"<p>{SHORT_NARRATION}</p>",
            ],
            [
                f"<p>{RU_LONG_NARRATION}</p>",
                "<p>— Кто?</p>",
                "<p>Она пожала плечами. — Никто из твоих знакомых, уж в этом я уверена.</p>",
                f"<p>{RU_SHORT_NARRATION}</p>",
            ],
        )

    @pytest.mark.parametrize(
        "source, reply",
        [
            # ch32_c01: narration ending in a colon, then the next paragraph.
            (
                "She shrugged. “Maybe someone wants us to stop looking for the seller.”",
                "Она пожала плечами:",
            ),
            # ch09_c06: no colon, the speech is simply not there.
            (
                "She stared at me. “After seeing that, sir, we could barely stand at all.”",
                "Она уставилась на меня.",
            ),
            # Single quotes, as in the British books.
            (
                "He sighed. ‘We have been over this and I will not say it again, Peter.’",
                "Он вздохнул.",
            ),
        ],
    )
    def test_speech_left_out_is_rejected(self, source, reply):
        with pytest.raises(ValueError, match="Paragraph 2 of 3: .*direct speech"):
            check_alignment(
                [f"<p>{LONG_NARRATION}</p>", f"<p>{source}</p>", f"<p>{SHORT_NARRATION}</p>"],
                [f"<p>{RU_LONG_NARRATION}</p>", f"<p>{reply}</p>", f"<p>{RU_SHORT_NARRATION}</p>"],
            )

    def test_every_paragraph_that_lost_speech_is_named(self):
        source = "<p>She shrugged. “Maybe someone wants us to stop looking for the seller.”</p>"
        with pytest.raises(ValueError, match="Paragraphs 1, 3 of 3"):
            check_alignment(
                [source, f"<p>{SHORT_NARRATION}</p>", source],
                [
                    "<p>Она пожала плечами:</p>",
                    f"<p>{RU_SHORT_NARRATION}</p>",
                    "<p>Она кивнула.</p>",
                ],
            )

    def test_speech_kept_in_guillemets_passes(self):
        check_alignment(
            ["<p>She said, “Go,” and left the room without looking back at any of us.</p>"],
            ["<p>Она сказала: «Иди» — и вышла, ни на кого не оглянувшись.</p>"],
        )

    def test_a_quiet_ending_with_a_colon_passes_when_the_source_has_no_speech(self):
        check_alignment(
            ["<p>He read out the list of the dead, and then he read it out once more:</p>"],
            ["<p>Он прочёл список погибших, а потом прочёл его ещё раз:</p>"],
        )

    def test_a_reply_shifted_down_the_source_is_rejected(self):
        """ch10_c03: one paragraph split, every later one a place late, the end cut off."""
        source = [
            LONG_NARRATION,
            SHORT_NARRATION,
            LONG_NARRATION,
            SHORT_NARRATION,
            LONG_NARRATION,
            SHORT_NARRATION,
        ]
        shifted = [
            RU_LONG_NARRATION[:30],
            RU_LONG_NARRATION,
            RU_SHORT_NARRATION,
            RU_LONG_NARRATION,
            RU_SHORT_NARRATION,
            RU_LONG_NARRATION,
        ]
        with pytest.raises(ValueError, match="no longer line up"):
            check_alignment(
                [f"<p>{text}</p>" for text in source], [f"<p>{text}</p>" for text in shifted]
            )

    def test_one_or_two_free_paragraphs_are_a_translators_right(self):
        """The check must not turn a compact or a wordy paragraph into a paid re-ask."""
        source = [LONG_NARRATION, LONG_NARRATION, SHORT_NARRATION, SHORT_NARRATION]
        reply = [RU_LONG_NARRATION[:50], RU_OTHER_NARRATION, RU_THIRD_NARRATION, RU_SHORT_NARRATION]
        check_alignment([f"<p>{t}</p>" for t in source], [f"<p>{t}</p>" for t in reply])

    def test_short_lines_are_not_compared_by_length(self):
        check_alignment(
            ["<p>No.</p>", "<p>Yes.</p>", "<p>Why?</p>", "<p>Go.</p>"],
            ["<p>— Нет, конечно же нет, сэр.</p>", "<p>Да</p>", "<p>Почему</p>", "<p>Ступай</p>"],
        )

    def test_markup_is_not_counted_as_text(self):
        attributes = 'class="a-very-long-class-name-that-is-not-text" id="also-not-text-at-all"'
        check_alignment(
            [f"<p {attributes}><i {attributes}>{t}</i></p>" for t in [LONG_NARRATION] * 3],
            [f"<p>{t}</p>" for t in (RU_LONG_NARRATION, RU_OTHER_NARRATION, RU_THIRD_NARRATION)],
        )


EPUB_NS = "http://www.idpf.org/2007/ops"
PAGE_BREAK = '<span epub:type="pagebreak" id="page_245" role="doc-pagebreak" title="245"/>'


class TestUndeclaredEpubPrefix:
    """ch36_c04: both replies were complete, and <p> had lost its xmlns:epub both times."""

    def test_a_missing_declaration_is_added_and_nothing_else_changes(self):
        element = parse_fragment(f'<p class="para-p">{PAGE_BREAK}— Известно, что это так.</p>')
        assert element[0].get(f"{{{EPUB_NS}}}type") == "pagebreak"
        assert element.get("class") == "para-p"
        assert "".join(element.itertext()) == "— Известно, что это так."

    def test_a_declared_prefix_is_left_as_it_is(self):
        element = parse_fragment(f'<p xmlns:epub="{EPUB_NS}" class="a">{PAGE_BREAK}Текст.</p>')
        assert element[0].get(f"{{{EPUB_NS}}}type") == "pagebreak"

    def test_a_paragraph_without_the_prefix_gets_no_declaration(self):
        assert parse_fragment("<p>Формат epub: то, что читает читалка.</p>").nsmap == {}

    @pytest.mark.parametrize(
        "fragment",
        [
            # The declaration doesn't turn a broken reply into a good one.
            f"<p>{PAGE_BREAK}Раз.</p>\n<p>Два.</p>",
            f"<p>{PAGE_BREAK}Обрыв",
            f"Постойте.\n<p>{PAGE_BREAK}Раз.</p>",
            # Only the prefix seen in books: another unknown one is still a broken reply.
            '<p><span foo:type="x"/>Раз.</p>',
        ],
    )
    def test_everything_else_stays_rejected(self, fragment):
        with pytest.raises(ValueError):
            parse_fragment(fragment)

    def test_a_split_paragraph_with_a_lost_declaration_is_still_joined(self):
        text = _reply(
            f'<p class="a">{PAGE_BREAK}Раз, два, три.</p>\n<p class="a">— Четыре, пять.</p>'
        )
        joined_text, joined = join_split_paragraphs(
            text, ["<p>One, two and three. “Four and five,” he said to all of them.</p>"]
        )
        assert joined == [1]
        element = parse_fragment(split_markers(joined_text)[0])
        assert element[0].get(f"{{{EPUB_NS}}}type") == "pagebreak"
        assert "".join(element.itertext()) == "Раз, два, три. — Четыре, пять."


class TestRepeatedText:
    """ch52_c02: paragraph 6 also held paragraph 7, and paragraph 7 came again after it.

    One paragraph of odd length is allowed, so the length check alone let this through.
    """

    FIRST = "Вы продумали, как всё это сделать, и никто ничего не заподозрил до самого конца."
    SECOND = "Вы составили план вместе с остальными и сами выбрали для этого подходящий вечер."

    def test_a_sentence_repeated_in_the_next_paragraph_is_rejected(self):
        source = [f"<p>{LONG_NARRATION}</p>", f"<p>{LONG_NARRATION}</p>"]
        reply = [f"<p>{self.FIRST} {self.SECOND}</p>", f"<p>{self.SECOND}</p>"]
        with pytest.raises(ValueError, match="Paragraph 1 of 2: .*next"):
            check_alignment(source, reply)

    def test_a_short_line_said_twice_is_ordinary_dialogue(self):
        check_alignment(["<p>“Yes.”</p>", "<p>“Yes.”</p>"], ["<p>— Да.</p>", "<p>— Да.</p>"])

    def test_the_same_sentence_further_apart_is_left_alone(self):
        """A refrain is the author's. Only a neighbour points at a mistake in the reply."""
        source = [f"<p>{LONG_NARRATION}</p>"] * 3
        reply = [f"<p>{self.SECOND}</p>", f"<p>{RU_LONG_NARRATION}</p>", f"<p>{self.SECOND}</p>"]
        check_alignment(source, reply)


def _reply(*fragments: str) -> str:
    return "\n".join(f"===PARAGRAPH {n}===\n{f}" for n, f in enumerate(fragments, start=1))


class TestRebuildOverSplitReply:
    """A first reply that is complete but has too many markers: ch09_c02, 48 for 43."""

    QUESTION = "“What are they doing with a house in this town?” I asked, quite bewildered."
    MIXED = "He shrugged. “Dunno, sir. Maybe they ran out of houses to buy elsewhere.”"
    AFTER = "We walked on down a long hallway that ended in a black door."

    RU_QUESTION = "— Что они делают с домом в этом городе? — спросил я, совсем озадаченный."
    RU_NARRATION = "Он пожал плечами:"
    RU_SPEECH = "— Не знаю, сэр. Может, им больше негде было покупать дома."
    RU_AFTER = "Мы пошли дальше по длинному коридору, упиравшемуся в чёрную дверь."

    def _source(self) -> list[str]:
        return [f"<p>{t}</p>" for t in (self.QUESTION, self.MIXED, self.AFTER)]

    def test_narration_and_its_speech_go_back_under_one_marker(self):
        text = _reply(
            f"<p>{self.RU_QUESTION}</p>",
            f"<p>{self.RU_NARRATION}</p>",
            f"<p>{self.RU_SPEECH}</p>",
            f"<p>{self.RU_AFTER}</p>",
        )

        rebuilt_text, rebuilt = rebuild_over_split_reply(text, self._source())

        assert rebuilt == [2]
        assert split_markers(rebuilt_text) == [
            f"<p>{self.RU_QUESTION}</p>",
            f"<p>{self.RU_NARRATION} {self.RU_SPEECH}</p>",
            f"<p>{self.RU_AFTER}</p>",
        ]
        assert len(parse_paragraphs(rebuilt_text, 3)) == 3
        check_alignment(self._source(), split_markers(rebuilt_text))

    def test_the_narration_goes_with_the_speech_after_it_not_the_line_before(self):
        """The first attempt at this attached «Он пожал плечами:» to the question above it."""
        text = _reply(
            f"<p>{self.RU_QUESTION}</p>",
            f"<p>{self.RU_NARRATION}</p>",
            f"<p>{self.RU_SPEECH}</p>",
            f"<p>{self.RU_AFTER}</p>",
        )
        fragments = split_markers(rebuild_over_split_reply(text, self._source())[0])
        assert fragments[0] == f"<p>{self.RU_QUESTION}</p>"

    def test_inline_markup_survives(self):
        text = _reply(
            f"<p>{self.RU_QUESTION}</p>",
            f'<p class="a"><span id="page_9"/>{self.RU_NARRATION}</p>',
            '<p class="a">— Не <i>знаю</i>, сэр. Может, им больше негде было покупать дома.</p>',
            f"<p>{self.RU_AFTER}</p>",
        )
        fragments = split_markers(rebuild_over_split_reply(text, self._source())[0])
        element = parse_fragment(fragments[1])
        assert [child.tag for child in element] == ["span", "i"]
        assert "".join(element.itertext()).startswith("Он пожал плечами: — Не знаю, сэр.")

    def test_the_right_count_is_left_for_the_other_checks(self):
        text = _reply(
            f"<p>{self.RU_QUESTION}</p>", f"<p>{self.RU_SPEECH}</p>", f"<p>{self.RU_AFTER}</p>"
        )
        assert rebuild_over_split_reply(text, self._source()) == (text, [])

    def test_too_few_markers_are_never_repaired(self):
        text = _reply(f"<p>{self.RU_QUESTION}</p>", f"<p>{self.RU_AFTER}</p>")
        assert rebuild_over_split_reply(text, self._source()) == (text, [])

    @pytest.mark.parametrize(
        "markers",
        [
            # ch44_c02: numbering starts over, so this is two answers, not one split.
            [
                "===PARAGRAPH 1===\n<p>{q}</p>",
                "===PARAGRAPH 2===\n<p>{n}</p>",
                "===PARAGRAPH 1===\n<p>{s}</p>",
                "===PARAGRAPH 2===\n<p>{a}</p>",
            ],
            # A number skipped: something went wrong that this cannot see.
            [
                "===PARAGRAPH 1===\n<p>{q}</p>",
                "===PARAGRAPH 2===\n<p>{n}</p>",
                "===PARAGRAPH 3===\n<p>{s}</p>",
                "===PARAGRAPH 5===\n<p>{a}</p>",
            ],
        ],
    )
    def test_numbering_that_is_not_one_to_m_is_refused(self, markers):
        text = "\n".join(markers).format(
            q=self.RU_QUESTION, n=self.RU_NARRATION, s=self.RU_SPEECH, a=self.RU_AFTER
        )
        assert rebuild_over_split_reply(text, self._source()) == (text, [])

    def test_the_models_commentary_is_refused(self):
        text = _reply(
            f"<p>{self.RU_QUESTION}</p>",
            f"<p>{self.RU_NARRATION}</p>\n\nWait: I have mis-numbered.",
            f"<p>{self.RU_SPEECH}</p>",
            f"<p>{self.RU_AFTER}</p>",
        )
        assert rebuild_over_split_reply(text, self._source()) == (text, [])

    def test_speech_then_narration_is_refused(self):
        """Only the shape measured: a source paragraph opening with narration, then speech."""
        source = ["<p>“Go,” he said. Then he turned and left without looking back at her.</p>"]
        text = _reply("<p>— Иди, — сказал он.</p>", "<p>Он повернулся и вышел, не оглянувшись.</p>")
        assert rebuild_over_split_reply(text, source) == (text, [])

    def test_a_source_paragraph_without_speech_is_never_merged(self):
        source = [f"<p>{LONG_NARRATION}</p>"]
        text = _reply(f"<p>{RU_LONG_NARRATION[:50]}</p>", f"<p>— {RU_LONG_NARRATION[50:]}</p>")
        assert rebuild_over_split_reply(text, source) == (text, [])

    def test_parts_far_from_their_part_of_the_source_are_refused(self):
        """Almost all of a long paragraph under the narration: not this split."""
        source = [
            "<p>He shrugged. “Dunno, sir. Maybe they ran out of houses to buy somewhere else, "
            "and maybe nobody in the whole city ever cared to ask them why.”</p>"
        ]
        text = _reply(f"<p>{RU_LONG_NARRATION}</p>", "<p>— Нет.</p>")
        assert rebuild_over_split_reply(text, source) == (text, [])

    def test_elements_of_different_look_are_refused(self):
        text = _reply(
            f"<p>{self.RU_QUESTION}</p>",
            f'<p class="a">{self.RU_NARRATION}</p>',
            f'<p class="b">{self.RU_SPEECH}</p>',
            f"<p>{self.RU_AFTER}</p>",
        )
        assert rebuild_over_split_reply(text, self._source()) == (text, [])

    def test_a_split_that_cannot_absorb_every_extra_marker_changes_nothing(self):
        text = _reply(
            f"<p>{self.RU_QUESTION}</p>",
            f"<p>{self.RU_NARRATION}</p>",
            f"<p>{self.RU_SPEECH}</p>",
            f"<p>{self.RU_AFTER}</p>",
            "<p>Лишний абзац, которому в оригинале ничего не соответствует вовсе.</p>",
        )
        assert rebuild_over_split_reply(text, self._source()) == (text, [])


class TestJoinSplitParagraphs:
    SOURCE = [
        f"<p>{SHORT_NARRATION}</p>",
        "<p>Kalista stood up. “Should I get tested? Is there a test for such a thing?”</p>",
        f"<p>{SHORT_NARRATION}</p>",
    ]

    def test_a_split_paragraph_becomes_one_element_and_the_rest_is_untouched(self):
        first = f'<p class="a">{RU_SHORT_NARRATION}</p>'
        last = f'<p class="a">{RU_SHORT_NARRATION}</p>'
        text = _reply(
            first, '<p class="a">Калиста встала.</p>\n<p class="a">— Мне пройти проверку?</p>', last
        )

        joined_text, joined = join_split_paragraphs(text, self.SOURCE)

        assert joined == [2]
        fragments = split_markers(joined_text)
        assert fragments[0] == first and fragments[2] == last
        assert fragments[1] == '<p class="a">Калиста встала. — Мне пройти проверку?</p>'
        assert len(parse_paragraphs(joined_text, 3)) == 3

    def test_a_reply_with_nothing_to_join_is_returned_byte_for_byte(self):
        text = "===PARAGRAPH 1===\n\n<p>Раз</p>\n\n===PARAGRAPH 2===\n<p>Два</p>\n"
        assert join_split_paragraphs(text, ["<p>One</p>", "<p>Two</p>"]) == (text, [])

    def test_three_parts_keep_their_order(self):
        text = _reply("<p>Раз.</p>\n<p>Два.</p>\n<p>Три.</p>")
        joined_text, joined = join_split_paragraphs(
            text, ["<p>One, and then two, and then three of them.</p>"]
        )
        assert joined == [1]
        assert "".join(parse_fragment(split_markers(joined_text)[0]).itertext()) == "Раз. Два. Три."

    def test_inline_markup_and_a_page_break_survive(self):
        text = _reply(
            '<p xmlns:epub="http://www.idpf.org/2007/ops" class="a">'
            '<span epub:type="pagebreak" id="page_1"/>Он встал.</p>\n'
            '<p class="a">— <i class="b">Нет</i>, сказал он &amp; вышел.</p>'
        )
        joined_text, joined = join_split_paragraphs(
            text, ["<p>He stood up. “No,” he said, and then he walked out of the room.</p>"]
        )
        assert joined == [1]
        element = parse_fragment(split_markers(joined_text)[0])
        assert "".join(element.itertext()) == "Он встал. — Нет, сказал он & вышел."
        assert [child.tag for child in element] == ["span", "i"]
        assert element[0].get("id") == "page_1"

    def test_a_bare_ampersand_does_not_stop_the_join(self):
        text = _reply("<p>Купил в M&S.</p>\n<p>— Зачем?</p>")
        joined_text, joined = join_split_paragraphs(
            text, ["<p>He bought it at M&amp;S. “Why?” she asked him.</p>"]
        )
        assert joined == [1]
        assert "".join(parse_fragment(split_markers(joined_text)[0]).itertext()) == (
            "Купил в M&S. — Зачем?"
        )

    @pytest.mark.parametrize(
        "fragment",
        [
            # Not the same kind of element: not one paragraph split in two.
            '<p class="a">Раз.</p>\n<p class="b">Два.</p>',
            "<p>Раз.</p>\n<h2>Два.</h2>",
            # The model's own commentary is not book text.
            "<p>Раз.</p>\nWait: I have mis-numbered.\n<p>Два.</p>",
            "Постойте.\n<p>Раз.</p>\n<p>Два.</p>",
            # Not well-formed at all.
            "<p>Раз.\n<p>Два.</p>",
        ],
    )
    def test_anything_else_is_left_alone_to_be_rejected(self, fragment):
        text = _reply(fragment)
        assert join_split_paragraphs(text, ["<p>One and two together.</p>"]) == (text, [])

    def test_far_more_text_than_the_source_paragraph_is_not_joined(self):
        """Two paragraphs of the source under one marker, or text nobody asked for."""
        text = _reply(f"<p>{RU_LONG_NARRATION}</p>\n<p>{RU_LONG_NARRATION}</p>")
        assert join_split_paragraphs(text, [f"<p>{SHORT_NARRATION}</p>"]) == (text, [])

    def test_half_again_as_long_as_the_source_is_not_joined(self):
        """Real splits came to 0.82-1.27 of the source; ch52_c02, which was no split, to 1.83."""
        text = _reply(f"<p>{RU_LONG_NARRATION}</p>\n<p>{RU_LONG_NARRATION[:80]}</p>")
        assert join_split_paragraphs(text, [f"<p>{LONG_NARRATION}</p>"]) == (text, [])

    NEXT_LINE = (
        "Следующий абзац книги, который модель написала здесь и потом ещё раз на своём месте."
    )

    @pytest.mark.parametrize("repeats", ["next", "previous"])
    def test_a_part_that_repeats_a_neighbouring_marker_is_not_joined(self, repeats):
        """ch52_c02: joining this would print a paragraph twice, whatever the length says."""
        long_source = f"<p>{LONG_NARRATION * 3}</p>"
        split = f"<p>{RU_LONG_NARRATION * 3}</p>\n<p>{self.NEXT_LINE}</p>"
        neighbour = f"<p>{self.NEXT_LINE}</p>"
        short_source = f"<p>{LONG_NARRATION}</p>"
        if repeats == "next":
            text = _reply(split, neighbour)
            source = [long_source, short_source]
        else:
            text = _reply(neighbour, split)
            source = [short_source, long_source]

        assert join_split_paragraphs(text, source) == (text, [])
        with pytest.raises(ValueError, match="not well-formed"):
            parse_paragraphs(text, 2)

    def test_a_wrong_number_of_markers_is_never_repaired(self):
        text = _reply("<p>Раз.</p>\n<p>Два.</p>", "<p>Три.</p>")
        assert join_split_paragraphs(text, ["<p>One and two.</p>"]) == (text, [])

    def test_no_markers_is_left_alone(self):
        assert join_split_paragraphs("<p>Раз.</p>", ["<p>One.</p>"]) == ("<p>Раз.</p>", [])


class TestParseParagraphs:
    def test_one_broken_paragraph_is_named(self):
        with pytest.raises(ValueError, match="Paragraph 2 of 3: "):
            parse_paragraphs(_reply("<p>Раз</p>", "<p>Два", "<p>Три</p>"), 3)

    def test_every_broken_paragraph_is_named_within_what_the_retry_note_keeps(self):
        """The Tainted Cup ch33_c01 had six; the model was told of one and left another."""
        with pytest.raises(ValueError) as caught:
            parse_paragraphs(_reply("<p>Раз", "<p>Два</p>", "<p>Три", "<p>Четыре"), 4)
        assert "Paragraphs 1, 3, 4 of 4" in str(caught.value)[:300]


def test_the_second_ask_forbids_dropping_text_to_fit_the_count():
    note = RETRY_NOTE.format(reason="x", expected=3)
    assert "exactly 3" in note
    assert "drop" in note.lower()


def test_the_second_ask_says_the_last_marker_holds_the_last_paragraph():
    """Told only «exactly N», the model kept splitting and stopped at marker N."""
    note = " ".join(RETRY_NOTE.format(reason="x", expected=3).split())
    assert "marker 3 holds the last source paragraph" in note
