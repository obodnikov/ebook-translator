"""Text a model damaged while every marker and tag stayed right.

All shapes come from the first run of A Drop of Corruption, with the text rewritten:
163 chunks, and five of those accepted held a broken character, a word in two
alphabets, an English word, or a paragraph that stopped in the middle.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from booktranslator.cache import Cache
from booktranslator.models import SeriesGlossary, SeriesGlossaryEntry
from booktranslator.postprocess import PostProcessor
from booktranslator.provider import CompletionResult
from booktranslator.repair import Repairer, RepairStats
from booktranslator.replies import ReplyChecker, check_text, untranslated_words
from booktranslator.series import translations_text

SOURCE = "<p>The old kettle began to burble on the stove. I watched the steam rise slowly.</p>"
GOOD = "<p>Старый чайник забулькал на плите. Я смотрел, как медленно поднимается пар.</p>"


class TestCheckText:
    def test_a_sound_translation_passes(self):
        check_text([SOURCE], [GOOD])

    @pytest.mark.parametrize(
        "reply, reason",
        [
            (GOOD.replace("медленно", "медл��нно"), "broken character"),
            (GOOD.replace("поднимается", "podнимается"), "two alphabets"),
            (GOOD.replace("забулькал", "burbling-забулькал"), "two alphabets"),
            # A Latin letter that looks like the Russian one.
            (GOOD.replace("плите", "плитe"), "two alphabets"),
            ("<p>Старый чайник на плите began to</p>", "cut short"),
        ],
    )
    def test_damage_is_rejected(self, reply, reason):
        with pytest.raises(ValueError, match=f"Paragraph 1 of 1: .*{reason}"):
            check_text([SOURCE], [reply])

    def test_every_damaged_paragraph_is_named(self):
        bad = GOOD.replace("плите", "плитe")
        with pytest.raises(ValueError, match="Paragraphs 1, 3 of 3"):
            check_text([SOURCE] * 3, [bad, GOOD, bad])

    @pytest.mark.parametrize(
        "words",
        [
            "IT-отдел",
            "GPS-навигатор",
            "Wi-Fi-роутер",
            "e-mail-адрес",
            "x-лучи",
            "web-сайт",
            "online-игра",
            "iPhone-чехол",
            "бо́льшая часть",
            "Тусим хо, Ki iha mire",
        ],
    )
    def test_latin_that_russian_uses_is_not_damage(self, words):
        check_text([SOURCE], [GOOD.replace("пар.", f"пар, {words}.")])

    def test_tags_between_words_do_not_join_them(self):
        reply = GOOD.replace("плите.", "плите<br/>posse<sup>a</sup>.")
        check_text([SOURCE], [reply])

    def test_a_word_of_the_source_is_not_damage(self):
        source = SOURCE.replace("kettle", "kettle of the Заводская-Бees")
        check_text([source], [GOOD.replace("чайник", "чайник Заводская-Бees")])

    def test_a_word_the_glossary_prescribes_is_not_damage(self):
        """Otherwise every chunk with the term would fail both asks and stop the book."""
        glossary = SeriesGlossary(
            series_slug="s",
            title="T",
            author="A",
            source_lang="en",
            target_lang="ru",
            entries=[SeriesGlossaryEntry(original="Bees", translation="Бees", type="concept")],
        )
        reply = GOOD.replace("чайник", "чайник Бees")
        check_text([SOURCE], [reply], allowed=translations_text(glossary))

    def test_a_short_line_without_a_full_stop_is_not_cut_short(self):
        check_text(["<p>The end of it all.</p>"], ["<p>Конец</p>"])

    def test_a_paragraph_the_source_leaves_open_is_not_cut_short(self):
        check_text([SOURCE.replace("slowly.", "slowly")], ["<p>Чайник забулькал на плите</p>"])


class TestEnglishWords:
    def test_a_word_left_in_english_is_found(self):
        reply = "<p>Потом они снова began стучать, и пальцы их trouble не знали.</p>"
        assert untranslated_words(reply) == ["began", "trouble"]

    @pytest.mark.parametrize(
        "reply",
        [
            "<p>— Ki iha mire lai koi hai?</p>",
            "<p>Запись доступна по адресу https://example.com/12</p>",
            "<p>Он открыл web версию журнала.</p>",
        ],
    )
    def test_foreign_speech_addresses_and_loanwords_are_not(self, reply):
        assert untranslated_words(reply) == []


def _completion(*fragments: str) -> CompletionResult:
    text = "\n".join(f"===PARAGRAPH {n}===\n{f}" for n, f in enumerate(fragments, start=1))
    return CompletionResult(
        text=text,
        input_tokens=1,
        output_tokens=1,
        total_tokens=2,
        model="m",
        raw={},
        finish_reason="stop",
    )


class TestReplyChecker:
    @staticmethod
    def _ask(complete):
        checker = ReplyChecker(
            stage="translate", model="m", prompt_version="1", cache_path=None, lock=threading.Lock()
        )
        stats = SimpleNamespace(chunks_total=10, retries=0, input_tokens=0, output_tokens=0)
        return checker.ask(complete, "s", "u", chunk_id="c1", source=[SOURCE], stats=stats)

    def test_a_damaged_reply_is_asked_for_again(self):
        complete = MagicMock(
            side_effect=[_completion(GOOD.replace("плите", "плитe")), _completion(GOOD)]
        )

        reply = self._ask(complete)

        assert reply.elements[0].text == GOOD[3:-4]
        assert "two alphabets" in complete.call_args.args[1]

    def test_an_english_word_is_only_reported(self, caplog):
        """It can be meant, as `nia` in a Pithian line, so it never costs a re-ask."""
        complete = MagicMock(return_value=_completion(GOOD.replace("забулькал", "began булькать")))

        self._ask(complete)

        assert complete.call_count == 1
        assert "began" in caplog.text
        assert "c1" in caplog.text


@pytest.fixture
def cache(tmp_path: Path):
    cache = Cache(tmp_path / "cache.sqlite")
    yield cache
    cache.close()


class TestEditsThatDamageText:
    @pytest.fixture
    def proofreader(self, cache):
        return PostProcessor(
            MagicMock(), Path("prompts/proofread.md"), cache, None, stage="proofread", model="m"
        )

    @pytest.mark.parametrize(
        "patch",
        [
            GOOD.replace("плите", "плитe"),
            GOOD.replace("медленно", "медл�нно"),
            "<p>Старый чайник на плите</p>",
        ],
    )
    def test_the_patch_is_dropped_and_the_others_stand(self, proofreader, patch):
        other = "<p>Я кивнул.</p>"
        patches = [{"p": 1, "text": patch}, {"p": 2, "text": other}]

        output, changed = proofreader._apply_patches(patches, [GOOD, "<p>Я кивнл.</p>"])

        assert output == [GOOD, other]
        assert changed

    def test_damage_that_was_already_there_does_not_block_an_edit(self, proofreader):
        before = GOOD.replace("плите", "плитe")
        after = before.replace("медленно", "неспешно")

        output, _ = proofreader._apply_patches([{"p": 1, "text": after}], [before])

        assert output == [after]

    def test_an_english_word_in_a_patch_is_only_reported(self, proofreader, caplog):
        after = GOOD.replace("забулькал", "began булькать")

        output, _ = proofreader._apply_patches([{"p": 1, "text": after}], [GOOD])

        assert output == [after]
        assert "began" in caplog.text

    def test_a_full_text_answer_keeps_the_paragraph_it_damaged(self, proofreader, cache):
        damaged = GOOD.replace("плите", "плитe")
        proofreader.provider.complete.return_value = _completion(damaged, "<p>Я кивнул.</p>")
        stats = MagicMock(chunks_processed=0, input_tokens=0, output_tokens=0, results=[])

        result = proofreader.process_chunk("c1", [GOOD, "<p>Я кивнл.</p>"], stats)

        assert "плитe" not in result.content
        assert "Я кивнул." in result.content
        [row] = cache.get_chunk_stages("c1")
        assert "плитe" not in row.content

    def test_a_repair_that_damages_text_counts_as_rejected(self, cache):
        provider = MagicMock()
        verdict = _completion()
        verdict.text = json.dumps([{"n": 1, "verdict": "accept", "why": "-"}])
        provider.complete.return_value = verdict
        repairer = Repairer(provider, Path("prompts/repair.md"), cache, None, model="m")

        result = repairer.repair_chunk(
            "c1", [GOOD], ["grammar: p.1 «на плите» → «на плитe»"], SOURCE, RepairStats()
        )

        assert result.paragraphs == [GOOD]
        assert not result.accepted
        assert "repair" not in cache.list_stages()
