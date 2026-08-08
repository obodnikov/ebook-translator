"""Tests for model_json.py: reading JSON out of a model's reply."""

from __future__ import annotations

import json

import pytest

from booktranslator import model_json

# The verdict that broke a real judging run: the model quoted English dialogue
# and left the straight quotes inside the JSON string unescaped.
REAL_BROKEN_VERDICT = (
    '{\n  "score": 3,\n  "issues": [\n'
    '    "accuracy: p.10 «и одновременно — Нет, — сказала Ада» — оригинал не '
    'указывает на одновременность; там просто «and "No," Ada said, but then '
    'felt»; смысл сдвинут",\n'
    '    "grammar: p.3 «было тридцать одного года» → «был тридцать один год»"\n'
    "  ]\n}"
)


class TestValidInput:
    def test_plain_object(self):
        assert model_json.loads('{"score": 4, "issues": []}') == {"score": 4, "issues": []}

    def test_code_fences(self):
        assert model_json.loads('```json\n{"score": 5, "issues": []}\n```')["score"] == 5

    def test_trailing_commentary(self):
        text = '{"score": 5, "issues": []}\n\nПояснение: перевод хороший.'
        assert model_json.loads(text)["score"] == 5

    def test_array_of_mixed_values_is_untouched(self):
        """A valid document must never be rewritten by the repair pass."""
        assert model_json.loads('["a", 3, null, {"b": "c"}]') == ["a", 3, None, {"b": "c"}]

    def test_existing_escapes_survive(self):
        assert model_json.loads(r'{"why": "он сказал \"да\""}') == {"why": 'он сказал "да"'}

    def test_backslash_before_closing_quote(self):
        assert model_json.loads(r'{"path": "C:\\dir\\"}') == {"path": "C:\\dir\\"}


class TestStrayQuotes:
    def test_the_verdict_that_broke_a_real_run(self):
        data = model_json.loads(REAL_BROKEN_VERDICT)
        assert data["score"] == 3
        assert len(data["issues"]) == 2
        # The quotation is preserved, not dropped or paraphrased.
        assert '«and "No," Ada said, but then felt»' in data["issues"][0]

    def test_stray_quote_before_a_comma(self):
        """The comma inside the quotation must not be read as a separator."""
        data = model_json.loads('[{"n": 1, "why": "он сказал "да", коротко"}]')
        assert data == [{"n": 1, "why": 'он сказал "да", коротко'}]

    def test_stray_quote_at_the_end_of_a_value(self):
        data = model_json.loads('{"why": "он сказал "да""}')
        assert data == {"why": 'он сказал "да"'}

    def test_several_stray_quotes_in_one_string(self):
        data = model_json.loads('{"issues": ["a "b" c "d" e"]}')
        assert data == {"issues": ['a "b" c "d" e']}


class TestFailures:
    def test_prose_raises(self):
        with pytest.raises(json.JSONDecodeError):
            model_json.loads("Не могу выполнить эту задачу.")

    def test_empty_raises(self):
        with pytest.raises(json.JSONDecodeError):
            model_json.loads("")

    def test_truncated_reply_raises(self):
        """Cut-off replies are not this module's job — callers salvage them."""
        with pytest.raises(json.JSONDecodeError):
            model_json.loads('{"score": 3, "issues": ["accuracy: p.1 «непол')


class TestStripCodeFences:
    def test_removes_fences(self):
        assert model_json.strip_code_fences("```json\n{}\n```") == "{}"

    def test_leaves_unfenced_text(self):
        assert model_json.strip_code_fences("  {}  ") == "{}"
