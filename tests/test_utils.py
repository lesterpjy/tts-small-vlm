"""Tests for src/utils.py - answer extraction, normalization, data classes."""

import json
import tempfile
from pathlib import Path

import pytest

from src.utils import (
    Candidate,
    PipelineResult,
    SelectionResult,
    extract_answer,
    load_config,
    normalize_answer_key,
    save_results,
)


# --- normalize_answer_key ---


class TestNormalizeAnswerKey:
    def test_latin_uppercase(self):
        assert normalize_answer_key("A") == "A"
        assert normalize_answer_key("E") == "E"

    def test_latin_lowercase(self):
        assert normalize_answer_key("b") == "B"
        assert normalize_answer_key("d") == "D"

    def test_cyrillic(self):
        assert normalize_answer_key("А") == "A"  # Cyrillic A
        assert normalize_answer_key("Б") == "B"
        assert normalize_answer_key("Д") == "E"

    def test_numeric(self):
        assert normalize_answer_key("1") == "A"
        assert normalize_answer_key("3") == "C"
        assert normalize_answer_key("5") == "E"

    def test_invalid(self):
        assert normalize_answer_key("F") is None
        assert normalize_answer_key("0") is None
        assert normalize_answer_key("6") is None
        assert normalize_answer_key("") is None

    def test_whitespace(self):
        assert normalize_answer_key("  A  ") == "A"


# --- extract_answer ---


class TestExtractAnswer:
    def test_explicit_pattern(self):
        assert extract_answer("The answer is B") == "B"
        assert extract_answer("The answer is C.") == "C"
        assert extract_answer("answer: D") == "D"

    def test_single_letter(self):
        assert extract_answer("A") == "A"
        assert extract_answer("C.") == "C"

    def test_no_explicit_answer_returns_none(self):
        # No "the answer is" / "final answer" pattern -> return None instead
        # of guessing the last letter (the previous Strategy-3 fallback
        # biased toward whatever the model was enumerating last at
        # truncation, causing a ~100% "A" collapse in the first smoke).
        assert extract_answer("I think it's between A and C, so C") is None

    def test_cot_with_answer(self):
        text = "Let me analyze... Option A is wrong. Option B seems right. The answer is B"
        assert extract_answer(text) == "B"

    def test_empty(self):
        assert extract_answer("") is None
        assert extract_answer(None) is None

    def test_no_answer(self):
        assert extract_answer("I don't know the answer to this question.") is None

    def test_case_insensitive(self):
        assert extract_answer("the answer is b") == "B"


# --- Candidate dataclass ---


class TestCandidate:
    def test_uid(self):
        c = Candidate(description_id=2, chain_id=3, description="desc",
                       reasoning="reason", answer="A")
        assert c.uid == "d2_c3"

    def test_optional_logprob(self):
        c = Candidate(description_id=0, chain_id=0, description="d",
                       reasoning="r", answer="B")
        assert c.logprob is None

        c2 = Candidate(description_id=0, chain_id=0, description="d",
                        reasoning="r", answer="B", logprob=-1.5)
        assert c2.logprob == -1.5


# --- load_config ---


class TestLoadConfig:
    def test_load_dtr_config(self):
        config = load_config("configs/search/q25_dtr_n4m4.yaml")
        assert "model" in config
        assert "describe" in config
        assert "reason" in config
        assert "verify" in config
        assert config["describe"]["n_samples"] == 4
        assert config["reason"]["m_samples"] == 4


# --- save_results ---


class TestSaveResults:
    def test_save_competition_format(self):
        results = [
            PipelineResult(question_id="q1", language="English", predicted_answer="A"),
            PipelineResult(question_id="q2", language="French", predicted_answer="C"),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "preds.json"
            save_results(results, path)

            with open(path) as f:
                data = json.load(f)

            assert len(data) == 2
            assert data[0]["id"] == "q1"
            assert data[0]["prediction"] == "A"
            assert data[0]["language"] == "English"
            assert data[1]["id"] == "q2"
            assert data[1]["prediction"] == "C"
