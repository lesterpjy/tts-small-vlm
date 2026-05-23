"""Tests for src/search.py - PRM-BAS beam-annealing search.

All tests use mocked generation and scoring (no GPU).
"""

from unittest.mock import MagicMock, patch

import pytest

from src.backend import GenerationOutput
from src.search import (
    Beam,
    _anneal,
    _check_terminated,
    _top_w,
    beams_to_candidates,
    prm_bas,
    record_prm_bas_metadata,
)


# --- Helpers ---


def _mock_gen_output(text: str, logprob: float = -1.0) -> GenerationOutput:
    return GenerationOutput(
        text=text, logprob=logprob,
        prompt_tokens=10, completion_tokens=5, latency_s=0.1,
    )


def _constant_scorer(value: float = 0.8):
    """Step scorer that always returns the same value."""
    def scorer(prior_steps, current_step):
        return value
    return scorer


def _length_scorer():
    """Synthetic scorer: prefers longer steps."""
    def scorer(prior_steps, current_step):
        return min(len(current_step) / 100.0, 1.0)
    return scorer


def _score_by_step_index():
    """Scorer that returns higher scores for later steps (more prior_steps)."""
    def scorer(prior_steps, current_step):
        return min((len(prior_steps) + 1) * 0.15, 1.0)
    return scorer


def _divergent_scorer():
    """Scorer returning values with large spread (> default tau=0.05)."""
    scores = iter([0.9, 0.8, 0.7, 0.1, 0.05, 0.02, 0.01, 0.005])
    def scorer(prior_steps, current_step):
        return next(scores, 0.5)
    return scorer


def _tight_scorer():
    """Scorer returning values within tau=0.05 (should NOT anneal)."""
    scores = iter([0.82, 0.81, 0.80, 0.79, 0.78, 0.80, 0.81, 0.79])
    def scorer(prior_steps, current_step):
        return next(scores, 0.8)
    return scorer


# --- Beam dataclass ---


class TestBeam:
    def test_final_score_is_mean_of_step_scores(self):
        beam = Beam(steps=["a", "b", "c"], scores=[0.6, 0.8, 1.0], logprobs=[-1, -1, -1])
        assert beam.final_score == pytest.approx(0.8)

    def test_final_score_empty(self):
        beam = Beam()
        assert beam.final_score == 0.0

    def test_reasoning_joins_steps(self):
        beam = Beam(steps=["Step 1", "Step 2", "Answer: A"], scores=[0.5, 0.6, 0.7])
        assert beam.reasoning == "Step 1\n\nStep 2\n\nAnswer: A"

    def test_depth_is_step_count(self):
        beam = Beam(steps=["a", "b", "c"], scores=[0.5, 0.6, 0.7])
        assert beam.depth == 3


# --- Helper functions ---


class TestCheckTerminated:
    def test_detects_answer_colon(self):
        assert _check_terminated("Therefore, Answer: B", "answer:")

    def test_case_insensitive(self):
        assert _check_terminated("ANSWER: C", "answer:")

    def test_no_match(self):
        assert not _check_terminated("Let me think about this", "answer:")

    def test_the_answer_is_pattern(self):
        assert _check_terminated("The answer is A", "the answer is")


class TestAnneal:
    def test_halves_when_spread_exceeds_tau(self):
        actives = [
            Beam(steps=["a"], scores=[0.9]),
            Beam(steps=["b"], scores=[0.3]),
        ]
        assert _anneal(actives, W=4, tau=0.05) == 2

    def test_keeps_width_when_spread_within_tau(self):
        actives = [
            Beam(steps=["a"], scores=[0.82]),
            Beam(steps=["b"], scores=[0.80]),
        ]
        assert _anneal(actives, W=4, tau=0.05) == 4

    def test_no_anneal_single_active(self):
        actives = [Beam(steps=["a"], scores=[0.9])]
        assert _anneal(actives, W=4, tau=0.05) == 4

    def test_no_anneal_empty(self):
        assert _anneal([], W=4, tau=0.05) == 4

    def test_minimum_width_is_one(self):
        actives = [
            Beam(steps=["a"], scores=[0.9]),
            Beam(steps=["b"], scores=[0.1]),
        ]
        assert _anneal(actives, W=1, tau=0.05) == 1


class TestTopW:
    def test_keeps_top_w_by_latest_score(self):
        beams = [
            Beam(steps=["a"], scores=[0.3]),
            Beam(steps=["b"], scores=[0.9]),
            Beam(steps=["c"], scores=[0.6]),
        ]
        result = _top_w(beams, 2)
        assert len(result) == 2
        assert result[0].scores[-1] == 0.9
        assert result[1].scores[-1] == 0.6


# --- Core PRM-BAS ---


class TestPrmBas:
    @patch("src.search.generate_n")
    def test_initial_beam_is_B0(self, mock_gen):
        """First generation call samples B0 continuations."""
        mock_gen.return_value = [
            _mock_gen_output(f"Step {i}") for i in range(4)
        ]
        terminals, meta = prm_bas(
            MagicMock(), MagicMock(), "test description",
            _constant_scorer(0.5),
            B0=4, B=2, max_depth=1, anneal_tau=0.05,
        )
        first_call_n = mock_gen.call_args_list[0][0][3]  # positional arg: n (model, processor, prompt, n)
        assert first_call_n == 4

    @patch("src.search.generate_n")
    def test_anneals_width_when_spread_exceeds_tau(self, mock_gen):
        """Width halves when score spread > tau."""
        call_count = [0]
        def gen_side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return [_mock_gen_output(f"Step {i}") for i in range(4)]
            return [_mock_gen_output(f"Continue {i}") for i in range(2)]

        mock_gen.side_effect = gen_side_effect

        scores = iter([0.9, 0.3, 0.2, 0.1])
        def divergent(prior, step):
            return next(scores, 0.5)

        terminals, meta = prm_bas(
            MagicMock(), MagicMock(), "test desc",
            divergent,
            B0=4, B=2, max_depth=2, anneal_tau=0.05,
        )
        assert meta["width_schedule"][0] < 4

    @patch("src.search.generate_n")
    def test_keeps_width_when_spread_within_tau(self, mock_gen):
        """Width stays if score spread <= tau."""
        call_count = [0]
        def gen_side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return [_mock_gen_output(f"Step {i}") for i in range(4)]
            return [_mock_gen_output(f"Continue {i}") for i in range(2)]

        mock_gen.side_effect = gen_side_effect

        scores = iter([0.82, 0.81, 0.80, 0.79])
        def tight(prior, step):
            return next(scores, 0.8)

        terminals, meta = prm_bas(
            MagicMock(), MagicMock(), "test desc",
            tight,
            B0=4, B=2, max_depth=2, anneal_tau=0.05,
        )
        assert meta["width_schedule"][0] == 4

    @patch("src.search.generate_n")
    def test_terminates_on_stop_marker(self, mock_gen):
        """Beams containing stop_marker are moved to terminals."""
        mock_gen.return_value = [
            _mock_gen_output("Let me think..."),
            _mock_gen_output("The option is... Answer: B"),
            _mock_gen_output("Step reasoning"),
            _mock_gen_output("Final: Answer: A"),
        ]
        terminals, meta = prm_bas(
            MagicMock(), MagicMock(), "test desc",
            _constant_scorer(0.7),
            B0=4, B=2, max_depth=1, stop_marker="answer:",
        )
        terminated_with_answer = [
            t for t in terminals
            if "answer:" in t.reasoning.lower()
        ]
        assert len(terminated_with_answer) == 2

    @patch("src.search.generate_n")
    def test_respects_max_depth(self, mock_gen):
        """Search stops at max_depth even without termination."""
        mock_gen.return_value = [_mock_gen_output("Step text")]

        terminals, meta = prm_bas(
            MagicMock(), MagicMock(), "test desc",
            _constant_scorer(0.5),
            B0=1, B=1, max_depth=3, stop_marker="NEVER_MATCH",
        )
        assert all(t.depth <= 3 for t in terminals)
        assert meta["max_depth_reached"] <= 3

    @patch("src.search.generate_n")
    def test_returns_sorted_terminals(self, mock_gen):
        """Terminals sorted descending by final_score."""
        mock_gen.return_value = [
            _mock_gen_output("Answer: A"),
            _mock_gen_output("Answer: B"),
            _mock_gen_output("Answer: C"),
        ]
        scores = iter([0.3, 0.9, 0.6])
        def scorer(prior, step):
            return next(scores, 0.5)

        terminals, meta = prm_bas(
            MagicMock(), MagicMock(), "test desc",
            scorer,
            B0=3, B=1, max_depth=1, stop_marker="answer:",
        )
        final_scores = [t.final_score for t in terminals]
        assert final_scores == sorted(final_scores, reverse=True)

    @patch("src.search.generate_n")
    def test_metadata_accounts_scorer_calls(self, mock_gen):
        """Metadata total_scorer_calls matches actual invocations."""
        call_count = [0]
        def gen_side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return [_mock_gen_output(f"Step {i}") for i in range(3)]
            return [_mock_gen_output("Answer: A")]

        mock_gen.side_effect = gen_side_effect

        scorer_calls = [0]
        def counting_scorer(prior, step):
            scorer_calls[0] += 1
            return 0.5

        terminals, meta = prm_bas(
            MagicMock(), MagicMock(), "test desc",
            counting_scorer,
            B0=3, B=1, max_depth=2, stop_marker="answer:",
        )
        assert meta["total_scorer_calls"] == scorer_calls[0]
        assert meta["total_scorer_calls"] > 0

    @patch("src.search.generate_n")
    def test_empty_steps_skipped(self, mock_gen):
        """Beams from empty generation outputs are skipped."""
        mock_gen.return_value = [
            _mock_gen_output(""),
            _mock_gen_output("   "),
            _mock_gen_output("Real step"),
        ]
        terminals, meta = prm_bas(
            MagicMock(), MagicMock(), "test desc",
            _constant_scorer(0.5),
            B0=3, B=1, max_depth=1,
        )
        assert all(len(t.steps[0].strip()) > 0 for t in terminals)

    @patch("src.search.generate_n")
    def test_width_schedule_monotonic_non_increasing(self, mock_gen):
        """Width schedule should be monotonic non-increasing."""
        call_count = [0]
        def gen_side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return [_mock_gen_output(f"Step {i}") for i in range(8)]
            return [_mock_gen_output(f"Continue {i}") for i in range(4)]

        mock_gen.side_effect = gen_side_effect

        scores = iter([0.95, 0.9, 0.85, 0.8, 0.1, 0.05, 0.02, 0.01] + [0.5] * 100)
        def scorer(prior, step):
            return next(scores, 0.5)

        terminals, meta = prm_bas(
            MagicMock(), MagicMock(), "test desc",
            scorer,
            B0=8, B=4, max_depth=3, anneal_tau=0.05,
        )
        schedule = meta["width_schedule"]
        for i in range(1, len(schedule)):
            assert schedule[i] <= schedule[i - 1]


# --- beams_to_candidates ---


class TestBeamsToCandidates:
    def test_converts_beams_to_candidates(self):
        beams = [
            Beam(
                steps=["Think about it", "Answer: A"],
                scores=[0.7, 0.9],
                logprobs=[-0.5, -0.3],
                terminated=True,
            ),
            Beam(
                steps=["Consider options", "Answer: B"],
                scores=[0.6, 0.8],
                logprobs=[-0.6, -0.4],
                terminated=True,
            ),
        ]
        candidates = beams_to_candidates(beams, description_id=2, description="test desc")
        assert len(candidates) == 2
        assert candidates[0].description_id == 2
        assert candidates[0].chain_id == 0
        assert candidates[1].chain_id == 1
        assert candidates[0].description == "test desc"
        assert "Answer: A" in candidates[0].reasoning

    def test_extracts_answer_from_reasoning(self):
        beams = [
            Beam(
                steps=["Step 1", "The answer is C"],
                scores=[0.5, 0.8],
                logprobs=[-1.0, -0.5],
                terminated=True,
            ),
        ]
        candidates = beams_to_candidates(beams, 0, "desc")
        assert candidates[0].answer == "C"

    def test_logprob_is_mean(self):
        beams = [
            Beam(
                steps=["a", "b"],
                scores=[0.5, 0.8],
                logprobs=[-2.0, -1.0],
                terminated=True,
            ),
        ]
        candidates = beams_to_candidates(beams, 0, "desc")
        assert candidates[0].logprob == pytest.approx(-1.5)


# --- Metadata ---


class TestRecordMetadata:
    def test_records_all_fields(self):
        terminals = [
            Beam(steps=["a", "b"], scores=[0.7, 0.8]),
            Beam(steps=["c"], scores=[0.6]),
        ]
        meta = record_prm_bas_metadata(terminals, [4, 2], 12, 5)
        assert meta["total_scorer_calls"] == 12
        assert meta["total_generate_calls"] == 5
        assert meta["width_schedule"] == [4, 2]
        assert meta["n_terminals"] == 2
        assert len(meta["terminal_scores"]) == 2
        assert meta["max_depth_reached"] == 2
