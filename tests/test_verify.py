"""Tests for src/verify.py - majority vote, tie-breaking, agentic verification."""

from unittest.mock import MagicMock, patch

import pytest

from src.utils import Candidate
from src.verify import majority_vote, select_answer


def make_candidate(answer: str, desc_id: int = 0, chain_id: int = 0,
                   logprob: float | None = None) -> Candidate:
    return Candidate(
        description_id=desc_id, chain_id=chain_id,
        description="test desc", reasoning="test reasoning",
        answer=answer, logprob=logprob,
    )


# --- Majority vote ---


class TestMajorityVote:
    def test_clear_majority(self):
        """8 correct, 4 wrong -> picks correct answer."""
        candidates = (
            [make_candidate("A", chain_id=i) for i in range(8)]
            + [make_candidate("B", chain_id=i + 8) for i in range(4)]
        )
        result = majority_vote(candidates)

        assert result.answer == "A"
        assert result.confidence == "high"
        assert result.vote_counts["A"] == 8
        assert result.vote_counts["B"] == 4

    def test_plurality(self):
        """6A, 5B, 5C -> picks A with medium confidence."""
        candidates = (
            [make_candidate("A", chain_id=i) for i in range(6)]
            + [make_candidate("B", chain_id=i + 6) for i in range(5)]
            + [make_candidate("C", chain_id=i + 11) for i in range(5)]
        )
        result = majority_vote(candidates)

        assert result.answer == "A"
        assert result.confidence == "medium"

    def test_exact_tie_breaks_by_logprob(self):
        """4A vs 4B with different logprobs -> picks answer with best logprob."""
        candidates = (
            [make_candidate("A", chain_id=i, logprob=-2.0) for i in range(4)]
            + [make_candidate("B", chain_id=i + 4, logprob=-1.0) for i in range(4)]
        )
        result = majority_vote(candidates)

        assert result.answer == "B"
        assert result.confidence == "low"
        assert "tiebreak" in result.metadata

    def test_all_disagree(self):
        """4 different answers -> low confidence, best logprob wins."""
        candidates = [
            make_candidate("A", chain_id=0, logprob=-3.0),
            make_candidate("B", chain_id=1, logprob=-1.0),
            make_candidate("C", chain_id=2, logprob=-2.0),
            make_candidate("D", chain_id=3, logprob=-4.0),
        ]
        result = majority_vote(candidates)

        assert result.answer == "B"
        assert result.confidence == "low"

    def test_no_valid_candidates(self):
        """All candidates have answer=None -> returns None."""
        candidates = [
            make_candidate(None, chain_id=0),
            make_candidate(None, chain_id=1),
        ]
        result = majority_vote(candidates)

        assert result.answer is None
        assert result.confidence == "very_low"

    def test_single_candidate(self):
        """One candidate -> high confidence."""
        result = majority_vote([make_candidate("C")])
        assert result.answer == "C"
        assert result.confidence == "high"

    def test_skips_none_answers_in_vote(self):
        """None answers are excluded from voting."""
        candidates = [
            make_candidate("A", chain_id=0),
            make_candidate("A", chain_id=1),
            make_candidate(None, chain_id=2),
        ]
        result = majority_vote(candidates)

        assert result.answer == "A"
        assert result.vote_counts["A"] == 2

    def test_empty_candidates(self):
        result = majority_vote([])
        assert result.answer is None


# --- select_answer dispatcher ---


class TestSelectAnswer:
    def test_majority_vote_dispatch(self):
        candidates = [make_candidate("A") for _ in range(4)]
        result = select_answer(candidates, method="majority_vote")
        assert result.answer == "A"
        assert result.method == "majority_vote"

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown verification method"):
            select_answer([], method="nonexistent")

    def test_agentic_requires_model(self):
        with pytest.raises(ValueError, match="model"):
            select_answer([make_candidate("A")], method="agentic")

    def test_visualprm_falls_back(self):
        """VisualPRM falls back to majority vote with a warning."""
        candidates = [make_candidate("B") for _ in range(4)]
        result = select_answer(candidates, method="visualprm")
        assert result.answer == "B"
        assert result.metadata.get("fallback_from") == "visualprm"

    def test_generative_requires_model(self):
        with pytest.raises(ValueError, match="model, processor, and image"):
            select_answer([make_candidate("A")], method="generative")


# --- Agentic verification ---


class TestAgenticVerify:
    def test_skips_when_majority_strong(self):
        """With 12/16 votes for A, agentic should skip verification."""
        from src.verify import agentic_verify

        candidates = (
            [make_candidate("A", chain_id=i) for i in range(12)]
            + [make_candidate("B", chain_id=i + 12) for i in range(4)]
        )

        from PIL import Image
        img = Image.new("RGB", (10, 10))

        # generate should NOT be called when majority is skipped
        with patch("src.verify.generate") as mock_gen:
            result = agentic_verify(MagicMock(), MagicMock(), img, candidates, skip_threshold=0.75)

        assert result.answer == "A"
        assert result.method == "agentic_skip"
        mock_gen.assert_not_called()


# --- Generative critic (GM-PRM-style) ---


def _fake_gen_output(text: str):
    """Minimal stand-in for `src.backend.GenerationOutput`, only the `text`
    attribute is read by the critic."""
    m = MagicMock()
    m.text = text
    m.prompt_tokens = 0
    m.completion_tokens = 0
    m.latency_s = 0.0
    m.logprob = None
    return m


def _img():
    from PIL import Image
    return Image.new("RGB", (10, 10))


class TestGenerativeCriticParser:
    def test_parses_fenced_json(self):
        from src.verify import _parse_critic_score
        score, reason = _parse_critic_score('```json\n{"score": 4, "reason": "ok"}\n```')
        assert score == 4
        assert reason == "ok"

    def test_parses_bare_json(self):
        from src.verify import _parse_critic_score
        score, _ = _parse_critic_score('some preamble {"score": 3, "reason": "fine"} trailing')
        assert score == 3

    def test_parses_score_regex_fallback(self):
        from src.verify import _parse_critic_score
        # Malformed JSON but a recognisable score key/value
        score, reason = _parse_critic_score('score broken: "score": 5 - see justification')
        assert score == 5
        assert reason is None

    def test_rejects_out_of_range(self):
        from src.verify import _parse_critic_score
        # 7 is outside 1..5 and the bare-object parser should reject it.
        # Regex fallback explicitly anchors to [1-5] so it won't match either.
        score, _ = _parse_critic_score('{"score": 7, "reason": "bogus"}')
        assert score is None

    def test_handles_empty_text(self):
        from src.verify import _parse_critic_score
        assert _parse_critic_score("") == (None, None)
        assert _parse_critic_score("I think this is fine.") == (None, None)


class TestGenerativeCriticScore:
    def test_averages_three_axes(self):
        """Per-axis scores [4, 2, 5] -> mapped [0.75, 0.25, 1.0] -> mean 0.667."""
        from src.verify import generative_critic_score
        texts = iter([
            '{"score": 4, "reason": "intent fine"}',
            '{"score": 2, "reason": "visual weak"}',
            '{"score": 5, "reason": "logic solid"}',
        ])
        with patch("src.verify.generate",
                   side_effect=lambda *a, **kw: _fake_gen_output(next(texts))):
            s = generative_critic_score(
                MagicMock(), MagicMock(), _img(),
                make_candidate("A"),
            )
        assert abs(s - (0.75 + 0.25 + 1.0) / 3) < 1e-9

    def test_handles_malformed_output_neutral(self):
        """Unparseable -> 0.5 neutral on that axis; all-axis failure -> 0.5 overall."""
        from src.verify import generative_critic_score
        with patch("src.verify.generate",
                   return_value=_fake_gen_output("I dunno, seems fine?")):
            s = generative_critic_score(
                MagicMock(), MagicMock(), _img(),
                make_candidate("A"),
            )
        assert s == 0.5

    def test_clamped_to_unit_interval(self):
        from src.verify import generative_critic_score
        # All 5s -> all map to 1.0 -> mean 1.0. Valid upper edge.
        with patch("src.verify.generate",
                   return_value=_fake_gen_output('{"score": 5, "reason": "perfect"}')):
            s = generative_critic_score(
                MagicMock(), MagicMock(), _img(),
                make_candidate("A"),
            )
        assert s == 1.0

    def test_respects_custom_axes(self):
        """Passing `axes=["step-intent"]` runs one call, scores one axis."""
        from src.verify import generative_critic_score
        with patch("src.verify.generate",
                   return_value=_fake_gen_output('{"score": 3, "reason": "ok"}')) as g:
            s = generative_critic_score(
                MagicMock(), MagicMock(), _img(),
                make_candidate("A"), axes=["step-intent"],
            )
        assert g.call_count == 1
        assert abs(s - 0.5) < 1e-9  # (3-1)/4 = 0.5


class TestGenerativeCriticRank:
    def test_picks_highest_scoring_candidate(self):
        from src.verify import generative_critic_rank
        # Three candidates with different answers; we force per-candidate means
        # of 0.3, 0.8, 0.5 by returning constant per-axis scores.
        score_sequence = {
            "d0_c0": 2,  # (2-1)/4 = 0.25; across 3 axes -> 0.25
            "d0_c1": 5,  # 1.0
            "d0_c2": 3,  # 0.5
        }
        cands = [
            make_candidate("A", desc_id=0, chain_id=0),
            make_candidate("B", desc_id=0, chain_id=1),
            make_candidate("C", desc_id=0, chain_id=2),
        ]

        call_log: list[str] = []

        def fake_generate(*a, **kw):
            # Extract which candidate is being scored by inspecting the prompt's
            # chain-text block (reasoning); we stuffed uid-identifying text on the
            # candidate reasoning field via make_candidate("B") -> "test reasoning".
            # Easier: count calls and cycle through candidates.
            idx = len(call_log) // 3
            call_log.append("x")
            uid = cands[idx].uid
            return _fake_gen_output(f'{{"score": {score_sequence[uid]}, "reason": "ok"}}')

        with patch("src.verify.generate", side_effect=fake_generate):
            result = generative_critic_rank(cands, MagicMock(), MagicMock(), _img())

        assert result.answer == "B"
        assert result.method == "generative"
        # Score gap B (1.0) - C (0.5) = 0.5 -> "high" confidence
        assert result.confidence == "high"
        scores = result.metadata["critic_scores"]
        assert scores["d0_c1"] == 1.0
        assert scores["d0_c0"] == 0.25

    def test_skips_unparseable_candidates(self):
        """Candidates with answer=None are excluded from scoring."""
        from src.verify import generative_critic_rank
        cands = [
            make_candidate(None, chain_id=0),
            make_candidate("A", chain_id=1),
        ]
        with patch("src.verify.generate",
                   return_value=_fake_gen_output('{"score": 4}')) as g:
            result = generative_critic_rank(cands, MagicMock(), MagicMock(), _img())
        # Only one candidate gets scored -> 3 axes -> 3 calls.
        assert g.call_count == 3
        assert result.answer == "A"

    def test_all_unparseable_returns_none(self):
        from src.verify import generative_critic_rank
        cands = [make_candidate(None, chain_id=i) for i in range(3)]
        result = generative_critic_rank(cands, MagicMock(), MagicMock(), _img())
        assert result.answer is None
        assert result.confidence == "very_low"
        assert result.metadata["reason"] == "no_parseable_candidates"

    def test_empty_list_returns_none(self):
        from src.verify import generative_critic_rank
        result = generative_critic_rank([], MagicMock(), MagicMock(), _img())
        assert result.answer is None
        assert result.confidence == "very_low"


class TestGenerativeDispatch:
    def test_dispatch_routes_to_generative(self):
        """method='generative' should reach generative_critic_rank."""
        cands = [make_candidate("A")]
        with patch("src.verify.generative_critic_rank") as mock_rank:
            mock_rank.return_value = MagicMock(spec=["answer", "method"])
            select_answer(
                cands, method="generative",
                model=MagicMock(), processor=MagicMock(), image=_img(),
            )
            mock_rank.assert_called_once()


class TestGenerativeCriticDeterminism:
    def test_same_inputs_same_score(self):
        """Determinism: same (image, chain) with identical mocked critic
        outputs must produce the same score across two invocations. Guards
        against future refactors that add hidden stateful behaviour."""
        from src.verify import generative_critic_score

        cand = make_candidate("A")
        with patch("src.verify.generate",
                   return_value=_fake_gen_output('{"score": 4, "reason": "x"}')):
            s1 = generative_critic_score(MagicMock(), MagicMock(), _img(), cand)
            s2 = generative_critic_score(MagicMock(), MagicMock(), _img(), cand)
        assert s1 == s2

    def test_axis_order_invariance(self):
        """The mean-aggregation across axes is symmetric, reordering the
        axes list must not change the final score."""
        from src.verify import generative_critic_score

        # Per-axis scores keyed by axis name (so order of `axes` argument
        # doesn't change which score comes back from which axis call).
        per_axis = {
            "step-intent": 4,
            "visual-alignment": 2,
            "logical-soundness": 5,
        }

        def make_side_effect(order):
            it = iter(order)
            def _side(*a, **kw):
                axis = next(it)
                return _fake_gen_output(f'{{"score": {per_axis[axis]}}}')
            return _side

        cand = make_candidate("A")
        order_a = ["step-intent", "visual-alignment", "logical-soundness"]
        order_b = ["logical-soundness", "step-intent", "visual-alignment"]

        with patch("src.verify.generate", side_effect=make_side_effect(order_a)):
            s_a = generative_critic_score(MagicMock(), MagicMock(), _img(), cand, axes=order_a)
        with patch("src.verify.generate", side_effect=make_side_effect(order_b)):
            s_b = generative_critic_score(MagicMock(), MagicMock(), _img(), cand, axes=order_b)

        assert abs(s_a - s_b) < 1e-9


# --- Qwen-VL-PRM wrapper ---
#
# All tests mocked, no GPU, no torch model loading. We patch
# src.verify._qwen_vl_prm_forward_score (the single forward-pass primitive)
# so individual tests can prescribe per-call P(+) values; this avoids
# fiddling with torch tensors in unit tests while still exercising the
# scoring + ranking + dispatch logic end-to-end.


def _fake_pmtok():
    """Mock tokenizer that resolves '+' / '-' to known ids."""
    m = MagicMock()
    m.encode = MagicMock(side_effect=lambda s, **kw: [101] if s == "+" else [102])
    return m


class TestQwenVlPrmStepSplit:
    def test_split_double_newline(self):
        from src.verify import _split_into_steps
        text = "Step 1: think\n\nStep 2: conclude\n\nAnswer: A"
        assert _split_into_steps(text) == [
            "Step 1: think", "Step 2: conclude", "Answer: A",
        ]

    def test_split_single_newlines_treated_as_one_step(self):
        from src.verify import _split_into_steps
        # No blank line => one step. Per-step PRM is then equivalent to one_shot.
        assert _split_into_steps("a\nb\nc") == ["a\nb\nc"]

    def test_empty_input_returns_empty_list(self):
        from src.verify import _split_into_steps
        assert _split_into_steps("") == []
        assert _split_into_steps("   ") == []

    def test_strips_whitespace_around_segments(self):
        from src.verify import _split_into_steps
        assert _split_into_steps("  one  \n\n   two   ") == ["one", "two"]


class TestQwenVlPrmTokenIds:
    def test_returns_pos_neg_ids(self):
        from src.verify import _qwen_vl_prm_pm_token_ids
        pos_id, neg_id = _qwen_vl_prm_pm_token_ids(_fake_pmtok())
        assert pos_id == 101
        assert neg_id == 102

    def test_raises_on_unresolvable(self):
        from src.verify import _qwen_vl_prm_pm_token_ids
        bad = MagicMock()
        bad.encode = MagicMock(return_value=[])
        with pytest.raises(RuntimeError):
            _qwen_vl_prm_pm_token_ids(bad)


class TestQwenVlPrmScore:
    def test_per_step_returns_float_per_step(self):
        from src.verify import qwen_vl_prm_score
        # Three steps, three forward calls; mock returns ascending P(+).
        with patch("src.verify._qwen_vl_prm_forward_score",
                   side_effect=[0.1, 0.5, 0.9]) as fwd:
            out = qwen_vl_prm_score(
                prm_model=MagicMock(), prm_processor=MagicMock(),
                prm_tokenizer=_fake_pmtok(), image=_img(),
                question="Q", steps=["a", "b", "c"],
            )
        assert out == [0.1, 0.5, 0.9]
        assert fwd.call_count == 3

    def test_empty_steps_returns_empty_list(self):
        from src.verify import qwen_vl_prm_score
        with patch("src.verify._qwen_vl_prm_forward_score") as fwd:
            out = qwen_vl_prm_score(
                prm_model=MagicMock(), prm_processor=MagicMock(),
                prm_tokenizer=_fake_pmtok(), image=_img(),
                question="Q", steps=[],
            )
        assert out == []
        fwd.assert_not_called()

    def test_one_shot_single_call(self):
        from src.verify import qwen_vl_prm_score_one_shot
        with patch("src.verify._qwen_vl_prm_forward_score",
                   return_value=0.73) as fwd:
            score = qwen_vl_prm_score_one_shot(
                prm_model=MagicMock(), prm_processor=MagicMock(),
                prm_tokenizer=_fake_pmtok(), image=_img(),
                question="Q", solution="full chain",
            )
        assert score == 0.73
        assert fwd.call_count == 1


class TestQwenVlPrmRank:
    def _three_chains(self):
        return [
            Candidate(description_id=-1, chain_id=0, description="",
                      reasoning="A-chain step 1\n\nA-chain step 2", answer="A"),
            Candidate(description_id=-1, chain_id=1, description="",
                      reasoning="B-chain step 1\n\nB-chain step 2", answer="B"),
            Candidate(description_id=-1, chain_id=2, description="",
                      reasoning="C-chain step 1\n\nC-chain step 2", answer="C"),
        ]

    def test_step_mean_picks_highest_average(self):
        from src.verify import qwen_vl_prm_rank
        # Three chains x two steps = six forward calls. Means: A=0.6, B=0.85, C=0.4
        with patch("src.verify._qwen_vl_prm_forward_score",
                   side_effect=[0.5, 0.7,  0.8, 0.9,  0.3, 0.5]):
            res = qwen_vl_prm_rank(
                self._three_chains(),
                image=_img(), question="Q",
                prm_model=MagicMock(), prm_processor=MagicMock(),
                prm_tokenizer=_fake_pmtok(),
                mode="step_mean",
            )
        assert res.answer == "B"
        assert res.method == "qwen_vl_prm"
        assert res.metadata["mode"] == "step_mean"
        assert "B-chain" in str(res.selected_candidate.reasoning)

    def test_step_min_picks_highest_worst_step(self):
        from src.verify import qwen_vl_prm_rank
        # Mins: A=0.4, B=0.5, C=0.45 -> B wins under step_min
        with patch("src.verify._qwen_vl_prm_forward_score",
                   side_effect=[0.4, 0.99,  0.5, 0.51,  0.45, 0.99]):
            res = qwen_vl_prm_rank(
                self._three_chains(),
                image=_img(), question="Q",
                prm_model=MagicMock(), prm_processor=MagicMock(),
                prm_tokenizer=_fake_pmtok(),
                mode="step_min",
            )
        assert res.answer == "B"

    def test_one_shot_mode_uses_one_call_per_chain(self):
        from src.verify import qwen_vl_prm_rank
        with patch("src.verify._qwen_vl_prm_forward_score",
                   side_effect=[0.3, 0.9, 0.6]) as fwd:
            res = qwen_vl_prm_rank(
                self._three_chains(),
                image=_img(), question="Q",
                prm_model=MagicMock(), prm_processor=MagicMock(),
                prm_tokenizer=_fake_pmtok(),
                mode="one_shot",
            )
        assert fwd.call_count == 3  # 1 per chain, not per step
        assert res.answer == "B"

    def test_skips_unparseable_candidates(self):
        from src.verify import qwen_vl_prm_rank
        cands = [
            Candidate(description_id=-1, chain_id=0, description="",
                      reasoning="x\n\ny", answer=None),  # unparseable, skipped
            Candidate(description_id=-1, chain_id=1, description="",
                      reasoning="x\n\ny", answer="B"),
        ]
        with patch("src.verify._qwen_vl_prm_forward_score",
                   side_effect=[0.6, 0.7]) as fwd:
            res = qwen_vl_prm_rank(
                cands, image=_img(), question="Q",
                prm_model=MagicMock(), prm_processor=MagicMock(),
                prm_tokenizer=_fake_pmtok(), mode="step_mean",
            )
        # Only the parseable B candidate is scored.
        assert fwd.call_count == 2
        assert res.answer == "B"

    def test_all_unparseable_returns_none(self):
        from src.verify import qwen_vl_prm_rank
        cands = [Candidate(description_id=-1, chain_id=i, description="",
                           reasoning="x", answer=None) for i in range(3)]
        with patch("src.verify._qwen_vl_prm_forward_score") as fwd:
            res = qwen_vl_prm_rank(
                cands, image=_img(), question="Q",
                prm_model=MagicMock(), prm_processor=MagicMock(),
                prm_tokenizer=_fake_pmtok(),
            )
        assert res.answer is None
        assert res.confidence == "very_low"
        fwd.assert_not_called()

    def test_empty_candidates_returns_none(self):
        from src.verify import qwen_vl_prm_rank
        res = qwen_vl_prm_rank(
            [], image=_img(), question="Q",
            prm_model=MagicMock(), prm_processor=MagicMock(),
            prm_tokenizer=_fake_pmtok(),
        )
        assert res.answer is None


class TestQwenVlPrmDispatch:
    def test_dispatch_routes_to_qwen_vl_prm(self):
        cands = [Candidate(description_id=-1, chain_id=0, description="",
                           reasoning="step", answer="A")]
        with patch("src.verify.qwen_vl_prm_rank") as mock_rank:
            mock_rank.return_value = MagicMock(spec=["answer", "method", "confidence",
                                                     "vote_counts", "metadata"])
            mock_rank.return_value.answer = "A"
            mock_rank.return_value.confidence = "high"
            mock_rank.return_value.vote_counts = {"A": 1}
            mock_rank.return_value.metadata = {}
            select_answer(
                cands, method="qwen_vl_prm",
                model=MagicMock(), processor=MagicMock(), image=_img(),
                verify_config={"tokenizer": _fake_pmtok(), "question": "Q"},
            )
            mock_rank.assert_called_once()

    def test_dispatch_requires_image(self):
        cands = [Candidate(description_id=-1, chain_id=0, description="",
                           reasoning="step", answer="A")]
        with pytest.raises(ValueError, match="qwen_vl_prm verification requires"):
            select_answer(
                cands, method="qwen_vl_prm",
                model=MagicMock(), processor=MagicMock(), image=None,
                verify_config={"tokenizer": _fake_pmtok()},
            )

    def test_dispatch_requires_tokenizer(self):
        cands = [Candidate(description_id=-1, chain_id=0, description="",
                           reasoning="step", answer="A")]
        bad_proc = MagicMock(spec=[])  # no .tokenizer attribute
        with pytest.raises(ValueError, match="tokenizer"):
            select_answer(
                cands, method="qwen_vl_prm",
                model=MagicMock(), processor=bad_proc, image=_img(),
                verify_config={"question": "Q"},  # tokenizer absent
            )

    def test_visualprm_now_warns_and_falls_back(self):
        """method='visualprm' is deprecated in favour of 'qwen_vl_prm';
        confirm it still falls back to majority vote and emits the
        deprecation warning."""
        cands = [Candidate(description_id=-1, chain_id=0, description="",
                           reasoning="x", answer="A")]
        result = select_answer(cands, method="visualprm")
        assert result.metadata.get("fallback_from") == "visualprm"
        assert result.answer == "A"  # majority over single candidate
