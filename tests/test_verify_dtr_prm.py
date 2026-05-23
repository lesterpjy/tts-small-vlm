"""Tests for DTR + Qwen-VL-PRM segmented scoring.

All tests are mocked, no GPU required. Validates:
  - Segmented scoring formats steps correctly and partitions scores
  - DTR ranking picks the highest-scoring candidate
  - Dispatch routes qwen_vl_prm_dtr method correctly
  - Offline rescorer joins descriptions with chains
"""

from unittest.mock import MagicMock, patch

import pytest

from src.utils import Candidate
from src.verify import (
    qwen_vl_prm_rank_dtr,
    qwen_vl_prm_score_dtr_segmented,
    select_answer,
)


def _make_dtr_candidate(
    answer: str,
    desc_id: int = 0,
    chain_id: int = 0,
    description: str = "The image shows a graph with a parabola.",
    reasoning: str = "Step 1: The graph shows y=x^2.\n\nStep 2: The vertex is at (0,0).\n\nThe answer is A.",
    logprob: float | None = None,
) -> Candidate:
    return Candidate(
        description_id=desc_id,
        chain_id=chain_id,
        description=description,
        reasoning=reasoning,
        answer=answer,
        logprob=logprob,
    )


class TestSegmentedScoring:
    """Tests for qwen_vl_prm_score_dtr_segmented."""

    @patch("src.verify.qwen_vl_prm_score")
    def test_basic_segmentation(self, mock_score):
        mock_score.return_value = [0.9, 0.8, 0.7]

        result = qwen_vl_prm_score_dtr_segmented(
            prm_model=MagicMock(),
            prm_processor=MagicMock(),
            prm_tokenizer=MagicMock(),
            image=MagicMock(),
            question="",
            description="The graph shows a parabola.",
            reasoning="Step 1: y=x^2.\n\nStep 2: Vertex at origin.",
        )

        assert result["n_perception_steps"] == 1
        assert result["n_reasoning_steps"] == 2
        assert result["perception_score"] == pytest.approx(0.9)
        assert result["reasoning_mean"] == pytest.approx(0.75)
        assert result["reasoning_min"] == pytest.approx(0.7)
        assert result["overall"] == pytest.approx(0.8)
        assert len(result["perception_steps"]) == 1
        assert len(result["reasoning_steps"]) == 2

        call_args = mock_score.call_args
        steps = call_args.kwargs["steps"]
        assert steps[0].startswith("[Perception]")
        assert steps[1].startswith("[Reasoning]")
        assert steps[2].startswith("[Reasoning]")

    @patch("src.verify.qwen_vl_prm_score")
    def test_empty_description(self, mock_score):
        mock_score.return_value = [0.6, 0.5]

        result = qwen_vl_prm_score_dtr_segmented(
            prm_model=MagicMock(),
            prm_processor=MagicMock(),
            prm_tokenizer=MagicMock(),
            image=MagicMock(),
            question="",
            description="",
            reasoning="Step 1: something.\n\nStep 2: answer.",
        )

        assert result["n_perception_steps"] == 0
        assert result["n_reasoning_steps"] == 2
        assert result["perception_score"] == 0.0
        assert result["reasoning_mean"] == pytest.approx(0.55)

    @patch("src.verify.qwen_vl_prm_score")
    def test_empty_reasoning(self, mock_score):
        mock_score.return_value = [0.8]

        result = qwen_vl_prm_score_dtr_segmented(
            prm_model=MagicMock(),
            prm_processor=MagicMock(),
            prm_tokenizer=MagicMock(),
            image=MagicMock(),
            question="",
            description="A graph.",
            reasoning="",
        )

        assert result["n_perception_steps"] == 1
        assert result["n_reasoning_steps"] == 0
        assert result["perception_score"] == pytest.approx(0.8)
        assert result["reasoning_mean"] == 0.0

    def test_both_empty(self):
        result = qwen_vl_prm_score_dtr_segmented(
            prm_model=MagicMock(),
            prm_processor=MagicMock(),
            prm_tokenizer=MagicMock(),
            image=MagicMock(),
            question="",
            description="",
            reasoning="",
        )

        assert result["overall"] == 0.0
        assert result["n_perception_steps"] == 0
        assert result["n_reasoning_steps"] == 0


class TestDTRRanking:
    """Tests for qwen_vl_prm_rank_dtr."""

    @patch("src.verify.qwen_vl_prm_score_dtr_segmented")
    def test_picks_highest_overall(self, mock_seg):
        mock_seg.side_effect = [
            {"overall": 0.6, "perception_score": 0.5, "reasoning_mean": 0.7,
             "reasoning_min": 0.5, "perception_steps": [0.5],
             "reasoning_steps": [0.7], "n_perception_steps": 1, "n_reasoning_steps": 1},
            {"overall": 0.9, "perception_score": 0.8, "reasoning_mean": 0.95,
             "reasoning_min": 0.9, "perception_steps": [0.8],
             "reasoning_steps": [0.95], "n_perception_steps": 1, "n_reasoning_steps": 1},
        ]

        candidates = [
            _make_dtr_candidate("A", chain_id=0),
            _make_dtr_candidate("B", chain_id=1),
        ]

        result = qwen_vl_prm_rank_dtr(
            candidates,
            image=MagicMock(),
            question="",
            prm_model=MagicMock(),
            prm_processor=MagicMock(),
            prm_tokenizer=MagicMock(),
        )

        assert result.answer == "B"
        assert result.method == "qwen_vl_prm_dtr"
        assert "segmented_scores" in result.metadata
        assert result.metadata["aggregation"] == "overall"

    @patch("src.verify.qwen_vl_prm_score_dtr_segmented")
    def test_rank_by_reasoning_mean(self, mock_seg):
        mock_seg.side_effect = [
            {"overall": 0.9, "perception_score": 0.95, "reasoning_mean": 0.3,
             "reasoning_min": 0.1, "perception_steps": [0.95],
             "reasoning_steps": [0.3], "n_perception_steps": 1, "n_reasoning_steps": 1},
            {"overall": 0.7, "perception_score": 0.5, "reasoning_mean": 0.8,
             "reasoning_min": 0.7, "perception_steps": [0.5],
             "reasoning_steps": [0.8], "n_perception_steps": 1, "n_reasoning_steps": 1},
        ]

        candidates = [
            _make_dtr_candidate("A", chain_id=0),
            _make_dtr_candidate("B", chain_id=1),
        ]

        result = qwen_vl_prm_rank_dtr(
            candidates,
            image=MagicMock(),
            question="",
            prm_model=MagicMock(),
            prm_processor=MagicMock(),
            prm_tokenizer=MagicMock(),
            aggregation="reasoning_mean",
        )

        assert result.answer == "B"
        assert result.metadata["aggregation"] == "reasoning_mean"

    def test_no_candidates(self):
        result = qwen_vl_prm_rank_dtr(
            [],
            image=MagicMock(),
            question="",
            prm_model=MagicMock(),
            prm_processor=MagicMock(),
            prm_tokenizer=MagicMock(),
        )
        assert result.answer is None
        assert result.confidence == "very_low"

    def test_no_parseable_candidates(self):
        candidates = [_make_dtr_candidate(None)]
        result = qwen_vl_prm_rank_dtr(
            candidates,
            image=MagicMock(),
            question="",
            prm_model=MagicMock(),
            prm_processor=MagicMock(),
            prm_tokenizer=MagicMock(),
        )
        assert result.answer is None
        assert result.confidence == "very_low"


class TestDispatchDTR:
    """Test that select_answer routes qwen_vl_prm_dtr correctly."""

    def test_dispatch_requires_model(self):
        candidates = [_make_dtr_candidate("A")]
        with pytest.raises(ValueError, match="qwen_vl_prm_dtr"):
            select_answer(candidates, method="qwen_vl_prm_dtr")

    def test_dispatch_requires_tokenizer(self):
        candidates = [_make_dtr_candidate("A")]
        proc = MagicMock(spec=[])
        with pytest.raises(ValueError, match="tokenizer"):
            select_answer(
                candidates, method="qwen_vl_prm_dtr",
                model=MagicMock(), processor=proc, image=MagicMock(),
            )


class TestDTRChainsToCandidate:
    """Test the offline rescorer's description-joining logic."""

    def test_joins_description_text(self):
        from scripts.rescore_dtr_with_prm import dtr_chains_to_candidates

        record = {
            "descriptions": [
                {"idx": 0, "text": "Description zero"},
                {"idx": 1, "text": "Description one"},
            ],
            "chains": [
                {"desc_idx": 0, "chain_idx": 0, "reasoning": "Step A",
                 "extracted_answer": "A", "logprob_mean": -1.0,
                 "prompt_tokens": 100, "completion_tokens": 50, "latency_s": 1.0},
                {"desc_idx": 1, "chain_idx": 0, "reasoning": "Step B",
                 "extracted_answer": "B", "logprob_mean": -2.0,
                 "prompt_tokens": 100, "completion_tokens": 50, "latency_s": 1.0},
            ],
        }
        cands = dtr_chains_to_candidates(record)
        assert len(cands) == 2
        assert cands[0].description == "Description zero"
        assert cands[1].description == "Description one"
        assert cands[0].reasoning == "Step A"

    def test_missing_description_index(self):
        from scripts.rescore_dtr_with_prm import dtr_chains_to_candidates

        record = {
            "descriptions": [{"idx": 0, "text": "Only one desc"}],
            "chains": [
                {"desc_idx": 5, "chain_idx": 0, "reasoning": "Step",
                 "extracted_answer": "C"},
            ],
        }
        cands = dtr_chains_to_candidates(record)
        assert len(cands) == 1
        assert cands[0].description == ""

    def test_skips_empty_reasoning(self):
        from scripts.rescore_dtr_with_prm import dtr_chains_to_candidates

        record = {
            "descriptions": [{"idx": 0, "text": "Desc"}],
            "chains": [
                {"desc_idx": 0, "chain_idx": 0, "reasoning": "",
                 "extracted_answer": "A"},
                {"desc_idx": 0, "chain_idx": 1, "reasoning": "Step",
                 "extracted_answer": "B"},
            ],
        }
        cands = dtr_chains_to_candidates(record)
        assert len(cands) == 1
        assert cands[0].answer == "B"
