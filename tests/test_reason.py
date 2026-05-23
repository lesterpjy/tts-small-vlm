"""Tests for src/reason.py - reasoning prompt building and chain generation."""

from unittest.mock import MagicMock, patch

import pytest

from src.reason import build_reason_prompt, reason, reason_all
from src.utils import Candidate


class TestBuildReasonPrompt:
    def test_cot_prompt(self):
        prompt = build_reason_prompt("This is a biology question about cells.", use_cot=True)
        assert "biology question about cells" in prompt
        assert "step by step" in prompt.lower()
        assert "The answer is" in prompt

    def test_strict_prompt(self):
        prompt = build_reason_prompt("A math question.", use_cot=False)
        assert "math question" in prompt
        assert "ONLY the single letter" in prompt
        assert "No explanation" in prompt

    def test_caption_substitution(self):
        caption = "Question: What is 2+2? Options: A) 3, B) 4, C) 5, D) 6"
        prompt = build_reason_prompt(caption)
        assert caption in prompt

    def test_mmmu_cot_style(self):
        prompt = build_reason_prompt("A question.", use_cot=True, cot_style="mmmu")
        assert "Answer: X" in prompt
        assert "The answer is" not in prompt

    def test_default_cot_style_unchanged(self):
        prompt = build_reason_prompt("A question.", use_cot=True, cot_style="default")
        assert "The answer is" in prompt


class TestReason:
    @patch("src.reason.generate_n")
    def test_returns_m_candidates(self, mock_generate_n):
        """Verify reason() returns exactly M candidates."""
        from src.backend import GenerationOutput

        mock_generate_n.return_value = [
            GenerationOutput(text="Let me think... The answer is A", logprob=-0.5),
            GenerationOutput(text="Analyzing options... The answer is B", logprob=-0.8),
            GenerationOutput(text="Step 1... The answer is A", logprob=-0.3),
            GenerationOutput(text="The answer is C", logprob=-1.0),
        ]

        candidates = reason(MagicMock(), MagicMock(), "A question", description_id=0, m=4)

        assert len(candidates) == 4
        assert all(isinstance(c, Candidate) for c in candidates)
        # One batched call, not M sequential calls.
        assert mock_generate_n.call_count == 1
        assert mock_generate_n.call_args[0][3] == 4  # positional `m`

    @patch("src.reason.generate_n")
    def test_answer_extraction(self, mock_generate_n):
        """Verify answers are correctly extracted from reasoning chains."""
        from src.backend import GenerationOutput

        mock_generate_n.return_value = [
            GenerationOutput(text="The answer is B", logprob=-0.5),
            GenerationOutput(text="A", logprob=-0.3),
        ]

        candidates = reason(MagicMock(), MagicMock(), "desc", description_id=0, m=2)

        assert candidates[0].answer == "B"
        assert candidates[1].answer == "A"

    @patch("src.reason.generate_n")
    def test_logprob_propagated(self, mock_generate_n):
        """Verify logprob from backend is propagated to candidates."""
        from src.backend import GenerationOutput

        mock_generate_n.return_value = [GenerationOutput(text="The answer is A", logprob=-1.5)]

        candidates = reason(MagicMock(), MagicMock(), "desc", description_id=0, m=1)

        assert candidates[0].logprob == pytest.approx(-1.5)

    @patch("src.reason.generate_n")
    def test_description_id_propagated(self, mock_generate_n):
        """Verify description_id is set correctly."""
        from src.backend import GenerationOutput

        mock_generate_n.return_value = [GenerationOutput(text="A", logprob=None)]

        candidates = reason(MagicMock(), MagicMock(), "desc", description_id=7, m=1)
        assert candidates[0].description_id == 7

    @patch("src.reason.generate_n")
    def test_handles_extraction_failure(self, mock_generate_n):
        """Verify candidates with unparseable answers get answer=None."""
        from src.backend import GenerationOutput

        mock_generate_n.return_value = [GenerationOutput(
            text="I cannot determine the answer from this.", logprob=-2.0,
        )]

        candidates = reason(MagicMock(), MagicMock(), "desc", description_id=0, m=1)
        assert candidates[0].answer is None

    @patch("src.reason.generate_n")
    def test_text_only_generation(self, mock_generate_n):
        """Verify reason calls generate_n with image=None (text-only)."""
        from src.backend import GenerationOutput

        mock_generate_n.return_value = [GenerationOutput(text="The answer is B", logprob=-0.5)]

        reason(MagicMock(), MagicMock(), "desc", description_id=0, m=1)

        call_kwargs = mock_generate_n.call_args[1]
        assert call_kwargs["image"] is None


class TestReasonAll:
    @patch("src.reason.generate_n")
    def test_generates_n_times_m_candidates(self, mock_generate_n):
        """Verify reason_all produces N*M total candidates via N batched calls."""
        from src.backend import GenerationOutput

        # Each description -> one batched generate_n call returning m=2 outputs.
        mock_generate_n.return_value = [
            GenerationOutput(text="The answer is A", logprob=-0.5),
            GenerationOutput(text="The answer is A", logprob=-0.5),
        ]

        descriptions = ["desc1", "desc2", "desc3"]
        candidates = reason_all(MagicMock(), MagicMock(), descriptions, m=2)

        assert len(candidates) == 6  # 3 descriptions * 2 chains each
        desc_ids = [c.description_id for c in candidates]
        assert desc_ids == [0, 0, 1, 1, 2, 2]
        # One batched call per description (3 total), not N*M individual calls.
        assert mock_generate_n.call_count == 3
