"""Tests for src/describe.py - description prompt building and generation."""

from unittest.mock import MagicMock, patch

import pytest

from src.describe import build_describe_prompt, describe


class TestBuildDescribePrompt:
    def test_english(self):
        prompt = build_describe_prompt("English")
        assert "English" in prompt
        assert "Do not answer the question" in prompt
        assert "A, B, C, D, E" in prompt

    def test_language_substitution(self):
        prompt = build_describe_prompt("French")
        assert "French" in prompt

    def test_contains_key_instructions(self):
        prompt = build_describe_prompt("English")
        assert "visual element" in prompt
        assert "mathematical notation" in prompt
        assert "Normalize" in prompt


class TestDescribe:
    @patch("src.describe.generate_n")
    def test_returns_n_descriptions(self, mock_generate_n):
        """Verify describe() returns exactly N descriptions."""
        from src.backend import GenerationOutput

        mock_generate_n.return_value = [
            GenerationOutput(
                text=f"Description {i}: biology question with options A-D.",
                logprob=-0.5,
            )
            for i in range(4)
        ]

        from PIL import Image
        img = Image.new("RGB", (100, 100), color="white")

        descriptions = describe(MagicMock(), MagicMock(), img, n=4)

        assert len(descriptions) == 4
        assert all(hasattr(d, "text") for d in descriptions)
        assert all(isinstance(d.text, str) for d in descriptions)
        # One batched call, not N sequential calls.
        assert mock_generate_n.call_count == 1
        assert mock_generate_n.call_args[0][3] == 4  # positional `n`

    @patch("src.describe.generate_n")
    def test_passes_image_to_generate(self, mock_generate_n):
        """Verify the image is passed to the generate_n function."""
        from src.backend import GenerationOutput

        mock_generate_n.return_value = [GenerationOutput(text="desc", logprob=None)]

        from PIL import Image
        img = Image.new("RGB", (10, 10))
        model, processor = MagicMock(), MagicMock()

        describe(model, processor, img, language="German", n=1, temperature=0.5, max_tokens=512)

        args, kwargs = mock_generate_n.call_args
        assert args[0] is model
        assert args[1] is processor
        assert args[3] == 1  # n
        assert kwargs["image"] is img
        assert kwargs["temperature"] == 0.5
        assert kwargs["max_new_tokens"] == 512

    @patch("src.describe.generate_n")
    def test_handles_empty_description(self, mock_generate_n):
        """Verify empty descriptions don't crash (just warn)."""
        from src.backend import GenerationOutput

        mock_generate_n.return_value = [GenerationOutput(text="", logprob=None)]

        from PIL import Image
        img = Image.new("RGB", (10, 10))

        descriptions = describe(MagicMock(), MagicMock(), img, n=1)
        assert len(descriptions) == 1
        assert descriptions[0].text == ""
