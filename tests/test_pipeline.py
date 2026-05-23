"""Tests for src/pipeline.py - end-to-end smoke test against the JSONL
record returned by `run_pipeline` under a DummyContext."""

import json
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

from PIL import Image

from src.pipeline import run_pipeline


class _DummySpan:
    def set_attribute(self, *_a, **_kw):
        pass

    def record_exception(self, *_a, **_kw):
        pass

    def set_status(self, *_a, **_kw):
        pass


class DummyContext:
    """Minimal RunContext stand-in for pipeline tests.

    Provides no-op `question_span` / `stage_span` / `llm_span` context
    managers plus the `set_output` / `write_question` helpers the pipeline
    calls. Captures written records for assertions.
    """

    def __init__(self, run_id: str = "test-run"):
        self.run_id = run_id
        self.records: list[dict] = []

    @contextmanager
    def question_span(self, _question):
        yield _DummySpan()

    @contextmanager
    def stage_span(self, _name, **_attrs):
        yield _DummySpan()

    @contextmanager
    def llm_span(self, _name, **_attrs):
        yield _DummySpan()

    @staticmethod
    def set_output(_span, _output):
        pass

    @staticmethod
    def set_llm_output(_span, _output, **_kwargs):
        pass

    def write_question(self, record: dict) -> None:
        self.records.append(record)


def _item(
    *, sample_id="q1", language="English", subject="geography",
    answer_key="B", question="What?"
):
    return {
        "sample_id": sample_id,
        "language": language,
        "subject": subject,
        "answer_key": answer_key,
        "question": question,
        "image": Image.new("RGB", (100, 100), color="white"),
    }


_BASE_CONFIG = {
    "model": {"name": "test-model"},
    "describe": {"n_samples": 2, "temperature": 0.7, "max_tokens": 256},
    "reason": {"m_samples": 2, "temperature": 0.7, "max_tokens": 128, "use_cot": True},
    "verify": {"method": "majority_vote"},
    "evaluation": {"seed": 42},
}


class TestRunPipeline:
    @patch("src.describe.generate_n")
    @patch("src.reason.generate_n")
    def test_smoke_record_shape(self, mock_reason_gen, mock_describe_gen):
        """End-to-end: pipeline returns a record with the expected shape."""
        from src.backend import GenerationOutput

        # describe(n=2) -> one batched call returning 2 outputs.
        mock_describe_gen.return_value = [
            GenerationOutput(
                text=(
                    "The image shows a multiple choice question. "
                    "Question: What is the capital of France? "
                    "Options: A) London B) Paris C) Berlin D) Madrid"
                ),
                logprob=-0.5, prompt_tokens=10, completion_tokens=20, latency_s=0.1,
            )
        ] * 2
        # reason(m=2) -> called once per description (2 times total), each
        # returning 2 outputs.
        mock_reason_gen.return_value = [
            GenerationOutput(
                text="Paris is the capital of France. The answer is B",
                logprob=-0.3, prompt_tokens=50, completion_tokens=30, latency_s=0.2,
            )
        ] * 2

        ctx = DummyContext(run_id="test-run")
        record = run_pipeline(
            MagicMock(), MagicMock(),
            _item(sample_id="test_q1", language="English", answer_key="B"),
            _BASE_CONFIG,
            ctx,
        )

        assert record["run_id"] == "test-run"
        assert record["question_id"] == "test_q1"
        assert record["language"] == "English"
        assert record["gold"] == "B"
        assert len(record["descriptions"]) == 2
        assert len(record["chains"]) == 4  # 2 x 2
        assert record["verifier"]["method"] == "majority_vote"
        assert record["verifier"]["selected_answer"] == "B"
        assert record["correct"] is True
        assert record["total_vlm_calls"] == 6  # 2 descriptions + 4 chains
        assert record["total_latency_s"] >= 0

    @patch("src.describe.generate_n")
    @patch("src.reason.generate_n")
    def test_competition_serializable(self, mock_reason_gen, mock_describe_gen):
        """The record projects cleanly to the competition JSON format."""
        from src.backend import GenerationOutput

        mock_describe_gen.return_value = [GenerationOutput(
            text="A question about geography.", logprob=None,
        )] * 2
        mock_reason_gen.return_value = [GenerationOutput(
            text="The answer is C", logprob=-0.5,
        )] * 2

        record = run_pipeline(
            MagicMock(), MagicMock(),
            _item(sample_id="q42", language="French", answer_key="C"),
            _BASE_CONFIG,
            DummyContext(),
        )

        entry = {
            "id": record["question_id"],
            "prediction": record["verifier"]["selected_answer"],
            "language": record["language"],
        }
        parsed = json.loads(json.dumps(entry))
        assert parsed == {"id": "q42", "prediction": "C", "language": "French"}

    @patch("src.describe.generate_n")
    @patch("src.reason.generate_n")
    def test_chain_traceability(self, mock_reason_gen, mock_describe_gen):
        """Each chain in the record carries valid desc_idx / chain_idx."""
        from src.backend import GenerationOutput

        # describe(n=3) -> one batched call returning 3 outputs.
        mock_describe_gen.return_value = [
            GenerationOutput(text="Question desc", logprob=None)
        ] * 3
        # reason(m=2) called once per description (3 times), each returns 2.
        mock_reason_gen.return_value = [
            GenerationOutput(text="The answer is A", logprob=-0.5)
        ] * 2

        config = {**_BASE_CONFIG, "describe": {**_BASE_CONFIG["describe"], "n_samples": 3}}
        record = run_pipeline(
            MagicMock(), MagicMock(),
            _item(sample_id="q1", answer_key="A"),
            config,
            DummyContext(),
        )

        assert len(record["chains"]) == 6  # 3 x 2
        desc_ids = {c["desc_idx"] for c in record["chains"]}
        assert desc_ids == {0, 1, 2}
        for c in record["chains"]:
            assert c["chain_idx"] in {0, 1}
            assert c["extracted_answer"] == "A"
