"""Integration smoke test: mock the VLM, run `run_pipeline` under a real
`RunContext`, assert JSONL is written, Phoenix spans were issued, and W&B
summary received `wall_clock_s`."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import jsonschema
import pytest
from PIL import Image

from src.pipeline import run_pipeline
from src.utils.logger import RunContext

_SCHEMA_PATH = Path(__file__).parent / "test_fixtures" / "candidate_record.schema.json"


class _SpyWandbRun:
    def __init__(self):
        self.summary: dict = {}
        self.logged: list[dict] = []
        self.finished: bool = False

    def log(self, d: dict) -> None:
        self.logged.append(d)

    def finish(self, *_, **__) -> None:
        self.finished = True


class _SpySpan:
    def __init__(self, name: str, recorder: list[str]):
        self.name = name
        self._rec = recorder

    def __enter__(self):
        self._rec.append(self.name)
        return self

    def __exit__(self, *_):
        return False

    def set_attribute(self, *_a, **_kw):
        pass

    def set_status(self, *_a, **_kw):
        pass

    def record_exception(self, *_a, **_kw):
        pass


class _SpyTracer:
    def __init__(self):
        self.names: list[str] = []

    def start_as_current_span(self, name: str, **_kw):
        return _SpySpan(name, self.names)


@patch("src.describe.generate_n")
@patch("src.reason.generate_n")
def test_three_layers_receive_data(mock_reason_gen, mock_describe_gen,
                                    tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from src.backend import GenerationOutput

    # describe(n=2) -> one batched call returning 2 outputs
    mock_describe_gen.return_value = [
        GenerationOutput(
            text="The image shows a math question. Options: A 1 B 2 C 3 D 4",
            logprob=-0.4, prompt_tokens=100, completion_tokens=50, latency_s=0.1,
        )
    ] * 2
    # reason(m=2) -> one batched call per description returning 2 outputs
    mock_reason_gen.return_value = [
        GenerationOutput(
            text="The answer is B",
            logprob=-0.2, prompt_tokens=80, completion_tokens=10, latency_s=0.05,
        )
    ] * 2

    spy_tracer = _SpyTracer()
    spy_wb = _SpyWandbRun()
    monkeypatch.setattr("src.utils.logger.init_phoenix", lambda *a, **k: spy_tracer)
    monkeypatch.setattr("src.utils.logger.init_wandb", lambda *a, **k: spy_wb)

    config = {
        "model": {"name": "test-model"},
        "describe": {"n_samples": 2, "temperature": 0.7, "max_tokens": 256},
        "reason": {"m_samples": 2, "temperature": 0.7, "max_tokens": 128, "use_cot": True},
        "verify": {"method": "majority_vote"},
        "evaluation": {"seed": 42},
    }
    item = {
        "sample_id": "int_q1",
        "language": "English",
        "subject": "math",
        "answer_key": "B",
        "question": "What is 1 + 1?",
        "image": Image.new("RGB", (100, 100)),
    }

    out_dir = tmp_path / "run"
    with RunContext(run_id="int-test", config=config, out_dir=out_dir) as ctx:
        record = run_pipeline(MagicMock(), MagicMock(), item, config, ctx)
        ctx.write_question(record)

    # Layer 1: JSONL
    jsonl = out_dir / "candidates.jsonl"
    assert jsonl.exists()
    lines = [json.loads(l) for l in jsonl.read_text().splitlines() if l]
    assert len(lines) == 1
    schema = json.loads(_SCHEMA_PATH.read_text())
    jsonschema.validate(lines[0], schema)
    assert lines[0]["verifier"]["selected_answer"] == "B"
    assert lines[0]["correct"] is True

    # Layer 2: Phoenix spans - expect question + describe + 2 x description.sample
    # + reason + 4 x reasoning.chain + verify + verifier.majority_vote
    names = spy_tracer.names
    assert "question" in names
    assert "describe" in names
    assert names.count("description.sample") == 2
    assert "reason" in names
    assert names.count("reasoning.chain") == 4
    assert "verify" in names
    assert "verifier.majority_vote" in names

    # Layer 3: W&B summary populated on exit
    assert spy_wb.finished is True
    assert "wall_clock_s" in spy_wb.summary
    assert spy_wb.summary["questions"] == 1
