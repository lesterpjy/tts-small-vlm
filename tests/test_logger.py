"""Tests for RunContext, make_run_id, and JSONL record shape."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.utils.logger import RunContext, make_run_id
from src.utils.records import (
    build_question_record,
    chain_entry,
    description_entry,
    verifier_entry,
)

_SCHEMA_PATH = Path(__file__).parent / "test_fixtures" / "candidate_record.schema.json"


def _minimal_record(run_id: str = "test-run") -> dict:
    return build_question_record(
        run_id=run_id,
        item={
            "sample_id": "q1",
            "subject": "physics",
            "language": "en",
            "gold": "B",
            "question": "What?",
            "A": "a", "B": "b", "C": "c", "D": "d", "E": "e",
        },
        descriptions=[
            description_entry(
                idx=0, text="caption",
                prompt_tokens=10, completion_tokens=20,
                logprob_mean=-0.4, latency_s=0.1,
                model="test-model", temperature=0.7, seed=42,
            )
        ],
        chains=[
            chain_entry(
                desc_idx=0, chain_idx=0, reasoning="chain",
                extracted_answer="B",
                prompt_tokens=30, completion_tokens=10,
                logprob_mean=-0.2, latency_s=0.2,
                model="test-model", temperature=0.7, seed=42,
            )
        ],
        verifier=verifier_entry(selection=None, latency_s=0.0)
        | {
            "method": "majority_vote",
            "selected_answer": "B",
            "cluster_sizes": {"B": 1},
            "confidence": "high",
        },
        correct=True,
        total_vlm_calls=2,
        total_latency_s=0.3,
    )


class TestMakeRunId:
    def test_format(self):
        config = {
            "model": {"name": "Qwen/Qwen3.5-4B"},
            "describe": {"n_samples": 4, "temperature": 0.7},
            "reason": {"m_samples": 4, "temperature": 0.7},
            "verify": {"method": "majority_vote"},
            "evaluation": {"seed": 0},
        }
        run_id = make_run_id(config, clock=0)
        parts = run_id.split("-")
        # pipeline, model_short, verifier, nNmM, tXX, seedX, timestamp
        assert parts[0] == "msa"
        assert "qwen35" in parts[1]
        assert parts[-5] == "majvote"
        assert parts[-4] == "n4m4"
        assert parts[-3] == "t07"
        assert parts[-2] == "seed0"
        assert len(parts[-1]) == 12  # YYYYMMDDhhmm

    def test_baseline_fallback(self):
        config = {
            "model": {"name": "Qwen/Qwen2.5-VL-7B-Instruct"},
            "evaluation": {"seed": 42},
        }
        run_id = make_run_id(config, clock=0)
        assert run_id.startswith("baseline-")
        assert "n1m1" in run_id
        assert "seed42" in run_id


class TestRunContext:
    def test_write_question_matches_schema(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """End-to-end: RunContext writes a JSONL line that validates against
        the spec JSON Schema."""
        import jsonschema
        schema = json.loads(_SCHEMA_PATH.read_text())

        # Keep Phoenix + W&B stubbed so this is a pure unit test.
        monkeypatch.setattr("src.utils.logger.init_phoenix", lambda *a, **k: _NoopTracer())
        monkeypatch.setattr("src.utils.logger.init_wandb", lambda *a, **k: None)

        ctx = RunContext(
            run_id="test-run",
            config={"model": {"name": "test"}},
            out_dir=tmp_path / "run",
        )
        with ctx:
            ctx.write_question(_minimal_record())

        jsonl_path = tmp_path / "run" / "candidates.jsonl"
        assert jsonl_path.exists()
        lines = [json.loads(l) for l in jsonl_path.read_text().splitlines() if l]
        assert len(lines) == 1
        jsonschema.validate(lines[0], schema)

    def test_config_yaml_written(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        import yaml
        monkeypatch.setattr("src.utils.logger.init_phoenix", lambda *a, **k: _NoopTracer())
        monkeypatch.setattr("src.utils.logger.init_wandb", lambda *a, **k: None)

        cfg = {"model": {"name": "test"}, "describe": {"n_samples": 2}}
        with RunContext(run_id="rid", config=cfg, out_dir=tmp_path / "r") as _ctx:
            pass
        loaded = yaml.safe_load((tmp_path / "r" / "config.yaml").read_text())
        assert loaded == cfg


# --- local noop tracer helper ---


class _NoopSpan:
    def set_attribute(self, *_a, **_kw):
        pass

    def set_status(self, *_a, **_kw):
        pass

    def record_exception(self, *_a, **_kw):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass


class _NoopTracer:
    def start_as_current_span(self, _name, **_kw):
        return _NoopSpan()
