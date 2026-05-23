"""Tests for scripts/analyze.py subsampling logic."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.analyze import (
    _candidates_for,
    compute_scaling_grid,
    compute_stratified,
)


def _chain(desc_idx: int, chain_idx: int, answer: str | None, logprob: float = -0.5) -> dict:
    return {
        "desc_idx": desc_idx,
        "chain_idx": chain_idx,
        "reasoning": f"chain {desc_idx}/{chain_idx}",
        "extracted_answer": answer,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "logprob_mean": logprob,
        "latency_s": 0.0,
        "model": "t",
        "temperature": 0.7,
        "seed": 0,
    }


def _desc(idx: int) -> dict:
    return {
        "idx": idx, "text": "d", "prompt_tokens": 0, "completion_tokens": 0,
        "logprob_mean": None, "latency_s": 0.0,
        "model": "t", "temperature": 0.7, "seed": 0,
    }


def _record(*, qid: str, gold: str, chains: list[dict], n_desc: int = 2, **overrides) -> dict:
    base = {
        "run_id": "r",
        "question_id": qid,
        "dataset": "exams_v",
        "split": "validation",
        "subject": "physics",
        "language": "en",
        "content_type": None,
        "image_path": None,
        "question_text": "",
        "options": {},
        "gold": gold,
        "descriptions": [_desc(i) for i in range(n_desc)],
        "chains": chains,
        "verifier": {
            "method": "majority_vote",
            "selected_answer": None,
            "cluster_sizes": {},
            "confidence": "high",
            "tie_break": None,
            "scored_candidates": None,
            "latency_s": 0.0,
        },
        "correct": None,
        "total_vlm_calls": len(chains),
        "total_latency_s": 0.0,
        "timestamp": "2026-04-11T00:00:00Z",
    }
    base.update(overrides)
    return base


class TestCandidatesFor:
    def test_slice_by_n_and_m(self):
        chains = [
            _chain(0, 0, "A"), _chain(0, 1, "B"), _chain(0, 2, "A"),
            _chain(1, 0, "C"), _chain(1, 1, "A"),
            _chain(2, 0, "B"),
        ]
        rec = _record(qid="q", gold="A", chains=chains, n_desc=3)

        cands = _candidates_for(rec, n=2, m=2)
        # First 2 descriptions x 2 chains each = 4 candidates
        assert len(cands) == 4
        desc_ids = sorted({c.description_id for c in cands})
        assert desc_ids == [0, 1]

    def test_deterministic_order(self):
        chains = [_chain(0, i, "A") for i in range(5)]
        rec = _record(qid="q", gold="A", chains=chains, n_desc=1)
        a = _candidates_for(rec, n=1, m=3)
        b = _candidates_for(rec, n=1, m=3)
        assert [c.chain_id for c in a] == [c.chain_id for c in b] == [0, 1, 2]


class TestScalingGrid:
    def test_majority_vote_subsampling(self):
        # q1: all chains say A, trivially correct at every (n, m)
        q1 = _record(
            qid="q1", gold="A",
            chains=[_chain(0, 0, "A"), _chain(0, 1, "A"),
                    _chain(1, 0, "A"), _chain(1, 1, "A")],
        )
        # q2: desc 0 votes B, desc 1 votes A. At (n=1, m=2) only desc 0 -> B (wrong).
        # At (n=2, m=2) it's a tie with logprobs -> depends on tie-break.
        q2 = _record(
            qid="q2", gold="A",
            chains=[_chain(0, 0, "B"), _chain(0, 1, "B"),
                    _chain(1, 0, "A", logprob=-0.1), _chain(1, 1, "A", logprob=-0.1)],
        )

        grid = compute_scaling_grid([q1, q2], n_max=2, m_max=2, method="majority_vote")
        # At (1, 2): q1 -> A, q2 -> B (only desc 0 is used) -> 1/2 = 0.5
        assert grid["n1_m2"] == pytest.approx(0.5)
        # At (2, 2): q1 -> A, q2 tie but A logprobs dominate -> 2/2 = 1.0
        assert grid["n2_m2"] == pytest.approx(1.0)


class TestStratified:
    def test_by_subject_and_language(self):
        records = [
            _record(qid="q1", gold="A", chains=[_chain(0, 0, "A")],
                    verifier={"method": "majority_vote", "selected_answer": "A",
                              "cluster_sizes": {"A": 1}, "confidence": "high",
                              "tie_break": None, "scored_candidates": None, "latency_s": 0.0},
                    subject="physics", language="en"),
            _record(qid="q2", gold="B", chains=[_chain(0, 0, "A")],
                    verifier={"method": "majority_vote", "selected_answer": "A",
                              "cluster_sizes": {"A": 1}, "confidence": "high",
                              "tie_break": None, "scored_candidates": None, "latency_s": 0.0},
                    subject="physics", language="bg"),
        ]
        strata = compute_stratified(records)
        assert strata["subject"]["physics"] == pytest.approx(0.5)
        assert strata["language"]["en"] == pytest.approx(1.0)
        assert strata["language"]["bg"] == pytest.approx(0.0)
