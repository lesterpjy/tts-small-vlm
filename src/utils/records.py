"""JSONL record schema + builder.

The JSONL file at `runs/{run_id}/candidates.jsonl` is the source of truth.
Phoenix and W&B are write-only mirrors; losing them loses nothing. One JSON
object per line, one line per question, candidates nested inside the record
so all of a question's data is readable in a single disk seek.
"""

from __future__ import annotations

import datetime as _dt
from typing import Any

from src.utils.core import Candidate, SelectionResult


def _iso_now_utc() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_question_record(
    *,
    run_id: str,
    item: dict,
    descriptions: list[dict],
    chains: list[dict],
    verifier: dict,
    correct: bool | None,
    total_vlm_calls: int,
    total_latency_s: float,
    dataset: str = "exams_v",
    split: str = "validation",
) -> dict:
    """Assemble one question record.

    `item` is the full HF dataset row (so `sample_id`, `subject`, `language`,
    `question`, options, gold are all reachable without a second fetch).
    `descriptions`, `chains`, and `verifier` are already-shaped dicts. This
    function does not inspect the model or the pipeline stages, it just binds
    them into the schema.

    `content_type` is set to `None` because EXAMS-V does not expose a
    content-type field; analysis scripts may derive one later without a schema
    change.

    `image_path` is `None` because EXAMS-V images are loaded in-memory from HF
    datasets and we do not cache them to disk.
    """
    options = {
        k: item.get(k) or item.get(k.lower())
        for k in ("A", "B", "C", "D", "E")
    }
    return {
        "run_id": run_id,
        "question_id": item.get("sample_id") or item.get("id"),
        "dataset": dataset,
        "split": split,
        "subject": item.get("subject"),
        "language": item.get("language"),
        "content_type": item.get("content_type"),
        "image_path": None,
        "question_text": item.get("question") or item.get("question_text"),
        "options": options,
        "gold": item.get("gold"),
        "descriptions": descriptions,
        "chains": chains,
        "verifier": verifier,
        "correct": correct,
        "total_vlm_calls": total_vlm_calls,
        "total_latency_s": total_latency_s,
        "timestamp": _iso_now_utc(),
    }


def description_entry(
    *,
    idx: int,
    text: str,
    prompt_tokens: int,
    completion_tokens: int,
    logprob_mean: float | None,
    latency_s: float,
    model: str,
    temperature: float,
    seed: int,
) -> dict:
    return {
        "idx": idx,
        "text": text,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "logprob_mean": logprob_mean,
        "latency_s": latency_s,
        "model": model,
        "temperature": temperature,
        "seed": seed,
    }


def chain_entry(
    *,
    desc_idx: int,
    chain_idx: int,
    reasoning: str,
    extracted_answer: str | None,
    prompt_tokens: int,
    completion_tokens: int,
    logprob_mean: float | None,
    latency_s: float,
    model: str,
    temperature: float,
    seed: int,
) -> dict:
    return {
        "desc_idx": desc_idx,
        "chain_idx": chain_idx,
        "reasoning": reasoning,
        "extracted_answer": extracted_answer,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "logprob_mean": logprob_mean,
        "latency_s": latency_s,
        "model": model,
        "temperature": temperature,
        "seed": seed,
    }


def verifier_entry(
    *,
    selection: SelectionResult | None,
    latency_s: float,
    scored_candidates: list[dict] | None = None,
) -> dict:
    if selection is None:
        return {
            "method": None,
            "selected_answer": None,
            "cluster_sizes": {},
            "confidence": None,
            "tie_break": None,
            "scored_candidates": scored_candidates,
            "latency_s": latency_s,
        }
    tie_break = selection.metadata.get("tiebreak") if selection.metadata else None
    return {
        "method": selection.method,
        "selected_answer": selection.answer,
        "cluster_sizes": dict(selection.vote_counts or {}),
        "confidence": selection.confidence,
        "tie_break": tie_break,
        "scored_candidates": scored_candidates,
        "latency_s": latency_s,
    }


def chain_entries_from_candidates(
    candidates: list[Candidate],
    *,
    model: str,
    temperature: float,
    seed: int,
) -> list[dict]:
    """Convert a list of `Candidate` objects to JSONL chain entries."""
    return [
        chain_entry(
            desc_idx=c.description_id,
            chain_idx=c.chain_id,
            reasoning=c.reasoning,
            extracted_answer=c.answer,
            prompt_tokens=c.prompt_tokens,
            completion_tokens=c.completion_tokens,
            logprob_mean=c.logprob,
            latency_s=c.latency_s,
            model=model,
            temperature=temperature,
            seed=seed,
        )
        for c in candidates
    ]
