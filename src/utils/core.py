"""Shared utilities for the pipeline.

Data classes, answer extraction, image encoding, and config loading.
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from loguru import logger
from PIL import Image

# --- Answer normalization ---

CYRILLIC_TO_LATIN = {
    "А": "A", "Б": "B", "В": "C", "Г": "D", "Д": "E",
    "а": "A", "б": "B", "в": "C", "г": "D", "д": "E",
}
CANONICAL_MCQ = {"A", "B", "C", "D", "E"}


def normalize_answer_key(raw: str) -> str | None:
    """Normalize answer keys from various formats to A-E."""
    x = str(raw).strip()
    if x.upper() in CANONICAL_MCQ:
        return x.upper()
    if x in CYRILLIC_TO_LATIN:
        return CYRILLIC_TO_LATIN[x]
    if x.isdigit() and 1 <= int(x) <= 5:
        return chr(ord("A") + int(x) - 1)
    return None


def extract_answer(text: str, choices: set[str] = CANONICAL_MCQ) -> str | None:
    """Extract a single answer letter from model output.

    The MCQ prompts explicitly ask the model to emit English
    "The answer is <letter>", so patterns are English-only. The one
    multilingual accommodation is a Cyrillic A-E to Latin substitution
    applied up front. Even when the model follows the English phrasing,
    it sometimes emits the letter itself in the source-question's script
    (e.g. Russian/Bulgarian/Serbian, "The answer is B"). This mirrors
    the Cyrillic handling on the gold side (`normalize_answer_key`).

    Strategies in order:
      1. "the answer is X" / "answer: X" / "final answer: X" (last match wins)
      2. Standalone single letter (entire response)
    """
    if not text:
        return None
    # Strip <think>...</think> blocks (DeepSeek-R1 family). The thinking
    # content often references answer letters speculatively; extracting from
    # only the post-think output prevents false matches.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    # Cyrillic A-E -> Latin. Cheap char substitution, no pattern gamble.
    for cyr, lat in CYRILLIC_TO_LATIN.items():
        text = text.replace(cyr, lat)
    text_upper = text.upper().strip()

    # Strategy 1: explicit English pattern. Uses findall and takes the LAST
    # match, since models often mention intermediate answers ("could be B")
    # before concluding with the final one ("Answer: C").
    matches = re.findall(
        r"(?:the answer is|final answer:?|correct answer:?|answer:?)\s*\**\s*([A-E])\b",
        text,
        flags=re.IGNORECASE,
    )
    if matches and matches[-1].upper() in choices:
        return matches[-1].upper()

    # Strategy 2: single letter response (e.g. "B" or "B.")
    trimmed = text_upper.rstrip(".").rstrip(":")
    if trimmed in choices:
        return trimmed

    # No Strategy 3 fallback. Picking "the last standalone A-E letter" from
    # free text systematically biases toward whatever the model was
    # enumerating last before truncation (e.g. "Option A..."), which gave
    # us a ~100% "A" collapse on the first smoke. If Strategies 1-2 fail we
    # return None; majority_vote filters None candidates cleanly.
    return None


# --- Data classes ---


@dataclass
class Candidate:
    """A single describe-then-reason candidate."""

    description_id: int
    chain_id: int
    description: str
    reasoning: str
    answer: str | None
    logprob: float | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0

    @property
    def uid(self) -> str:
        return f"d{self.description_id}_c{self.chain_id}"


@dataclass
class SelectionResult:
    """Result of the verification/selection stage."""

    answer: str | None
    method: str
    confidence: str  # "high", "medium", "low", "very_low"
    vote_counts: dict[str, int] = field(default_factory=dict)
    selected_candidate: Candidate | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class PipelineResult:
    """Full result for a single question."""

    question_id: str
    language: str
    predicted_answer: str | None
    selection: SelectionResult | None = None
    candidates: list[Candidate] = field(default_factory=list)
    descriptions: list[str] = field(default_factory=list)


# --- Config loading ---


def load_config(path: str | Path) -> dict:
    """Load a YAML configuration file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


# --- Result I/O ---


def save_results(results: list[PipelineResult], output_path: str | Path):
    """Save pipeline results in the ImageCLEF 2026 MCQ submission format.

    Matches what all three official baselines (molmo.py, smolvlm.py, olmo.py)
    emit: {id, language, answer_key}. evaluate_mcq.py reads `answer_key`
    as the prediction field. Note: the official repo's format_checker.py
    contradicts this by requiring `prediction` instead; we follow the
    baselines since they're what actually produces scored submissions.
    """
    competition_format = []
    for r in results:
        competition_format.append({
            "id": r.question_id,
            "language": r.language,
            "answer_key": r.predicted_answer or "",
        })

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(competition_format, f, indent=2, ensure_ascii=False)
    logger.info("Saved {} predictions to {}", len(competition_format), output_path)


def save_detailed_results(results: list[PipelineResult], output_path: str | Path):
    """Save detailed results including candidates and reasoning for analysis."""
    detailed = []
    for r in results:
        entry = {
            "question_id": r.question_id,
            "language": r.language,
            "predicted_answer": r.predicted_answer,
            "descriptions": r.descriptions,
            "candidates": [
                {
                    "uid": c.uid,
                    "answer": c.answer,
                    "reasoning": c.reasoning[:500],
                    "logprob": c.logprob,
                }
                for c in r.candidates
            ],
        }
        if r.selection:
            entry["selection"] = {
                "method": r.selection.method,
                "confidence": r.selection.confidence,
                "vote_counts": r.selection.vote_counts,
                "metadata": r.selection.metadata,
            }
        detailed.append(entry)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(detailed, f, indent=2, ensure_ascii=False)
    logger.info("Saved detailed results to {}", output_path)
