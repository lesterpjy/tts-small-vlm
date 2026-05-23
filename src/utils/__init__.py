"""Shared utilities for the pipeline.

Core symbols are re-exported here so `from src.utils import Candidate` etc.
keeps working unchanged.
"""

from src.utils.core import (
    CANONICAL_MCQ,
    CYRILLIC_TO_LATIN,
    Candidate,
    PipelineResult,
    SelectionResult,
    extract_answer,
    load_config,
    normalize_answer_key,
    save_detailed_results,
    save_results,
)
from src.utils.subsets import stratified_indices

__all__ = [
    "CANONICAL_MCQ",
    "CYRILLIC_TO_LATIN",
    "Candidate",
    "PipelineResult",
    "SelectionResult",
    "extract_answer",
    "load_config",
    "normalize_answer_key",
    "save_detailed_results",
    "save_results",
    "stratified_indices",
]
