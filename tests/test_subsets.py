"""Tests for stratified dev subset sampling."""

from __future__ import annotations

from collections import Counter

from src.utils.subsets import stratified_indices


def _fake_dataset() -> list[dict]:
    """EXAMS-V-like validation set: 13 languages x varying subjects.

    Mirrors the expected skew: English is the largest language, Chinese
    and Arabic medium, Hungarian small. Total = 2000 items.
    """
    rows: list[dict] = []
    layout = {
        "English": {"Biology": 300, "Chemistry": 300, "Physics": 200, "History": 100},
        "Chinese": {"Biology": 80, "Chemistry": 80, "Physics": 40},
        "Arabic": {"Biology": 60, "Chemistry": 40, "History": 50},
        "French": {"Biology": 40, "History": 40, "Geography": 30},
        "German": {"Biology": 30, "Chemistry": 30, "History": 20},
        "Spanish": {"Biology": 30, "Chemistry": 30},
        "Italian": {"Biology": 25, "Physics": 25},
        "Polish": {"Biology": 20, "History": 30},
        "Russian": {"Biology": 20, "Chemistry": 20},
        "Bulgarian": {"Biology": 15, "History": 15},
        "Croatian": {"Biology": 15, "History": 10},
        "Serbian": {"Biology": 10, "History": 10},
        "Hungarian": {"Biology": 10, "History": 5},
    }
    for lang, subjects in layout.items():
        for subj, n in subjects.items():
            for _ in range(n):
                rows.append({"language": lang, "subject": subj})
    return rows


def test_returns_exact_subset_size():
    items = _fake_dataset()
    indices, alloc = stratified_indices(items, subset_size=200, seed=42)
    assert len(indices) == 200
    assert sum(alloc.values()) == 200


def test_every_language_has_minimum():
    """Chinese and Arabic should each get >=10 samples (the min_per_language floor)."""
    items = _fake_dataset()
    indices, _ = stratified_indices(items, subset_size=200, seed=42, min_per_language=10)

    by_language = Counter(items[i]["language"] for i in indices)
    languages = {row["language"] for row in items}
    for lang in languages:
        assert by_language[lang] >= 10, f"{lang} only got {by_language[lang]}"


def test_reproducible_with_same_seed():
    items = _fake_dataset()
    a, _ = stratified_indices(items, subset_size=200, seed=42)
    b, _ = stratified_indices(items, subset_size=200, seed=42)
    assert a == b


def test_different_seed_differs():
    items = _fake_dataset()
    a, _ = stratified_indices(items, subset_size=200, seed=42)
    b, _ = stratified_indices(items, subset_size=200, seed=7)
    assert a != b


def test_subject_diversity_within_language():
    """Within each language, multiple subjects should be represented when possible."""
    items = _fake_dataset()
    indices, _ = stratified_indices(items, subset_size=200, seed=42)

    per_lang_subjects: dict[str, set[str]] = {}
    for i in indices:
        row = items[i]
        per_lang_subjects.setdefault(row["language"], set()).add(row["subject"])

    # English has 4 subjects in the fake data; stratified sampling should cover at least 3
    assert len(per_lang_subjects["English"]) >= 3


def test_no_duplicate_indices():
    items = _fake_dataset()
    indices, _ = stratified_indices(items, subset_size=200, seed=42)
    assert len(indices) == len(set(indices))


def test_handles_small_subset():
    """With subset_size < n_languages * min_per_language, fall back gracefully."""
    items = _fake_dataset()
    indices, alloc = stratified_indices(items, subset_size=20, seed=42, min_per_language=10)
    assert len(indices) == 20
    assert sum(alloc.values()) == 20
