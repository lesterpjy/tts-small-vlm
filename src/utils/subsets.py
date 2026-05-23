"""Stratified dev subset sampling.

The 200-question dev subset is stratified by language (and secondarily by
subject) rather than i.i.d., so that small languages like Chinese and Arabic
are not under-sampled and we can measure cross-lingual TTS gaps reliably.

Allocation strategy:
  1. Reserve `min_per_language` (default 10) samples per non-empty language.
     This guarantees every language has enough examples for crude accuracy
     stats.
  2. Distribute the remainder proportional to language frequency.
  3. Within each language's budget, allocate across subjects proportional
     to (language, subject) frequency, with largest-remainder rounding to
     hit exact counts.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Iterable


def stratified_indices(
    items: Iterable[dict],
    subset_size: int,
    *,
    seed: int = 42,
    min_per_language: int = 10,
) -> tuple[list[int], dict[str, int]]:
    """Return `subset_size` indices stratified by (language, subject).

    Args:
        items: iterable of dict-like dataset rows with `language` and `subject` keys.
        subset_size: total number of indices to select.
        seed: RNG seed for within-stratum shuffling.
        min_per_language: floor for each non-empty language. Set to 0 to
            disable and fall back to purely proportional allocation.

    Returns:
        (indices, allocation) where `indices` is sorted ascending and
        `allocation` maps "{language}__{subject}" -> count for logging/debugging.
    """
    rng = random.Random(seed)

    by_language: dict[str, list[int]] = defaultdict(list)
    by_stratum: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i, item in enumerate(items):
        lang = (item.get("language") or "unknown").strip()
        subj = (item.get("subject") or "unknown").strip()
        by_language[lang].append(i)
        by_stratum[(lang, subj)].append(i)

    n_languages = len(by_language)
    reserved = min(min_per_language, subset_size // max(n_languages, 1)) if n_languages else 0
    reserved_total = reserved * n_languages
    if reserved_total > subset_size:
        reserved = subset_size // n_languages
        reserved_total = reserved * n_languages

    # Per-language budget: reserved floor + proportional share of the remainder
    remainder = subset_size - reserved_total
    total_items = sum(len(v) for v in by_language.values())
    lang_budget: dict[str, int] = {}
    lang_float: dict[str, float] = {}
    for lang, idxs in by_language.items():
        frac = len(idxs) / total_items if total_items else 0
        lang_float[lang] = reserved + remainder * frac
        lang_budget[lang] = reserved + int(remainder * frac)

    # Largest-remainder rounding to hit exact subset_size at the language level
    diff = subset_size - sum(lang_budget.values())
    remainders = sorted(
        by_language.keys(),
        key=lambda l: lang_float[l] - int(lang_float[l]),
        reverse=True,
    )
    if diff > 0:
        for lang in remainders:
            if diff == 0:
                break
            if lang_budget[lang] < len(by_language[lang]):
                lang_budget[lang] += 1
                diff -= 1
    elif diff < 0:
        diff = -diff
        for lang in sorted(by_language, key=lambda l: -lang_budget[l]):
            if diff == 0:
                break
            if lang_budget[lang] > 0:
                take = min(diff, lang_budget[lang])
                lang_budget[lang] -= take
                diff -= take

    # Clip to available items per language
    for lang in lang_budget:
        lang_budget[lang] = min(lang_budget[lang], len(by_language[lang]))

    # Within each language, allocate across subjects proportional to stratum size
    stratum_budget: dict[tuple[str, str], int] = {}
    for lang, lang_n in lang_budget.items():
        lang_strata = [(k, v) for k, v in by_stratum.items() if k[0] == lang]
        if not lang_strata or lang_n == 0:
            continue
        lang_total = sum(len(v) for _, v in lang_strata)
        floats = {k: lang_n * len(v) / lang_total for k, v in lang_strata}
        ints = {k: int(f) for k, f in floats.items()}
        remaining = lang_n - sum(ints.values())
        # Largest remainder distributes remaining units
        for k, _ in sorted(
            floats.items(), key=lambda x: x[1] - int(x[1]), reverse=True,
        ):
            if remaining == 0:
                break
            if ints[k] < len(by_stratum[k]):
                ints[k] += 1
                remaining -= 1
        stratum_budget.update(ints)

    # Sample within each stratum
    chosen: list[int] = []
    for stratum, n in stratum_budget.items():
        if n == 0:
            continue
        pool = list(by_stratum[stratum])
        rng.shuffle(pool)
        chosen.extend(pool[:n])

    chosen.sort()
    allocation = {f"{lang}__{subj}": n for (lang, subj), n in stratum_budget.items() if n > 0}
    return chosen, allocation
