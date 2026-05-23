"""Analyze parse-fail patterns in SC candidates.jsonl files.

Reports per-language: question-level parse-fail rate, chain-level parse-fail,
truncation at max_tokens, token count distribution, and accuracy on valid
vs all questions (when gold is available).

Usage:
    python scripts/analyze_parse_fail.py runs/<run>/candidates.jsonl
    python scripts/analyze_parse_fail.py file1.jsonl file2.jsonl  # compare
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path


def analyze(path: Path) -> None:
    records = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    print(f"\n{'='*70}")
    print(f"  {path.name}  ({len(records)} questions)")
    print(f"{'='*70}")

    lang_stats: dict[str, dict] = defaultdict(lambda: {
        "n": 0, "sc_none": 0, "correct": 0, "correct_valid": 0, "valid": 0,
        "chains": 0, "chain_fail": 0, "chain_at_max": 0,
        "comp_tokens_ok": [], "comp_tokens_fail": [],
    })

    for r in records:
        lang = r.get("language", "?")
        ls = lang_stats[lang]
        ls["n"] += 1

        pred = (r.get("verifier") or {}).get("selected_answer")
        gold = r.get("gold")
        correct = (pred == gold) if pred and gold else False

        ls["correct"] += int(correct)
        if pred is None:
            ls["sc_none"] += 1
        else:
            ls["valid"] += 1
            ls["correct_valid"] += int(correct)

        for c in r.get("chains", []):
            ls["chains"] += 1
            comp = c.get("completion_tokens", 0)
            ans = c.get("extracted_answer")
            if ans is None:
                ls["chain_fail"] += 1
                ls["comp_tokens_fail"].append(comp)
            else:
                ls["comp_tokens_ok"].append(comp)
            if comp >= 1020:
                ls["chain_at_max"] += 1

    has_gold = any(r.get("gold") for r in records)

    header = f"{'Language':12s} | {'N':>5s} {'SC_nil':>5s} {'%':>6s} | {'ChFail':>7s} {'@Max':>6s} {'Trunc%':>6s} | {'Mean_OK':>7s} {'Mean_F':>7s}"
    if has_gold:
        header += f" | {'Acc_all':>7s} {'Acc_ok':>7s} {'Delta':>6s}"
    print(f"\n{header}")
    print("-" * len(header))

    for lang in sorted(lang_stats, key=lambda l: -lang_stats[l]["n"]):
        ls = lang_stats[lang]
        n = ls["n"]
        sc_none = ls["sc_none"]
        cf = ls["chain_fail"]
        at_max = ls["chain_at_max"]
        sv = ls["valid"]

        mean_ok = statistics.mean(ls["comp_tokens_ok"]) if ls["comp_tokens_ok"] else 0
        mean_f = statistics.mean(ls["comp_tokens_fail"]) if ls["comp_tokens_fail"] else 0

        line = f"{lang:12s} | {n:5d} {sc_none:5d} {sc_none/n*100:5.1f}% | {cf:7d} {at_max:6d} {at_max/max(cf,1)*100:5.1f}% | {mean_ok:6.0f}t {mean_f:6.0f}t"

        if has_gold:
            acc_all = ls["correct"] / n * 100
            acc_ok = ls["correct_valid"] / sv * 100 if sv else 0
            delta = acc_ok - acc_all
            line += f" | {acc_all:6.1f}% {acc_ok:6.1f}% +{delta:4.1f}"

        print(line)

    total = sum(ls["n"] for ls in lang_stats.values())
    total_none = sum(ls["sc_none"] for ls in lang_stats.values())
    total_cf = sum(ls["chain_fail"] for ls in lang_stats.values())
    total_at_max = sum(ls["chain_at_max"] for ls in lang_stats.values())
    print(f"\nOverall: {total_none}/{total} questions parse-fail ({total_none/total*100:.1f}%), "
          f"{total_cf} chain parse-fail ({total_at_max} at max tokens = {total_at_max/max(total_cf,1)*100:.0f}%)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="+", type=Path)
    args = parser.parse_args()
    for p in args.files:
        analyze(p)


if __name__ == "__main__":
    main()
