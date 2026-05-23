#!/usr/bin/env python3
"""
Stratified accuracy evaluation for development.

This script computes accuracy on the validation set with optional stratification
by subject, language, or any other metadata field. It is intended for quick
development iteration, NOT for generating competition submissions.

Usage:
    python eval/evaluate.py --predictions preds.json --references refs.json
    python eval/evaluate.py --predictions preds.json --references refs.json --stratify-by subject language
"""

import argparse
import json
import sys
from collections import defaultdict


def load_json(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list, got {type(data).__name__}")
    return data


def accuracy(correct: int, total: int) -> float:
    return correct / total if total > 0 else 0.0


def evaluate(predictions: list[dict], references: list[dict], stratify_by: list[str] | None = None):
    """Compute overall and optionally stratified accuracy.

    Args:
        predictions: list of {"question_id": ..., "predicted_answer": ...}
                     Also accepts {"id": ..., "prediction": ...} (competition format).
        references:  list of {"question_id": ..., "answer": ..., <metadata fields>}
        stratify_by: optional list of metadata field names to group by.

    Returns:
        dict with "overall" and per-stratum results.
    """
    # Normalize prediction keys
    pred_map = {}
    for p in predictions:
        qid = p.get("question_id") or p.get("id")
        ans = p.get("predicted_answer") or p.get("prediction")
        if qid is None or ans is None:
            raise ValueError(f"Prediction missing question_id/id or predicted_answer/prediction: {p}")
        pred_map[str(qid)] = str(ans).strip().upper()

    # Build reference map
    ref_map = {}
    ref_meta = {}
    for r in references:
        qid = str(r.get("question_id") or r.get("id"))
        ans = str(r.get("answer") or r.get("answer_key", "")).strip().upper()
        ref_map[qid] = ans
        ref_meta[qid] = r

    # Check alignment
    missing = set(ref_map.keys()) - set(pred_map.keys())
    extra = set(pred_map.keys()) - set(ref_map.keys())
    if missing:
        print(f"Warning: {len(missing)} reference questions have no prediction.", file=sys.stderr)
    if extra:
        print(f"Warning: {len(extra)} predictions have no matching reference.", file=sys.stderr)

    common_ids = set(ref_map.keys()) & set(pred_map.keys())

    # Overall accuracy
    correct = sum(1 for qid in common_ids if pred_map[qid] == ref_map[qid])
    total = len(common_ids)
    results = {
        "overall": {
            "accuracy": accuracy(correct, total),
            "correct": correct,
            "total": total,
        }
    }

    # Stratified accuracy
    if stratify_by:
        for field in stratify_by:
            buckets = defaultdict(lambda: {"correct": 0, "total": 0})
            for qid in common_ids:
                value = ref_meta.get(qid, {}).get(field, "unknown")
                buckets[value]["total"] += 1
                if pred_map[qid] == ref_map[qid]:
                    buckets[value]["correct"] += 1

            strata = {}
            for value, counts in sorted(buckets.items(), key=lambda x: -x[1]["total"]):
                strata[value] = {
                    "accuracy": accuracy(counts["correct"], counts["total"]),
                    "correct": counts["correct"],
                    "total": counts["total"],
                }
            results[f"by_{field}"] = strata

    return results


def print_results(results: dict):
    ov = results["overall"]
    print(f"\nOverall accuracy: {ov['accuracy']:.4f} ({ov['correct']}/{ov['total']})")

    for key, strata in results.items():
        if key == "overall":
            continue
        print(f"\n{key}:")
        print(f"  {'Value':<30s} {'Accuracy':>8s}  {'N':>5s}")
        print(f"  {'-'*30} {'-'*8}  {'-'*5}")
        for value, counts in strata.items():
            print(f"  {str(value):<30s} {counts['accuracy']:>8.4f}  {counts['total']:>5d}")


def main():
    parser = argparse.ArgumentParser(description="Stratified accuracy evaluation for development.")
    parser.add_argument("--predictions", required=True, help="Path to predictions JSON file.")
    parser.add_argument("--references", required=True, help="Path to references/gold JSON file.")
    parser.add_argument("--stratify-by", nargs="+", default=None,
                        help="Metadata fields to stratify by (e.g., subject language).")
    parser.add_argument("--output", default=None, help="Optional path to save results JSON.")
    args = parser.parse_args()

    predictions = load_json(args.predictions)
    references = load_json(args.references)

    results = evaluate(predictions, references, stratify_by=args.stratify_by)
    print_results(results)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
