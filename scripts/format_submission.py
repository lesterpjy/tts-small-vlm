"""Emit an ImageCLEF 2026 MCQ submission JSON from a candidates.jsonl.

Reads runs/<run_id>/candidates.jsonl (one row per question, schema written
by src/utils/records.py::build_question_record), selects a per-question
prediction letter, and writes the JSON in the AI4Media-Bench platform's
canonical schema.

Output schema per row (defensive, all plausible answer-field aliases):
  {
    "question_id":      "<hex>",
    "predicted_answer": "A",
    "answer_key":       "A",
    "prediction":       "A",
    "language":         "Bulgarian"
  }

Container: AI4Media-Bench rejects raw .json uploads. Zip the output with
the JSON named `predictions.json` at the archive root before uploading.

  cd submissions
  cp my_submission.json predictions.json
  zip my_submission.zip predictions.json
  rm predictions.json

Selection:
  --field verifier.selected_answer (default) uses the SC majority winner /
    critic top-pick / PRM top-pick already written into the verifier block.
  --field chains.majority recomputes majority over chains[*].extracted_answer
    (useful for sanity-checking a stale verifier block).

Fallback chain (per question):
  1. value at --field, if a single A-E letter
  2. majority over chains[*].extracted_answer, ignoring None
  3. if none parseable, --fallback (default "A"), submission spec rejects null

Each prediction is upper-cased and asserted in {A,B,C,D,E}; ids deduped; total
must match --expected-count if supplied. Optionally invokes the official
format_checker.py inline.

Usage:
  python scripts/format_submission.py \\
      --input runs/<run_id>/candidates.jsonl \\
      --output submissions/<arm>.json \\
      --expected-count 1117 \\
      --check ../ImageCLEF-MultimodalReasoning/2026/src/utils/format_checker.py
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

VALID_LETTERS = {"A", "B", "C", "D", "E"}


def _get_path(record: dict, dotted_path: str):
    """Walk a dotted path through a nested dict; return None if any segment missing."""
    obj = record
    for seg in dotted_path.split("."):
        if not isinstance(obj, dict):
            return None
        obj = obj.get(seg)
        if obj is None:
            return None
    return obj


def _normalize_letter(value) -> str | None:
    """Accept str (any case) or None; return upper-case letter in A-E or None."""
    if value is None:
        return None
    if not isinstance(value, str):
        return None
    s = value.strip().upper()
    return s if s in VALID_LETTERS else None


def _majority_over_chains(record: dict) -> str | None:
    """Recompute majority over chains[*].extracted_answer."""
    chains = record.get("chains") or []
    votes = [_normalize_letter(c.get("extracted_answer")) for c in chains]
    votes = [v for v in votes if v is not None]
    if not votes:
        return None
    return Counter(votes).most_common(1)[0][0]


def select_prediction(
    record: dict, *, field: str, fallback: str
) -> tuple[str, str]:
    """Return (letter, source) where source describes which fallback fired."""
    if field == "chains.majority":
        winner = _majority_over_chains(record)
        if winner is not None:
            return winner, "chains.majority"
    else:
        primary = _normalize_letter(_get_path(record, field))
        if primary is not None:
            return primary, field
        winner = _majority_over_chains(record)
        if winner is not None:
            return winner, "fallback:chains.majority"
    return fallback, "fallback:default"


def build_submission(
    records, *, field: str, fallback: str
) -> tuple[list[dict], dict]:
    """Emit the AI4Media-Bench-canonical schema (defensive: all four
    plausible answer-field aliases present per row, plus language).

    The platform's deployed scorer expects `question_id` for the id field;
    its answer-field name is one of {predicted_answer, answer_key,
    prediction} and the read_data implementation tolerates extra keys,
    so we emit all three with the same letter, whichever the platform
    reads, it gets the right answer.

    Container: ZIP with this JSON named `predictions.json` at the
    archive root (CodaBench convention). The platform's file picker
    rejects raw .json uploads.
    """
    items: list[dict] = []
    seen: set[str] = set()
    counts = Counter()
    for r in records:
        qid = r.get("question_id") or r.get("id")
        if not qid:
            raise ValueError(f"record missing question_id: {r!r}")
        if qid in seen:
            raise ValueError(f"duplicate id: {qid}")
        seen.add(qid)
        letter, source = select_prediction(r, field=field, fallback=fallback)
        if letter not in VALID_LETTERS:
            raise ValueError(f"non-letter prediction {letter!r} for {qid}")
        items.append({
            "question_id":      qid,
            "predicted_answer": letter,
            "answer_key":       letter,
            "prediction":       letter,
            "language":         r.get("language", ""),
        })
        counts[source] += 1
    return items, dict(counts)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, required=True, help="candidates.jsonl path")
    p.add_argument("--output", type=Path, required=True, help="submission JSON path")
    p.add_argument(
        "--field", default="verifier.selected_answer",
        help="dotted path to the prediction; or 'chains.majority' to recompute",
    )
    p.add_argument(
        "--fallback", default="A",
        choices=sorted(VALID_LETTERS),
        help="letter to emit if all candidates fail to parse (spec rejects null)",
    )
    p.add_argument("--expected-count", type=int, default=None,
                   help="assert len(items) == this (e.g. 1117 for the test split)")
    p.add_argument("--check", type=Path, default=None,
                   help="path to ImageCLEF format_checker.py; if set, runs it on the output")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.input.is_file():
        print(f"[ERR] input not found: {args.input}", file=sys.stderr)
        return 2

    records: list[dict] = []
    for ln, line in enumerate(args.input.read_text().splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"[ERR] {args.input}:{ln} not JSON: {e}", file=sys.stderr)
            return 2

    items, source_counts = build_submission(
        records, field=args.field, fallback=args.fallback,
    )

    if args.expected_count is not None and len(items) != args.expected_count:
        print(
            f"[ERR] item count {len(items)} != expected {args.expected_count}",
            file=sys.stderr,
        )
        return 3

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(items, indent=2, ensure_ascii=False))
    print(f"[OK] wrote {len(items)} predictions -> {args.output}")
    print(f"     selection sources: {source_counts}")
    print(f"     letter histogram : {dict(Counter(i['prediction'] for i in items))}")

    if args.check is not None:
        if not args.check.is_file():
            print(f"[WARN] format_checker not found at {args.check}; skipping")
            return 0
        rc = subprocess.run(
            [sys.executable, str(args.check), "--input_file", str(args.output)],
            check=False,
        ).returncode
        if rc != 0:
            print(f"[ERR] format_checker rejected the file (rc={rc})", file=sys.stderr)
            return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
