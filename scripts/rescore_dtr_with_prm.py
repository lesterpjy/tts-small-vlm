"""Offline rescore DTR candidates with Qwen-VL-PRM-7B.

Reads a DTR candidates.jsonl (from experiment.py's dtr runner), joins each
chain's description from the record's descriptions array, and scores with
Qwen-VL-PRM using segmented [Perception]/[Reasoning] tagging. Writes
augmented JSONL with:
  - PRM-selected answer (verifier)
  - Per-candidate perception/reasoning attribution scores
  - Original majority-vote verifier preserved as verifier_majority

Two scoring modes:
  - segmented (default): [Perception] + [Reasoning] step tagging, gives
    per-stage attribution for error analysis
  - flat: description+reasoning concatenated, scored as one solution;
    comparable to one_shot on SC chains

CLI:
    python scripts/rescore_dtr_with_prm.py \\
        --input  runs/.../candidates.jsonl \\
        --output runs/.../dtr_prm_scored.jsonl \\
        --dataset examsv_validation \\
        --mode segmented \\
        --subset 200       # dev smoke; omit for full pool

    # Local smoke without GPU:
    python scripts/rescore_dtr_with_prm.py \\
        --input  runs/.../candidates.jsonl \\
        --output /tmp/dtr_prm_mock.jsonl \\
        --mock
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from scripts.score_with_critic import (  # noqa: E402
    _DATASET_HF,
    load_image_lookup,
    passthrough_high_conf,
)


def dtr_chains_to_candidates(record: dict):
    """Convert DTR JSONL chains to Candidate objects, joining description text.

    Unlike SC chains (which have no description stage), DTR chains reference
    a description by `desc_idx`. We join the description text from the
    record's `descriptions` array so the PRM can score it.
    """
    from src.utils import Candidate

    descriptions = record.get("descriptions") or []
    chains = record.get("chains") or []
    cands: list[Candidate] = []
    for c in chains:
        reasoning = c.get("reasoning")
        if not reasoning:
            continue
        desc_idx = int(c.get("desc_idx", 0))
        desc_text = ""
        if desc_idx < len(descriptions):
            desc_text = descriptions[desc_idx].get("text", "")
        cands.append(Candidate(
            description_id=desc_idx,
            chain_id=int(c.get("chain_idx", 0)),
            description=desc_text,
            reasoning=reasoning,
            answer=c.get("extracted_answer"),
            logprob=c.get("logprob_mean"),
            prompt_tokens=int(c.get("prompt_tokens", 0)),
            completion_tokens=int(c.get("completion_tokens", 0)),
            latency_s=float(c.get("latency_s", 0.0)),
        ))
    return cands


def score_record_dtr_segmented(
    record: dict,
    image,
    *,
    prm_model,
    prm_processor,
    prm_tokenizer,
    question: str,
    system_prompt: str | None,
    aggregation: str,
):
    """Score a DTR record with segmented [Perception]/[Reasoning] PRM scoring."""
    from src.verify import qwen_vl_prm_rank_dtr

    candidates = dtr_chains_to_candidates(record)
    if not candidates:
        return record, {"status": "no_candidates", "n_chains": 0}

    sel = qwen_vl_prm_rank_dtr(
        candidates,
        image=image,
        question=question,
        prm_model=prm_model,
        prm_processor=prm_processor,
        prm_tokenizer=prm_tokenizer,
        system_prompt=system_prompt,
        aggregation=aggregation,
    )

    new = dict(record)
    new["verifier_majority"] = record.get("verifier")
    new["verifier"] = {
        "method": "qwen_vl_prm_dtr",
        "selected_answer": sel.answer,
        "cluster_sizes": sel.vote_counts or {},
        "confidence": sel.confidence,
        "tie_break": None,
        "scored_candidates": sel.metadata.get("prm_scores"),
        "segmented_scores": sel.metadata.get("segmented_scores"),
        "aggregation": sel.metadata.get("aggregation"),
        "top_score": sel.metadata.get("top_score"),
        "score_gap": sel.metadata.get("score_gap"),
        "latency_s": 0.0,
    }
    gold = record.get("gold")
    if gold is not None and sel.answer is not None:
        new["correct"] = sel.answer == gold
    elif gold is None:
        new["correct"] = None

    seg_scores = sel.metadata.get("segmented_scores") or {}
    best_uid = sel.selected_candidate.uid if sel.selected_candidate else None
    best_seg = seg_scores.get(best_uid, {})

    return new, {
        "status": "ok",
        "n_chains": len(candidates),
        "prm_winner": sel.answer,
        "majority_winner": (record.get("verifier") or {}).get("selected_answer"),
        "agree": sel.answer == (record.get("verifier") or {}).get("selected_answer"),
        "top_score": sel.metadata.get("top_score"),
        "perception_score": best_seg.get("perception_score"),
        "reasoning_mean": best_seg.get("reasoning_mean"),
    }


def score_record_dtr_flat(
    record: dict,
    image,
    *,
    prm_model,
    prm_processor,
    prm_tokenizer,
    question: str,
    system_prompt: str | None,
    mode: str,
):
    """Score DTR candidates with flat (non-segmented) PRM, comparable to SC rescoring."""
    from src.verify import qwen_vl_prm_rank

    candidates = dtr_chains_to_candidates(record)
    if not candidates:
        return record, {"status": "no_candidates", "n_chains": 0}

    for c in candidates:
        if c.description:
            c.reasoning = f"{c.description}\n\n{c.reasoning}"

    sel = qwen_vl_prm_rank(
        candidates,
        image=image,
        question=question,
        prm_model=prm_model,
        prm_processor=prm_processor,
        prm_tokenizer=prm_tokenizer,
        system_prompt=system_prompt,
        mode=mode,
    )

    new = dict(record)
    new["verifier_majority"] = record.get("verifier")
    new["verifier"] = {
        "method": "qwen_vl_prm_dtr_flat",
        "selected_answer": sel.answer,
        "cluster_sizes": sel.vote_counts or {},
        "confidence": sel.confidence,
        "tie_break": None,
        "scored_candidates": sel.metadata.get("prm_scores"),
        "per_chain_step_scores": sel.metadata.get("per_chain_step_scores"),
        "mode": sel.metadata.get("mode"),
        "top_score": sel.metadata.get("top_score"),
        "score_gap": sel.metadata.get("score_gap"),
        "latency_s": 0.0,
    }
    gold = record.get("gold")
    if gold is not None and sel.answer is not None:
        new["correct"] = sel.answer == gold
    elif gold is None:
        new["correct"] = None

    return new, {
        "status": "ok",
        "n_chains": len(candidates),
        "prm_winner": sel.answer,
        "majority_winner": (record.get("verifier") or {}).get("selected_answer"),
        "agree": sel.answer == (record.get("verifier") or {}).get("selected_answer"),
        "top_score": sel.metadata.get("top_score"),
    }


def _mock_score_record(record, image):
    """Mock PRM: picks the first parseable chain; for local smoke."""
    chains = record.get("chains") or []
    descriptions = record.get("descriptions") or []
    valid = [c for c in chains if c.get("extracted_answer")]
    pick = valid[0]["extracted_answer"] if valid else None

    new = dict(record)
    new["verifier_majority"] = record.get("verifier")
    new["verifier"] = {
        "method": "qwen_vl_prm_dtr_mock",
        "selected_answer": pick,
        "cluster_sizes": dict(Counter(c.get("extracted_answer") for c in valid)),
        "confidence": "low",
        "tie_break": None,
        "scored_candidates": {
            f"d{c.get('desc_idx', 0)}_c{c.get('chain_idx', 0)}": 0.9 if i == 0 else 0.1
            for i, c in enumerate(valid)
        },
        "segmented_scores": {
            f"d{c.get('desc_idx', 0)}_c{c.get('chain_idx', 0)}": {
                "overall": 0.9 if i == 0 else 0.1,
                "perception_score": 0.8 if i == 0 else 0.2,
                "reasoning_mean": 0.9 if i == 0 else 0.1,
                "reasoning_min": 0.7 if i == 0 else 0.0,
                "n_perception_steps": 1,
                "n_reasoning_steps": 2,
            }
            for i, c in enumerate(valid)
        },
        "aggregation": "overall",
        "top_score": 0.9 if pick else 0.0,
        "score_gap": None,
        "latency_s": 0.0,
    }
    gold = record.get("gold")
    new["correct"] = (pick == gold) if (gold is not None and pick is not None) else None
    return new, {
        "status": "mock",
        "n_chains": len(valid),
        "prm_winner": pick,
        "majority_winner": (record.get("verifier") or {}).get("selected_answer"),
        "agree": pick == (record.get("verifier") or {}).get("selected_answer"),
        "top_score": 0.9 if pick else 0.0,
        "perception_score": 0.8,
        "reasoning_mean": 0.9,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--dataset", default="examsv_validation",
        choices=sorted(_DATASET_HF),
    )
    p.add_argument("--prm-model-id", default="ob11/Qwen-VL-PRM-7B")
    p.add_argument(
        "--mode", default="segmented",
        choices=["segmented", "flat_one_shot", "flat_step_mean"],
        help="segmented: [Perception]/[Reasoning] tagging with attribution; "
             "flat_*: description+reasoning concatenated, scored as one chain",
    )
    p.add_argument(
        "--aggregation", default="overall",
        choices=["overall", "reasoning_mean", "reasoning_min"],
        help="(segmented mode only) which score to rank candidates by",
    )
    p.add_argument("--question", default="")
    p.add_argument("--system-prompt", default=None)
    p.add_argument("--subset", type=int, default=None)
    p.add_argument(
        "--skip-high-confidence", action="store_true",
        help="pass through records where majority confidence is 'high'",
    )
    p.add_argument(
        "--mock", action="store_true",
        help="bypass model loading; deterministic mock for local smoke",
    )
    p.add_argument(
        "--resume", action="store_true",
        help="skip records already present in --output (matched by question_id)",
    )
    return p.parse_args()


def _load_prm(args):
    from src.verify import load_qwen_vl_prm
    return load_qwen_vl_prm(args.prm_model_id)


def main() -> int:
    args = parse_args()
    if not args.input.is_file():
        print(f"[ERR] input not found: {args.input}", file=sys.stderr)
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.mock:
        ds = None
        qid_to_idx: dict = {}
        prm_model = prm_processor = prm_tokenizer = None
    else:
        ds, qid_to_idx = load_image_lookup(args.dataset)
        prm_model, prm_processor, prm_tokenizer = _load_prm(args)

    done_qids: set[str] = set()
    if args.resume and args.output.is_file():
        with args.output.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                qid = rec.get("question_id") or rec.get("id")
                if qid:
                    done_qids.add(qid)
        print(f"[resume] {len(done_qids)} records already in {args.output}")

    n_in = n_out = n_skipped = n_changed = n_skip_high = n_resumed = 0
    perception_scores: list[float] = []
    reasoning_scores: list[float] = []
    summary_letters: Counter = Counter()
    t0 = time.perf_counter()

    open_mode = "a" if (args.resume and done_qids) else "w"
    with args.input.open() as fin, args.output.open(open_mode) as fout:
        for ln, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            n_in += 1
            if args.subset is not None and n_in > args.subset:
                break
            record = json.loads(line)
            qid = record.get("question_id") or record.get("id")

            if args.resume and qid in done_qids:
                n_resumed += 1
                continue

            if args.skip_high_confidence and (record.get("verifier") or {}).get("confidence") == "high":
                pass_through = passthrough_high_conf(
                    record, method_label="qwen_vl_prm_dtr_skipped_high_conf",
                )
                fout.write(json.dumps(pass_through, ensure_ascii=False) + "\n")
                n_out += 1
                n_skip_high += 1
                pick = (pass_through.get("verifier") or {}).get("selected_answer")
                if pick:
                    summary_letters[pick] += 1
                continue

            if args.mock:
                image = None
            else:
                idx = qid_to_idx.get(qid)
                image = ds[idx]["image"] if idx is not None else None
            if not args.mock and image is None:
                print(
                    f"[WARN] {args.input}:{ln} qid={qid!r} not in image index; "
                    "writing through unchanged",
                    file=sys.stderr,
                )
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                n_skipped += 1
                continue

            if args.mock:
                new_record, stats = _mock_score_record(record, image)
            elif args.mode == "segmented":
                new_record, stats = score_record_dtr_segmented(
                    record, image,
                    prm_model=prm_model,
                    prm_processor=prm_processor,
                    prm_tokenizer=prm_tokenizer,
                    question=args.question,
                    system_prompt=args.system_prompt,
                    aggregation=args.aggregation,
                )
            else:
                flat_mode = "one_shot" if args.mode == "flat_one_shot" else "step_mean"
                new_record, stats = score_record_dtr_flat(
                    record, image,
                    prm_model=prm_model,
                    prm_processor=prm_processor,
                    prm_tokenizer=prm_tokenizer,
                    question=args.question,
                    system_prompt=args.system_prompt,
                    mode=flat_mode,
                )

            fout.write(json.dumps(new_record, ensure_ascii=False) + "\n")
            n_out += 1
            if stats.get("agree") is False:
                n_changed += 1
            if stats.get("prm_winner"):
                summary_letters[stats["prm_winner"]] += 1
            if stats.get("perception_score") is not None:
                perception_scores.append(stats["perception_score"])
            if stats.get("reasoning_mean") is not None:
                reasoning_scores.append(stats["reasoning_mean"])
            if n_in % 50 == 0:
                elapsed = time.perf_counter() - t0
                n_scored_now = n_out - n_skip_high
                print(
                    f"[{n_in}] {elapsed:.1f}s  "
                    f"scored={n_scored_now} skipped_high={n_skip_high}  "
                    f"changed-vs-maj={n_changed}",
                )

    elapsed = time.perf_counter() - t0
    n_scored = n_out - n_skip_high
    print(f"\n=== DTR + PRM scoring done ===")
    print(f"  mode:                {args.mode}")
    print(f"  aggregation:         {args.aggregation}")
    print(f"  read:                {n_in}")
    print(f"  resumed (skipped):   {n_resumed}")
    print(f"  written:             {n_out}")
    print(f"  passthrough (high):  {n_skip_high}")
    print(f"  scored (selector):   {n_scored}")
    print(f"  skipped (no image):  {n_skipped}")
    print(f"  changed-vs-majority: {n_changed} ({n_changed / max(n_scored,1) * 100:.1f}%)")
    print(f"  letter histogram:    {dict(summary_letters)}")
    if perception_scores:
        avg_p = sum(perception_scores) / len(perception_scores)
        avg_r = sum(reasoning_scores) / len(reasoning_scores) if reasoning_scores else 0.0
        print(f"  mean perception P(+): {avg_p:.3f}")
        print(f"  mean reasoning P(+):  {avg_r:.3f}")
    print(f"  wall: {elapsed:.1f}s "
          f"({elapsed / max(n_scored,1):.2f} s/scored-record, "
          f"{elapsed / max(n_out,1):.2f} s/record-overall)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
