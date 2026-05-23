"""Offline rescore an SC-N=8 candidate pool with Qwen-VL-PRM-7B.

Mirror of scripts/score_with_critic.py but uses the trained discriminative
PRM (ob11/Qwen-VL-PRM-7B) instead of the policy-backbone generative critic.
This is the cross-model discriminative comparator: a sycophancy-resistant
control (where the critic and the policy share weights and could prefer
their own generations per Panickssery et al. 2024).

Reads candidates.jsonl, joins to source HF split for images, scores each
chain's per-step P(+), aggregates to a chain-level scalar (step_mean by
default), picks the highest, and writes augmented JSONL with the original
SC-majority verifier preserved alongside.

CLI:
    python scripts/score_with_prm.py \\
        --input  runs/.../candidates.jsonl \\
        --output runs/.../prm_scored.jsonl \\
        --dataset examsv_validation \\
        --mode step_mean \\
        --subset 200       # dev smoke; omit for full pool
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
    chains_to_candidates,
    load_image_lookup,
    passthrough_high_conf,
)


def score_record_with_prm(
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
    from src.verify import qwen_vl_prm_rank

    candidates = chains_to_candidates(record.get("chains") or [])
    if not candidates:
        return record, {"status": "no_candidates", "n_chains": 0}

    sel = qwen_vl_prm_rank(
        candidates,
        image=image, question=question,
        prm_model=prm_model, prm_processor=prm_processor,
        prm_tokenizer=prm_tokenizer,
        system_prompt=system_prompt,
        mode=mode,
    )

    new = dict(record)
    new["verifier_majority"] = record.get("verifier")
    new["verifier"] = {
        "method": "qwen_vl_prm",
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--dataset", default="examsv_validation", choices=sorted(_DATASET_HF))
    p.add_argument("--prm-model-id", default="ob11/Qwen-VL-PRM-7B")
    p.add_argument(
        "--mode", default="step_mean",
        choices=["step_mean", "step_min", "one_shot"],
        help="aggregation across PRM per-step scores; one_shot scores the "
             "full chain in a single forward (~n_steps x cheaper)",
    )
    p.add_argument(
        "--question", default="",
        help="MCQ question text passed in the PRM prompt's '### Question' "
             "section. EXAMS-V questions are rendered IN the image (text-in-"
             "image), so an empty string is fine, the image carries the "
             "question. Override only if you have a textual question to inject.",
    )
    p.add_argument(
        "--system-prompt", default=None,
        help="override the default Qwen-VL-PRM system prompt; default uses "
             "the one in src.verify.QWEN_VL_PRM_SYSTEM_PROMPT",
    )
    p.add_argument("--subset", type=int, default=None)
    p.add_argument(
        "--skip-high-confidence", action="store_true",
        help="pass through records where SC majority confidence is 'high' "
             "without rescoring. Cuts compute by ~70%% on EXAMS-V val. "
             "Operationally defines a tiered selector: PRM for SC-uncertain, "
             "majority for SC-strong.",
    )
    p.add_argument(
        "--mock", action="store_true",
        help="bypass model loading + use a deterministic mock PRM that scores "
             "first-listed parseable chain at 0.9 and others at 0.1; for local "
             "schema/loop smoke without GPU.",
    )
    return p.parse_args()


def _load_prm(args):
    from src.verify import load_qwen_vl_prm
    return load_qwen_vl_prm(args.prm_model_id)


def _mock_score_record(record, image):
    """Mock PRM picks the first parseable chain with a high P(+).

    Useful only as a structural smoke test.
    """
    chains = record.get("chains") or []
    valid = [c for c in chains if c.get("extracted_answer")]
    pick = valid[0]["extracted_answer"] if valid else None
    new = dict(record)
    new["verifier_majority"] = record.get("verifier")
    new["verifier"] = {
        "method": "qwen_vl_prm_mock",
        "selected_answer": pick,
        "cluster_sizes": dict(Counter(c.get("extracted_answer") for c in valid)),
        "confidence": "low",
        "tie_break": None,
        "scored_candidates": {
            f"d{c.get('desc_idx', -1)}_c{c.get('chain_idx', 0)}":
                0.9 if i == 0 else 0.1
            for i, c in enumerate(valid)
        },
        "per_chain_step_scores": {},
        "mode": "mock",
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
    }


def main() -> int:
    args = parse_args()
    if not args.input.is_file():
        print(f"[ERR] input not found: {args.input}", file=sys.stderr)
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Lazy image lookup (see scripts/score_with_critic.py for rationale).
    if args.mock:
        ds = None
        qid_to_idx: dict = {}
        prm_model = prm_processor = prm_tokenizer = None
    else:
        ds, qid_to_idx = load_image_lookup(args.dataset)
        prm_model, prm_processor, prm_tokenizer = _load_prm(args)

    n_in = n_out = n_skipped = n_changed = n_skip_high = 0
    summary_letters: Counter = Counter()
    t0 = time.perf_counter()
    with args.input.open() as fin, args.output.open("w") as fout:
        for ln, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            n_in += 1
            if args.subset is not None and n_in > args.subset:
                break
            record = json.loads(line)
            qid = record.get("question_id") or record.get("id")

            # Skip-on-high-confidence: pass through SC-high records.
            if args.skip_high_confidence and (record.get("verifier") or {}).get("confidence") == "high":
                pass_through = passthrough_high_conf(
                    record, method_label="qwen_vl_prm_skipped_high_conf"
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
            else:
                new_record, stats = score_record_with_prm(
                    record, image,
                    prm_model=prm_model,
                    prm_processor=prm_processor,
                    prm_tokenizer=prm_tokenizer,
                    question=args.question,
                    system_prompt=args.system_prompt,
                    mode=args.mode,
                )
            fout.write(json.dumps(new_record, ensure_ascii=False) + "\n")
            n_out += 1
            if stats.get("agree") is False:
                n_changed += 1
            if stats.get("prm_winner"):
                summary_letters[stats["prm_winner"]] += 1
            if n_in % 50 == 0:
                elapsed = time.perf_counter() - t0
                n_scored_now = n_out - n_skip_high
                print(
                    f"[{n_in}] {elapsed:.1f}s  "
                    f"scored={n_scored_now} skipped_high={n_skip_high}  "
                    f"changed-vs-maj-among-scored={n_changed}",
                )

    elapsed = time.perf_counter() - t0
    n_scored = n_out - n_skip_high
    print(f"\n=== PRM scoring done ===")
    print(f"  read:                {n_in}")
    print(f"  written:             {n_out}")
    print(f"  passthrough (SC-high): {n_skip_high}")
    print(f"  scored (selector):   {n_scored}")
    print(f"  skipped (no image):  {n_skipped}")
    print(f"  changed-vs-majority (among scored): {n_changed} ({n_changed / max(n_scored,1) * 100:.1f}%)")
    print(f"  letter histogram (final picks): {dict(summary_letters)}")
    print(f"  wall: {elapsed:.1f}s "
          f"({elapsed / max(n_scored,1):.2f} s/scored-record, "
          f"{elapsed / max(n_out,1):.2f} s/record-overall)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
