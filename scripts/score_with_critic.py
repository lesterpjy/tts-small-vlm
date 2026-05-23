"""Offline rescore an SC-N=8 candidate pool with the generative critic.

Reads a candidates.jsonl (written by experiment.py's self_consistency runner),
joins each row to the source dataset by question_id to recover the image,
converts the row's `chains` into Candidate objects, and runs
src.verify.generative_critic_rank to pick a winner per question. Writes an
augmented JSONL with the critic-selected answer + per-chain critic scores
preserved alongside the original majority-vote verifier so downstream
analysis can compare the two heads on identical chains.

This experiment performs offline re-scoring of the already-generated
self-consistency candidate pool with the generative critic, with pairwise
comparison against majority vote stratified by language and subject.

CLI:
    python scripts/score_with_critic.py \\
        --input  runs/baseline-self_consistency_n8_full_val-*/candidates.jsonl \\
        --output runs/baseline-self_consistency_n8_full_val-*/critic_scored.jsonl \\
        --dataset examsv_validation \\
        --model Qwen/Qwen2.5-VL-7B-Instruct \\
        --subset 200       # dev smoke; omit for full pool
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

# Ensure project root is on sys.path so src.* imports work whether the script
# is invoked from anywhere.
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

# Match the dataset registry in scripts/experiment.py; keep these in sync.
_DATASET_HF = {
    "examsv_validation": ("MBZUAI/EXAMS-V", "validation", "sample_id"),
    "imageclef_mcq_test": (
        "SU-FMI-AI/ImageCLEF-MR2026-MCQ-Visual",
        "test",
        "question_id",
    ),
}


def load_image_lookup(dataset_name: str):
    """Return (dataset, qid_to_idx) for lazy per-row image lookup.

    Earlier versions materialised a {qid: PIL.Image} dict upfront. That works
    on the 1117-row, ~1332x561 test split (~10 GB peak) but blows past 120 GB
    of memory on EXAMS-V validation (4651 rows x ~6000x3000, approximately
    250 GB decoded RGB), causing OOM-kill before the policy model even loads.

    We instead read only the id column (cheap, strings) and build a position
    map. The HF Dataset is Arrow-backed; ds[idx]["image"] decodes one image
    on demand and the result is GC-eligible after the per-row score() call.
    """
    from datasets import load_dataset
    if dataset_name not in _DATASET_HF:
        raise ValueError(
            f"Unknown --dataset {dataset_name!r}; valid: {sorted(_DATASET_HF)}"
        )
    hf_id, split, id_col = _DATASET_HF[dataset_name]
    print(f"Loading {hf_id}[{split}] (lazy image lookup)...")
    ds = load_dataset(hf_id)[split]
    # ds[id_col] returns a list[str], pulls only the id column, no images.
    qid_to_idx = {qid: i for i, qid in enumerate(ds[id_col])}
    print(f"Built qid->idx for {len(qid_to_idx)} rows; images will decode on demand.")
    return ds, qid_to_idx


# Backwards-compat shim: callers (e.g. score_with_prm.py) imported
# load_image_index. Keep the name resolvable, but it now returns the lazy
# (ds, qid_to_idx) pair.
load_image_index = load_image_lookup


def chains_to_candidates(chains: list[dict]):
    """JSONL chains -> list[Candidate]. Skips chains missing reasoning text."""
    from src.utils import Candidate
    cands: list[Candidate] = []
    for c in chains:
        reasoning = c.get("reasoning")
        if not reasoning:
            continue
        cands.append(Candidate(
            description_id=int(c.get("desc_idx", -1)),
            chain_id=int(c.get("chain_idx", 0)),
            description=c.get("description", ""),
            reasoning=reasoning,
            answer=c.get("extracted_answer"),
            logprob=c.get("logprob_mean"),
            prompt_tokens=int(c.get("prompt_tokens", 0)),
            completion_tokens=int(c.get("completion_tokens", 0)),
            latency_s=float(c.get("latency_s", 0.0)),
        ))
    return cands


def score_record(record: dict, image, *, model, processor, axes, temperature, max_tokens):
    """Apply generative_critic_rank to one record's chains. Mutates a copy.

    Returns (new_record, stats) where stats is a small dict for run summary
    accounting (n_chains, n_skipped, critic_winner, critic_top_score, ...).
    """
    from src.verify import generative_critic_rank

    candidates = chains_to_candidates(record.get("chains") or [])
    if not candidates:
        # Preserve original verifier so downstream comparators don't choke;
        # mark explicitly that the rescoring produced nothing.
        return record, {"status": "no_candidates", "n_chains": 0}

    sel = generative_critic_rank(
        candidates, model, processor, image,
        axes=axes, temperature=temperature, max_tokens=max_tokens,
    )

    # Embed the new selector alongside the original; never overwrite the
    # original verifier (we need it for the critic-vs-majority comparison).
    new = dict(record)
    new["verifier_majority"] = record.get("verifier")  # snapshot the original
    new["verifier"] = {
        "method": "generative_critic",
        "selected_answer": sel.answer,
        "cluster_sizes": sel.vote_counts or {},
        "confidence": sel.confidence,
        "tie_break": None,
        "scored_candidates": sel.metadata.get("critic_scores"),
        "axes": sel.metadata.get("axes"),
        "top_score": sel.metadata.get("top_score"),
        "score_gap": sel.metadata.get("score_gap"),
        "latency_s": 0.0,  # accumulated below
    }
    # Recompute correctness against gold (None on held-out splits).
    gold = record.get("gold")
    if gold is not None and sel.answer is not None:
        new["correct"] = sel.answer == gold
    elif gold is None:
        new["correct"] = None

    return new, {
        "status": "ok",
        "n_chains": len(candidates),
        "critic_winner": sel.answer,
        "majority_winner": (record.get("verifier") or {}).get("selected_answer"),
        "agree": sel.answer == (record.get("verifier") or {}).get("selected_answer"),
        "top_score": sel.metadata.get("top_score"),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, required=True, help="candidates.jsonl path")
    p.add_argument("--output", type=Path, required=True, help="critic_scored.jsonl path")
    p.add_argument(
        "--dataset", default="examsv_validation",
        choices=sorted(_DATASET_HF), help="HF split to load images from",
    )
    p.add_argument(
        "--model", default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="policy model whose backbone runs the critic prompt",
    )
    p.add_argument("--backend", default="vllm", choices=["vllm", "hf"])
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--max-model-len", type=int, default=16384)
    p.add_argument(
        "--axes", nargs="+", default=None,
        help="critic axis names to score (default: all from GENERATIVE_CRITIC_AXES)",
    )
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument(
        "--subset", type=int, default=None,
        help="rescore only the first N records (dev smoke). default: all",
    )
    p.add_argument(
        "--skip-high-confidence", action="store_true",
        help="pass through records where SC majority confidence is 'high' "
             "without rescoring. Cuts compute by ~70%% on EXAMS-V val. "
             "Operationally defines a tiered selector: critic for SC-uncertain, "
             "majority for SC-strong.",
    )
    p.add_argument(
        "--mock", action="store_true",
        help="bypass model loading and use a deterministic mock critic; for "
             "local schema/loop smoke without GPU. Picks the first chain's "
             "answer as the 'critic winner' regardless of content.",
    )
    return p.parse_args()


def passthrough_high_conf(record: dict, *, method_label: str) -> dict:
    """Construct a write-through copy of an SC-high record, preserving the
    original SC verifier as `verifier_majority` and stamping a sentinel
    `verifier.method` so downstream analysis can identify skipped rows.
    """
    new = dict(record)
    orig = record.get("verifier") or {}
    new["verifier_majority"] = orig
    new["verifier"] = {
        "method": method_label,
        "selected_answer": orig.get("selected_answer"),
        "cluster_sizes": orig.get("cluster_sizes", {}),
        "confidence": orig.get("confidence"),
        "tie_break": orig.get("tie_break"),
        "scored_candidates": None,
        "axes": None,
        "top_score": None,
        "score_gap": None,
        "latency_s": 0.0,
        "skipped_high_conf": True,
    }
    # `correct` is None on held-out test; on val we'd need to recompute against
    # gold (selected_answer is unchanged from majority, so correctness is too).
    return new


def _load_model(args):
    """Defer the heavy import so --mock smoke works without torch installed."""
    from src.backend import load_model
    print(f"Loading {args.model} (backend={args.backend}, dtype={args.dtype})...")
    kwargs = {"backend": args.backend, "dtype": args.dtype}
    if args.backend == "vllm":
        kwargs["max_model_len"] = args.max_model_len
    return load_model(args.model, **kwargs)


def _mock_score_record(record, image):
    """Deterministic stand-in for score_record: picks the first parseable
    chain's letter; lets us smoke the loop without a model."""
    chains = record.get("chains") or []
    valid = [c for c in chains if c.get("extracted_answer")]
    pick = valid[0]["extracted_answer"] if valid else None
    new = dict(record)
    new["verifier_majority"] = record.get("verifier")
    new["verifier"] = {
        "method": "generative_critic_mock",
        "selected_answer": pick,
        "cluster_sizes": dict(Counter(c.get("extracted_answer") for c in valid)),
        "confidence": "low",
        "tie_break": None,
        "scored_candidates": {
            f"d{c.get('desc_idx', -1)}_c{c.get('chain_idx', 0)}": 0.5
            for c in valid
        },
        "axes": ["mock"],
        "top_score": 0.5,
        "score_gap": None,
        "latency_s": 0.0,
    }
    gold = record.get("gold")
    new["correct"] = (pick == gold) if (gold is not None and pick is not None) else None
    return new, {
        "status": "mock",
        "n_chains": len(valid),
        "critic_winner": pick,
        "majority_winner": (record.get("verifier") or {}).get("selected_answer"),
        "agree": pick == (record.get("verifier") or {}).get("selected_answer"),
        "top_score": 0.5,
    }


def main() -> int:
    args = parse_args()
    if not args.input.is_file():
        print(f"[ERR] input not found: {args.input}", file=sys.stderr)
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Lazy image lookup: build qid->idx (cheap), decode each image on demand.
    if args.mock:
        ds = None
        qid_to_idx: dict = {}
    else:
        ds, qid_to_idx = load_image_lookup(args.dataset)

    if args.mock:
        model, processor = None, None
    else:
        model, processor = _load_model(args)

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

            # Skip-on-high-confidence: pass through SC-high records unchanged.
            if args.skip_high_confidence and (record.get("verifier") or {}).get("confidence") == "high":
                pass_through = passthrough_high_conf(
                    record, method_label="generative_critic_skipped_high_conf"
                )
                fout.write(json.dumps(pass_through, ensure_ascii=False) + "\n")
                n_out += 1
                n_skip_high += 1
                pick = (pass_through.get("verifier") or {}).get("selected_answer")
                if pick:
                    summary_letters[pick] += 1
                continue

            # Lazy image fetch: ds[idx]["image"] decodes one image; freed on
            # next iteration once the per-record critic call completes.
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
                new_record, stats = score_record(
                    record, image,
                    model=model, processor=processor,
                    axes=args.axes,
                    temperature=args.temperature, max_tokens=args.max_tokens,
                )
            fout.write(json.dumps(new_record, ensure_ascii=False) + "\n")
            n_out += 1
            if stats.get("agree") is False:
                n_changed += 1
            if stats.get("critic_winner"):
                summary_letters[stats["critic_winner"]] += 1
            if n_in % 50 == 0:
                elapsed = time.perf_counter() - t0
                print(
                    f"[{n_in}] {elapsed:.1f}s  "
                    f"agree={n_in - n_changed - n_skipped}/{n_in - n_skipped}  "
                    f"changed={n_changed}",
                )

    elapsed = time.perf_counter() - t0
    n_scored = n_out - n_skip_high
    print(f"\n=== Critic scoring done ===")
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
