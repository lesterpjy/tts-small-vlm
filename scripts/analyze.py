"""Post-hoc analysis of a run's JSONL.

Reads `runs/{run_id}/candidates.jsonl`, computes the (N, M) scaling grid by
subsampling the max-budget candidate set per question, plus stratified
accuracies by subject / language / content_type. Writes
`runs/{run_id}/summary.json`. Optionally resumes the W&B run by name and
pushes the scaling scalars and stratified summary there.

The JSONL is never modified. `analyze.py` is re-runnable and reproduces
every number without touching the VLM.

Usage:
    python scripts/analyze.py --run-dir runs/<run_id>
    python scripts/analyze.py --run-dir runs/<run_id> --wandb-resume
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

# project-root on sys.path
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import yaml
from loguru import logger

from src.utils import Candidate
from src.verify import majority_vote


def load_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _candidates_for(record: dict, n: int, m: int) -> list[Candidate]:
    """Subsample up to first `n` descriptions and `m` chains per description.

    Returns a flat `Candidate` list that `src.verify.majority_vote` can
    consume. Only fields relevant to majority_vote are populated.
    """
    desc_ids = {d["idx"] for d in record.get("descriptions", [])[:n]}
    out: list[Candidate] = []
    per_desc: dict[int, int] = defaultdict(int)
    for ch in record.get("chains", []):
        di = ch["desc_idx"]
        if di not in desc_ids and di != -1:
            continue
        if per_desc[di] >= m:
            continue
        per_desc[di] += 1
        out.append(Candidate(
            description_id=di,
            chain_id=ch["chain_idx"],
            description="",
            reasoning=ch.get("reasoning", ""),
            answer=ch.get("extracted_answer"),
            logprob=ch.get("logprob_mean"),
        ))
    return out


def compute_scaling_grid(
    records: list[dict],
    *,
    n_max: int,
    m_max: int,
    method: str,
) -> dict[str, float]:
    """Return `{"n{n}_m{m}": accuracy}` for every `(n, m)` in the grid.

    For `majority_vote`, the verifier is a pure function of candidates so
    subsampling is exact. Other methods are flagged and fall back to the
    recorded prediction (only the max-budget accuracy is correct).
    """
    grid: dict[str, float] = {}
    for n in range(1, n_max + 1):
        for m in range(1, m_max + 1):
            correct = 0
            total = 0
            for rec in records:
                gold = rec.get("gold")
                if gold is None:
                    continue
                total += 1
                if method == "majority_vote":
                    cands = _candidates_for(rec, n, m)
                    sel = majority_vote(cands)
                    pred = sel.answer
                else:
                    # Non-pure verifier: only the max-budget accuracy is correct
                    pred = rec.get("verifier", {}).get("selected_answer")
                if pred == gold:
                    correct += 1
            grid[f"n{n}_m{m}"] = correct / total if total else 0.0
    return grid


def compute_stratified(records: list[dict]) -> dict[str, dict[str, float]]:
    buckets: dict[str, dict[str, list[int]]] = {
        "subject": defaultdict(lambda: [0, 0]),
        "language": defaultdict(lambda: [0, 0]),
        "content_type": defaultdict(lambda: [0, 0]),
    }
    for r in records:
        gold = r.get("gold")
        if gold is None:
            continue
        pred = r.get("verifier", {}).get("selected_answer")
        hit = 1 if pred == gold else 0
        for dim in buckets:
            key = r.get(dim) or "unknown"
            buckets[dim][key][0] += hit
            buckets[dim][key][1] += 1
    out: dict[str, dict[str, float]] = {}
    for dim, bucket in buckets.items():
        out[dim] = {k: (hits / total) if total else 0.0 for k, (hits, total) in bucket.items()}
    return out


def main():
    parser = argparse.ArgumentParser(description="Post-hoc analysis of a run's JSONL")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--wandb-resume", action="store_true",
                        help="Resume the W&B run (by name) and log scaling scalars + stratified summary.")
    args = parser.parse_args()

    run_dir: Path = args.run_dir
    jsonl_path = run_dir / "candidates.jsonl"
    config_path = run_dir / "config.yaml"
    assert jsonl_path.exists(), f"missing {jsonl_path}"
    assert config_path.exists(), f"missing {config_path}"

    records = load_jsonl(jsonl_path)
    config = yaml.safe_load(config_path.read_text())
    run_id = run_dir.name
    n_max = config.get("describe", {}).get("n_samples", 1)
    m_max = config.get("reason", {}).get("m_samples", 1)
    method = config.get("verify", {}).get("method", "majority_vote")

    if method != "majority_vote":
        logger.warning(
            "Verifier method '{}' is not pure; scaling grid will be a flat line "
            "at the (N_max, M_max) accuracy.",
            method,
        )

    grid = compute_scaling_grid(records, n_max=n_max, m_max=m_max, method=method)
    strata = compute_stratified(records)
    overall = grid.get(f"n{n_max}_m{m_max}", 0.0)
    total_calls = sum(r.get("total_vlm_calls", 0) for r in records)
    total_latency_s = sum(r.get("total_latency_s", 0.0) for r in records)

    summary = {
        "run_id": run_id,
        "method": method,
        "questions": len(records),
        "overall_accuracy": overall,
        "total_vlm_calls": total_calls,
        "total_latency_s": total_latency_s,
        "scaling_grid": grid,
        "stratified": strata,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    logger.info("Wrote {}", run_dir / "summary.json")
    logger.info("Overall {:.1%} on {} questions", overall, len(records))

    if args.wandb_resume:
        try:
            import wandb
            run = wandb.init(
                project=os.environ.get("WANDB_PROJECT", "tts-small-vlm"),
                entity=os.environ.get("WANDB_ENTITY") or None,
                name=run_id,
                resume="allow",
                id=None,  # resume by name, allow W&B to find the existing run
            )
            for key, acc in grid.items():
                n = int(key.split("_")[0][1:])
                m = int(key.split("_")[1][1:])
                wandb.log({
                    "scaling/accuracy": acc,
                    "scaling/N": n,
                    "scaling/M": m,
                }, step=n * m)
            for dim, bucket in strata.items():
                for key, acc in bucket.items():
                    wandb.summary[f"stratified/{dim}/{key}"] = acc
            wandb.summary["overall_accuracy"] = overall
            wandb.finish()
            logger.info("W&B resumed and scaling/stratified metrics pushed.")
        except Exception:
            logger.exception("W&B resume failed; summary.json is still authoritative.")


if __name__ == "__main__":
    main()
