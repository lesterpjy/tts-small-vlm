"""Repair parse-fail questions by running guided answer extraction.

For questions where all N chains failed to produce an extractable answer letter,
this script:
1. Loads the same model via vLLM
2. Loads the test dataset to get images
3. For each failed question's chains, appends an answer suffix and runs
   guided decoding (max_tokens=1, guided_choice=A-E) to force an answer
4. Majority-votes the forced answers
5. Writes an updated candidates.jsonl

This is a post-hoc repair using guided decoding (Phase 2 of the two-phase
generation approach in backend_vllm.py).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from loguru import logger


def main():
    parser = argparse.ArgumentParser(description="Repair parse-fail questions via guided decoding")
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl")
    parser.add_argument("--dataset", default="imageclef_mcq_test", help="Dataset name")
    parser.add_argument("--answer-suffix", default="\n\nAnswer: ", help="Suffix before forced letter")
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct", help="Model name")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--dry-run", action="store_true", help="Identify failures without running model")
    args = parser.parse_args()

    candidates_path = Path(args.candidates)
    records = []
    with open(candidates_path) as f:
        for line in f:
            records.append(json.loads(line))

    failed_indices = []
    for i, r in enumerate(records):
        chains = r.get("chains", [])
        has_answer = any(c.get("extracted_answer") for c in chains)
        if not has_answer and chains:
            failed_indices.append(i)

    logger.info("Total records: {}, parse-fail: {}", len(records), len(failed_indices))

    if not failed_indices:
        logger.info("No parse failures to repair.")
        return

    if args.dry_run:
        for idx in failed_indices:
            r = records[idx]
            logger.info("  FAIL: qid={} lang={} subj={} chains={}",
                        r["question_id"][:12], r.get("language"), r.get("subject"),
                        len(r.get("chains", [])))
        return

    # --- Load model ---
    logger.info("Loading model {}...", args.model)
    from src.backend_vllm import load_model, _build_prompt_inputs, _build_sampling_params

    model, processor = load_model(
        args.model,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    # --- Load dataset (for images) ---
    logger.info("Loading dataset {}...", args.dataset)
    from datasets import load_dataset as hf_load

    _SPECS = {
        "imageclef_mcq_test": {
            "hf_id": "SU-FMI-AI/ImageCLEF-MR2026-MCQ-Visual",
            "split": "test",
            "id_column": "question_id",
        },
        "examsv_validation": {
            "hf_id": "MBZUAI/EXAMS-V",
            "split": "validation",
            "id_column": "sample_id",
        },
    }
    spec = _SPECS[args.dataset]
    ds = hf_load(spec["hf_id"])[spec["split"]]
    id_col = spec["id_column"]
    id_to_idx = {row[id_col]: i for i, row in enumerate(ds)}

    # The MMMU-style CoT prompt (same as the original run)
    PROMPT = (
        "Answer the following multiple-choice question based on the image. "
        "The question and answer options are shown in the image.\n\n"
        "Think step by step. End your answer with: Answer: X (where X is A, B, "
        "C, D, or E)."
    )
    ANSWER_CHOICES = ["A", "B", "C", "D", "E"]

    # --- Guided decoding repair ---
    t0_all = time.perf_counter()
    repaired = 0

    for fi, rec_idx in enumerate(failed_indices):
        rec = records[rec_idx]
        qid = rec["question_id"]
        chains = rec["chains"]

        ds_idx = id_to_idx.get(qid)
        if ds_idx is None:
            logger.warning("Question {} not found in dataset, skipping", qid[:12])
            continue

        image = ds[ds_idx]["image"]

        # Build the base prompt (chat-template formatted)
        base_inputs = _build_prompt_inputs(processor, PROMPT, image)
        base_prompt_text = base_inputs["prompt"]

        # For each chain, build a guided decoding prompt
        p2_inputs = []
        for chain in chains:
            reasoning = chain.get("reasoning", "")
            p2_text = base_prompt_text + reasoning + args.answer_suffix
            p2_inputs.append({
                "prompt": p2_text,
                "multi_modal_data": {"image": image},
            })

        sp = _build_sampling_params(
            temperature=0.0,
            max_new_tokens=1,
            n=1,
            guided_choice=ANSWER_CHOICES,
        )

        t0 = time.perf_counter()
        outputs = model.generate(p2_inputs, sp)
        latency = time.perf_counter() - t0

        forced_answers = []
        for j, out in enumerate(outputs):
            letter = out.outputs[0].text.strip()
            forced_answers.append(letter)
            chains[j]["extracted_answer"] = letter
            chains[j]["phase2_forced"] = True

        votes = Counter(forced_answers)
        winner, count = votes.most_common(1)[0]

        # Update verifier block
        rec["verifier"]["selected_answer"] = winner
        rec["verifier"]["confidence"] = (
            "high" if count > len(chains) * 0.5 else
            "medium" if count == votes.most_common(2)[-1][1] else "low"
        )
        rec["verifier"]["cluster_sizes"] = dict(votes)
        rec["verifier"]["phase2_repair"] = True

        # Update correct field if gold is available
        gold = rec.get("gold")
        if gold:
            rec["correct"] = (winner.upper() == gold.upper())

        repaired += 1
        logger.info(
            "[{}/{}] qid={} -> {} (votes={}, {:.1f}s)",
            fi + 1, len(failed_indices), qid[:12], winner, dict(votes), latency,
        )

    total_time = time.perf_counter() - t0_all
    logger.info("Repaired {}/{} questions in {:.1f}s", repaired, len(failed_indices), total_time)

    # --- Write updated candidates ---
    backup = candidates_path.with_suffix(".jsonl.bak")
    shutil.copy2(candidates_path, backup)
    logger.info("Backup: {}", backup)

    with open(candidates_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info("Updated: {}", candidates_path)


if __name__ == "__main__":
    main()
