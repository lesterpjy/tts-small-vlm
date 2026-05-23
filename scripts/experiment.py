"""Experiment runner (config-driven).

Runs experiments on the dev subset:
  - Baselines: zero-shot, CoT, self-consistency (N=8)
  - DTR + majority vote at (N=2,M=2), (N=4,M=1), (N=4,M=4)

Single model load, shared across all experiments. Each experiment variant
gets its own `run_id` and its own `runs/{run_id}/candidates.jsonl`, plus a
matching Phoenix project filter and W&B run. This gives clean cross-variant
comparison in the W&B UI.

Usage:
    uv run python scripts/experiment.py                      # all experiments
    uv run python scripts/experiment.py --only baseline      # baselines only
    uv run python scripts/experiment.py --only dtr           # DTR only
    uv run python scripts/experiment.py --only smoke         # smoke test
    uv run python scripts/experiment.py --subset 10          # quick test on 10 q's
    uv run python scripts/experiment.py --push-to-hub        # + push JSONL to HF
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

# Load .env before observability backends read env vars. Strips surrounding
# quotes from values so phoenix-otel etc. don't see `"https://..."` (literal
# quotes included) and fail to parse the URL. Skips unfilled placeholders
# (values still starting with `<`).
_env = Path(__file__).resolve().parent.parent / ".env"
if _env.is_file():
    for line in _env.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if v and not v.startswith("<"):
                os.environ.setdefault(k, v)

# Ensure project root is on sys.path
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from loguru import logger
from tqdm.auto import tqdm

from src.backend import GenerationOutput, generate, generate_n, load_model
from src.pipeline import run_pipeline
from src.utils import extract_answer, normalize_answer_key, stratified_indices
from src.utils.logger import make_run_id, run_context
from src.utils.records import (
    build_question_record,
    chain_entry,
    description_entry,
    verifier_entry,
)

SEED = 42
_WALL_ID = time.strftime("%Y%m%d_%H%M%S")


def setup_logging(verbose: bool = False):
    logger.remove()
    level = "DEBUG" if verbose else "INFO"
    logger.add(
        sys.stderr, level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<7}</level> | <level>{message}</level>",
    )
    Path("logs").mkdir(exist_ok=True)
    logger.add(f"logs/experiment_{_WALL_ID}.log", level="DEBUG")
    logger.info("Experiment wall id: {}", _WALL_ID)


# ---------------------------------------------------------------------------
# Baseline prompts
# ---------------------------------------------------------------------------

MCQ_ZERO_SHOT_PROMPT = (
    "Look at the image and answer the following multiple-choice question. "
    "The question and answer options are shown in the image. "
    "Choose the correct answer from the options.\n\n"
    "Answer with only the letter of the correct option (A, B, C, D, or E)."
)

MCQ_COT_PROMPT = (
    "Look at the image and answer the following multiple-choice question. "
    "The question and answer options are shown in the image.\n\n"
    "Think step by step before answering. First describe what you see in the "
    "image, then reason through each option. "
    "End your response with: The answer is <letter>."
)

# MMMU-style CoT prompt (Yue et al. 2024, arXiv 2311.16502). The de-facto
# multimodal-MCQ CoT evaluation shape: Qwen2.5-VL, InternVL, VisualPRM,
# Athena-PRM, and Ahmadpour et al. "Limits of TTS for VLMs" all report MMMU
# numbers using this prompt family. The rigid "Answer: X" closer (no "the",
# no punctuation before the letter) is far more parser-friendly than the
# original "The answer is <letter>", as it is an exact-match target for
# extract_answer Strategy 1.
MCQ_COT_MMMU_PROMPT = (
    "Answer the following multiple-choice question based on the image. "
    "The question and answer options are shown in the image.\n\n"
    "Think step by step. End your answer with: Answer: X (where X is A, B, "
    "C, D, or E)."
)

_COT_PROMPT_STYLES = {
    "default": MCQ_COT_PROMPT,
    "mmmu": MCQ_COT_MMMU_PROMPT,
}

# Prompts reproduced verbatim from the official ImageCLEF 2026 baselines
# (ImageCLEF-MultimodalReasoning/2026/src/baselines/*.py). Using these lets
# us directly compare our policy against the molmo / smolvlm / olmo numbers
# published with the same wording and the same guided_choice constraint.

MCQ_OFFICIAL_SHORT_PROMPT = (
    "Analyze the image of a multiple-choice question. Identify the question, "
    "all answer options (even if there are more than four), and any relevant "
    "visuals like graphs or tables. Choose the correct answer based only on "
    "the image. Reply with just the letter of the correct option, no explanation."
)

MCQ_OFFICIAL_LONG_PROMPT = (
    "You are a sophisticated Vision-Language Model (VLM) capable of analyzing "
    "images containing multiple-choice questions, regardless of language. To "
    "guide your analysis, you may adopt the following process:\n"
    "1. Examine the image carefully for all textual and visual information.\n"
    "2. Identify the question text, even if it's in a different language.\n"
    "3. Extract all answer options (note: there may be more than four).\n"
    "4. Look for additional visual elements such as tables, diagrams, charts, or graphs.\n"
    "5. Ensure to consider any multilingual content present in the image.\n"
    "6. Analyze the complete context and data provided.\n"
    "7. Select the correct answer(s) based solely on your analysis.\n"
    "8. Respond by outputting only the corresponding letter(s) without any extra explanation."
)

_PROMPT_STYLES = {
    "default": MCQ_ZERO_SHOT_PROMPT,
    "official_short": MCQ_OFFICIAL_SHORT_PROMPT,
    "official_long": MCQ_OFFICIAL_LONG_PROMPT,
}


def _baseline_record(
    *,
    run_id: str,
    item: dict,
    gen: GenerationOutput,
    answer: str | None,
    gold: str | None,
    method_label: str,
    model_name: str,
    temperature: float,
    seed: int,
    total_latency_s: float,
    verifier_override: dict | None = None,
    descriptions_entries: list[dict] | None = None,
    chain_entries: list[dict] | None = None,
) -> dict:
    """Shape a baseline result as a record (N=1 description, M=1 chain)."""
    if descriptions_entries is None:
        descriptions_entries = []
    if chain_entries is None:
        chain_entries = [
            chain_entry(
                desc_idx=0,
                chain_idx=0,
                reasoning=gen.text,
                extracted_answer=answer,
                prompt_tokens=gen.prompt_tokens,
                completion_tokens=gen.completion_tokens,
                logprob_mean=gen.logprob,
                latency_s=gen.latency_s,
                model=model_name,
                temperature=temperature,
                seed=seed,
            )
        ]
    verifier = verifier_override or {
        "method": method_label,
        "selected_answer": answer,
        "cluster_sizes": {answer: 1} if answer else {},
        "confidence": "high" if answer else "very_low",
        "tie_break": None,
        "scored_candidates": None,
        "latency_s": 0.0,
    }
    correct = (answer == gold) if (answer is not None and gold is not None) else None
    # Honour dataset/split labels stamped on the row by the per-question loop.
    # Falls back to build_question_record's defaults (exams_v / validation) so
    # legacy callers keep working.
    rec_kwargs = {
        "run_id": run_id,
        "item": {**item, "gold": gold},
        "descriptions": descriptions_entries,
        "chains": chain_entries,
        "verifier": verifier,
        "correct": correct,
        "total_vlm_calls": len(chain_entries) + len(descriptions_entries),
        "total_latency_s": total_latency_s,
    }
    if "_dataset" in item:
        rec_kwargs["dataset"] = item["_dataset"]
    if "_split" in item:
        rec_kwargs["split"] = item["_split"]
    return build_question_record(**rec_kwargs)


# ---------------------------------------------------------------------------
# Baseline runners (emit spans via ctx + return a JSONL record)
# ---------------------------------------------------------------------------


def _run_single_llm_baseline(
    *, model, processor, item, ctx, method_label, prompt, temperature, max_new_tokens,
    model_name, seed, guided_choice=None,
):
    """Single-generation baseline. `guided_choice` (vLLM only) constrains
    output to one of the given strings, used by the official MCQ variants
    to force A/B/C/D/E without free-text extraction."""
    qid = item.get("sample_id") or item.get("id")
    gold = normalize_answer_key(item.get("answer_key", "")) if item.get("answer_key") else None
    span_item = {**item, "gold": gold}

    t0 = time.perf_counter()
    with ctx.question_span(span_item) as qspan:
        with ctx.stage_span("reason", m_samples=1, n_descriptions=0):
            with ctx.llm_span(
                "reasoning.chain",
                desc_idx=-1, chain_idx=0,
                model=model_name, temperature=temperature, seed=seed,
                prompt=prompt,
            ) as leaf:
                gen = generate(
                    model, processor, prompt, image=item["image"],
                    temperature=temperature, max_new_tokens=max_new_tokens,
                    guided_choice=guided_choice, seed=seed,
                )
                answer = extract_answer(gen.text)
                ctx.set_llm_output(
                    leaf, gen.text,
                    prompt_tokens=gen.prompt_tokens,
                    completion_tokens=gen.completion_tokens,
                    latency_s=gen.latency_s,
                    logprob=gen.logprob,
                    extracted_answer=answer,
                )
        try:
            qspan.set_attribute("predicted", answer or "")
            qspan.set_attribute("correct", bool(answer == gold) if gold else False)
            qspan.set_attribute("total_vlm_calls", 1)
            qspan.set_attribute("total_prompt_tokens", int(gen.prompt_tokens or 0))
            qspan.set_attribute("total_completion_tokens", int(gen.completion_tokens or 0))
            qspan.set_attribute("total_tokens",
                                int((gen.prompt_tokens or 0) + (gen.completion_tokens or 0)))
            qspan.set_attribute("n_chains", 1)
            qspan.set_attribute("n_descriptions", 0)
            if gold is not None:
                qspan.set_attribute("gold", gold)
        except Exception:
            pass
    total_latency_s = time.perf_counter() - t0

    return _baseline_record(
        run_id=ctx.run_id,
        item=item,
        gen=gen,
        answer=answer,
        gold=gold,
        method_label=method_label,
        model_name=model_name,
        temperature=temperature,
        seed=seed,
        total_latency_s=total_latency_s,
    )


def _gen_params(cfg: dict, *, default_temp: float, default_max_tokens: int) -> tuple[float, int]:
    """Pull generation knobs from a variant config with baseline defaults."""
    gen = cfg.get("generation", {})
    return (
        float(gen.get("temperature", default_temp)),
        int(gen.get("max_new_tokens", default_max_tokens)),
    )


def run_zero_shot_q(model, processor, item, ctx, model_name, seed, cfg):
    temperature, max_new_tokens = _gen_params(cfg, default_temp=0.0, default_max_tokens=512)
    gen_cfg = cfg.get("generation", {})
    prompt_style = gen_cfg.get("prompt_style", "default")
    prompt = _PROMPT_STYLES.get(prompt_style, MCQ_ZERO_SHOT_PROMPT)
    # guided_choice (vLLM) forces the output to be exactly one of these strings.
    # Matches the official baselines' use of vLLM's guided_choice=[A..E] so
    # our zero-shot numbers are directly comparable. List is optional;
    # omit to get free-text decoding.
    guided_choice = gen_cfg.get("guided_choice")
    method_label = f"zero_shot_{prompt_style}" if prompt_style != "default" else "zero_shot"
    return _run_single_llm_baseline(
        model=model, processor=processor, item=item, ctx=ctx,
        method_label=method_label,
        prompt=prompt, temperature=temperature, max_new_tokens=max_new_tokens,
        model_name=model_name, seed=seed, guided_choice=guided_choice,
    )


def run_cot_q(model, processor, item, ctx, model_name, seed, cfg):
    temperature, max_new_tokens = _gen_params(cfg, default_temp=0.0, default_max_tokens=1024)
    prompt_style = cfg.get("generation", {}).get("prompt_style", "default")
    prompt = _COT_PROMPT_STYLES.get(prompt_style, MCQ_COT_PROMPT)
    return _run_single_llm_baseline(
        model=model, processor=processor, item=item, ctx=ctx,
        method_label="cot",
        prompt=prompt, temperature=temperature, max_new_tokens=max_new_tokens,
        model_name=model_name, seed=seed,
    )


def run_self_consistency_q(model, processor, item, ctx, model_name, seed, cfg, n=8):
    from src.utils import Candidate
    from src.verify import majority_vote

    temperature, max_new_tokens = _gen_params(cfg, default_temp=0.7, default_max_tokens=1024)
    prompt_style = cfg.get("generation", {}).get("prompt_style", "default")
    prompt = _COT_PROMPT_STYLES.get(prompt_style, MCQ_COT_PROMPT)
    qid = item.get("sample_id")
    gold = normalize_answer_key(item.get("answer_key", "")) if item.get("answer_key") else None
    span_item = {**item, "gold": gold}

    chains_meta: list[dict] = []  # for JSONL chain_entries
    candidates: list[Candidate] = []  # for majority_vote with logprob tie-break

    t0 = time.perf_counter()
    with ctx.question_span(span_item) as qspan:
        with ctx.stage_span("reason", m_samples=n, n_descriptions=0):
            # Single batched vLLM call -> N CoT chains. Image encode + prefill
            # are paid once; only decoding is N-wise. HF backend loops.
            gens = generate_n(
                model, processor, prompt, n, image=item["image"],
                temperature=temperature, max_new_tokens=max_new_tokens,
                seed=seed,
            )
            for i, gen in enumerate(gens):
                ans = extract_answer(gen.text)
                with ctx.llm_span(
                    "reasoning.chain",
                    desc_idx=-1, chain_idx=i,
                    model=model_name, temperature=temperature, seed=seed,
                    prompt=prompt,
                ) as leaf:
                    ctx.set_llm_output(
                        leaf, gen.text,
                        prompt_tokens=gen.prompt_tokens,
                        completion_tokens=gen.completion_tokens,
                        latency_s=gen.latency_s,
                        logprob=gen.logprob,
                        extracted_answer=ans,
                    )
                candidates.append(Candidate(
                    description_id=-1, chain_id=i,
                    description="", reasoning=gen.text, answer=ans,
                    logprob=gen.logprob,
                    prompt_tokens=gen.prompt_tokens,
                    completion_tokens=gen.completion_tokens,
                    latency_s=gen.latency_s,
                ))
                chains_meta.append(chain_entry(
                    desc_idx=-1, chain_idx=i, reasoning=gen.text,
                    extracted_answer=ans,
                    prompt_tokens=gen.prompt_tokens,
                    completion_tokens=gen.completion_tokens,
                    logprob_mean=gen.logprob, latency_s=gen.latency_s,
                    model=model_name, temperature=temperature, seed=seed,
                ))

        # Reuse the same majority_vote (with logprob tie-break) that the DTR
        # pipeline uses, instead of Counter.most_common(1) (which resolves ties
        # by insertion order). Wraps chains in a "verify" span to match the
        # Phoenix span tree conventions for DTR.
        with ctx.stage_span("verify", method="majority_vote"):
            with ctx.stage_span("verifier.majority_vote"):
                selection = majority_vote(candidates)

        winner = selection.answer

        try:
            q_prompt_toks = sum(int(c.prompt_tokens or 0) for c in candidates)
            q_completion_toks = sum(int(c.completion_tokens or 0) for c in candidates)
            qspan.set_attribute("predicted", winner or "")
            qspan.set_attribute("correct", bool(winner == gold) if gold else False)
            qspan.set_attribute("total_vlm_calls", len(candidates))
            qspan.set_attribute("total_prompt_tokens", q_prompt_toks)
            qspan.set_attribute("total_completion_tokens", q_completion_toks)
            qspan.set_attribute("total_tokens", q_prompt_toks + q_completion_toks)
            qspan.set_attribute("n_chains", len(candidates))
            qspan.set_attribute("n_descriptions", 0)
            if gold is not None:
                qspan.set_attribute("gold", gold)
        except Exception:
            pass
    total_latency_s = time.perf_counter() - t0

    verifier = {
        "method": "self_consistency",
        "selected_answer": winner,
        "cluster_sizes": dict(selection.vote_counts),
        "confidence": selection.confidence,
        "tie_break": selection.metadata.get("tiebreak"),
        "scored_candidates": None,
        "latency_s": 0.0,
    }
    return _baseline_record(
        run_id=ctx.run_id,
        item=item,
        gen=GenerationOutput(text=""),  # placeholder; chain_entries supplied directly
        answer=winner,
        gold=gold,
        method_label="self_consistency",
        model_name=model_name,
        temperature=temperature,
        seed=seed,
        total_latency_s=total_latency_s,
        verifier_override=verifier,
        descriptions_entries=[],
        chain_entries=chains_meta,
    )


def run_visuothink_q(model, processor, item, ctx, model_name, seed, cfg, n=8):
    """Self-consistency over N VisuoThink reasoning chains (image re-examination)."""
    from src.utils import Candidate
    from src.verify import majority_vote
    from src.visuothink import visuothink_sc

    temperature, max_new_tokens = _gen_params(cfg, default_temp=0.7, default_max_tokens=512)
    vt_cfg = cfg.get("visuothink", {})
    max_steps = int(vt_cfg.get("max_steps", 6))
    re_examine_every = int(vt_cfg.get("re_examine_every", 2))
    max_verify_tokens = int(vt_cfg.get("max_verify_tokens", 128))

    qid = item.get("sample_id")
    gold = normalize_answer_key(item.get("answer_key", "")) if item.get("answer_key") else None
    span_item = {**item, "gold": gold}

    t0 = time.perf_counter()
    with ctx.question_span(span_item) as qspan:
        with ctx.stage_span("visuothink", n_chains=n, max_steps=max_steps):
            candidates, all_meta = visuothink_sc(
                model, processor, item["image"],
                n=n,
                max_steps=max_steps,
                re_examine_every=re_examine_every,
                temperature=temperature,
                max_step_tokens=max_new_tokens,
                max_verify_tokens=max_verify_tokens,
                seed=seed,
                ctx=ctx,
                model_name=model_name,
            )

        chains_meta = []
        for i, (c, meta) in enumerate(zip(candidates, all_meta)):
            n_verify = sum(1 for m in meta if m["type"] == "verify")
            n_correct = sum(1 for m in meta if m["type"] == "correct")
            chains_meta.append(chain_entry(
                desc_idx=-1, chain_idx=i, reasoning=c.reasoning,
                extracted_answer=c.answer,
                prompt_tokens=c.prompt_tokens,
                completion_tokens=c.completion_tokens,
                logprob_mean=c.logprob, latency_s=c.latency_s,
                model=model_name, temperature=temperature, seed=seed,
            ))

        with ctx.stage_span("verify", method="majority_vote"):
            with ctx.stage_span("verifier.majority_vote"):
                selection = majority_vote(candidates)

        winner = selection.answer

        try:
            q_prompt_toks = sum(int(c.prompt_tokens or 0) for c in candidates)
            q_completion_toks = sum(int(c.completion_tokens or 0) for c in candidates)
            qspan.set_attribute("predicted", winner or "")
            qspan.set_attribute("correct", bool(winner == gold) if gold else False)
            qspan.set_attribute("n_chains", len(candidates))
            if gold is not None:
                qspan.set_attribute("gold", gold)
        except Exception:
            pass
    total_latency_s = time.perf_counter() - t0

    verifier = {
        "method": "visuothink_sc",
        "selected_answer": winner,
        "cluster_sizes": dict(selection.vote_counts),
        "confidence": selection.confidence,
        "tie_break": selection.metadata.get("tiebreak"),
        "scored_candidates": None,
        "latency_s": 0.0,
    }
    return _baseline_record(
        run_id=ctx.run_id,
        item=item,
        gen=GenerationOutput(text=""),
        answer=winner,
        gold=gold,
        method_label="visuothink",
        model_name=model_name,
        temperature=temperature,
        seed=seed,
        total_latency_s=total_latency_s,
        verifier_override=verifier,
        descriptions_entries=[],
        chain_entries=chains_meta,
    )


# ---------------------------------------------------------------------------
# Experiment orchestration
# ---------------------------------------------------------------------------


# Dataset registry. `id_column` is the row field carrying the per-question
# identifier; SC-N=8 / DTR / baselines all read it via item["sample_id"], so
# the runner projects whatever the upstream calls it onto sample_id below.
# `has_gold=False` flags challenge test sets where answer_key is held out.
_DATASET_SPECS = {
    "examsv_validation": {
        "hf_id": "MBZUAI/EXAMS-V",
        "split": "validation",
        "id_column": "sample_id",
        "dataset_label": "exams_v",
        "split_label": "validation",
        "has_gold": True,
    },
    "imageclef_mcq_test": {
        "hf_id": "SU-FMI-AI/ImageCLEF-MR2026-MCQ-Visual",
        "split": "test",
        "id_column": "question_id",
        "dataset_label": "imageclef_mr2026_mcq_visual",
        "split_label": "test",
        "has_gold": False,
    },
}


def load_dataset_split(
    name: str,
    subset_size,
    *,
    stratify: bool = True,
    min_per_language: int = 10,
):
    """Load a registered dataset split + select indices.

    `subset_size=None` returns the full split (no sampling). Otherwise
    stratify by (language, subject) so small languages aren't under-sampled.
    `stratify=False` falls back to i.i.d. shuffle.

    Returns (dataset, indices, meta) where meta carries the dataset/split
    labels and the upstream id-column name; the per-question loop projects
    these onto each row dict so downstream record builders see a stable
    schema regardless of source.
    """
    import json

    from datasets import load_dataset

    if name not in _DATASET_SPECS:
        raise ValueError(
            f"Unknown evaluation.dataset={name!r}; "
            f"valid options: {sorted(_DATASET_SPECS)}"
        )
    spec = _DATASET_SPECS[name]
    logger.info("Loading {} split={} ({})...", spec["hf_id"], spec["split"], name)
    ds = load_dataset(spec["hf_id"])[spec["split"]]
    meta = {
        "dataset": spec["dataset_label"],
        "split": spec["split_label"],
        "id_column": spec["id_column"],
        "has_gold": spec["has_gold"],
        "name": name,
    }

    if subset_size is None:
        indices = list(range(ds.num_rows))
        logger.info(
            "Full {} split: {} questions (no sampling)",
            name, len(indices),
        )
        return ds, indices, meta

    if stratify:
        rows = [{"language": r["language"], "subject": r["subject"]} for r in ds]
        indices, allocation = stratified_indices(
            rows, subset_size=subset_size, seed=SEED, min_per_language=min_per_language,
        )
        logger.info(
            "Stratified subset of {}: {} questions across {} (lang, subject) strata (seed={})",
            name, len(indices), len(allocation), SEED,
        )
        Path("runs").mkdir(exist_ok=True)
        manifest = Path("runs") / f"subset_{_WALL_ID}.json"
        manifest.write_text(json.dumps({
            "dataset": name,
            "subset_size": len(indices),
            "seed": SEED,
            "min_per_language": min_per_language,
            "stratify": True,
            "indices": indices,
            "allocation": allocation,
        }, indent=2))
        logger.info("Subset manifest: {}", manifest)
    else:
        random.seed(SEED)
        all_idx = list(range(ds.num_rows))
        random.shuffle(all_idx)
        indices = all_idx[:subset_size]
        logger.info("I.I.D. subset of {}: {} questions (seed={})", name, len(indices), SEED)

    return ds, indices, meta


def load_dev_subset(subset_size, *, stratify: bool = True, min_per_language: int = 10):
    """Backwards-compatible wrapper: EXAMS-V validation only.

    New code should call load_dataset_split("examsv_validation", ...) directly,
    or set evaluation.dataset in the variant YAML to dispatch to a different
    split (e.g. "imageclef_mcq_test" for the challenge test set).
    """
    ds, indices, _meta = load_dataset_split(
        "examsv_validation", subset_size,
        stratify=stratify, min_per_language=min_per_language,
    )
    return ds, indices


def _make_variant_config(base: dict, variant_name: str, tags: list[str]) -> dict:
    """Build a per-variant config so `make_run_id` picks the right tags."""
    return {**base, "variant": variant_name, "tags": tags}


def _base_model_config(model_name: str, quantization: str) -> dict:
    return {
        "model": {"name": model_name, "quantization": quantization},
        "evaluation": {"seed": SEED},
    }


def run_variant(
    *,
    variant_name: str,
    pipeline_label: str,
    config: dict,
    tags: list[str],
    per_question,
    model,
    processor,
    dataset,
    indices,
    base_out: Path,
    push_to_hub: bool,
    dataset_meta: dict | None = None,
    resume_dir: Path | None = None,
):
    if resume_dir:
        out_dir = resume_dir
        run_id = resume_dir.name
    else:
        run_id = make_run_id(config, pipeline=pipeline_label, extra=variant_name)
        out_dir = base_out / run_id
    logger.info("=== {} :: {} ({} questions) ===", pipeline_label, variant_name, len(indices))

    meta = dataset_meta or {}
    id_column = meta.get("id_column", "sample_id")
    has_gold = meta.get("has_gold", True)

    done_ids: set[str] = set()
    if resume_dir:
        jsonl_path = out_dir / "candidates.jsonl"
        if jsonl_path.exists():
            lines = jsonl_path.read_text().splitlines()
            for i, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    # Tolerate a truncated final line (job killed mid-write).
                    # Anything earlier is a real corruption, so raise.
                    if i == len(lines) - 1:
                        logger.warning(
                            "Discarding malformed final line of {} (likely "
                            "killed mid-write); record will be re-run.",
                            jsonl_path,
                        )
                        continue
                    raise
                qid = rec.get("question_id")
                if qid:
                    done_ids.add(qid)
            logger.info("Resuming: {} questions already processed, {} remaining",
                        len(done_ids), len(indices) - len(done_ids))

    correct = 0
    total = 0
    with run_context(
        run_id, config, out_dir, tags=tags,
        group=f"{pipeline_label}-{config.get('verify', {}).get('method', 'majvote')}",
        variant_name=variant_name,
        push_to_hub=push_to_hub,
    ) as ctx:
        for idx in tqdm(indices, desc=variant_name):
            item = dict(dataset[idx])
            # Project upstream id-column to sample_id so downstream record code
            # (records.py, logger.py) finds it under the canonical name.
            if "sample_id" not in item:
                item["sample_id"] = item.get(id_column) or item.get("id") or item.get("question_id")
            if item.get("sample_id") in done_ids:
                continue
            # Stamp dataset/split labels so the JSONL records aren't mislabelled
            # as exams_v/validation when running on the challenge test set.
            if "dataset" in meta:
                item["_dataset"] = meta["dataset"]
            if "split" in meta:
                item["_split"] = meta["split"]
            try:
                record = per_question(model, processor, item, ctx)
                ctx.write_question(record)
                total += 1
                if record.get("correct"):
                    correct += 1
                if has_gold:
                    logger.info(
                        "[{}] gold={} pred={} {}",
                        record["question_id"], record["gold"],
                        record["verifier"]["selected_answer"],
                        "ok" if record.get("correct") else "WRONG",
                    )
                else:
                    # No gold available (challenge test set), accuracy log line
                    # would be misleading; just log the prediction.
                    logger.info(
                        "[{}] pred={} (no gold)",
                        record["question_id"],
                        record["verifier"]["selected_answer"],
                    )
            except Exception:
                logger.exception("Failed on {}", item.get("sample_id"))
        if has_gold:
            acc = correct / total if total else 0.0
            logger.info("=== {}: {:.1f}% ({}/{}) ===", variant_name, acc * 100, correct, total)
        else:
            logger.info("=== {}: {} predictions written (no gold) ===", variant_name, total)


def _load_variant_config(path: Path) -> dict:
    """Load a variant YAML and validate required keys."""
    import yaml
    cfg = yaml.safe_load(path.read_text())
    for required in ("variant", "runner", "model", "evaluation"):
        if required not in cfg:
            raise ValueError(f"{path} missing required key: {required}")
    cfg.setdefault("pipeline", "baseline" if cfg["runner"] != "dtr" else "msa")
    cfg.setdefault("tags", ["phase1"])
    return cfg


def _build_runner(cfg: dict, model_name: str):
    """Return a per-question callable dispatched by cfg['runner'].

    The cfg is captured in the closure so baseline runners can read
    `cfg.generation.temperature / .max_new_tokens` instead of using
    hardcoded defaults, keeping the YAML the single source of truth.
    """
    runner = cfg["runner"]
    if runner == "zero_shot":
        def _inner_zs(m, p, item, ctx):
            return run_zero_shot_q(m, p, item, ctx, model_name, SEED, cfg)
        return _inner_zs
    if runner == "cot":
        def _inner_cot(m, p, item, ctx):
            return run_cot_q(m, p, item, ctx, model_name, SEED, cfg)
        return _inner_cot
    if runner == "self_consistency":
        n = int(cfg.get("self_consistency", {}).get("n", 8))
        def _inner_sc(m, p, item, ctx):
            return run_self_consistency_q(m, p, item, ctx, model_name, SEED, cfg, n=n)
        return _inner_sc
    if runner == "visuothink":
        n = int(cfg.get("visuothink", {}).get("n", 8))
        def _inner_vt(m, p, item, ctx):
            return run_visuothink_q(m, p, item, ctx, model_name, SEED, cfg, n=n)
        return _inner_vt
    if runner == "dtr":
        search_resources = _maybe_load_search_resources(cfg)
        reason_resources = _maybe_load_reason_model(cfg)
        def _inner_dtr(m, p, item, ctx):
            return run_pipeline(
                m, p, item, cfg, ctx,
                search_resources=search_resources,
                reason_model=reason_resources[0] if reason_resources else None,
                reason_processor=reason_resources[1] if reason_resources else None,
            )
        return _inner_dtr
    raise ValueError(f"Unknown runner: {runner}")


def _maybe_load_search_resources(cfg: dict) -> dict | None:
    """Load PRM model if the config needs it for search or verification."""
    needs_prm = False

    # PRM-BAS search with qwen_vl_prm step scorer
    search_cfg = cfg.get("search", {})
    if search_cfg.get("method") == "prm_bas":
        scorer_method = search_cfg.get("step_scorer", {}).get("method", "generative")
        if scorer_method == "qwen_vl_prm":
            needs_prm = True

    # PRM-based verification methods
    verify_method = cfg.get("verify", {}).get("method", "majority_vote")
    if verify_method in ("qwen_vl_prm", "qwen_vl_prm_dtr", "prm_decomposed"):
        needs_prm = True

    if not needs_prm:
        return None
    from src.verify import load_qwen_vl_prm
    # NOTE: verify.prm.{backend,gpu_memory_utilization,enforce_eager,max_model_len}
    # config keys are reserved for a future vLLM PRM backend but not honored
    # today. src.verify.load_qwen_vl_prm only accepts `model_id`. The HF
    # backend is the validated path for PRM loading.
    logger.info("Loading Qwen-VL-PRM for search/verification (HF backend)...")
    prm_model, prm_processor, prm_tokenizer = load_qwen_vl_prm()
    return {
        "prm_model": prm_model,
        "prm_processor": prm_processor,
        "prm_tokenizer": prm_tokenizer,
    }


def _maybe_load_reason_model(cfg: dict) -> tuple | None:
    """Load a separate reasoning model if reason.model is specified in config.

    Enables hybrid DTR: VLM for describe (perception), dedicated text LLM for
    reasoning. The reasoning model is loaded co-resident on the same GPU, as
    both models fit comfortably since the pipeline is sequential (only one
    model does forward passes at a time; vLLM KV cache is not shared).

    Returns (model, processor) tuple or None if no separate model configured.
    """
    reason_cfg = cfg.get("reason", {})
    reason_model_cfg = reason_cfg.get("model")
    if not reason_model_cfg:
        return None

    reason_name = reason_model_cfg.get("name")
    if not reason_name:
        return None

    # Don't double-load if it's the same model as the primary
    primary_name = cfg.get("model", {}).get("name", "")
    if reason_name == primary_name:
        logger.info("reason.model == primary model ({}); skipping separate load", reason_name)
        return None

    reason_backend = reason_model_cfg.get("backend", "hf")
    reason_quant = reason_model_cfg.get("quantization", "none")
    reason_dtype = reason_model_cfg.get("dtype", "bfloat16")
    reason_max_len = int(reason_model_cfg.get("max_model_len", 16384))
    reason_text_only = reason_model_cfg.get("text_only", False)

    logger.info(
        "Loading reasoning model: {} (backend={}, quant={}, dtype={}, max_len={}, text_only={})",
        reason_name, reason_backend, reason_quant, reason_dtype, reason_max_len,
        reason_text_only,
    )

    load_kwargs = {
        "quantization": reason_quant,
        "dtype": reason_dtype,
        "backend": reason_backend,
    }
    if reason_backend == "vllm":
        load_kwargs["max_model_len"] = reason_max_len
        load_kwargs["gpu_memory_utilization"] = float(
            reason_model_cfg.get("gpu_memory_utilization", 0.45)
        )
        load_kwargs["text_only"] = reason_text_only
        if reason_model_cfg.get("enforce_eager"):
            load_kwargs["enforce_eager"] = True
        if reason_model_cfg.get("num_gpu_blocks_override"):
            load_kwargs["num_gpu_blocks_override"] = int(
                reason_model_cfg["num_gpu_blocks_override"]
            )

    r_model, r_processor = load_model(reason_name, **load_kwargs)
    return (r_model, r_processor)


def main():
    parser = argparse.ArgumentParser(description="Experiment runner (config-driven)")
    parser.add_argument(
        "--config", type=Path,
        help="Path to a variant YAML (e.g. configs/phase1/cot.yaml).",
    )
    parser.add_argument(
        "--config-dir", type=Path,
        help="Run every *.yaml in this directory (sharing one model load).",
    )
    parser.add_argument(
        "--subset-override", type=int, default=None,
        help="Override evaluation.subset_size for quick runs (e.g. 5).",
    )
    parser.add_argument(
        "--resume-dir", type=Path,
        help="Resume into an existing run directory, skipping already-processed questions.",
    )
    parser.add_argument(
        "--start-idx", type=int, default=None,
        help="Slice the resolved indices list at [start:end]. Use with SLURM "
             "array jobs to shard a long val-full run across tasks.",
    )
    parser.add_argument(
        "--end-idx", type=int, default=None,
        help="End of the index slice (exclusive). See --start-idx.",
    )
    parser.add_argument(
        "--variant-suffix", type=str, default=None,
        help="Appended to each variant name to disambiguate run dirs across "
             "SLURM array shards (e.g. 'shard0'). Run IDs are otherwise "
             "minute-precision and collide for tasks starting concurrently.",
    )
    parser.add_argument("--push-to-hub", action="store_true",
                        help="Push each run's JSONL to HF_RUNS_REPO as a new branch.")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    if not args.config and not args.config_dir:
        parser.error("Pass --config <path> or --config-dir <dir>")
    if args.resume_dir and args.config_dir:
        parser.error("--resume-dir is per-variant; use --config <path>, not --config-dir.")
    if args.resume_dir and not args.resume_dir.is_dir():
        parser.error(f"--resume-dir path does not exist or is not a directory: {args.resume_dir}")

    setup_logging(args.verbose)
    base_out = Path("runs")
    base_out.mkdir(exist_ok=True)
    logger.info("Results will be saved under {}/<run_id>/", base_out)

    # Collect configs
    if args.config_dir:
        config_paths = sorted(p for p in args.config_dir.glob("*.yaml"))
    else:
        config_paths = [args.config]
    if not config_paths:
        parser.error("No configs found.")

    configs = [_load_variant_config(p) for p in config_paths]
    for cfg, p in zip(configs, config_paths):
        cfg["_source_path"] = str(p)
        if args.subset_override is not None:
            cfg["evaluation"]["subset_size"] = args.subset_override
        if args.variant_suffix:
            cfg["variant"] = f"{cfg['variant']}-{args.variant_suffix}"

    # Sanity: all configs must agree on model+backend (one load shared across variants)
    model_names = {c["model"]["name"] for c in configs}
    backends = {c["model"].get("backend", "vllm") for c in configs}
    if len(model_names) > 1:
        parser.error(f"All configs must use the same model; got {model_names}")
    if len(backends) > 1:
        parser.error(f"All configs must use the same backend; got {backends}")
    model_name = configs[0]["model"]["name"]
    backend = configs[0]["model"].get("backend", "vllm")
    quantization = configs[0]["model"].get("quantization", "none")
    dtype = configs[0]["model"].get("dtype", "bfloat16")
    max_model_len = int(configs[0]["model"].get("max_model_len", 8192))

    # If any config specifies a separate reasoning model, reduce the primary
    # VLM's gpu_memory_utilization to leave room for co-residency.
    has_reason_model = any(
        c.get("reason", {}).get("model", {}).get("name")
        and c["reason"]["model"]["name"] != model_name
        for c in configs
    )
    default_gpu_util = 0.45 if has_reason_model else 0.9

    logger.info(
        "Loading model: {} (backend={}, quantization={}, dtype={}, max_model_len={})",
        model_name, backend, quantization, dtype, max_model_len,
    )
    load_kwargs = {"quantization": quantization, "dtype": dtype, "backend": backend}
    if backend == "vllm":
        load_kwargs["max_model_len"] = max_model_len
        gpu_mem_util = float(configs[0]["model"].get("gpu_memory_utilization", default_gpu_util))
        load_kwargs["gpu_memory_utilization"] = gpu_mem_util
    model, processor = load_model(model_name, **load_kwargs)

    # Subsets are variant-local: different subset_size, stratify, or dataset
    # per config are all allowed; identical specs share one load across variants.
    subset_cache: dict[tuple, tuple] = {}
    for cfg in configs:
        eval_cfg = cfg["evaluation"]
        dataset_name = eval_cfg.get("dataset", "examsv_validation")
        key = (
            dataset_name,
            eval_cfg["subset_size"],
            eval_cfg.get("stratify", True),
            eval_cfg.get("min_per_language", 10),
            eval_cfg.get("seed", SEED),
        )
        if key not in subset_cache:
            subset_cache[key] = load_dataset_split(
                dataset_name,
                eval_cfg["subset_size"],
                stratify=eval_cfg.get("stratify", True),
                min_per_language=eval_cfg.get("min_per_language", 10),
            )
        dataset, dev_indices, dataset_meta = subset_cache[key]

        if args.start_idx is not None or args.end_idx is not None:
            start = args.start_idx or 0
            end = args.end_idx if args.end_idx is not None else len(dev_indices)
            sliced = dev_indices[start:end]
            logger.info(
                "Sharding indices: [{}:{}] -> {} of {} questions",
                start, end, len(sliced), len(dev_indices),
            )
            dev_indices = sliced

        run_variant(
            variant_name=cfg["variant"],
            pipeline_label=cfg["pipeline"],
            config=cfg,
            tags=cfg["tags"],
            per_question=_build_runner(cfg, model_name),
            model=model, processor=processor,
            dataset=dataset, indices=dev_indices,
            base_out=base_out, push_to_hub=args.push_to_hub,
            dataset_meta=dataset_meta,
            resume_dir=args.resume_dir,
        )


if __name__ == "__main__":
    main()
