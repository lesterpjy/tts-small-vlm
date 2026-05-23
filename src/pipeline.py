"""Pipeline orchestrator: describe -> reason -> verify -> select.

Ties together all three stages and provides entry points for
single-question inference and full-dataset experiments.

When called through `RunContext`, `run_pipeline` emits the full
question/describe/reason/verify span tree and returns a JSONL
record ready for `ctx.write_question(record)`.
"""

from __future__ import annotations

import random
import time
from pathlib import Path
from typing import TYPE_CHECKING

from datasets import load_dataset
from loguru import logger
from tqdm.auto import tqdm

from src.backend import load_model
from src.describe import describe
from src.reason import reason_all
from src.utils import (
    PipelineResult,
    load_config,
    normalize_answer_key,
)
from src.utils.logger import make_run_id, run_context
from src.utils.records import (
    build_question_record,
    chain_entries_from_candidates,
    description_entry,
    verifier_entry,
)
from src.verify import select_answer

if TYPE_CHECKING:
    from src.utils.logger import RunContext


def _run_prm_bas_search(
    model, processor, image, descriptions,
    *, search_cfg, reason_cfg, seed, search_resources,
):
    """Run PRM-BAS beam search per description, returning flat candidate list."""
    from src.backend import GenerationOutput
    from src.search import (
        beams_to_candidates,
        make_generative_step_scorer,
        make_prm_step_scorer,
        prm_bas,
    )

    bas_cfg = search_cfg.get("prm_bas", {})
    scorer_method = search_cfg.get("step_scorer", {}).get("method", "generative")
    cot_style = reason_cfg.get("cot_style", "mmmu")

    all_candidates = []
    for desc_id, desc in enumerate(descriptions):
        desc_text = desc.text if isinstance(desc, GenerationOutput) else desc

        if scorer_method == "qwen_vl_prm" and search_resources:
            step_scorer = make_prm_step_scorer(
                search_resources["prm_model"],
                search_resources["prm_processor"],
                search_resources["prm_tokenizer"],
                image=image,
                question=desc_text,
            )
        else:
            step_scorer = make_generative_step_scorer(model, processor, image)

        beams, meta = prm_bas(
            model, processor, desc_text, step_scorer,
            image=None,
            B0=bas_cfg.get("B0", 8),
            B=bas_cfg.get("B", 4),
            anneal_tau=bas_cfg.get("anneal_tau", 0.05),
            max_depth=bas_cfg.get("max_depth", 10),
            stop_marker=bas_cfg.get("stop_marker", "answer:"),
            temperature=reason_cfg.get("temperature", 0.7),
            max_step_tokens=bas_cfg.get("max_step_tokens", 256),
            use_cot=reason_cfg.get("use_cot", True),
            cot_style=cot_style,
            seed=seed,
        )
        all_candidates.extend(beams_to_candidates(beams, desc_id, desc_text))

    logger.info(
        "PRM-BAS: {} candidates from {} descriptions",
        len(all_candidates), len(descriptions),
    )
    return all_candidates


def run_pipeline(
    model,
    processor,
    item: dict,
    config: dict,
    ctx: "RunContext",
    *,
    search_resources: dict | None = None,
    reason_model=None,
    reason_processor=None,
) -> dict:
    """Run the DTR pipeline on a single question and return a JSONL record.

    Stages:
        1. Describe: generate N image descriptions (VLM)
        2. Reason: generate M reasoning chains per description (text-only;
           uses reason_model if provided, otherwise falls back to the VLM)
        3. Verify & Select: pick the best answer

    `item` is the full HF dataset row. Opens the `question` span on `ctx`
    and returns the record. Caller is responsible for calling
    `ctx.write_question(record)`.
    """
    cfg = config or {}
    desc_cfg = cfg.get("describe", {})
    reason_cfg = cfg.get("reason", {})
    verify_cfg = cfg.get("verify", {})
    model_cfg = cfg.get("model", {})
    model_name = model_cfg.get("name", "")
    reason_model_name = reason_cfg.get("model", {}).get("name", "") or model_name
    seed = cfg.get("evaluation", {}).get("seed", 0)

    r_model = reason_model if reason_model is not None else model
    r_processor = reason_processor if reason_processor is not None else processor

    image = item.get("image")
    language = item.get("language") or "English"
    gold = normalize_answer_key(item.get("answer_key", "")) if item.get("answer_key") else None
    item_for_span = {**item, "gold": gold}

    t0 = time.perf_counter()
    with ctx.question_span(item_for_span) as qspan:
        # --- Stage 1: Describe ---
        desc_temp = desc_cfg.get("temperature", 0.7)
        descriptions = describe(
            model, processor, image, language=language,
            n=desc_cfg.get("n_samples", 4),
            temperature=desc_temp,
            max_tokens=desc_cfg.get("max_tokens", 1024),
            ctx=ctx, model_name=model_name, seed=seed,
        )
        description_entries = [
            description_entry(
                idx=i,
                text=d.text,
                prompt_tokens=d.prompt_tokens,
                completion_tokens=d.completion_tokens,
                logprob_mean=d.logprob,
                latency_s=d.latency_s,
                model=model_name,
                temperature=desc_temp,
                seed=seed,
            )
            for i, d in enumerate(descriptions)
        ]

        # --- Stage 2: Reason ---
        search_cfg = cfg.get("search", {})
        search_method = search_cfg.get("method", "flat_dtr")
        reason_temp = reason_cfg.get("temperature", 0.7)

        if search_method == "prm_bas":
            candidates = _run_prm_bas_search(
                model, processor, image, descriptions,
                search_cfg=search_cfg, reason_cfg=reason_cfg,
                seed=seed, search_resources=search_resources,
            )
        else:
            candidates = reason_all(
                r_model, r_processor, descriptions,
                m=reason_cfg.get("m_samples", 4),
                temperature=reason_temp,
                max_tokens=reason_cfg.get("max_tokens", 512),
                use_cot=reason_cfg.get("use_cot", True),
                cot_style=reason_cfg.get("cot_style", "default"),
                ctx=ctx, model_name=reason_model_name, seed=seed,
            )

        chain_entries = chain_entries_from_candidates(
            candidates,
            model=reason_model_name,
            temperature=reason_temp,
            seed=seed,
        )

        # --- Stage 3: Verify & Select ---
        method = verify_cfg.get("method", "majority_vote")
        verify_t0 = time.perf_counter()

        # PRM-based verification methods need the PRM model, not the VLM.
        # Thread search_resources through to the verify config.
        method_cfg = dict(verify_cfg.get(method, {}))
        if method in ("qwen_vl_prm", "qwen_vl_prm_dtr", "prm_decomposed"):
            if search_resources:
                method_cfg.setdefault("tokenizer", search_resources.get("prm_tokenizer"))

        selection = select_answer(
            candidates,
            method=method,
            model=search_resources["prm_model"] if search_resources and method in (
                "qwen_vl_prm", "qwen_vl_prm_dtr", "prm_decomposed",
            ) else model,
            processor=search_resources["prm_processor"] if search_resources and method in (
                "qwen_vl_prm", "qwen_vl_prm_dtr", "prm_decomposed",
            ) else processor,
            image=image,
            verify_config=method_cfg,
            ctx=ctx,
        )
        verify_latency = time.perf_counter() - verify_t0
        verifier_dict = verifier_entry(selection=selection, latency_s=verify_latency)

        # --- Totals and per-question summary on the root span ---
        total_vlm_calls = len(descriptions) + len(candidates)
        total_latency_s = time.perf_counter() - t0
        correct = (
            (selection.answer == gold) if (selection and gold is not None) else None
        )
        try:
            q_prompt_toks = (
                sum(int(c.prompt_tokens or 0) for c in candidates)
                + sum(int(d.prompt_tokens or 0) for d in descriptions)
            )
            q_completion_toks = (
                sum(int(c.completion_tokens or 0) for c in candidates)
                + sum(int(d.completion_tokens or 0) for d in descriptions)
            )
            qspan.set_attribute("predicted", selection.answer or "")
            qspan.set_attribute("correct", bool(correct) if correct is not None else False)
            qspan.set_attribute("total_vlm_calls", total_vlm_calls)
            qspan.set_attribute("total_latency_s", total_latency_s)
            qspan.set_attribute("total_prompt_tokens", q_prompt_toks)
            qspan.set_attribute("total_completion_tokens", q_completion_toks)
            qspan.set_attribute("total_tokens", q_prompt_toks + q_completion_toks)
            qspan.set_attribute("n_chains", len(candidates))
            qspan.set_attribute("n_descriptions", len(descriptions))
            if gold is not None:
                qspan.set_attribute("gold", gold)
        except Exception:
            pass

    return build_question_record(
        run_id=ctx.run_id,
        item=item_for_span,
        descriptions=description_entries,
        chains=chain_entries,
        verifier=verifier_dict,
        correct=correct,
        total_vlm_calls=total_vlm_calls,
        total_latency_s=total_latency_s,
    )


def run_experiment(
    config_path: str | Path,
    output_dir: str | Path | None = None,
) -> list[dict]:
    """Run the DTR pipeline on the EXAMS-V dev subset under a RunContext.

    Loads model via HF Transformers, opens a `RunContext`, runs the pipeline
    on each question, and appends one JSONL record per question. Returns the
    list of records (also available on disk at `runs/{run_id}/candidates.jsonl`).
    """
    config = load_config(config_path)
    eval_cfg = config.get("evaluation", {})
    model_cfg = config.get("model", {})

    subset_size = eval_cfg.get("subset_size", 200)
    seed = eval_cfg.get("seed", 42)
    base_dir = Path(output_dir or eval_cfg.get("output_dir", "runs"))

    random.seed(seed)

    model, processor = load_model(
        model_cfg["name"],
        quantization=model_cfg.get("quantization", "4bit"),
        dtype=model_cfg.get("dtype", "bfloat16"),
    )

    logger.info("Loading EXAMS-V validation set...")
    examsv = load_dataset("MBZUAI/EXAMS-V")
    examsv_val = examsv["validation"]

    indices = list(range(examsv_val.num_rows))
    random.shuffle(indices)
    dev_indices = indices[:subset_size]
    logger.info("Running on {} questions (seed={})", len(dev_indices), seed)

    run_id = make_run_id(config)
    out_dir = base_dir / run_id
    records: list[dict] = []

    with run_context(run_id, config, out_dir, tags=config.get("tags", [])) as ctx:
        for idx in tqdm(dev_indices, desc="DTR pipeline"):
            item = examsv_val[idx]
            try:
                record = run_pipeline(model, processor, dict(item), config, ctx)
                ctx.write_question(record)
                records.append(record)
                logger.info(
                    "[{}] gold={} pred={} correct={}",
                    record["question_id"],
                    record["gold"],
                    record["verifier"]["selected_answer"],
                    record["correct"],
                )
            except Exception:
                logger.exception("Failed on question {}", item.get("sample_id"))

    return records
