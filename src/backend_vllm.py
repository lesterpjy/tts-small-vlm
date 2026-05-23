"""vLLM backend - identical function signatures to src/backend.py so the
rest of the pipeline doesn't need to care which engine is running.

Why vLLM: the official ImageCLEF 2026 baselines (molmo, smolvlm, olmo) all
run on a vLLM OpenAI-compatible server and pass `guided_choice=["A","B",
"C","D","E"]` to force the output to exactly one letter. That eliminates
the extract_answer fragility entirely for constrained-MCQ baselines. vLLM
also wins for self-consistency (N=8 sampled chains in one batched call).

This module uses the in-process `LLM` class rather than the server. Same
semantics, no port orchestration, drop-in for our sbatch.
"""

from __future__ import annotations

import time
from typing import Any

from loguru import logger
from PIL import Image

from src.backend import GenerationOutput  # reuse dataclass for uniform records


def load_model(
    model_name: str,
    quantization: str = "none",
    dtype: str = "bfloat16",
    max_model_len: int = 16384,
    limit_mm_per_prompt: dict[str, int] | None = None,
    max_image_pixels: int = 1280 * 28 * 28,
    min_image_pixels: int = 4 * 28 * 28,
    gpu_memory_utilization: float = 0.9,
    text_only: bool = False,
    enforce_eager: bool = False,
    num_gpu_blocks_override: int | None = None,
) -> tuple[Any, Any]:
    """Load a model into a vLLM in-process engine.

    Supports both VLMs (Qwen2.5-VL, Qwen3.5) and text-only LLMs
    (DeepSeek-R1-Distill-Qwen-7B). Set ``text_only=True`` for models
    without vision support, where multimodal args are skipped.

    Args:
        model_name: HF model ID.
        quantization: "none" / "awq" / "gptq" / "fp8" etc.
        dtype: "bfloat16" (default) or "float16".
        max_model_len: context window cap.
        limit_mm_per_prompt: per-request multi-modal caps (VLM only).
        max_image_pixels / min_image_pixels: image resolution bounds (VLM only).
        gpu_memory_utilization: fraction of total GPU memory for this instance.
        text_only: skip all multimodal configuration (for text-only LLMs).
        enforce_eager: disable CUDA graph compilation (saves ~500 MB VRAM;
            use for co-resident models where memory is tight).
        num_gpu_blocks_override: force KV cache block count (bypasses vLLM's
            memory profiler; use for co-resident models).

    Returns:
        (llm, processor) where `llm` is a vLLM LLM instance and `processor`
        is the HF AutoProcessor/AutoTokenizer used for chat-template formatting.
    """
    from vllm import LLM

    llm_kwargs: dict[str, Any] = {
        "model": model_name,
        "trust_remote_code": True,
        "dtype": dtype,
        "max_model_len": max_model_len,
        "gpu_memory_utilization": gpu_memory_utilization,
    }
    if enforce_eager:
        llm_kwargs["enforce_eager"] = True
    if num_gpu_blocks_override is not None:
        llm_kwargs["num_gpu_blocks_override"] = num_gpu_blocks_override

    if not text_only:
        from transformers import AutoProcessor

        if limit_mm_per_prompt is None:
            limit_mm_per_prompt = {"image": 1}
        llm_kwargs["limit_mm_per_prompt"] = limit_mm_per_prompt
        llm_kwargs["mm_processor_kwargs"] = {
            "min_pixels": min_image_pixels,
            "max_pixels": max_image_pixels,
        }

        llm = LLM(**llm_kwargs)
        processor = AutoProcessor.from_pretrained(
            model_name, trust_remote_code=True,
            min_pixels=min_image_pixels, max_pixels=max_image_pixels,
        )
        logger.info(
            "vLLM: loaded {} (dtype={}, max_model_len={}, max_image_pixels={}, "
            "limit_mm_per_prompt={}, gpu_mem_util={})",
            model_name, dtype, max_model_len, max_image_pixels,
            limit_mm_per_prompt, gpu_memory_utilization,
        )
    else:
        from transformers import AutoTokenizer

        llm = LLM(**llm_kwargs)
        processor = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True,
        )
        logger.info(
            "vLLM: loaded {} (text-only, dtype={}, max_model_len={}, gpu_mem_util={})",
            model_name, dtype, max_model_len, gpu_memory_utilization,
        )
    return llm, processor


def _resolve_constraint_cls():
    """Resolve vLLM's structured-output constraint class across versions.

    vLLM 0.19 renamed GuidedDecodingParams to StructuredOutputsParams and
    the SamplingParams field guided_decoding to structured_outputs.
    """
    try:
        from vllm.sampling_params import StructuredOutputsParams as cls
        return cls, "structured_outputs"
    except ImportError:
        from vllm.sampling_params import GuidedDecodingParams as cls  # type: ignore
        return cls, "guided_decoding"


def _build_prompt_inputs(processor, prompt: str, image: Image.Image | None):
    """Chat-template render + vLLM-input dict builder (shared by generate /
    generate_n).

    Uses structured content (list of dicts) only when an image is present;
    plain string content otherwise, ensuring compatibility with both VLM
    processors and text-only tokenizers.
    """
    if image is not None:
        content = [{"type": "image"}, {"type": "text", "text": prompt}]
    else:
        content = prompt
    messages = [{"role": "user", "content": content}]

    chat_kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
    # Qwen3-family: disable native hidden reasoning so we get visible CoT.
    tok = getattr(processor, "tokenizer", None) or processor
    tok_name = getattr(tok, "name_or_path", "") or ""
    if "qwen3" in tok_name.lower():
        chat_kwargs["enable_thinking"] = False
    text = processor.apply_chat_template(messages, **chat_kwargs)

    inputs: dict[str, Any] = {"prompt": text}
    if image is not None:
        inputs["multi_modal_data"] = {"image": image}
    return inputs


def _build_sampling_params(
    *, temperature: float, max_new_tokens: int, n: int = 1,
    seed: int | None = None, guided_choice: list[str] | None = None,
    stop: list[str] | None = None,
):
    from vllm import SamplingParams
    sp_kwargs: dict[str, Any] = {
        "temperature": temperature,
        "max_tokens": max_new_tokens,
        "logprobs": 1,          # top-1 logprob per token for logprob_mean
        "n": n,
    }
    if temperature > 0:
        sp_kwargs["top_p"] = 0.95
    if seed is not None:
        sp_kwargs["seed"] = seed
    if guided_choice:
        cls, key = _resolve_constraint_cls()
        sp_kwargs[key] = cls(choice=list(guided_choice))
    if stop:
        sp_kwargs["stop"] = list(stop)
    return SamplingParams(**sp_kwargs)


def generate(
    model,
    processor,
    prompt: str,
    image: Image.Image | None = None,
    temperature: float = 0.0,
    max_new_tokens: int = 512,
    *,
    guided_choice: list[str] | None = None,
    seed: int | None = None,
    stop: list[str] | None = None,
) -> GenerationOutput:
    """Generate a single completion via vLLM.

    Extra kwargs beyond the HF `generate()` signature:
        guided_choice: restrict the output to exactly one string in this
            list (via vLLM's guided decoding). Used for the official MCQ
            baselines that constrain output to {A, B, C, D, E}.
        seed: per-call RNG seed for sampling reproducibility.
        stop: stop strings; generation halts at the first occurrence.
    """
    inputs = _build_prompt_inputs(processor, prompt, image)
    sampling_params = _build_sampling_params(
        temperature=temperature, max_new_tokens=max_new_tokens, n=1,
        seed=seed, guided_choice=guided_choice, stop=stop,
    )

    t0 = time.perf_counter()
    outputs = model.generate([inputs], sampling_params)
    latency_s = time.perf_counter() - t0

    request_output = outputs[0]
    completion = request_output.outputs[0]

    return GenerationOutput(
        text=completion.text,
        logprob=_mean_top1_logprob(completion),
        prompt_tokens=len(request_output.prompt_token_ids),
        completion_tokens=len(completion.token_ids),
        latency_s=latency_s,
    )


def generate_n(
    model,
    processor,
    prompt: str,
    n: int,
    image: Image.Image | None = None,
    temperature: float = 0.7,
    max_new_tokens: int = 512,
    *,
    guided_choice: list[str] | None = None,
    seed: int | None = None,
    stop: list[str] | None = None,
) -> list[GenerationOutput]:
    """Generate N completions from the same prompt in a single batched vLLM
    call. Image encode + prompt prefill are paid once; decoding is N-wise.

    Expected wall-clock speedup over N sequential `generate()` calls is
    roughly 3-5x for image-bearing prompts at N=4-8 (prefill dominates there).
    Used by Stage 1 describe (N captions), Stage 2 reason (M chains per
    description), and self-consistency (N CoT chains).

    Per-sample accounting:
        text, logprob, completion_tokens are genuinely per-sample.
        prompt_tokens is the same value for all samples (full prefill length);
            the batched call paid prefill once, but we report the logical prompt
            length each sample "saw" to keep per-sample metrics comparable
            with the unbatched path.
        latency_s is batched wall-clock divided evenly across N, so summing
            across samples reconstructs the true wall-clock (unlike assigning
            the full batch latency to every sample, which would N-fold count).

    Args:
        n: number of completions (n >= 1). At n=1 this is equivalent to
            `generate(...)` and incurs no extra overhead.
        temperature: must be > 0 when n > 1 to yield diverse samples (vLLM
            derives per-sample seeds from the base seed, but at temperature=0
            all samples collapse to the same greedy trajectory).
    """
    if n < 1:
        raise ValueError(f"generate_n: n must be >= 1, got {n}")
    if n > 1 and temperature <= 0.0:
        # At T=0 with n>1, vLLM returns N identical greedy samples. The batch
        # is wasted and any downstream diversity assumption (self-consistency,
        # DTR sampling, BoN) is silently violated. Fail loud; callers that want
        # diverse samples must pass temperature > 0.
        raise ValueError(
            f"generate_n: temperature={temperature:.3f} with n={n} yields "
            f"N identical greedy samples (batch wasted, diversity broken). "
            f"Set temperature > 0 or call generate() for n=1."
        )

    inputs = _build_prompt_inputs(processor, prompt, image)
    sampling_params = _build_sampling_params(
        temperature=temperature, max_new_tokens=max_new_tokens, n=n,
        seed=seed, guided_choice=guided_choice, stop=stop,
    )

    t0 = time.perf_counter()
    outputs = model.generate([inputs], sampling_params)
    latency_s = time.perf_counter() - t0

    request_output = outputs[0]
    prompt_toks = len(request_output.prompt_token_ids)
    # vLLM returns completions in the order they finished, not sample index.
    # Sort by `index` to keep deterministic per-seed mapping.
    completions = sorted(
        request_output.outputs, key=lambda c: getattr(c, "index", 0),
    )
    per_sample_latency = latency_s / max(len(completions), 1)

    return [
        GenerationOutput(
            text=c.text,
            logprob=_mean_top1_logprob(c),
            prompt_tokens=prompt_toks,
            completion_tokens=len(c.token_ids),
            latency_s=per_sample_latency,
        )
        for c in completions
    ]


def generate_two_phase(
    model,
    processor,
    prompt: str,
    image: Image.Image | None = None,
    temperature: float = 0.7,
    max_new_tokens: int = 2048,
    *,
    seed: int | None = None,
    answer_suffix: str = "\n\nThe answer is ",
    answer_choices: list[str] | None = None,
) -> GenerationOutput:
    """Two-phase constrained generation: free reasoning + forced answer letter.

    Solves catastrophic parse-fail on long-reasoning models (e.g. Qwen3.5-4B
    SC N=8, 45.9% fail on Chinese) where all failures are truncation at
    max_tokens with no answer letter produced.

    Phase 1: Generate reasoning freely (unconstrained) with the given prompt,
        temperature, and max_new_tokens.
    Phase 2: Append ``answer_suffix`` to the Phase 1 output, then call vLLM
        with ``guided_choice`` to force exactly one letter from
        ``answer_choices`` (default ["A", "B", "C", "D"]) with max_tokens=1.

    The returned GenerationOutput contains:
        text: full reasoning + suffix + forced answer letter (the complete
            generation a downstream parser would see).
        logprob: mean top-1 logprob from Phase 1 only (Phase 2 is a single
            forced token, so its logprob reflects constraint strength, not
            model confidence over reasoning).
        prompt_tokens: Phase 1 prompt tokens (Phase 2 prompt is the full
            Phase 1 context, but prefill is cached by vLLM's automatic
            prefix caching so the marginal cost is near-zero).
        completion_tokens: Phase 1 completion tokens + 1 (the forced letter).
        latency_s: wall-clock for both phases combined.

    Args:
        model: vLLM LLM instance (from load_model with backend="vllm").
        processor: HF AutoProcessor for chat template formatting.
        prompt: user prompt (will be wrapped in chat template).
        image: optional PIL image for vision-language input.
        temperature: sampling temperature for Phase 1 (Phase 2 uses greedy).
        max_new_tokens: max tokens for Phase 1 free reasoning.
        seed: RNG seed for Phase 1 sampling reproducibility.
        answer_suffix: text appended between reasoning and forced answer.
            Default ``"\\n\\nThe answer is "`` matches the MMMU prompt closer.
        answer_choices: valid answer letters for guided_choice constraint.
            Default ``["A", "B", "C", "D"]``.

    Returns:
        GenerationOutput with full text (reasoning + suffix + letter),
        Phase 1 logprob, combined token counts, and combined latency.
    """
    if answer_choices is None:
        answer_choices = ["A", "B", "C", "D"]

    t0 = time.perf_counter()

    # --- Phase 1: unconstrained reasoning ---
    inputs_p1 = _build_prompt_inputs(processor, prompt, image)
    sp_p1 = _build_sampling_params(
        temperature=temperature, max_new_tokens=max_new_tokens, n=1,
        seed=seed,
    )
    outputs_p1 = model.generate([inputs_p1], sp_p1)
    req_p1 = outputs_p1[0]
    comp_p1 = req_p1.outputs[0]
    reasoning_text = comp_p1.text
    p1_logprob = _mean_top1_logprob(comp_p1)
    p1_prompt_tokens = len(req_p1.prompt_token_ids)
    p1_completion_tokens = len(comp_p1.token_ids)

    # --- Phase 2: forced answer letter ---
    # Build a new prompt = original prompt + assistant reasoning + suffix,
    # then constrain to exactly one token from answer_choices.
    #
    # We reconstruct the full conversation so far as raw text (the Phase 1
    # prompt text + the model's reasoning output + the suffix) and pass it
    # as a continuation prompt. vLLM's automatic prefix caching means the
    # overlapping prefix (the entire Phase 1 prompt + reasoning) is not
    # re-computed.
    p2_prompt_text = inputs_p1["prompt"] + reasoning_text + answer_suffix
    inputs_p2: dict[str, Any] = {"prompt": p2_prompt_text}
    if image is not None:
        inputs_p2["multi_modal_data"] = {"image": image}

    sp_p2 = _build_sampling_params(
        temperature=0.0,  # greedy for the forced token
        max_new_tokens=1,
        n=1,
        guided_choice=answer_choices,
    )
    outputs_p2 = model.generate([inputs_p2], sp_p2)
    req_p2 = outputs_p2[0]
    comp_p2 = req_p2.outputs[0]
    forced_letter = comp_p2.text.strip()

    latency_s = time.perf_counter() - t0

    # Compose full text: reasoning + suffix + letter
    full_text = reasoning_text + answer_suffix + forced_letter

    return GenerationOutput(
        text=full_text,
        logprob=p1_logprob,
        prompt_tokens=p1_prompt_tokens,
        completion_tokens=p1_completion_tokens + 1,
        latency_s=latency_s,
    )


def _mean_top1_logprob(completion) -> float | None:
    """Mean of the top-1 token logprobs along the generated sequence."""
    lp_sequence = getattr(completion, "logprobs", None)
    if not lp_sequence:
        return None
    vals: list[float] = []
    for step in lp_sequence:
        if not step:
            continue
        # step is a dict {token_id: Logprob}; take the first entry
        first = next(iter(step.values()))
        lp = getattr(first, "logprob", None)
        if lp is not None:
            vals.append(float(lp))
    if not vals:
        return None
    return sum(vals) / len(vals)
