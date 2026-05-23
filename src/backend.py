"""HuggingFace Transformers backend for model loading and generation.

Model class is resolved via `AutoModelForImageTextToText` (falling back to
`AutoModelForCausalLM`) so the same backend handles Qwen2.5-VL-7B (the
control policy for S1-S4) and Qwen3.5-4B (hybrid Gated DeltaNet +
sparse MoE with integrated vision encoder). `trust_remote_code=True`
because Qwen3.5 requires custom modeling code from the HF Hub.

Image handling intentionally avoids `qwen_vl_utils` so the backend is not
tied to the Qwen2.5-VL message format. Images are extracted from the
message list and passed directly to the processor.
"""

import time
from dataclasses import dataclass

from loguru import logger
from PIL import Image

# torch / transformers are imported lazily inside load_model / generate so
# this module (and downstream pipeline modules that import `GenerationOutput`)
# can be loaded for tests without torch installed (e.g. Intel-Mac local dev).


@dataclass
class GenerationOutput:
    """Single generation result with per-call metrics.

    `logprob` is the mean token log-probability (feeds `logprob_mean` in the
    JSONL schema). Token counts and latency are captured for the JSONL record.
    """

    text: str
    logprob: float | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0


def load_model(
    model_name: str,
    quantization: str = "none",
    dtype: str = "bfloat16",
    backend: str = "vllm",
    **backend_kwargs,
) -> tuple:
    """Load a multimodal causal LM + processor.

    Args:
        model_name: HuggingFace model ID (e.g. "Qwen/Qwen2.5-VL-7B-Instruct").
        quantization: "4bit", "8bit", or "none" (HF backend only). Qwen3.5's
            hybrid architecture may not be fully compatible with BitsAndBytes.
        dtype: Compute dtype (e.g. "bfloat16").
        backend: "vllm" (default) for the fast inference engine that matches
            the official ImageCLEF 2026 baseline stack, or "hf" for direct
            HuggingFace Transformers (what we used in earlier smoke tests).
        backend_kwargs: forwarded to the chosen backend's load_model,
            e.g. max_model_len for vLLM.

    Returns:
        (model, processor) tuple. `model` is a vLLM LLM or HF model depending
        on the backend. Treat it as opaque and pass straight to `generate()`.
    """
    if backend == "vllm":
        from src.backend_vllm import load_model as load_vllm
        return load_vllm(model_name, quantization=quantization, dtype=dtype, **backend_kwargs)
    if backend != "hf":
        raise ValueError(f"Unknown backend: {backend!r} (expected 'vllm' or 'hf')")

    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig

    compute_dtype = getattr(torch, dtype, torch.bfloat16)

    load_kwargs: dict = {
        "device_map": "auto",
        "trust_remote_code": True,
    }

    if quantization == "4bit":
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
        )
    elif quantization == "8bit":
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    else:
        load_kwargs["dtype"] = compute_dtype  # torch_dtype deprecated in transformers

    # AutoModelForImageTextToText is the correct autoclass for multimodal
    # (vision+text) causal LMs: its generate() accepts pixel_values /
    # image_grid_thw. AutoModelForCausalLM would pick the text-only variant
    # of Qwen3.5 and reject vision kwargs. We fall back to AutoModelForCausalLM
    # only if the model has no image-text-to-text mapping registered.
    try:
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(model_name, **load_kwargs)
        logger.info("Loaded {} via AutoModelForImageTextToText", model_name)
    except (ValueError, KeyError) as exc:
        logger.warning(
            "AutoModelForImageTextToText could not resolve {}; falling back to "
            "AutoModelForCausalLM (text-only). Cause: {}",
            model_name, exc,
        )
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    logger.info("quantization={}, dtype={}", quantization, dtype)
    return model, processor


def _extract_images(messages: list[dict]) -> list[Image.Image]:
    """Pull image objects out of chat messages in order."""
    images: list[Image.Image] = []
    for msg in messages:
        content = msg.get("content", [])
        if isinstance(content, str):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image":
                img = part.get("image")
                if img is not None:
                    images.append(img)
    return images


def _truncate_at_stop(text: str, stop_strings: list[str]) -> str:
    """Truncate at the earliest occurrence of any stop string."""
    earliest = len(text)
    for s in stop_strings:
        idx = text.find(s)
        if idx != -1 and idx < earliest:
            earliest = idx
    return text[:earliest]


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
    """Generate a single completion.

    Dispatches to vLLM when `model` is a vLLM `LLM` instance (duck-typed
    on the presence of a vLLM-specific method), or to the HF path
    otherwise. This keeps pipeline/describe/reason/verify agnostic to
    which engine is running.

    Extra kwargs that only affect the vLLM path:
        guided_choice: restrict output to one of these strings (used for
            constrained MCQ baselines). Silently ignored under HF.
        seed: per-call RNG seed. Silently ignored under HF.
    """
    # Duck-type detect vLLM LLM (HF models don't have .generate accepting
    # list-of-dicts input). `llm_engine` is the public attribute exposed
    # by vllm.LLM and not present on HF modules.
    if hasattr(model, "llm_engine"):
        from src.backend_vllm import generate as vllm_generate
        return vllm_generate(
            model, processor, prompt, image=image,
            temperature=temperature, max_new_tokens=max_new_tokens,
            guided_choice=guided_choice, seed=seed, stop=stop,
        )

    import torch

    if image is not None:
        content = [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]
    else:
        # Plain string content for text-only models (e.g. DeepSeek-R1-Distill)
        # whose chat templates don't support the multi-modal list format.
        content = prompt

    messages = [{"role": "user", "content": content}]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )

    images = _extract_images(messages)
    inputs = processor(
        text=[text],
        images=images or None,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    gen_kwargs: dict = dict(
        max_new_tokens=max_new_tokens,
        output_scores=True,
        return_dict_in_generate=True,
    )
    if temperature > 0:
        gen_kwargs.update(do_sample=True, temperature=temperature, top_p=0.95)
    else:
        gen_kwargs.update(do_sample=False)

    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
    latency_s = time.perf_counter() - t0

    input_len = inputs["input_ids"].shape[1]
    generated_ids = outputs.sequences[0, input_len:]
    decoded = processor.decode(generated_ids, skip_special_tokens=True)

    logprob = _compute_mean_logprob(model, outputs, input_len)

    if stop:
        decoded = _truncate_at_stop(decoded, stop)

    return GenerationOutput(
        text=decoded,
        logprob=logprob,
        prompt_tokens=int(input_len),
        completion_tokens=int(generated_ids.shape[0]),
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
) -> list["GenerationOutput"]:
    """Batched N-sample generation from a shared prompt.

    On vLLM, one `LLM.generate` call with `SamplingParams(n=n)` pays prefill
    and image encode once; only decoding is N-wise. Expected wall speedup
    3-5x over N sequential `generate()` calls at N=4-8. See
    `backend_vllm.generate_n` for per-sample accounting semantics.

    On the HF backend, falls back to N sequential `generate()` calls
    (seeded by `seed`, `seed+1`, ... when seed is provided) since HF's
    `generate(num_return_sequences=n)` semantics differ from vLLM's per-
    sample seeding. No speedup, but keeps the call-site interface uniform.
    """
    if n < 1:
        raise ValueError(f"generate_n: n must be >= 1, got {n}")

    if hasattr(model, "llm_engine"):
        from src.backend_vllm import generate_n as vllm_generate_n
        return vllm_generate_n(
            model, processor, prompt, n, image=image,
            temperature=temperature, max_new_tokens=max_new_tokens,
            guided_choice=guided_choice, seed=seed, stop=stop,
        )

    outs: list[GenerationOutput] = []
    for i in range(n):
        sub_seed = None if seed is None else seed + i
        outs.append(generate(
            model, processor, prompt, image=image,
            temperature=temperature, max_new_tokens=max_new_tokens,
            guided_choice=guided_choice, seed=sub_seed, stop=stop,
        ))
    return outs


def _compute_mean_logprob(model, outputs, input_len: int) -> float | None:
    """Mean token log-probability from generation scores."""
    if not outputs.scores:
        return None
    try:
        transition_scores = model.compute_transition_scores(
            outputs.sequences, outputs.scores, normalize_logits=True,
        )
        scores = transition_scores[0]
        if len(scores) == 0:
            return None
        return scores.mean().item()
    except Exception:
        return None
