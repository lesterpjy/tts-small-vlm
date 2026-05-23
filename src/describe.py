"""Stage 1: N-sample image description generation.

Generates N diverse descriptions of an exam image, replacing MSA's
single Gemini 2.5 Flash description + Gemini 1.5 Pro refinement.

When called through `RunContext`, emits one parent `describe` span plus N
`description.sample` LLM spans with OpenInference input/output attributes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger
from PIL import Image

from src.backend import GenerationOutput, generate_n

if TYPE_CHECKING:
    from src.utils.logger import RunContext

DESCRIBE_PROMPT_TEMPLATE = (
    "Extract the complete question text and all answer options from this exam image. "
    "Provide a detailed description of every visual element (diagrams, graphs, tables, "
    "chemical structures, maps). Preserve mathematical notation, subscripts, and symbols. "
    "Normalize answer option labels to A, B, C, D, E. "
    "Output in {language}. Do not answer the question."
)


def build_describe_prompt(language: str) -> str:
    """Build the Stage 1 describe prompt for a given language."""
    return DESCRIBE_PROMPT_TEMPLATE.format(language=language)


def describe(
    model,
    processor,
    image: Image.Image,
    language: str = "English",
    n: int = 4,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    *,
    ctx: "RunContext | None" = None,
    model_name: str = "",
    seed: int = 0,
) -> list[GenerationOutput]:
    """Generate N diverse descriptions of an exam image.

    Loops N times with temperature sampling to produce diverse descriptions.
    Each description captures a different reading of the image content.

    Returns:
        List of N `GenerationOutput` objects (text + per-call metrics).
    """
    prompt = build_describe_prompt(language)

    def _sample_batch() -> list[GenerationOutput]:
        # One batched vLLM call produces N samples sharing prefill + image encode.
        # HF backend falls back to N sequential calls via the dispatcher.
        outs = generate_n(
            model, processor, prompt, n,
            image=image,
            temperature=temperature,
            max_new_tokens=max_tokens,
            seed=seed,
        )
        if ctx is not None:
            # Emit one llm_span per sample for trace-tree compatibility with
            # the unbatched path (Phoenix / W&B). Per-sample latency_s is the
            # amortized per-sample share of the batch wall-clock.
            for i, output in enumerate(outs):
                with ctx.llm_span(
                    "description.sample",
                    idx=i,
                    model=model_name,
                    temperature=temperature,
                    seed=seed,
                    prompt=prompt,
                ) as leaf:
                    ctx.set_llm_output(
                        leaf, output.text,
                        prompt_tokens=output.prompt_tokens,
                        completion_tokens=output.completion_tokens,
                        latency_s=output.latency_s,
                        logprob=output.logprob,
                    )
        return outs

    if ctx is not None:
        with ctx.stage_span("describe", n_samples=n):
            descriptions = _sample_batch()
    else:
        descriptions = _sample_batch()

    # Sanity: warn if any description is empty or very short
    for i, out in enumerate(descriptions):
        if len(out.text.strip()) < 20:
            logger.warning(
                "Description {} is very short ({} chars): {}",
                i, len(out.text), out.text[:50],
            )

    logger.info(
        "Generated {} descriptions (lengths: {})",
        len(descriptions),
        [len(d.text) for d in descriptions],
    )

    return descriptions
