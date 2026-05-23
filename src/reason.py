"""Stage 2: M-sample text-only reasoning.

Generates M reasoning chains per description. The reasoner receives only
the caption text (no image), mirroring MSA's architecture.

When called through `RunContext`, emits one parent `reason` span plus
N*M `reasoning.chain` LLM spans with OpenInference input/output attributes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

from loguru import logger

from src.backend import GenerationOutput, generate_n
from src.utils import Candidate, extract_answer

if TYPE_CHECKING:
    from src.utils.logger import RunContext

# Strict letter-only prompt (higher accuracy per MSA ablation)
REASON_STRICT_TEMPLATE = (
    "You are given a multiple-choice question extracted from an exam.\n"
    "The question description is:\n{caption}\n\n"
    "1. Identify the main question being asked.\n"
    "2. Extract all available answer options.\n"
    "3. Analyze all information including any described visual data.\n"
    "4. Select the correct answer based on your analysis.\n\n"
    "Your response MUST be ONLY the single letter of the correct answer "
    "[A, B, C, D, E]. No explanation or other text."
)

# CoT prompt, default closer (kept for backwards compat; high parse-fail)
REASON_COT_TEMPLATE = (
    "You are given a multiple-choice question extracted from an exam.\n"
    "The question description is:\n{caption}\n\n"
    "Think step by step. First analyze the question, then reason through "
    "each option.\nEnd with: The answer is <letter>."
)

# CoT prompt, MMMU closer (recommended; cuts parse-fail from ~9% to <2%)
REASON_COT_MMMU_TEMPLATE = (
    "You are given a multiple-choice question extracted from an exam.\n"
    "The question description is:\n{caption}\n\n"
    "Think step by step. First analyze the question, then reason through "
    "each option.\nEnd your answer with: Answer: X (where X is A, B, C, D, "
    "or E)."
)

_REASON_COT_STYLES = {
    "default": REASON_COT_TEMPLATE,
    "mmmu": REASON_COT_MMMU_TEMPLATE,
}


def build_reason_prompt(
    caption: str, use_cot: bool = True, cot_style: str = "default",
) -> str:
    """Build the Stage 2 reasoning prompt from a description."""
    if not use_cot:
        return REASON_STRICT_TEMPLATE.format(caption=caption)
    template = _REASON_COT_STYLES.get(cot_style, REASON_COT_TEMPLATE)
    return template.format(caption=caption)


def _as_text(desc: str | GenerationOutput) -> str:
    return desc.text if isinstance(desc, GenerationOutput) else desc


def reason(
    model,
    processor,
    description: str | GenerationOutput,
    description_id: int,
    m: int = 4,
    temperature: float = 0.7,
    max_tokens: int = 512,
    use_cot: bool = True,
    cot_style: str = "default",
    *,
    ctx: "RunContext | None" = None,
    model_name: str = "",
    seed: int = 0,
) -> list[Candidate]:
    """Generate M reasoning chains for a single description.

    Text-only call (no image). Loops M times with temperature sampling.
    Accepts either a raw string or a `GenerationOutput` (whose `.text` is
    used) for convenience when chaining from `describe()`.
    """
    description_text = _as_text(description)
    prompt = build_reason_prompt(description_text, use_cot=use_cot, cot_style=cot_style)

    # One batched vLLM call produces M chains sharing the text prefill. HF
    # backend falls back to M sequential calls via the dispatcher.
    outputs = generate_n(
        model, processor, prompt, m,
        image=None,
        temperature=temperature,
        max_new_tokens=max_tokens,
        seed=seed,
    )

    candidates: list[Candidate] = []
    for chain_id, output in enumerate(outputs):
        answer = extract_answer(output.text)
        if ctx is not None:
            with ctx.llm_span(
                "reasoning.chain",
                desc_idx=description_id,
                chain_idx=chain_id,
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
                    extracted_answer=answer,
                )

        candidates.append(Candidate(
            description_id=description_id,
            chain_id=chain_id,
            description=description_text,
            reasoning=output.text,
            answer=answer,
            logprob=output.logprob,
            prompt_tokens=output.prompt_tokens,
            completion_tokens=output.completion_tokens,
            latency_s=output.latency_s,
        ))

    n_failed = sum(1 for c in candidates if c.answer is None)
    if n_failed > 0:
        logger.warning(
            "Failed to extract answer from {}/{} chains (desc_id={})",
            n_failed, len(candidates), description_id,
        )

    return candidates


def reason_all(
    model,
    processor,
    descriptions: Sequence[str | GenerationOutput],
    m: int = 4,
    temperature: float = 0.7,
    max_tokens: int = 512,
    use_cot: bool = True,
    cot_style: str = "default",
    *,
    ctx: "RunContext | None" = None,
    model_name: str = "",
    seed: int = 0,
) -> list[Candidate]:
    """Generate M reasoning chains for each of N descriptions.

    Returns a flat list of N*M `Candidate` objects.
    """
    def _loop() -> list[Candidate]:
        all_candidates: list[Candidate] = []
        for desc_id, desc in enumerate(descriptions):
            all_candidates.extend(reason(
                model, processor, desc, desc_id,
                m=m, temperature=temperature, max_tokens=max_tokens,
                use_cot=use_cot, cot_style=cot_style,
                ctx=ctx, model_name=model_name, seed=seed,
            ))
        return all_candidates

    if ctx is not None:
        with ctx.stage_span(
            "reason",
            m_samples=m,
            n_descriptions=len(descriptions),
        ):
            all_candidates = _loop()
    else:
        all_candidates = _loop()

    valid = sum(1 for c in all_candidates if c.answer is not None)
    logger.info(
        "Generated {} candidates ({} with valid answers) from {} descriptions",
        len(all_candidates), valid, len(descriptions),
    )
    return all_candidates
