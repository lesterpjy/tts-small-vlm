"""PRM-BAS: beam-annealing search over reasoning steps.

Reimplemented from Hu et al. (arXiv 2504.10222). Wide beam at early
reasoning steps (B0), shrinks as PRM score spread exceeds threshold tau.

Step scorer protocol: any callable (list[str], str) -> float that scores
a reasoning step given the prior steps. Image and question context are
captured in the closure at construction time (see make_*_step_scorer
factories below).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

from loguru import logger

from src.backend import GenerationOutput, generate_n
from src.reason import build_reason_prompt
from src.utils import Candidate, extract_answer

if TYPE_CHECKING:
    from PIL import Image

StepScorer = Callable[[list[str], str], float]


@dataclass
class Beam:
    """A single beam in the PRM-BAS search tree."""

    steps: list[str] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    logprobs: list[float] = field(default_factory=list)
    terminated: bool = False

    @property
    def final_score(self) -> float:
        return sum(self.scores) / len(self.scores) if self.scores else 0.0

    @property
    def reasoning(self) -> str:
        return "\n\n".join(self.steps)

    @property
    def depth(self) -> int:
        return len(self.steps)


def _check_terminated(text: str, stop_marker: str) -> bool:
    return stop_marker.lower() in text.lower()


def prm_bas(
    model,
    processor,
    description: str,
    step_scorer: StepScorer,
    *,
    image: "Image.Image | None" = None,
    B0: int = 8,
    B: int = 4,
    anneal_tau: float = 0.05,
    max_depth: int = 10,
    stop_marker: str = "answer:",
    temperature: float = 0.7,
    max_step_tokens: int = 256,
    use_cot: bool = True,
    cot_style: str = "mmmu",
    seed: int | None = None,
) -> tuple[list[Beam], dict]:
    """Beam-annealing search over CoT reasoning steps.

    Per-description search: builds a reasoning prompt from the
    description, generates step-by-step with beam pruning guided by
    step_scorer scores, and returns terminal beams sorted by final_score.

    Args:
        model, processor: Policy model for step generation.
        description: Stage-1 caption (question context for reasoning).
        step_scorer: Callable (prior_steps, current_step) -> float.
        image: Passed to generate_n only if the model needs it for
            reasoning (default None = text-only, matching flat DTR).
        B0: Initial beam width (number of step-1 samples).
        B: Expansion width (continuations per surviving beam per depth).
        anneal_tau: Score-spread threshold for halving beam width.
        max_depth: Maximum reasoning depth (steps).
        stop_marker: Substring indicating a terminal reasoning step.
        temperature: Sampling temperature for step generation.
        max_step_tokens: Max tokens per reasoning step.
        use_cot, cot_style: Reasoning prompt configuration.
        seed: RNG seed for reproducibility.

    Returns:
        (terminals, metadata) where terminals is a list of Beam sorted
        descending by final_score, and metadata records search statistics.
    """
    base_prompt = build_reason_prompt(description, use_cot=use_cot, cot_style=cot_style)

    W = B0
    total_scorer_calls = 0
    total_generate_calls = 0
    width_schedule: list[int] = []
    terminals: list[Beam] = []

    # --- Depth 0: sample B0 first steps ---
    step1_outs = generate_n(
        model, processor, base_prompt, B0,
        image=image,
        temperature=temperature,
        max_new_tokens=max_step_tokens,
        stop=["\n\n"],
        seed=seed,
    )
    total_generate_calls += 1

    actives: list[Beam] = []
    for out in step1_outs:
        step_text = out.text.strip()
        if not step_text:
            continue
        score = step_scorer([], step_text)
        total_scorer_calls += 1
        beam = Beam(
            steps=[step_text],
            scores=[score],
            logprobs=[out.logprob if out.logprob is not None else 0.0],
            terminated=_check_terminated(step_text, stop_marker),
        )
        if beam.terminated:
            terminals.append(beam)
        else:
            actives.append(beam)

    W = _anneal(actives, W, anneal_tau)
    width_schedule.append(W)
    actives = _top_w(actives, W)

    # --- Depths 1..max_depth-1 ---
    for depth in range(1, max_depth):
        if not actives:
            break

        last_depth = (depth == max_depth - 1)

        new_beams: list[Beam] = []
        for beam in actives:
            prefix = base_prompt + "\n\n" + beam.reasoning + "\n\n"
            step_outs = generate_n(
                model, processor, prefix, B,
                image=image,
                temperature=temperature,
                max_new_tokens=max_step_tokens * 4 if last_depth else max_step_tokens,
                stop=None if last_depth else ["\n\n"],
                seed=seed,
            )
            total_generate_calls += 1

            for out in step_outs:
                step_text = out.text.strip()
                if not step_text:
                    continue
                score = step_scorer(beam.steps, step_text)
                total_scorer_calls += 1
                new_beams.append(Beam(
                    steps=[*beam.steps, step_text],
                    scores=[*beam.scores, score],
                    logprobs=[*beam.logprobs, out.logprob if out.logprob is not None else 0.0],
                    terminated=_check_terminated(step_text, stop_marker),
                ))

        terminals.extend(b for b in new_beams if b.terminated)
        actives = [b for b in new_beams if not b.terminated]

        W = _anneal(actives, W, anneal_tau)
        width_schedule.append(W)
        actives = _top_w(actives, W)

    # Force-terminate remaining actives at max_depth
    for b in actives:
        b.terminated = True
    terminals.extend(actives)

    terminals.sort(key=lambda b: b.final_score, reverse=True)

    metadata = record_prm_bas_metadata(terminals, width_schedule,
                                        total_scorer_calls, total_generate_calls)
    logger.debug(
        "PRM-BAS: {} terminals, {} scorer calls, width schedule {}",
        len(terminals), total_scorer_calls, width_schedule,
    )
    return terminals, metadata


def _anneal(actives: list[Beam], W: int, tau: float) -> int:
    """Halve W if the score spread among top-W actives exceeds tau."""
    if len(actives) < 2:
        return W
    top_scores = sorted((b.scores[-1] for b in actives), reverse=True)[:W]
    if len(top_scores) < 2:
        return W
    if top_scores[0] - top_scores[-1] > tau:
        return max(W // 2, 1)
    return W


def _top_w(actives: list[Beam], W: int) -> list[Beam]:
    """Keep the top-W beams by latest step score."""
    actives.sort(key=lambda b: b.scores[-1], reverse=True)
    return actives[:W]


def record_prm_bas_metadata(
    terminals: list[Beam],
    width_schedule: list[int],
    total_scorer_calls: int,
    total_generate_calls: int,
) -> dict:
    """Build metadata dict for JSONL logging."""
    return {
        "total_scorer_calls": total_scorer_calls,
        "total_generate_calls": total_generate_calls,
        "width_schedule": width_schedule,
        "n_terminals": len(terminals),
        "terminal_scores": [b.final_score for b in terminals],
        "max_depth_reached": max((b.depth for b in terminals), default=0),
    }


def beams_to_candidates(
    beams: list[Beam],
    description_id: int,
    description: str,
) -> list[Candidate]:
    """Convert terminal beams to Candidate objects for Stage 3 verify."""
    candidates: list[Candidate] = []
    for i, beam in enumerate(beams):
        reasoning = beam.reasoning
        answer = extract_answer(reasoning)
        mean_lp = sum(beam.logprobs) / len(beam.logprobs) if beam.logprobs else None
        candidates.append(Candidate(
            description_id=description_id,
            chain_id=i,
            description=description,
            reasoning=reasoning,
            answer=answer,
            logprob=mean_lp,
        ))
    return candidates


# --- Step scorer factories ---


def make_prm_step_scorer(
    prm_model,
    prm_processor,
    prm_tokenizer,
    image: "Image.Image",
    question: str,
) -> StepScorer:
    """Build a step scorer backed by Qwen-VL-PRM-7B.

    One PRM forward pass per call, scoring the full chain up to and
    including the current step and returning P(+) at the final position.
    """
    from src.verify import _qwen_vl_prm_forward_score

    def scorer(prior_steps: list[str], current_step: str) -> float:
        solution = "\n\n".join([*prior_steps, current_step])
        return _qwen_vl_prm_forward_score(
            prm_model=prm_model,
            prm_processor=prm_processor,
            prm_tokenizer=prm_tokenizer,
            image=image,
            question=question,
            solution=solution,
        )

    return scorer


def make_generative_step_scorer(
    model,
    processor,
    image: "Image.Image",
) -> StepScorer:
    """Build a step scorer backed by the generative critic (training-free).

    Scores the full accumulated reasoning chain on three axes using the
    policy backbone. Expensive (~3 generate calls per score); use
    Qwen-VL-PRM when available.
    """
    from src.verify import generative_critic_score

    def scorer(prior_steps: list[str], current_step: str) -> float:
        reasoning = "\n\n".join([*prior_steps, current_step])
        candidate = Candidate(
            description_id=0, chain_id=0,
            description="", reasoning=reasoning, answer=None,
        )
        return generative_critic_score(model, processor, image, candidate)

    return scorer
